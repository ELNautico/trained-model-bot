import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import toml
from sklearn.preprocessing import MinMaxScaler
from twelvedata import TDClient
import joblib

from core.features import enrich_features, get_feature_columns
from core.sensitivity import compute_feature_sensitivity
from core.risk import generate_certificate
from core.utils import directional_accuracy
from train.core import train_and_save_model, load_model, predict_price
from train.evaluate import backtest_strategy
from storage import save_forecast, save_evaluation

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
config_path = Path(__file__).resolve().parent.parent / "config.toml"
api_key = toml.load(config_path)["twelvedata"]["api_key"]

td = TDClient(apikey=api_key)
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Cache settings
CACHE_TTL = timedelta(hours=24)
MAX_RETURN_THRESHOLD = 0.1   # 10% daily return
MAX_STALE_DAYS = 2

# ─────────────────────────────────────────────────────────────────────────────
# DATA HANDLING
# ─────────────────────────────────────────────────────────────────────────────


def _download_and_cache(ticker: str, path: Path) -> pd.DataFrame:
    """
    Internal helper: download, normalize, cache, return DataFrame.
    """
    df = td.time_series(
        symbol=ticker,
        interval="1day",
        outputsize=5000,
        timezone="UTC"
    ).as_pandas()

    if df.empty:
        raise ValueError(f"Downloaded data for {ticker} is empty.")

    # Standardize column names
    df.columns = [col.strip().lower() for col in df.columns]
    df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }, inplace=True)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    logging.info(f"✅ Downloaded and normalized data for {ticker}")

    joblib.dump(df, path)
    return df


def cached_download(ticker: str) -> pd.DataFrame:
    """
    Downloads OHLCV data, caching to disk with a 24h TTL. Also removes outliers and stale rows.
    """
    key = hashlib.md5(ticker.encode()).hexdigest()
    path = CACHE_DIR / f"{key}.pkl"

    # Load from cache if fresh
    if path.exists():
        mtime = datetime.utcfromtimestamp(path.stat().st_mtime)
        if datetime.utcnow() - mtime < CACHE_TTL:
            logging.info(f"📦 Loaded cached data for {ticker}")
            df = joblib.load(path)
        else:
            logging.info(f"🕑 Cache expired for {ticker}, re-downloading...")
            df = _download_and_cache(ticker, path)
    else:
        df = _download_and_cache(ticker, path)

    # Outlier detection: drop extreme returns
    df['return'] = df['Close'].pct_change()
    outliers = df['return'].abs() > MAX_RETURN_THRESHOLD
    if outliers.any():
        logging.warning(f"⚠️ Dropping {outliers.sum()} outlier rows for {ticker}")
        df = df.loc[~outliers]

    # Staleness detection: drop gaps > MAX_STALE_DAYS
    gaps = df.index.to_series().diff()
    stale = gaps > pd.Timedelta(days=MAX_STALE_DAYS)
    if stale.any():
        logging.warning(f"⚠️ Dropping {stale.sum()} stale rows for {ticker}")
        df = df.loc[~stale]

    df.drop(columns=['return'], inplace=True)
    return df


def prepare_data_and_split(df: pd.DataFrame, window_size: int = 60, test_ratio: float = 0.2):
    """
    Enrich features, apply caching outlier/stale-cleaned data,
    then scale inputs+target with a single MinMaxScaler for stability.

    Returns:
        X_train, y_train, X_test, y_test, scaler, train_size
    """
    # Feature engineering
    df = enrich_features(df)

    # Subset and clean
    feature_cols = get_feature_columns()
    df = df[feature_cols].copy().ffill().bfill().dropna()

    if len(df) < window_size + 5:
        raise ValueError(
            f"Not enough data to build sequences. Got only {len(df)} rows after cleaning."
        )

    # Train/test split
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Scale both features+target
    scaler = MinMaxScaler()
    scaler.fit(train_df)
    train_scaled = scaler.transform(train_df)
    test_scaled = scaler.transform(test_df)

    # Identify target column index
    target_idx = feature_cols.index('Close')

    # Build sequences for X and y (scaled target)
    def build_sequences(scaled_array):
        X, y = [], []
        for i in range(window_size, len(scaled_array)):
            X.append(scaled_array[i - window_size:i])
            y.append(scaled_array[i][target_idx])
        return np.array(X), np.array(y)

    X_train, y_train = build_sequences(train_scaled)
    X_test, y_test = build_sequences(test_scaled)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "Train/Test sequences are empty. Check window size or data size."
        )

    return X_train, y_train, X_test, y_test, scaler, len(X_train)

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def train_predict_for_ticker(ticker, use_ensemble=True, account_balance=100_000, risk_per_trade=0.01):
    logging.info(f"🚀 Starting full pipeline for {ticker}")
    df = cached_download(ticker)

    if len(df) < 100:
        raise ValueError(f"Not enough data for {ticker}. Only {len(df)} rows downloaded.")

    window_size = 60 if len(df) >= 600 else 30
    X_train, y_train, X_test, y_test, scaler, train_size = prepare_data_and_split(df, window_size)

    try:
        model = load_model(ticker)
        logging.info("📦 Loaded existing model.")
    except FileNotFoundError:
        logging.info("🛠️ No saved model found. Training from scratch...")
        model, _, _ = train_and_save_model(X_train, y_train, X_train.shape[1:], ticker)

    last_seq = X_test[-1:]
    pred_price = predict_price(model, last_seq, scaler)
    current_price = float(df['Close'].iloc[-1])
    ensemble_pred = 0.9 * pred_price + 0.1 * current_price if use_ensemble else pred_price

    hist_vol = float(df['Close'].pct_change().std())
    clamp = min(hist_vol * 2, 0.05)
    pmin, pmax = current_price * (1 - clamp), current_price * (1 + clamp)
    final_price = float(np.clip(ensemble_pred, pmin, pmax))

    acc, cum_ret = backtest_strategy(model, X_test, scaler, df, window_size)

    ts = df.index[-1]
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=ZoneInfo("UTC"))
    ts = ts.astimezone(ZoneInfo("Europe/Vienna"))

    pct_change = (final_price / current_price - 1) * 100
    confidence = round(min(abs(pct_change) * 10, 100), 1)
    direction = "Buy" if pct_change > 0 else "Sell" if pct_change < 0 else "Hold"
    cert = generate_certificate(current_price, final_price, hist_vol)
    sensitivity = compute_feature_sensitivity(model, last_seq, get_feature_columns())

    try:
        save_forecast({
            "ts": datetime.utcnow().strftime("%Y-%m-%d"),
            "ticker": ticker,
            "current_px": current_price,
            "predicted_px": final_price,
            "direction": direction,
            "confidence": confidence,
            "model_tag": "v2025Q2"
        })
        logging.info("📩 Saved forecast to DB.")
    except Exception as e:
        logging.warning(f"⚠️ Failed to save forecast: {e}")

    try:
        save_evaluation({
            "ts": datetime.utcnow().strftime("%Y-%m-%d"),
            "ticker": ticker,
            "predicted_px": final_price,
            "actual_px": current_price,
            "error": final_price - current_price,
            "pct_error": abs(final_price - current_price) / current_price * 100,
            "model_tag": "v2025Q2"
        })
        logging.info("📊 Saved evaluation to DB.")
    except Exception as e:
        logging.warning(f"⚠️ Failed to save evaluation: {e}")

    return {
        "Ticker": ticker,
        "Timestamp": ts.strftime('%Y-%m-%d %H:%M %Z'),
        "Current Price": round(current_price, 2),
        "Predicted Price": round(final_price, 2),
        "Predicted % Change": round(pct_change, 2),
        "Confidence": confidence,
        "Volatility": round(hist_vol, 4),
        "Directional Accuracy": round(acc, 4),
        "Cumulative Return": round(cum_ret, 4),
        "Recommended Certificate": cert,
        "Feature Sensitivity": sensitivity
    }, df

# Alias to support jobs.py imports
download_data = cached_download
