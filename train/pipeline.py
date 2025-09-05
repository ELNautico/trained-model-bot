"""
train/pipeline.py
End-to-end pipeline with automatic *feature pruning*.

Flow
----
1. Download & cache OHLCV
2. Feature-engineer full set
3. If a pruned-feature file exists → use it
4. Hyper-parameter tuning (walk-forward, via transformer_utils)
5. Sensitivity analysis → drop weakest P% features
6. Retrain ensemble on pruned features
7. Forecast price & σ̂ and forward them to risk module
"""
from __future__ import annotations
from core.features import enrich_features, get_feature_columns

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import toml
from sklearn.preprocessing import MinMaxScaler
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from twelvedata import TDClient

from core.features import enrich_features, get_feature_columns
from core.risk import generate_certificate
from core.sensitivity import compute_feature_sensitivity
from train.core import load_model, predict_price, train_and_save_model
from train.evaluate import backtest_strategy

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
_cfg_path = Path(__file__).resolve().parent.parent / "config.toml"
api_key = toml.load(_cfg_path)["twelvedata"]["api_key"]
td = TDClient(apikey=api_key)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

CACHE_TTL = timedelta(hours=24)
# Drop only truly absurd daily jumps (likely bad data). Adjusted prices should
# already account for splits/dividends; keep this generous to avoid losing days.
MAX_RETURN_THRESHOLD = 0.50
MAX_STALE_DAYS = 2
PRUNE_PCT = 30                      # bottom X % features to drop

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers ­– I/O
# ─────────────────────────────────────────────────────────────────────────────
def _pruned_path(ticker: str) -> Path:
    """models/TKR/LATEST_pruned_features.json"""
    return Path("models") / ticker / "LATEST_pruned_features.json"


def load_pruned_features(ticker: str) -> list[str] | None:
    path = _pruned_path(ticker)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            logging.warning("⚠️  Failed to read pruned-feature list for %s", ticker)
    return None


def save_pruned_features(ticker: str, pruned: list[str], version_dir: Path):
    latest = _pruned_path(ticker)
    try:
        latest.write_text(json.dumps(pruned, indent=2))
        (version_dir / "pruned_features.json").write_text(json.dumps(pruned, indent=2))
    except Exception as e:
        logging.warning("⚠️  Could not persist pruned features: %s", e)

# ─────────────────────────────────────────────────────────────────────────────
#  Raw data download + clean
# ─────────────────────────────────────────────────────────────────────────────
@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
)
def _download_raw_ohlcv(ticker: str) -> pd.DataFrame:
    try:
        df = td.time_series(
            symbol=ticker,
            interval="1day",
            outputsize=5000,
            timezone="UTC",
            adjustment="all",  # use adjusted data to avoid split/dividend jumps
        ).as_pandas()
    except TypeError:
        # Some versions of twelvedata client don't accept 'adjustment' kwarg.
        logging.warning("⚠️  TDClient.time_series doesn't support 'adjustment'; falling back to unadjusted data.")
        df = td.time_series(
            symbol=ticker,
            interval="1day",
            outputsize=5000,
            timezone="UTC",
        ).as_pandas()
    if df.empty:
        raise ValueError(f"Empty OHLCV for {ticker}")
    df.columns = [c.capitalize() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def cached_download(ticker: str) -> pd.DataFrame:
    key = hashlib.md5(ticker.encode()).hexdigest()
    path = CACHE_DIR / f"{key}.pkl"

    if path.exists() and datetime.utcnow() - datetime.utcfromtimestamp(path.stat().st_mtime) < CACHE_TTL:
        df = joblib.load(path)
    else:
        df = _download_raw_ohlcv(ticker)
        joblib.dump(df, path)

    # Soft filter for extreme outliers; log how many rows would be dropped.
    df["ret"] = df["Close"].pct_change()
    bad_mask = df["ret"].abs() > MAX_RETURN_THRESHOLD
    bad_count = int(bad_mask.sum())
    if bad_count:
        logging.warning("⚠️  %s: dropping %d rows with > %.0f%% daily change",
                        ticker, bad_count, MAX_RETURN_THRESHOLD * 100)
        df = df.loc[~bad_mask]
    df = df.drop(columns="ret")

    # Do not drop rows after long gaps; retain data and let models handle.
    return df

# ─────────────────────────────────────────────────────────────────────────────
#  Sequence builder (d1 & d5 only)         << UPDATED
# ─────────────────────────────────────────────────────────────────────────────
def build_sequences(
    df: pd.DataFrame,
    feats: list[str],
    *,
    window_size: int,
    test_ratio: float = 0.2,
):
    close = df["Close"].values

    df = df[feats].ffill().bfill()

    split = int(len(df) * (1 - test_ratio))
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    scaler = MinMaxScaler()
    train_sc = scaler.fit_transform(train_df)
    test_sc  = scaler.transform(test_df)
    sc_all   = np.vstack([train_sc, test_sc])

    X_tr, X_te = [], []
    y1_tr, y5_tr = [], []
    y1_te, y5_te = [], []

    last_idx = len(df) - 5

    def _append(X, y1, y5, i):
        window = sc_all[i - window_size : i]
        # Only append if window is correct shape
        if window.shape == (window_size, len(feats)):
            X.append(window)
            y1.append(np.log(close[i] / close[i - 1]))        # 1-day return
            y5.append(np.log(close[i + 4] / close[i - 1]))    # 5-day return

    for i in range(window_size, split - 5):
        _append(X_tr, y1_tr, y5_tr, i)
    for i in range(split, last_idx):
        _append(X_te, y1_te, y5_te, i)

    y_train = {"d1": np.array(y1_tr), "d5": np.array(y5_tr)}
    y_test  = {"d1": np.array(y1_te), "d5": np.array(y5_te)}

    return (
        np.array(X_tr),
        y_train,
        np.array(X_te),
        y_test,
        scaler,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def train_predict_for_ticker(
    ticker: str,
    use_ensemble: bool = True,      # ← works both positionally *and* as keyword
    account_balance: float = 100_000.0,
    risk_per_trade: float = 0.01,
):
    logging.info("🚀 %s – pipeline start", ticker)

    df = cached_download(ticker)
    # Pass ticker to DataFrame attrs for sentiment feature
    df.attrs['ticker'] = ticker
    df = enrich_features(df)

    # ── feature subset (may be pruned)
    # Find latest model version directory
    model_base = Path("models") / ticker
    import re
    ts_pattern = re.compile(r"^\d{8}_\d{6}$")
    versions = [d for d in model_base.iterdir() if d.is_dir() and ts_pattern.match(d.name)]
    versions = sorted(versions, reverse=True)
    if not versions:
        logging.error(f"❌ No versioned subdirectory found for {ticker} in models/")
        raise FileNotFoundError(f"No versioned subdirectory found for {ticker} in models/")
    latest_version_dir = versions[0]
    # Load actual feature list from metadata.json in latest model dir
    metadata_path = latest_version_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        # If features are saved as a list of names, use them; if shape, fallback to get_feature_columns()
        feats_in_use = metadata.get("feature_names") if "feature_names" in metadata else get_feature_columns()
    else:
        feats_in_use = get_feature_columns()
    win = 60 if len(df) >= 600 else 30
    X_tr, y_tr, X_te, y_te, scaler = build_sequences(df, feats_in_use, window_size=win)

    # ── model load only
    model_base = Path("models") / ticker
    if not model_base.exists():
        logging.error(f"❌ No model directory found for {ticker} in models/")
        raise FileNotFoundError(f"No model directory found for {ticker} in models/")
    # Only use timestamped subdirectories (ignore files like LATEST_pruned_features.json)
    import re
    ts_pattern = re.compile(r"^\d{8}_\d{6}$")
    versions = [d for d in model_base.iterdir() if d.is_dir() and ts_pattern.match(d.name)]
    versions = sorted(versions, reverse=True)
    if not versions:
        logging.error(f"❌ No versioned subdirectory found for {ticker} in models/")
        raise FileNotFoundError(f"No versioned subdirectory found for {ticker} in models/")
    expected_model_path = versions[0] / "model.h5"
    logging.info(f"🔎 Checking for model at: {expected_model_path}")
    if expected_model_path.exists():
        logging.info(f"✅ Found model for {ticker} at {expected_model_path}")
        try:
            latest_version = versions[0].name
            model = load_model(ticker, version_timestamp=latest_version)
            logging.info(f"📦 model loaded from {expected_model_path}")
        except FileNotFoundError as e:
            logging.error(f"❌ Model loading failed: {e}")
            raise
    else:
        logging.error(f"❌ Model file missing at {expected_model_path}")
        raise FileNotFoundError(f"Model file missing at {expected_model_path}")

    # ── forecast
    current_px = float(df["Close"].iloc[-1])
    last_seq = X_te[-1:][...]
    pred_px, pred_vol = predict_price(model, last_seq, current_px)
    final_px = 0.9 * pred_px + 0.1 * current_px if use_ensemble else pred_px

    hist_vol = float(df["Close"].pct_change().std())
    vol_used = float(np.clip(pred_vol, 0.5 * hist_vol, 2.0 * hist_vol))

    pct_change = (final_px / current_px - 1) * 100
    confidence = round(min(abs(pct_change) * 10, 100), 1)
    direction = "Buy" if pct_change > 0 else "Sell" if pct_change < 0 else "Hold"
    acc, cum_ret = backtest_strategy(model, X_te, df, win)

    ts_local = df.index[-1].tz_convert(ZoneInfo("Europe/Vienna"))
    cert = generate_certificate(current_px, final_px, vol_used)
    sens_out = compute_feature_sensitivity(model, last_seq, feats_in_use)

    return (
        {
            "Ticker": ticker,
            "Timestamp": ts_local.strftime("%Y-%m-%d %H:%M %Z"),
            "Current Price": round(current_px, 2),
            "Predicted Price": round(final_px, 2),
            "Predicted % Change": round(pct_change, 2),
            "Confidence": confidence,
            "Volatility": round(vol_used, 4),
            "Directional Accuracy": round(acc, 4),
            "Cumulative Return": round(cum_ret, 4),
            "Recommended Certificate": cert,
            "Feature Sensitivity": sens_out,
            "Feature Count": len(feats_in_use),
        },
        df,
    )

# --------------------------------------------------------------------------
# Back-compat aliases (used by jobs.py  and Telegram-Bot)
# --------------------------------------------------------------------------

def prepare_data_and_split(
    df: pd.DataFrame,
    *,
    window_size: int = 60,
    test_ratio: float = 0.2,
):
    """
    Legacy-Wrapper für alten Code (retrain, Bot).
    Rechnet Features, ruft build_sequences() und liefert
    dieselbe 6-teilige Tuple-Struktur wie früher zurück.
    """
    df = enrich_features(df)

    # Ensure all required feature columns exist, fill missing with NaN
    for col in get_feature_columns():
        if col not in df.columns:
            df[col] = np.nan
    feats = [f for f in get_feature_columns() if f in df.columns]
    X_tr, y_tr, X_te, y_te, scaler = build_sequences(
        df,
        feats,
        window_size=window_size,
        test_ratio=test_ratio,
    )
    train_size = len(X_tr)
    return X_tr, y_tr, X_te, y_te, scaler, train_size


# keep old alias
download_data = cached_download
