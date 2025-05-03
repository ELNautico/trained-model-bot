# train/pipeline.py

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from core.indicators import add_basic_indicators, add_richer_features, get_feature_columns

import tensorflow as tf
import yfinance as yf

from train.core import train_and_save_model, load_model, predict_price
from train.evaluate import backtest_strategy
from core.indicators import get_feature_columns
from core.sensitivity import compute_feature_sensitivity
from core.risk import generate_certificate
from core.utils import directional_accuracy
from storage import save_forecast, save_evaluation


def prepare_data_and_split(df: pd.DataFrame, window_size: int = 60, test_ratio: float = 0.2):
    """
    Adds indicators, scales, splits, and builds LSTM sequences.
    Returns X_train, y_train, X_test, y_test, scaler, train_size.
    """
    df = add_basic_indicators(df)
    df = add_richer_features(df)

    feature_cols = get_feature_columns()
    df = df[feature_cols].copy().ffill().bfill().dropna()

    if len(df) < window_size + 5:
        raise ValueError(f"Not enough data to build sequences. Got only {len(df)} rows after cleaning.")

    split_idx = int(len(df) * (1 - test_ratio))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    # Fit scaler only on training data
    scaler = MinMaxScaler()
    scaler.fit(train_df)
    train_scaled = scaler.transform(train_df)
    test_scaled = scaler.transform(test_df)

    def build_sequences(scaled_data, unscaled_df):
        X, y = [], []
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i - window_size:i])
            y.append(unscaled_df.iloc[i]['Close'])  # Unscaled close as target
        return np.array(X), np.array(y)

    X_train, y_train = build_sequences(train_scaled, train_df)
    X_test, y_test = build_sequences(test_scaled, test_df)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Train/Test sequences are empty. Check window size or data size.")

    return X_train, y_train, X_test, y_test, scaler, len(X_train)


def download_data(ticker, years=30):
    end = datetime.today()
    start = end - timedelta(days=years * 365)
    df = yf.download(
        ticker,
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        interval='1d',
        progress=False
    )
    df = df[~df.index.duplicated()]
    return df


def train_predict_for_ticker(ticker, use_ensemble=True, account_balance=100_000, risk_per_trade=0.01):
    logging.info(f"🚀 Starting full pipeline for {ticker}")
    df = download_data(ticker)

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

    # Clamp extreme jumps
    hist_vol = float(df['Close'].pct_change().std())
    clamp = min(hist_vol * 2, 0.05)
    pmin, pmax = current_price * (1 - clamp), current_price * (1 + clamp)
    final_price = float(np.clip(ensemble_pred, pmin, pmax))

    # Backtest for accuracy stats
    acc, cum_ret = backtest_strategy(model, X_test, scaler, df, window_size)

    # Timestamp conversion
    ts = df.index[-1]
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=ZoneInfo("UTC"))
    ts = ts.astimezone(ZoneInfo("Europe/Vienna"))

    pct_change = (final_price / current_price - 1) * 100
    confidence = round(min(abs(pct_change) * 10, 100), 1)
    direction = "Buy" if pct_change > 0 else "Sell" if pct_change < 0 else "Hold"
    cert = generate_certificate(current_price, final_price, hist_vol)
    sensitivity = compute_feature_sensitivity(model, last_seq, get_feature_columns())

    # ✅ Save forecast
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

    # ✅ Save evaluation (current price as the closest proxy to actual, for now)
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
