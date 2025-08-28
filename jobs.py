import logging
import sqlite3
import time
import pathlib
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

from storage import (
    init_db,
    save_forecast,
    save_evaluation,
    DB,
    get_watchlist,
    update_watchlist_timestamp,
    get_recent_errors,
)
from alert import send
from train.pipeline import train_predict_for_ticker, prepare_data_and_split, download_data
from train.core import train_and_save_model, load_model
from tensorflow.keras.optimizers import Adam
from mlops.monitor import drift_job
import tensorflow as tf

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_TAG = "v2025Q2"
ACCT_BAL = 100_000
RISK = 0.01
MODEL_DIR = pathlib.Path("models")
ERROR_THRESHOLD = 2.0  # Only fine-tune if avg pct_error over recent days ≥ 2%

CALENDARS = {
    "SPX": "NYSE",
    "NAS100": "NASDAQ",
    "DEU40": "XETR",
    "ATX": "XETR",
}

# ─────────────────────────────────────────────────────────────────────────────
def is_trading_day_all() -> bool:
    today = datetime.utcnow().date()
    for cal_name in set(CALENDARS.values()):
        cal = mcal.get_calendar(cal_name)
        sched = cal.schedule(start_date=today, end_date=today)
        if sched.empty:
            return False
    return True

# ─────────────────────────────────────────────────────────────────────────────
def forecast_job():
    if not is_trading_day_all():
        logging.info("Market closed today – skipping forecast job.")
        send("🛑 Market closed today – skipping forecast job.")
        return

    for ticker in get_watchlist():
        time.sleep(10)

        # 1) Notify if no versioned model exists
        version_dir = MODEL_DIR / ticker
        if not (version_dir.exists() and any(version_dir.iterdir())):
            send(f"🛠️ No model found for {ticker}. Training a new one now…")

        try:
            # 2) Run pipeline (it will load or train as needed)
            result, _ = train_predict_for_ticker(
                ticker,
                use_ensemble=True,
                account_balance=ACCT_BAL,
                risk_per_trade=RISK,
            )

            current   = result["Current Price"]
            predicted = result["Predicted Price"]
            conf      = result["Confidence"]
            pct_chg   = result["Predicted % Change"]

            # 3) Compute recommendation
            direction = "Buy" if pct_chg > 0 else "Sell" if pct_chg < 0 else "Hold"

            # 4) Persist forecast
            save_forecast({
                "ts": datetime.utcnow().date().isoformat(),
                "ticker": ticker,
                "current_px": current,
                "predicted_px": predicted,
                "direction": direction,
                "confidence": conf,
                "model_tag": MODEL_TAG,
            })
            update_watchlist_timestamp(ticker, "last_forecast")

            # 5) Local‐time formatting
            now   = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC")) \
                                      .astimezone(ZoneInfo("Europe/Vienna"))
            ts_str = now.strftime("%d.%m at %H:%M")

            # 6) Send final message
            msg = (
                f"📈 Stock: {ticker}\n"
                f"📊 Prediction on {ts_str}\n"
                f"📋 Current Price: {current:.2f}\n"
                f"🧠 Predicted Close: {predicted:.2f} ({conf:.1f}%)\n\n"
                f"⚠️ ️Recommendation:\n"
                f"{direction}"
            )
            send(msg)

        except Exception as e:
            logging.error(f"{ticker}: Forecast failed – {e}")
            send(f"❌ {ticker}: Forecast failed – {e}")

# ─────────────────────────────────────────────────────────────────────────────
def evaluate_job():
    if not is_trading_day_all():
        logging.info("Market closed today – skipping evaluation job.")
        send("🛑 Market closed today – skipping evaluation job.")
        return

    init_db()
    today = datetime.utcnow().date().isoformat()

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(
        "SELECT ticker, predicted_px, model_tag FROM forecast WHERE ts = ?", (today,)
    )
    forecasts = cur.fetchall()
    conn.close()

    for tkr, predicted_px, model_tag in forecasts:
        try:
            df = download_data(tkr)
            actual_close = float(df["Close"].iloc[-1])
            error = actual_close - predicted_px
            pct_error = (error / actual_close) * 100

            save_evaluation({
                "ts": today,
                "ticker": tkr,
                "predicted_px": predicted_px,
                "actual_px": actual_close,
                "error": error,
                "pct_error": pct_error,
                "model_tag": model_tag,
            })
            send(
                f"✅ {tkr} Evaluation:\n"
                f"Predicted: {predicted_px:.2f}, Actual: {actual_close:.2f}\n"
                f"Error: {error:.2f} ({pct_error:.1f}%)"
            )
            logging.info(
                f"Saved evaluation for {tkr} – error {error:.2f} ({pct_error:.1f}%)"
            )
        except Exception as e:
            logging.error(f"❌ Evaluation failed for {tkr}: {e}")
            send(f"❌ Evaluation failed for {tkr}: {e}")

    # Run drift detection after evaluation
    logging.info("🔎 Running drift detection after evaluation job...")
    drift_job()

# ─────────────────────────────────────────────────────────────────────────────
def retrain_job(force: bool = False):
    MODEL_DIR.mkdir(exist_ok=True)
    for ticker in get_watchlist():
        logging.info(f"🔄 Retraining model for {ticker}")

        # 1) Prepare data
        try:
            df = download_data(ticker)
            X_train, y_train, *_ = prepare_data_and_split(df, window_size=60)
        except Exception as e:
            logging.error(f"❌ Data prep failed for {ticker}: {e}")
            continue

        if len(X_train) < 100:
            logging.warning(f"⚠️ Not enough samples for {ticker}, skipping.")
            continue

        # 2) Error‐based skip
        if not force:
            recent = get_recent_errors(5)
            avg_err = (sum(recent)/len(recent)) if recent else 0.0
            if avg_err < ERROR_THRESHOLD:
                logging.info(
                    f"{ticker}: avg error {avg_err:.2f}% < threshold {ERROR_THRESHOLD}% → skipping retrain."
                )
                continue

        model_path = MODEL_DIR / f"{ticker}_model.h5"

        # 3) Full retrain
        if not model_path.exists() or force:
            model, _, _ = train_and_save_model(
                X_train, y_train, X_train.shape[1:], ticker
            )
            model.save(model_path)
            update_watchlist_timestamp(ticker, "last_trained")
            send(f"✅ {ticker}: Full retrain complete.")
            continue

        # 4) Fine‐tune
        try:
            model = load_model(ticker)
            logging.info(f"✏️ Fine-tuning existing model for {ticker}")
            if not model.loss:  # legacy model saved w/ compile=False
                model.compile(
                    optimizer="adam",
                    loss={"d1": tf.keras.losses.MeanSquaredError(),
                          "d5": tf.keras.losses.MeanSquaredError()},
                    metrics={"d1": [tf.keras.metrics.MeanAbsoluteError()],
                             "d5": [tf.keras.metrics.MeanAbsoluteError()]},
                )
            model.compile(
                optimizer=Adam(learning_rate=1e-5),
                loss=model.loss,
                metrics=model.metrics
            )
            history = model.fit(
                X_train, y_train,
                epochs=5,
                validation_split=0.1,
                verbose=1
            )
            model.save(model_path)
            update_watchlist_timestamp(ticker, "last_trained")
            send(f"✅ {ticker}: Fine-tune complete ({len(history.history['loss'])} epochs).")
        except Exception as e:
            logging.error(f"❌ Fine-tune failed for {ticker}: {e}")
            send(f"❌ Fine-tune failed for {ticker}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
from mlops.utils import retrain_job  # Importing retrain_job from mlops/utils.py

def help_job():
    help_text = (
        "🧠 *Available Commands:*\n\n"
        "📈 /forecast – Run model and send today's market predictions.\n\n"
        "📊 /evaluate – Compare today's forecasts to actual prices.\n\n"
        "🔄 /retrain – Retrain models only if missing.\n\n"
        "⚠️ /retrain_force – Force retraining of all models, even if trained.\n\n"
        "🚨 /drift – Run drift-detection; retrains only models that show data drift.\n\n"
        "📥 /add TICKER – Add a stock ticker to your watchlist.\n\n"
        "🗑️ /remove TICKER – Remove a stock from your watchlist.\n\n"
        "📋 /watchlist – Show your current watchlist.\n\n"
        "❓ /help – Show this message."
    )
    send(help_text)

# ─────────────────────────────────────────────────────────────────────────────
def _cli():
    import sys
    init_db()
    job = sys.argv[1] if len(sys.argv) > 1 else None

    if job == "forecast":
        forecast_job()
    elif job == "evaluate":
        evaluate_job()
    elif job == "retrain":
        retrain_job(force=False)
    elif job == "retrain_force":
        retrain_job(force=True)
    elif job == "help":
        help_job()
    elif job == "drift":
        drift_job()
    else:
        print("Usage: jobs.py [forecast|evaluate|retrain|retrain_force|help|drift]")
