# jobs.py

import logging
import sqlite3
import time
import pathlib
from datetime import datetime

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
        time.sleep(3)
        try:
            result, _ = train_predict_for_ticker(
                ticker,
                use_ensemble=True,
                account_balance=ACCT_BAL,
                risk_per_trade=RISK,
            )
            res = {
                "Current Price": result["Current Price"],
                "Predicted Price": result["Predicted Price"],
                "Predicted % Change": result["Predicted % Change"],
                "Confidence": result["Confidence"],
            }

            direction = (
                "Buy" if res["Predicted % Change"] > 0
                else "Sell" if res["Predicted % Change"] < 0
                else "Hold"
            )

            row = dict(
                ts=datetime.utcnow().date().isoformat(),
                ticker=ticker,
                current_px=res["Current Price"],
                predicted_px=res["Predicted Price"],
                direction=direction,
                confidence=res["Confidence"],
                model_tag=MODEL_TAG,
            )
            save_forecast(row)
            update_watchlist_timestamp(ticker, "last_forecast")
            send(
                f"📈 {ticker}: {direction}\n"
                f"Predicted Close: {res['Predicted Price']:.2f} "
                f"(conf {res['Confidence']:.1f}%)"
            )
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

            eval_row = dict(
                ts=today,
                ticker=tkr,
                predicted_px=predicted_px,
                actual_px=actual_close,
                error=error,
                pct_error=pct_error,
                model_tag=model_tag,
            )
            save_evaluation(eval_row)

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

# ─────────────────────────────────────────────────────────────────────────────
def retrain_job(force: bool = False):
    MODEL_DIR.mkdir(exist_ok=True)
    for ticker in get_watchlist():
        logging.info(f"🔄 Retraining model for {ticker}")

        # 1) Load or prepare data
        try:
            df = download_data(ticker)
            X_train, y_train, *_ = prepare_data_and_split(df, window_size=60)
        except Exception as e:
            logging.error(f"❌ Data prep failed for {ticker}: {e}")
            continue

        if len(X_train) < 100:
            logging.warning(f"⚠️ Not enough samples for {ticker}, skipping.")
            continue

        # 2) Conditional skip based on recent error
        if not force:
            recent = get_recent_errors(5)
            avg_err = sum(recent) / len(recent) if recent else 0.0
            if avg_err < ERROR_THRESHOLD:
                logging.info(f"{ticker}: avg error {avg_err:.2f}% < threshold {ERROR_THRESHOLD}% → skipping retrain.")
                continue

        model_path = MODEL_DIR / f"{ticker}_model.h5"

        # 3) Full retrain if missing or forced
        if not model_path.exists() or force:
            model, _, _ = train_and_save_model(
                X_train, y_train, X_train.shape[1:], ticker
            )
            model.save(model_path)
            update_watchlist_timestamp(ticker, "last_trained")
            send(f"✅ {ticker}: Full retrain complete.")
            continue

        # 4) Otherwise fine-tune existing model
        try:
            model = load_model(ticker)
            logging.info(f"✏️ Fine-tuning existing model for {ticker}")

            # Compile with a lower LR
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
def help_job():
    help_text = (
        "🧠 *Available Commands:*\n\n"
        "📈 /forecast – Run model and send today's market predictions.\n\n"
        "📊 /evaluate – Compare today's forecasts to actual prices.\n\n"
        "🔄 /retrain – Retrain models only if missing.\n\n"
        "⚠️ /retrain_force – Force retraining of all models, even if already trained.\n\n"
        "📥 /add TICKER – Add a stock ticker to your watchlist (e.g. `/add AAPL`).\n\n"
        "🗑️ /remove TICKER – Remove a stock from your watchlist.\n\n"
        "📋 /watchlist – Show all currently watched tickers.\n\n"
        "❓ /help – Show this list of available commands.\n\n"
        "📅 *Note:* All times are UTC and forecasts are stored in the database."
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
    else:
        print("Usage: jobs.py [forecast|evaluate|retrain|retrain_force|help]")

if __name__ == "__main__":
    _cli()
