# jobs.py

import logging
import sqlite3
import os
import pathlib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

from storage import init_db, save_forecast, save_evaluation, DB
from alert import send
from train.pipeline import train_predict_for_ticker, download_data
from train.core import train_and_save_model
from core.utils import timestamp_now
from core.indicators import add_basic_indicators
from train.pipeline import prepare_data_and_split
from train.core import predict_price
from train.evaluate import backtest_strategy

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = {
    "DAX": "^GDAXI",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "ATX": "^ATX",
}
MODEL_TAG = "v2025Q2"
ACCT_BAL = 100_000
RISK = 0.01
MODEL_DIR = pathlib.Path("models")

CALENDARS = {
    "^GSPC": "NYSE",
    "^IXIC": "NASDAQ",
    "^GDAXI": "XETR",
    "^ATX": "XETR",
}

# ─────────────────────────────────────────────────────────────────────────────
# Check if all exchanges are open today
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
# FORECAST JOB
# ─────────────────────────────────────────────────────────────────────────────
def forecast_job():
    if not is_trading_day_all():
        logging.info("Market closed today – skipping forecast job.")
        send("🛑 Market closed today – skipping forecast job.")
        return

    for label, tkr in TICKERS.items():
        try:
            result, _ = train_predict_for_ticker(
                tkr,
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
        except ValueError as e:
            if "Not enough data" in str(e):
                logging.warning(f"{tkr}: fallback to SMA-20 due to insufficient data.")
                data = download_data(tkr)
                data = add_basic_indicators(data)
                sma20 = data["SMA_20"].iloc[-1]
                current = float(data["Close"].iloc[-1])

                ts = data.index[-1]
                ts = ts if ts.tzinfo else ts.replace(tzinfo=ZoneInfo("UTC"))
                ts_local = ts.astimezone(ZoneInfo("Europe/Vienna"))

                res = {
                    "Current Price": current,
                    "Predicted Price": float(sma20),
                    "Predicted % Change": 0.0,
                    "Confidence": 0.0,
                }
            else:
                raise

        direction = (
            "Buy" if res["Predicted % Change"] > 0
            else "Sell" if res["Predicted % Change"] < 0
            else "Hold"
        )

        row = dict(
            ts=datetime.utcnow().date().isoformat(),
            ticker=tkr,
            current_px=res["Current Price"],
            predicted_px=res["Predicted Price"],
            direction=direction,
            confidence=res["Confidence"],
            model_tag=MODEL_TAG,
        )
        save_forecast(row)
        send(
            f"📈 {label}: {row['direction']}\n"
            f"Predicted Closing Price: {row['predicted_px']:.2f} "
            f"(conf {row['confidence']:.1f}%)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE JOB
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_job():
    if not is_trading_day_all():
        logging.info("Market closed today – skipping forecast job.")
        send("🛑 Market closed today – skipping forecast job.")
        return

    init_db()
    today = datetime.utcnow().date().isoformat()

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT ticker, predicted_px, model_tag FROM forecast WHERE ts = ?", (today,))
    forecasts = cur.fetchall()
    conn.close()

    for tkr, predicted_px, model_tag in forecasts:
        try:
            data = download_data(tkr)
            actual_close = float(data["Close"].iloc[-1])
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
            logging.info(f"Saved evaluation for {tkr} – error {error:.2f} ({pct_error:.1f}%)")
        except Exception as e:
            logging.error(f"❌ Evaluation failed for {tkr}: {e}")
            send(f"❌ Evaluation failed for {tkr}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# RETRAIN JOB
# ─────────────────────────────────────────────────────────────────────────────
def retrain_job(force: bool = False):
    MODEL_DIR.mkdir(exist_ok=True)
    for label, tkr in TICKERS.items():
        logging.info(f"🔄 Retraining model for {label} ({tkr})")
        try:
            data = download_data(tkr)
            X, y, scaler = prepare_data_and_split(data, window_size=60)[0:3]
        except Exception as e:
            logging.error(f"❌ Failed to prepare data for {tkr}: {e}")
            continue

        if len(X) < 100:
            logging.warning(f"⚠️ Not enough samples to train {tkr}, skipping.")
            continue

        model_path = MODEL_DIR / f"{tkr}_model.h5"
        if model_path.exists() and not force:
            logging.info(f"✔️ Model for {tkr} already exists. Skipping (use force=True to retrain).")
            continue

        try:
            model, _, _ = train_and_save_model(X, y, X.shape[1:], tkr)
            model.save(model_path)
            send(f"✅ {label}: Model retrained and saved.")
        except Exception as e:
            logging.error(f"❌ Training error for {tkr}: {e}")
            send(f"❌ Error retraining {label}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# HELP COMMAND JOB
# ─────────────────────────────────────────────────────────────────────────────
def help_job():
    help_text = (
        "🧠 *Available Commands:*\n\n"
        "/forecast – Run model and send today's market predictions.\n"
        "/evaluate – Compare today's forecasts to actual prices.\n"
        "/retrain – Force retraining of all models.\n"
        "/retrain_force – Same as retrain, but overrides existing models.\n"
        "/help – Show this list of available commands.\n"
        "\nAll times are UTC and predictions are saved to the database."
    )
    send(help_text)

# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRYPOINT
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
