# jobs.py
from datetime import datetime
import logging
import sqlite3
import os
import pathlib

import pandas_market_calendars as mcal

from storage import init_db, save_forecast, save_evaluation, DB
from alert import send
from bot_core import (
    train_predict_for_ticker,
    prepare_data_and_split,
    tune_and_train_model,
    download_daily_data,
    prepare_data_full,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = {
    "S&P 500": "^GSPC",
    "DAX": "^GDAXI",
    "NASDAQ": "^IXIC",
    "ATX": "^ATX",
}
MODEL_TAG = "v2025Q2"
ACCT_BAL = 100_000
RISK = 0.01
MODEL_DIR = pathlib.Path(__file__).with_name("models")

# map each ticker to its exchange calendar
CALENDARS = {
    "^GSPC": "NYSE",
    "^IXIC": "NASDAQ",
    "^GDAXI": "XETR",
    "^ATX": "XETR",       # Vienna will use XETR calendar as a proxy
}


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: skip non‐trading days
# ─────────────────────────────────────────────────────────────────────────────
def is_trading_day_all() -> bool:
    today = datetime.utcnow().date()
    for cal_name in set(CALENDARS.values()):
        cal = mcal.get_calendar(cal_name)
        # get schedule for today
        sched = cal.schedule(start_date=today, end_date=today)
        if sched.empty:
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# JOBS
# ─────────────────────────────────────────────────────────────────────────────
def forecast_job():
    if not is_trading_day_all():
        logging.info("Market closed today – skipping forecast_job.")
        send("🛑 Market is closed today")
        return

    for label, tkr in TICKERS.items():
        res, _ = train_predict_for_ticker(
            tkr,
            use_ensemble=True,
            account_balance=ACCT_BAL,
            risk_per_trade=RISK,
        )
        row = dict(
            ts=datetime.utcnow().date().isoformat(),
            ticker=tkr,
            current_px=res["Current Price"],
            predicted_px=res["Predicted Price for Close"],
            direction=res["Trade Decision"],
            confidence=res["Signal Confidence"],
            model_tag=MODEL_TAG,
        )
        save_forecast(row)
        send(
            f"📈 {label}: {row['direction']}\n"
            f"Predicted Closing Price: {row['predicted_px']:.2f} (conf {row['confidence']:.1f}%)"
        )


def evaluate_job():
    if not is_trading_day_all():
        logging.info("Market closed today – skipping evaluate_job.")
        send("🛑 Market is closed today")
        return

    init_db()
    today = datetime.utcnow().date().isoformat()

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(
        "SELECT ticker, predicted_px, model_tag FROM forecast WHERE ts = ?",
        (today,),
    )
    forecasts = cur.fetchall()
    conn.close()

    for tkr, predicted_px, model_tag in forecasts:
        data = download_daily_data(tkr)
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
            f"✅ {tkr}: Predicted {predicted_px:.2f}, Actual {actual_close:.2f}, "
            f"Err {error:.2f} ({pct_error:.1f}%)"
        )


def retrain_job():
    MODEL_DIR.mkdir(exist_ok=True)
    for label, tkr in TICKERS.items():
        data = download_daily_data(tkr)
        X, y, scaler = prepare_data_full(data, window_size=60)

        model, _, best_hp = tune_and_train_model(
            X,
            y,
            input_shape=(X.shape[1], X.shape[2]),
            project_name=f"live_{tkr}",
        )
        tag = datetime.utcnow().strftime("%Y%m%d_%H%M")
        fname = MODEL_DIR / f"{tkr}_{tag}.h5"
        model.save(fname)

        import joblib

        joblib.dump(scaler, pathlib.Path(fname).with_suffix(".scaler"))
        send(f"🆕 {label} model retrained")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
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
        retrain_job()
    else:
        print("Usage: jobs.py [forecast|evaluate|retrain]")


if __name__ == "__main__":
    _cli()
