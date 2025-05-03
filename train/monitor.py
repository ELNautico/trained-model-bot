# train/monitor.py

import logging
import time
from core.risk import determine_trade_signal, determine_position_size
from train.pipeline import train_predict_for_ticker


def monitor_and_update_model(
    ticker,
    iterations=1,
    update_interval_seconds=10,
    account_balance=100_000,
    risk_per_trade=0.01
):
    """
    Monitoring loop to:
    - Download latest data
    - Retrain model if needed
    - Log updated prediction, confidence, and trading signals
    """
    for i in range(iterations):
        try:
            logging.info(f"🔄 Iteration {i + 1} for ticker: {ticker}")
            results, _ = train_predict_for_ticker(
                ticker,
                use_ensemble=True,
                account_balance=account_balance,
                risk_per_trade=risk_per_trade
            )

            # --- Log Key Metrics ---
            logging.info("🔍 Latest Forecast Summary:")
            for key, value in results.items():
                logging.info(f"  {key}: {value}")

            print(f"\n🔔 Updated Results for {ticker} (Iteration {i + 1})")
            for key, value in results.items():
                print(f"  {key}: {value}")

        except Exception as e:
            logging.error(f"❌ Error during iteration {i + 1} for {ticker}: {e}")

        if i < iterations - 1:
            logging.info(f"🕒 Sleeping {update_interval_seconds} seconds before next update...\n")
            time.sleep(update_interval_seconds)
