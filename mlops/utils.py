from train.pipeline import train_predict_for_ticker, prepare_data_and_split, download_data
from storage import get_watchlist, update_watchlist_timestamp, get_recent_errors
from train.core import train_and_save_model
from alert import send
import logging
import pathlib

MODEL_DIR = pathlib.Path("models")
ERROR_THRESHOLD = 2.0

# Utility retrain_job for use by both jobs.py and mlops.monitor

def retrain_job(force: bool = False):
    MODEL_DIR.mkdir(exist_ok=True)
    for ticker in get_watchlist():
        logging.info(f"🔄 Retraining model for {ticker}")
        try:
            df = download_data(ticker)
            X_train, y_train, *_ = prepare_data_and_split(df, window_size=60)
        except Exception as e:
            logging.error(f"❌ Data prep failed for {ticker}: {e}")
            continue
        if len(X_train) < 100:
            logging.warning(f"⚠️ Not enough samples for {ticker}, skipping.")
            continue
        if not force:
            recent = get_recent_errors(5)
            avg_err = (sum(recent)/len(recent)) if recent else 0.0
            if avg_err < ERROR_THRESHOLD:
                logging.info(
                    f"{ticker}: avg error {avg_err:.2f}% < threshold {ERROR_THRESHOLD}% → skipping retrain."
                )
                continue
        model_path = MODEL_DIR / f"{ticker}_model.h5"
        if not model_path.exists() or force:
            model, _, _ = train_and_save_model(
                X_train, y_train, X_train.shape[1:], ticker
            )
            model.save(model_path)
            update_watchlist_timestamp(ticker, "last_trained")
            send(f"✅ {ticker}: Full retrain complete.")
            continue
        try:
            # ...existing code for fine-tune...
            pass
        except Exception as e:
            logging.error(f"❌ Fine-tune failed for {ticker}: {e}")
            continue
