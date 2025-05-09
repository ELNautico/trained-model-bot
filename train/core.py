import os
import json
import numpy as np
import tensorflow as tf
import logging
from pathlib import Path
from datetime import datetime

from core.build import tune_model

def train_and_save_model(X_train, y_train, input_shape, ticker):
    """
    Tunes and trains model, then versions it under
    models/{ticker}/{YYYYMMDD_HHMMSS}/model.h5
    and writes hyperparams JSON there.
    """
    logging.info(f"🔧 Tuning model for {ticker}...")
    model, best_hp = tune_model(X_train, y_train, input_shape, project_name=f"lstm_{ticker}")

    logging.info("📈 Training final model with best hyperparameters...")
    # add TensorBoard for the final training, too
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / "training" / ticker / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1)

    callbacks = [
        tb_cb,
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=30,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # versioned save
    model_version_dir = Path("models") / ticker / timestamp
    model_version_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_version_dir / "model.h5"
    model.save(model_path)
    logging.info(f"✅ Model saved to {model_path}")

    # store best hyperparameters
    hp_path = model_version_dir / "hyperparameters.json"
    with open(hp_path, "w") as f:
        json.dump(best_hp.values, f, indent=4)
    logging.info(f"📝 Hyperparameters written to {hp_path}")

    return model, history, best_hp


def load_model(ticker, version_timestamp: str = None):
    """
    Load a specific version, or the latest if version_timestamp is None.
    """
    base = Path("models") / ticker
    if version_timestamp:
        model_path = base / version_timestamp / "model.h5"
    else:
        # pick the newest folder by timestamp
        versions = sorted(base.iterdir(), reverse=True)
        model_path = versions[0] / "model.h5"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return tf.keras.models.load_model(model_path)


def predict_price(model, X_last, scaler):
    """
    Makes price prediction using the final sequence.
    Returns predicted price and rescaled version.
    """
    pred_scaled = model.predict(X_last, verbose=0)[0][0]
    cmin, cmax = scaler.data_min_[3], scaler.data_max_[3]
    return pred_scaled * (cmax - cmin) + cmin