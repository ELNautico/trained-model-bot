import os
import numpy as np
import tensorflow as tf
import logging
from pathlib import Path
from datetime import datetime

from core.build import tune_model


def train_and_save_model(X_train, y_train, input_shape, ticker):
    """
    Tunes and trains model, saves the best one to disk.
    """
    logging.info(f"🔧 Tuning model for {ticker}...")
    model, best_hp = tune_model(X_train, y_train, input_shape, project_name=f"lstm_{ticker}")

    logging.info("📈 Training final model with best hyperparameters...")
    callbacks = [
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

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    path = model_dir / f"{ticker}_model.h5"
    model.save(path)
    logging.info(f"✅ Model saved to {path}")

    return model, history, best_hp


def load_model(ticker):
    """
    Loads a pre-trained model from disk.
    """
    path = Path("models") / f"{ticker}_model.h5"
    if not path.exists():
        raise FileNotFoundError(f"Model for {ticker} not found at {path}")
    model = tf.keras.models.load_model(path)
    return model


def predict_price(model, X_last, scaler):
    """
    Makes price prediction using the final sequence.
    Returns predicted price and rescaled version.
    """
    pred_scaled = model.predict(X_last, verbose=0)[0][0]
    cmin, cmax = scaler.data_min_[3], scaler.data_max_[3]
    predicted_price = pred_scaled * (cmax - cmin) + cmin
    return predicted_price
