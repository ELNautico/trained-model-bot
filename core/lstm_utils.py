# core/lstm_utils.py
"""
Legacy helpers required by core/build.py
────────────────────────────────────────
    • estimate_huber_delta(y_train)
    • build_lstm_model(hp, input_shape, huber_delta)

They implement a single-output LSTM forecaster (next-day log-return).
Nothing in the Transformer pipeline imports this file.
"""

from __future__ import annotations
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers
import keras_tuner as kt


# ── robust δ for Huber loss ────────────────────────────────────────────────
def estimate_huber_delta(y_train: np.ndarray) -> float:
    med = np.median(y_train)
    mad = np.median(np.abs(y_train - med))
    return float(5.0 * mad if mad > 0 else 1e-3)   # fallback if MAD==0


# ── tiny LSTM factory compatible with core/build.py ────────────────────────
def build_lstm_model(
    hp: kt.HyperParameters,
    input_shape,
    huber_delta: float | None = None,
):
    model = tf.keras.Sequential(name="lstm_forecaster")
    model.add(layers.Input(shape=input_shape))

    # first LSTM
    units = hp.Int("units", 32, 128, step=32)
    model.add(layers.LSTM(units, return_sequences=hp.Boolean("use_second_lstm")))
    model.add(layers.Dropout(hp.Float("dropout_rate", 0.0, 0.5, step=0.1)))

    # optional second LSTM
    if hp.get("use_second_lstm"):
        units2 = hp.Int("units2", 32, 128, step=32)
        model.add(layers.LSTM(units2))
        model.add(layers.Dropout(hp.get("dropout_rate")))

    # output
    model.add(layers.Dense(1, name="log_return"))

    # compile
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    loss_fn = losses.Huber(delta=huber_delta) if huber_delta else "mse"

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=["mae", "mape"],
    )
    return model
