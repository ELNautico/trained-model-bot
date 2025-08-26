from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def backtest_strategy(model: tf.keras.Model, X_test, data: pd.DataFrame, window_size: int):
    """
    Given a model that forecasts *log-return*, compute

      • directional accuracy
      • cumulative (log) return of a naïve long/short strategy
      • returns a tuple  (accuracy, cumulative_return)
    """
    log_rets_hat = model.predict(X_test, verbose=0)[0].flatten()

    start_idx = len(data) - (len(X_test) + window_size)
    test_idx = np.arange(start_idx + window_size, len(data))

    prev_px = data["Close"].iloc[test_idx - 1].values  # P(t)
    actual_px = data["Close"].iloc[test_idx].values    # P(t+1)
    pred_px = prev_px * np.exp(log_rets_hat)           # P̂(t+1)

    fee = 0.0002
    actual_ret = actual_px / prev_px - 1.0 - fee
    pred_ret = pred_px / prev_px - 1.0  # what the model *thinks* will happen

    accuracy = np.mean(np.sign(actual_ret) == np.sign(pred_ret))

    # naïve “go long if model says up, go short if down”
    strategy_ret = np.where(pred_ret > 0, actual_ret, -actual_ret)
    strategy_ret = np.clip(strategy_ret, -0.99, 0.99)  # guard rails
    cum_return = np.expm1(np.log1p(strategy_ret).sum())

    logging.info(f"🎯 Directional accuracy: {accuracy:.2%}")
    logging.info(f"📈 Cumulative return  : {cum_return:.2%}")

    return float(accuracy), float(cum_return)


def plot_predictions(pred_px: np.ndarray, actual_px: np.ndarray):
    plt.figure(figsize=(12, 5))
    plt.plot(actual_px, label="Actual", lw=2)
    plt.plot(pred_px, label="Predicted", lw=2)
    plt.title("Predicted vs. Actual Close")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
