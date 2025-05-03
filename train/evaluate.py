import numpy as np
import matplotlib.pyplot as plt
import logging


def backtest_strategy(model, X_test, scaler, data, window_size):
    """
    Evaluates strategy with directional accuracy and cumulative return.
    """
    predicted = model.predict(X_test, verbose=0).flatten()

    # Rescale predictions to original Close prices
    cmin, cmax = scaler.data_min_[3], scaler.data_max_[3]
    predicted_prices = predicted * (cmax - cmin) + cmin

    start_idx = len(data) - (len(X_test) + window_size)
    test_indices = np.arange(start_idx + window_size, len(data))

    actual_prices = data['Close'].iloc[test_indices].values
    prev_prices = data['Close'].iloc[test_indices - 1].values

    actual_returns = (actual_prices / prev_prices) - 1.0
    predicted_returns = (predicted_prices / actual_prices) - 1.0

    # Directional accuracy
    accuracy = np.mean(np.sign(predicted_returns) == np.sign(actual_returns))

    # Strategy returns
    strategy_returns = np.where(predicted_returns > 0, actual_returns, -actual_returns)
    strategy_returns = np.clip(strategy_returns, -0.99, 0.99)
    log_returns = np.log1p(strategy_returns)
    cum_return = np.expm1(np.sum(log_returns))

    logging.info(f"🎯 Directional Accuracy: {accuracy:.2%}")
    logging.info(f"📈 Cumulative Return: {cum_return:.2%}")

    return accuracy, cum_return


def plot_predictions(predicted_prices, actual_prices):
    """
    Plots predicted vs. actual prices.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(actual_prices, label="Actual", linewidth=2)
    plt.plot(predicted_prices, label="Predicted", linewidth=2)
    plt.title("Predicted vs Actual Close Prices")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
