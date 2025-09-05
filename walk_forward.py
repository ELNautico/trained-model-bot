from core.features import enrich_features
## FOR LATER BACKTESTING (SEE PERFORMANCE)

import sys
import numpy as np
import matplotlib.pyplot as plt
from train.pipeline import download_data, prepare_data_and_split
from train.core     import train_and_save_model, load_model, predict_price
import logging
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from alert import send
import tensorflow as tf

def train_in_memory(X_train, y_train, input_shape, ticker):
    # Minimal model for walk-forward: same as train_and_save_model, but does NOT save to disk
    # You can customize this to match your actual model architecture
    from tensorflow.keras import layers, models, optimizers
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(1)
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='mse')
    # Use only d1 for walk-forward
    y = y_train["d1"] if isinstance(y_train, dict) and "d1" in y_train else y_train
    model.fit(X_train, y, epochs=5, batch_size=32, verbose=0)
    # Always return a single value for d1 prediction
    def predict_d1(model, X_last, prev_close):
        pred = model.predict(X_last, verbose=0)
        # If pred is shape (1,1), flatten
        if hasattr(pred, 'shape') and pred.shape == (1, 1):
            pred = pred[0, 0]
        elif hasattr(pred, 'shape') and pred.shape[0] == 1:
            pred = pred[0]
        return float(prev_close * np.exp(pred))
    model._predict_d1 = lambda X_last, prev_close: predict_d1(model, X_last, prev_close)
    return model, None, None

def walk_forward(
    ticker: str,
    window_size: int = 60,
    retrain_every: int = 10  # set >1 to only retrain periodically
):
    df = download_data(ticker)
    closes = df["Close"].values

    errors, pct_errs, dirs = [], [], []
    model = None
    last_retrain_idx = None

    # Start after enough history
    total_steps = len(df) - (window_size + 5)
    for i, t in enumerate(range(window_size + 5, len(df))):
        if i % 100 == 0:
            print(f"Progress: t={t} ({i+1}/{total_steps})")
        sub = df.iloc[:t]
        if len(sub) < window_size + 5:
            continue

        sub = enrich_features(sub)
        if sub is None or sub.shape[0] == 0:
            continue
        X_train, y_train, X_test, y_test, scaler, _ = prepare_data_and_split(sub, window_size=window_size)

        # skip if X_train is empty or 1D
        if X_train.shape[0] == 0 or len(X_train.shape) < 2:
            continue

        # retrain on first step or every N steps
        if model is None or ((t - (window_size+5)) % retrain_every == 0):
            if len(X_train.shape) == 3:
                input_shape = X_train.shape[1:]
            else:
                input_shape = (window_size, X_train.shape[1])
            # Use in-memory training for validation (no disk writes)
            model, _hist, _hp = train_in_memory(X_train, y_train, input_shape, ticker)
            last_retrain_idx = t

        # If model is None, skip prediction and metrics for this window
        if model is None:
            continue

        # predict next day (the last row of X_test)
        # Use custom d1 prediction for in-memory model, else fallback to predict_price
        if hasattr(model, '_predict_d1'):
            pred = model._predict_d1(X_test[-1:], closes[t-1])
        else:
            pred, _ = predict_price(model, X_test[-1:], closes[t-1])
        actual = closes[t]

        err = pred - actual
        errors.append(err)
        pct_errs.append(err / actual * 100)
        dirs.append((pred - closes[t-1]) * (actual - closes[t-1]) > 0)


    errors   = np.array(errors)
    pct_errs = np.array(pct_errs)
    dir_acc  = np.mean(dirs)

    # Diagnostics for troubleshooting
    abs_pct_errs = np.abs(pct_errs)
    worst_idx = np.argsort(-abs_pct_errs)[:3]  # indices of 3 worst errors
    # Map t index to date
    error_dates = [df.index[window_size + 5 + i] for i in range(len(errors))]
    worst_dates = [str(error_dates[i]) for i in worst_idx]
    worst_vals = [pct_errs[i] for i in worst_idx]
    n_big_errs = int((abs_pct_errs > 10).sum())
    avg_pct = float(np.mean(abs_pct_errs))
    std_pct = float(np.std(abs_pct_errs))

    summary = (
        f"\n=== Walk-Forward Backtest: {ticker} ===\n"
        f"Days simulated: {len(errors)}\n"
        f"MAE:  {np.mean(np.abs(errors)):.4f}\n"
        f"MAPE: {np.mean(np.abs(pct_errs)):.2f}%\n"
        f"RMSE: {np.sqrt(np.mean(errors**2)):.4f}\n"
        f"Dir Acc: {dir_acc:.2%}\n"
        f"Window size: {window_size}, Retrain every: {retrain_every}\n"
        f"Avg abs % error: {avg_pct:.2f}%, Std: {std_pct:.2f}%\n"
        f"Days >10% abs error: {n_big_errs}\n"
        f"Worst 3 dates: {[(d, f'{v:.2f}%') for d, v in zip(worst_dates, worst_vals)]}\n"
    )
    print(summary)

    # Log to file
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"walk_forward_{ticker}.log")
    with open(log_path, "a") as f:
        f.write(summary)

    # Send Telegram alert if performance is poor, with diagnostics
    alert_needed = False
    alert_reasons = []
    if avg_pct > 3.0:
        alert_needed = True
        alert_reasons.append(f"MAPE > 3% ({avg_pct:.2f}%)")
    if dir_acc < 0.55:
        alert_needed = True
        alert_reasons.append(f"Dir Acc < 55% ({dir_acc:.2%})")
    if alert_needed:
        msg = (
            f"🚨 Walk-forward validation for {ticker} signals model review needed!\n"
            f"Reasons: {', '.join(alert_reasons)}\n"
            f"Window: {window_size}, Retrain every: {retrain_every}\n"
            f"MAE: {np.mean(np.abs(errors)):.4f}, MAPE: {avg_pct:.2f}%, Dir Acc: {dir_acc:.2%}\n"
            f"Avg abs % error: {avg_pct:.2f}%, Std: {std_pct:.2f}%\n"
            f"Days >10% abs error: {n_big_errs}\n"
            f"Worst 3 dates: {[(d, f'{v:.2f}%') for d, v in zip(worst_dates, worst_vals)]}"
        )
        send(msg)

    # error distribution
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    plot_path = os.path.join(log_dir, f"walk_forward_{ticker}_error_hist.png")
    plt.figure()
    plt.hist(pct_errs, bins=30)
    plt.title(f"{ticker} % Error Distribution")
    plt.xlabel("% Error")
    plt.ylabel("Frequency")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved error histogram to {plot_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python walk_forward.py TICKER [WINDOW] [RETRAIN_EVERY]")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    period = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    walk_forward(ticker, window, period)
