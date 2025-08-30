from core.features import enrich_features
## FOR LATER BACKTESTING (SEE PERFORMANCE)

import sys
import numpy as np
import matplotlib.pyplot as plt
from train.pipeline import download_data, prepare_data_and_split
from train.core     import train_and_save_model, load_model, predict_price

def walk_forward(
    ticker: str,
    window_size: int = 30,
    retrain_every: int = 1  # set >1 to only retrain periodically
):
    df = download_data(ticker)
    closes = df["Close"].values

    errors, pct_errs, dirs = [], [], []
    model = None
    last_retrain_idx = None

    # Start after enough history
    for t in range(window_size + 5, len(df)):
        sub = df.iloc[:t]
        if len(sub) < window_size + 5:
            print(f"Skipping t={t}: sub too small for window_size ({len(sub)} < {window_size + 5})")
            continue
        sub = enrich_features(sub)
        print(f"sub columns at t={t}:", sub.columns)
        print(f"sub shape at t={t}:", sub.shape)
        print(f"Any NaNs at t={t}:", sub.isnull().any().any())
        if sub is None or sub.shape[0] == 0:
            print(f"Skipping t={t}: feature engineering resulted in empty DataFrame.")
            continue
        # Pass raw DataFrame slice to prepare_data_and_split
        print(f"sub columns at t={t}:", sub.columns)
        X_train, y_train, X_test, y_test, scaler, _ = prepare_data_and_split(sub, window_size=window_size)

        # skip if X_train is empty or 1D
        if X_train.shape[0] == 0 or len(X_train.shape) < 2:
            print(f"Skipping t={t}: X_train.shape={X_train.shape}")
            continue

        # retrain on first step or every N steps
        if model is None or ((t - (window_size+5)) % retrain_every == 0):
            if len(X_train.shape) == 3:
                input_shape = X_train.shape[1:]
            else:
                input_shape = (window_size, X_train.shape[1])
            model, _hist, _hp = train_and_save_model(X_train, y_train, input_shape, ticker)
            last_retrain_idx = t

        # If model is None, skip prediction and metrics for this window
        if model is None:
            print(f"Skipping prediction at t={t}: model not trained due to insufficient data.")
            continue

        # predict next day (the last row of X_test)
        pred = predict_price(model, X_test[-1:], closes[t-1])
        actual = closes[t]

        err = pred - actual
        errors.append(err)
        pct_errs.append(err / actual * 100)
        dirs.append((pred - closes[t-1]) * (actual - closes[t-1]) > 0)

    errors   = np.array(errors)
    pct_errs = np.array(pct_errs)
    dir_acc  = np.mean(dirs)

    print(f"\n=== Walk-Forward Backtest: {ticker} ===")
    print(f"Days simulated: {len(errors)}")
    print(f"MAE:  {np.mean(np.abs(errors)):.4f}")
    print(f"MAPE: {np.mean(np.abs(pct_errs)):.2f}%")
    print(f"RMSE: {np.sqrt(np.mean(errors**2)):.4f}")
    print(f"Dir Acc: {dir_acc:.2%}")

    # error distribution
    plt.hist(pct_errs, bins=30)
    plt.title(f"{ticker} % Error Distribution")
    plt.xlabel("% Error")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python walk_forward.py TICKER [WINDOW] [RETRAIN_EVERY]")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    period = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    walk_forward(ticker, window, period)
