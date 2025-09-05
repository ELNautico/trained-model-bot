import pandas as pd
import numpy as np
from train.pipeline import download_data

def baseline_next_day_close(ticker="AAPL"):
    df = download_data(ticker)
    df = df.sort_index()
    closes = df["Close"].values
    # Predict next day's close as today's close (naive baseline)
    pred = closes[:-1]
    actual = closes[1:]
    dates = df.index[1:]
    errors = pred - actual
    pct_errs = (errors / actual) * 100
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(pct_errs))
    rmse = np.sqrt(np.mean(errors**2))
    # Directional accuracy: does sign of (actual - today) match sign of (today - yesterday)?
    if len(closes) > 2:
        prev_changes = closes[1:-1] - closes[:-2]  # today - yesterday
        next_changes = closes[2:] - closes[1:-1]   # tomorrow - today
        dir_acc = np.mean(np.sign(next_changes) == np.sign(prev_changes))
    else:
        dir_acc = np.nan
    n_big_errs = int((np.abs(pct_errs) > 10).sum())
    avg_pct = float(np.mean(np.abs(pct_errs)))
    std_pct = float(np.std(np.abs(pct_errs)))
    worst_idx = np.argsort(-np.abs(pct_errs))[:10]
    worst_dates = [str(dates[i]) for i in worst_idx]
    worst_vals = [pct_errs[i] for i in worst_idx]
    print(f"\n=== Baseline: Next Day = Today Close ({ticker}) ===")
    print(f"Days: {len(actual)}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.4f}")
    print(f"Dir Acc: {dir_acc:.2%}")
    print(f"Avg abs % error: {avg_pct:.2f}%, Std: {std_pct:.2f}%")
    print(f"Days >10% abs error: {n_big_errs}")
    print(f"Worst 3 dates: {[(d, f'{v:.2f}%') for d, v in zip(worst_dates, worst_vals)]}")

if __name__ == "__main__":
    baseline_next_day_close()
