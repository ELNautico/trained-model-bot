"""
walk_forward.py

Replaced: this is now a signal backtest driver (triple-barrier classifier + state machine).

Usage:
  python walk_forward.py TSLA
"""

import sys
import numpy as np

from signals.config import SignalConfig
from signals.model import SignalModel
from train.pipeline import download_data
from core.features import enrich_features


def quick_backtest_one_train(ticker: str, cfg: SignalConfig) -> str:
    """
    Minimal, honest baseline backtest:
      - train model on first (1 - test_ratio) portion
      - run state machine on holdout
      - report trade count and simple PnL

    This is not a production-grade backtester; it’s a sanity check.
    """
    df = download_data(ticker)
    df.attrs["ticker"] = ticker
    df = enrich_features(df)

    model, meta = SignalModel.train_from_df(ticker, df, cfg)

    # Build dataset again to get aligned decision dates, then split by time
    from signals.dataset import build_dataset
    ds = build_dataset(df, cfg)

    n = ds.X.shape[0]
    split = int(n * (1 - cfg.test_ratio))
    X_te = ds.X[split:]
    dates_te = ds.decision_dates[split:]

    # Simulate "flat/long" using predicted probs only (no stop/target simulation here),
    # just as a quick directional/edge proxy.
    pos = 0
    trades = 0
    pnl = 0.0

    close = df["Close"]
    # Map decision date -> index position in df
    idx_pos = {ts: i for i, ts in enumerate(df.index)}

    for x, d in zip(X_te, dates_te):
        i = idx_pos.get(d)
        if i is None or i + 1 >= len(df):
            continue

        proba = model.clf.predict_proba(x.reshape(1, -1))[0]
        p_stop, p_timeout, p_profit = proba[0], proba[1], proba[2]
        edge = p_profit - p_stop

        # execute at next open in principle; we approximate with next close for this quick test
        px_now = float(close.iloc[i])
        px_next = float(close.iloc[i + 1])

        if pos == 0:
            if (p_profit >= cfg.entry_min_p_profit) and (edge >= cfg.entry_min_edge):
                pos = 1
                trades += 1
        else:
            # soft exit
            if (p_stop >= cfg.exit_min_p_stop) or ((p_stop - p_profit) >= cfg.exit_min_edge):
                pos = 0
                trades += 1

        # mark-to-market daily
        if pos == 1:
            pnl += (px_next - px_now)

    return (
        f"{ticker} quick backtest\n"
        f"Holdout decisions: {len(X_te)}\n"
        f"Trade actions (entries+exits): {trades}\n"
        f"Approx PnL (close-to-close, 1 share): {pnl:.2f}\n"
        f"Meta holdout_accuracy (informational): {meta.get('holdout_accuracy')}\n"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python walk_forward.py TICKER")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    cfg = SignalConfig()
    print(quick_backtest_one_train(ticker, cfg))
