from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from signals.config import SignalConfig


@dataclass(frozen=True)
class BarrierLevels:
    """
    Barrier levels for one decision point.

    entry_px:  assumed entry price (we use the decision-time Close for labeling simplicity)
    stop_px:   entry - stop_loss_atr * ATR
    target_px: entry + take_profit_atr * ATR
    """
    entry_px: float
    stop_px: float
    target_px: float
    horizon_days: int


def compute_barriers_from_row(row: pd.Series, *, entry_px: float, cfg: SignalConfig) -> BarrierLevels:
    """
    Compute stop/target from ATR at the decision time.

    IMPORTANT:
      - This assumes ATR is already correctly computed (True ATR).
      - If ATR is missing/zero, we fall back to a small fraction of price to avoid crashes.
    """
    atr = float(row.get("ATR", np.nan))
    if not np.isfinite(atr) or atr <= 0:
        atr = max(0.005 * float(entry_px), 1e-6)  # fallback: 0.5% of price

    stop_px = float(entry_px - cfg.stop_loss_atr * atr)
    target_px = float(entry_px + cfg.take_profit_atr * atr)
    return BarrierLevels(entry_px=entry_px, stop_px=stop_px, target_px=target_px, horizon_days=cfg.horizon_days)


def build_triple_barrier_labels(
    df: pd.DataFrame,
    cfg: SignalConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Build triple-barrier labels for supervised learning.

    Labels:
      +1  profit-first: target touched before stop within horizon
      -1  stop-first:   stop touched before target within horizon
       0  timeout:      neither touched within horizon

    Decision time i:
      - features come from window [i-window_size, i)
      - entry is assumed at Close[i] (same bar close)
      - forward path uses bars (i+1 .. i+horizon_days)

    Returns:
      y:              np.ndarray of labels in {-1,0,+1}
      decision_index: np.ndarray of integer row indices in df corresponding to each label
      meta:           dict with debug arrays (entry/stop/target)
    """
    if df is None or df.empty:
        return np.array([], dtype=int), np.array([], dtype=int), {}

    df = df.sort_index()

    required = {"High", "Low", "Close", "ATR"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"build_triple_barrier_labels: missing required columns: {sorted(missing)}")

    n = len(df)
    ws = int(cfg.window_size)
    h = int(cfg.horizon_days)

    # We need i+1..i+h in range, so i <= n-h-1
    i_start = ws
    i_end = n - h - 1
    if i_end <= i_start:
        return np.array([], dtype=int), np.array([], dtype=int), {}

    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)

    y: List[int] = []
    decision_rows: List[int] = []
    entry_list: List[float] = []
    stop_list: List[float] = []
    target_list: List[float] = []

    for i in range(i_start, i_end + 1):
        entry_px = float(closes[i])
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue

        levels = compute_barriers_from_row(df.iloc[i], entry_px=entry_px, cfg=cfg)

        # Scan forward horizon bars
        label = 0  # timeout by default
        for j in range(i + 1, i + 1 + h):
            lo = float(lows[j])
            hi = float(highs[j])

            hit_stop = lo <= levels.stop_px
            hit_target = hi >= levels.target_px

            # Conservative OHLC rule:
            # If both hit on same bar, treat as stop-first.
            if hit_stop and hit_target:
                label = -1
                break
            if hit_stop:
                label = -1
                break
            if hit_target:
                label = +1
                break

        y.append(int(label))
        decision_rows.append(int(i))
        entry_list.append(levels.entry_px)
        stop_list.append(levels.stop_px)
        target_list.append(levels.target_px)

    meta = {
        "entry_px": np.asarray(entry_list, dtype=float),
        "stop_px": np.asarray(stop_list, dtype=float),
        "target_px": np.asarray(target_list, dtype=float),
        "horizon_days": h,
    }
    return np.asarray(y, dtype=int), np.asarray(decision_rows, dtype=int), meta


def infer_feature_columns(df: pd.DataFrame, cfg: SignalConfig) -> List[str]:
    """
    Choose numeric features for the signal model.

    Rules:
      - Use only numeric columns
      - Drop obvious non-features
      - Optionally exclude sentiment unless cfg.use_sentiment=True
    """
    if df is None or df.empty:
        return []

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    drop = set()

    # Do not include raw future labels or helper columns if present
    for c in ["ret", "label", "y", "signal"]:
        if c in numeric_cols:
            drop.add(c)

    # You may keep OHLCV + indicators; we do keep Close (common and useful).
    # Sentiment is optional due to timing/noise concerns.
    if not cfg.use_sentiment and "sentiment" in numeric_cols:
        drop.add("sentiment")

    feats = [c for c in numeric_cols if c not in drop]
    return feats


def build_feature_matrix(
    df: pd.DataFrame,
    cfg: SignalConfig,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build X,y for training.

    X[i] is the flattened window of length cfg.window_size ending at decision time t:
      df[feature_cols].iloc[t-window_size:t]  -> shape (window_size, n_features) -> flatten

    y[i] is triple-barrier label computed at decision time t.

    Returns:
      X: (n_samples, window_size*n_features)
      y: (n_samples,)
      idx: DatetimeIndex aligned to decision time (df.index[t])
    """
    df = df.sort_index()

    y_all, decision_rows, _meta = build_triple_barrier_labels(df, cfg)
    if len(y_all) == 0:
        return np.zeros((0, 0), dtype=float), np.array([], dtype=int), pd.DatetimeIndex([])

    if feature_cols is None:
        feature_cols = infer_feature_columns(df, cfg)

    if not feature_cols:
        raise ValueError("No feature columns available for signal model.")

    ws = int(cfg.window_size)
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    idx_list: List[pd.Timestamp] = []

    feat_df = df[feature_cols].astype(float)

    for label, t in zip(y_all, decision_rows):
        # Features use [t-ws, t) window
        start = int(t) - ws
        end = int(t)
        if start < 0:
            continue

        window = feat_df.iloc[start:end]
        if len(window) != ws:
            continue

        # Must be finite to avoid poisoning training
        arr = window.to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            continue

        X_list.append(arr.reshape(-1))  # flatten
        y_list.append(int(label))
        idx_list.append(df.index[int(t)])

    if not X_list:
        return np.zeros((0, 0), dtype=float), np.array([], dtype=int), pd.DatetimeIndex([])

    X = np.vstack(X_list).astype(float)
    y = np.asarray(y_list, dtype=int)
    idx = pd.DatetimeIndex(idx_list)

    return X, y, idx
