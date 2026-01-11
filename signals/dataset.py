from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from signals.config import SignalConfig
from signals.labeling import build_triple_barrier_labels


@dataclass(frozen=True)
class Dataset:
    """
    Flattened window dataset for tree-based models.
    """
    X: np.ndarray
    y: np.ndarray  # encoded classes {0,1,2} for {-1,0,+1}
    decision_dates: List[pd.Timestamp]
    feature_cols: List[str]
    window_size: int
    class_map: dict


def select_feature_columns(df: pd.DataFrame, cfg: SignalConfig) -> List[str]:
    """
    Select numeric feature columns in a stable way.

    We intentionally avoid automatically pulling *all* numeric columns without control,
    but we also don't want to maintain a huge manual list.

    Policy:
      - include numeric columns
      - exclude obvious non-features
      - optionally drop sentiment unless cfg.use_sentiment=True
    """
    exclude = {
        "ret", "label", "target", "stop",
    }

    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if not cfg.use_sentiment and c.lower() == "sentiment":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)

    # Ensure essential OHLCV are present (and at the front for readability)
    preferred = ["Open", "High", "Low", "Close", "Volume"]
    ordered = [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]
    return ordered


def make_flat_window_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    cfg: SignalConfig,
    decision_dates: List[pd.Timestamp],
) -> np.ndarray:
    """
    Build flattened window matrix X for each decision_date.

    Each sample uses rows [t-window+1 ... t] inclusive, flattened row-major:
      (window_size, n_features) -> (window_size * n_features,)
    """
    win = cfg.window_size
    n_feat = len(feature_cols)

    # Build a mapping from timestamp to integer index for O(1) lookup
    index_pos = {ts: i for i, ts in enumerate(df.index)}

    X = np.zeros((len(decision_dates), win * n_feat), dtype=np.float32)

    for j, ts in enumerate(decision_dates):
        t = index_pos.get(ts)
        if t is None or t < win - 1:
            continue

        window_df = df.iloc[t - win + 1 : t + 1][feature_cols]

        # Robust fill (tree models tolerate scale issues; still need no NaNs)
        window_arr = window_df.to_numpy(dtype=np.float32, copy=True)
        # Replace infs/nans
        window_arr[~np.isfinite(window_arr)] = 0.0

        X[j, :] = window_arr.reshape(-1)

    return X


def build_dataset(df: pd.DataFrame, cfg: SignalConfig) -> Dataset:
    """
    Build the supervised dataset for training:
      - compute triple barrier labels aligned to decision dates
      - build flattened window matrix X
      - encode y from {-1,0,+1} to {0,1,2}

    Encoding:
      -1 -> 0   (stop first)
       0 -> 1   (timeout)
      +1 -> 2   (profit first)
    """
    feature_cols = select_feature_columns(df, cfg)

    y_raw, _barriers, decision_dates = build_triple_barrier_labels(df, cfg)
    X = make_flat_window_matrix(df, feature_cols, cfg, decision_dates)

    class_map = {-1: 0, 0: 1, 1: 2}
    y = np.asarray([class_map.get(int(v), 1) for v in y_raw], dtype=int)

    # Filter out any rows that are all-zeros (can happen if window was invalid)
    keep = np.isfinite(X).all(axis=1) & (np.abs(X).sum(axis=1) > 0)
    X = X[keep]
    y = y[keep]
    decision_dates = [d for k, d in enumerate(decision_dates) if bool(keep[k])]

    return Dataset(
        X=X,
        y=y,
        decision_dates=decision_dates,
        feature_cols=feature_cols,
        window_size=cfg.window_size,
        class_map=class_map,
    )
