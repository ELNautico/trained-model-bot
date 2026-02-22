"""
Unit tests for signals/labeling.py.

Covers:
  - compute_barriers_from_row   : normal ATR, NaN ATR, zero ATR
  - build_triple_barrier_labels : profit-first, stop-first, timeout,
                                  both-same-bar (stop-first rule), edge cases
  - infer_feature_columns       : drop-list, sentiment toggle
  - build_feature_matrix        : shape, NaN exclusion, empty output
"""
import numpy as np
import pandas as pd
import pytest

from signals.config import SignalConfig
from signals.labeling import (
    BarrierLevels,
    build_feature_matrix,
    build_triple_barrier_labels,
    compute_barriers_from_row,
    infer_feature_columns,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────

# Config with small window / horizon so we can build minimal test DataFrames.
_CFG = SignalConfig(window_size=2, horizon_days=3)

# OHLC bar templates (High, Low, Close); entry=100, ATR=10 → stop=90, target=115
_NEUTRAL = (104, 96, 100)       # no barrier touch
_PROFIT  = (116, 96, 100)       # High > target=115 → +1
_STOP    = (104, 88, 100)       # Low  < stop=90   → -1
_BOTH    = (116, 88, 100)       # both hit same bar → stop-first = -1


def _make_df(rows, atr=10.0):
    """Build a minimal OHLC + ATR DataFrame from (H, L, C) tuples."""
    dates = pd.date_range("2020-01-01", periods=len(rows), freq="B")
    df = pd.DataFrame(
        [(c, h, l, c) for h, l, c in rows],
        columns=["Open", "High", "Low", "Close"],
        index=dates,
    )
    df["ATR"] = atr
    return df


def _single_decision_df(forward_rows):
    """
    Return a 7-row DataFrame.  With window_size=2, horizon_days=3:

      i_start = 2, i_end = 7-3-1 = 3  → decision rows [2, 3]
      Forward scan for i=2: rows 3, 4, 5  (supplied by caller)
      Forward scan for i=3: rows 4, 5, 6  (all neutral — padding row)

    y[0] therefore reflects the barrier outcome for decision row i=2.
    Rows 0,1 are the feature window (neutral).
    Row 2 is the decision bar (Close=100, ATR=10).
    Rows 3,4,5 are supplied by the caller.
    Row 6 is a neutral padding row so i_end > i_start.
    """
    assert len(forward_rows) == 3, "need exactly 3 forward bars"
    rows = [_NEUTRAL, _NEUTRAL, _NEUTRAL] + list(forward_rows) + [_NEUTRAL]
    return _make_df(rows)


# ── compute_barriers_from_row ──────────────────────────────────────────────────

class TestComputeBarriersFromRow:

    def test_normal_atr(self):
        row = pd.Series({"ATR": 10.0})
        cfg = SignalConfig()
        b = compute_barriers_from_row(row, entry_px=100.0, cfg=cfg)
        assert isinstance(b, BarrierLevels)
        assert b.entry_px == pytest.approx(100.0)
        assert b.stop_px  == pytest.approx(100.0 - cfg.stop_loss_atr * 10.0)
        assert b.target_px == pytest.approx(100.0 + cfg.take_profit_atr * 10.0)
        assert b.horizon_days == cfg.horizon_days

    def test_nan_atr_uses_fallback(self):
        row = pd.Series({"ATR": float("nan")})
        b = compute_barriers_from_row(row, entry_px=200.0, cfg=SignalConfig())
        fallback_atr = 0.005 * 200.0  # = 1.0
        assert b.stop_px < 200.0
        assert b.target_px > 200.0
        # Barriers computed with fallback_atr=1.0
        cfg = SignalConfig()
        assert b.stop_px   == pytest.approx(200.0 - cfg.stop_loss_atr * fallback_atr)
        assert b.target_px == pytest.approx(200.0 + cfg.take_profit_atr * fallback_atr)

    def test_zero_atr_uses_fallback(self):
        row = pd.Series({"ATR": 0.0})
        b = compute_barriers_from_row(row, entry_px=100.0, cfg=SignalConfig())
        fallback_atr = 0.005 * 100.0  # = 0.5
        cfg = SignalConfig()
        assert b.stop_px   == pytest.approx(100.0 - cfg.stop_loss_atr * fallback_atr)
        assert b.target_px == pytest.approx(100.0 + cfg.take_profit_atr * fallback_atr)

    def test_missing_atr_column_uses_fallback(self):
        row = pd.Series({"Close": 50.0})  # no ATR key
        b = compute_barriers_from_row(row, entry_px=50.0, cfg=SignalConfig())
        assert b.stop_px < 50.0
        assert b.target_px > 50.0


# ── build_triple_barrier_labels ────────────────────────────────────────────────

class TestBuildTripleBarrierLabels:

    def test_profit_first_at_first_bar(self):
        df = _single_decision_df([_PROFIT, _NEUTRAL, _NEUTRAL])
        y, rows, meta = build_triple_barrier_labels(df, _CFG)
        assert len(y) >= 1      # 7-row df → 2 decision rows; we care about y[0]
        assert y[0] == +1
        assert rows[0] == 2

    def test_profit_first_at_last_bar(self):
        df = _single_decision_df([_NEUTRAL, _NEUTRAL, _PROFIT])
        y, _, _ = build_triple_barrier_labels(df, _CFG)
        assert y[0] == +1

    def test_stop_first_at_first_bar(self):
        df = _single_decision_df([_STOP, _NEUTRAL, _NEUTRAL])
        y, _, _ = build_triple_barrier_labels(df, _CFG)
        assert y[0] == -1

    def test_stop_first_at_last_bar(self):
        df = _single_decision_df([_NEUTRAL, _NEUTRAL, _STOP])
        y, _, _ = build_triple_barrier_labels(df, _CFG)
        assert y[0] == -1

    def test_timeout_when_no_barrier_touched(self):
        df = _single_decision_df([_NEUTRAL, _NEUTRAL, _NEUTRAL])
        y, _, _ = build_triple_barrier_labels(df, _CFG)
        assert y[0] == 0

    def test_both_hit_same_bar_is_stop_first(self):
        """Conservative rule: both on same bar → stop-first (-1)."""
        df = _single_decision_df([_BOTH, _NEUTRAL, _NEUTRAL])
        y, _, _ = build_triple_barrier_labels(df, _CFG)
        assert y[0] == -1

    def test_stop_before_profit(self):
        """Stop hit before profit in later bar → stop-first."""
        df = _single_decision_df([_STOP, _PROFIT, _NEUTRAL])
        y, _, _ = build_triple_barrier_labels(df, _CFG)
        assert y[0] == -1

    def test_profit_before_stop(self):
        """Profit hit before stop in later bar → profit-first."""
        df = _single_decision_df([_PROFIT, _STOP, _NEUTRAL])
        y, _, _ = build_triple_barrier_labels(df, _CFG)
        assert y[0] == +1

    def test_meta_contains_barrier_levels(self):
        df = _single_decision_df([_NEUTRAL, _NEUTRAL, _NEUTRAL])
        y, rows, meta = build_triple_barrier_labels(df, _CFG)
        assert "entry_px" in meta
        assert "stop_px" in meta
        assert "target_px" in meta
        assert meta["entry_px"][0] == pytest.approx(100.0)
        assert meta["stop_px"][0]  < 100.0
        assert meta["target_px"][0] > 100.0
        assert meta["horizon_days"] == _CFG.horizon_days

    def test_decision_row_index_is_correct(self):
        df = _single_decision_df([_PROFIT, _NEUTRAL, _NEUTRAL])
        y, rows, _ = build_triple_barrier_labels(df, _CFG)
        assert rows[0] == 2  # window_size=2 → first decision at index 2

    def test_multiple_decision_rows(self):
        """8-row df with window_size=2, horizon_days=3 → decision rows [2,3,4]."""
        cfg = SignalConfig(window_size=2, horizon_days=3)
        # 8 rows: i_start=2, i_end=8-3-1=4 → rows 2,3,4
        rows = [_NEUTRAL] * 8
        df = _make_df(rows)
        y, dec_rows, _ = build_triple_barrier_labels(df, cfg)
        assert len(y) == 3
        np.testing.assert_array_equal(dec_rows, [2, 3, 4])
        np.testing.assert_array_equal(y, [0, 0, 0])  # all neutral → timeout

    def test_empty_dataframe_returns_empty_arrays(self):
        y, rows, meta = build_triple_barrier_labels(pd.DataFrame(), _CFG)
        assert len(y) == 0
        assert len(rows) == 0

    def test_none_dataframe_returns_empty_arrays(self):
        y, rows, meta = build_triple_barrier_labels(None, _CFG)
        assert len(y) == 0

    def test_missing_required_columns_raises(self):
        df = pd.DataFrame({"Close": [100.0] * 10})
        with pytest.raises(ValueError, match="missing required columns"):
            build_triple_barrier_labels(df, _CFG)

    def test_too_short_df_returns_empty(self):
        """DataFrame shorter than window_size + horizon_days + 1 → no labels."""
        cfg = SignalConfig(window_size=5, horizon_days=3)
        # Need n > 5+3+1=9; n=8 is too short
        df = _make_df([_NEUTRAL] * 8)
        # i_start=5, i_end=8-3-1=4 → i_end <= i_start → empty
        y, rows, _ = build_triple_barrier_labels(df, cfg)
        assert len(y) == 0

    def test_invalid_entry_price_skipped(self):
        """Rows with Close=NaN or Close<=0 should be skipped gracefully."""
        rows = [_NEUTRAL, _NEUTRAL, (104, 96, float("nan")), _NEUTRAL, _NEUTRAL, _NEUTRAL]
        df = _make_df(rows)
        # Decision at i=2 has NaN close → skipped; no other decision row exists
        y, _, _ = build_triple_barrier_labels(df, _CFG)
        assert len(y) == 0


# ── infer_feature_columns ──────────────────────────────────────────────────────

class TestInferFeatureColumns:

    def _rich_df(self):
        """DataFrame with OHLC, ATR, helper cols, sentiment, and a feature."""
        n = 10
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.DataFrame({
            "Open":      100.0,
            "High":      105.0,
            "Low":        95.0,
            "Close":     100.0,
            "ATR":        10.0,
            "feat_a":    1.0,
            "label":     0,       # should be dropped
            "ret":       0.01,    # should be dropped
            "signal":    1,       # should be dropped
            "y":         0,       # should be dropped
            "sentiment": 0.5,     # dropped when use_sentiment=False
            "text_col":  "abc",   # non-numeric → excluded by dtype
        }, index=dates)

    def test_drops_label_columns(self):
        df = self._rich_df()
        cols = infer_feature_columns(df, SignalConfig(use_sentiment=False))
        for forbidden in ("label", "ret", "signal", "y"):
            assert forbidden not in cols

    def test_drops_sentiment_when_disabled(self):
        df = self._rich_df()
        cols = infer_feature_columns(df, SignalConfig(use_sentiment=False))
        assert "sentiment" not in cols

    def test_keeps_sentiment_when_enabled(self):
        df = self._rich_df()
        cols = infer_feature_columns(df, SignalConfig(use_sentiment=True))
        assert "sentiment" in cols

    def test_excludes_non_numeric_columns(self):
        df = self._rich_df()
        cols = infer_feature_columns(df, SignalConfig())
        assert "text_col" not in cols

    def test_keeps_ohlc_and_atr(self):
        df = self._rich_df()
        cols = infer_feature_columns(df, SignalConfig(use_sentiment=False))
        for expected in ("Open", "High", "Low", "Close", "ATR", "feat_a"):
            assert expected in cols

    def test_empty_df_returns_empty(self):
        assert infer_feature_columns(pd.DataFrame(), SignalConfig()) == []

    def test_none_df_returns_empty(self):
        assert infer_feature_columns(None, SignalConfig()) == []


# ── build_feature_matrix ───────────────────────────────────────────────────────

class TestBuildFeatureMatrix:

    def _build_df(self, n=20):
        """20-row OHLC + ATR + one extra feature column."""
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.DataFrame({
            "Open":   100.0,
            "High":   104.0,
            "Low":     96.0,
            "Close":  100.0,
            "ATR":    10.0,
            "feat":   np.arange(n, dtype=float),
        }, index=dates)

    def test_output_shape(self):
        df = self._build_df(20)
        cfg = SignalConfig(window_size=2, horizon_days=3)
        feat_cols = ["Close", "ATR"]
        X, y, idx = build_feature_matrix(df, cfg, feature_cols=feat_cols)
        # n=20, ws=2, h=3: decision rows 2..16 → 15 decisions, all valid windows
        assert X.ndim == 2
        # Each sample is a flattened window of shape (ws=2, n_feats=2) = 4
        assert X.shape[1] == 2 * 2
        assert len(y) == X.shape[0]
        assert isinstance(idx, pd.DatetimeIndex)
        assert len(idx) == X.shape[0]

    def test_y_values_in_valid_set(self):
        df = self._build_df(20)
        cfg = SignalConfig(window_size=2, horizon_days=3)
        X, y, _ = build_feature_matrix(df, cfg, feature_cols=["Close", "ATR"])
        assert set(y).issubset({-1, 0, 1})

    def test_nan_windows_excluded(self):
        df = self._build_df(20)
        # Inject NaN into first several Close values (will poison windows ending there)
        df.loc[df.index[3], "Close"] = float("nan")
        cfg = SignalConfig(window_size=2, horizon_days=3)
        X, y, _ = build_feature_matrix(df, cfg, feature_cols=["Close", "ATR"])
        assert np.isfinite(X).all(), "NaN windows must be excluded from X"

    def test_returns_empty_when_df_too_short(self):
        df = self._build_df(5)  # too short for ws=5, h=5
        cfg = SignalConfig(window_size=5, horizon_days=5)
        X, y, idx = build_feature_matrix(df, cfg)
        assert X.shape[0] == 0

    def test_raises_when_no_feature_columns(self):
        """Explicitly passing feature_cols=[] on a df that produces labels must raise."""
        df = self._build_df(20)
        cfg = SignalConfig(window_size=2, horizon_days=3)
        with pytest.raises(ValueError, match="No feature columns"):
            build_feature_matrix(df, cfg, feature_cols=[])
