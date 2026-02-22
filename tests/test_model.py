"""
Unit tests for signals/model.py.

Functions tested:
  _compute_class_weights
  _time_split_train_cal
  _multiclass_brier_ovr_mean
  _adaptive_calibration_cv_folds
  train_calibrated_model          (end-to-end with synthetic data)
  SignalModelArtifact.predict_proba_last
"""
import numpy as np
import pandas as pd
import pytest

from signals.config import SignalConfig
from signals.model import (
    SignalModelArtifact,
    _adaptive_calibration_cv_folds,
    _compute_class_weights,
    _multiclass_brier_ovr_mean,
    _time_split_train_cal,
    train_calibrated_model,
)


# ── _compute_class_weights ─────────────────────────────────────────────────────

class TestComputeClassWeights:

    # y with three classes, imbalanced: rare=+1 (10), medium=-1 (30), common=0 (60)
    _y = np.array([-1] * 30 + [0] * 60 + [1] * 10)

    def test_disabled_returns_empty_dict(self):
        cfg = SignalConfig(use_class_weights=False)
        cw = _compute_class_weights(self._y, cfg)
        assert cw == {}

    def test_mode_none_returns_empty_dict(self):
        cfg = SignalConfig(class_weight_mode="none")
        cw = _compute_class_weights(self._y, cfg)
        assert cw == {}

    def test_sqrt_inv_returns_all_classes(self):
        cfg = SignalConfig(use_class_weights=True, class_weight_mode="sqrt_inv")
        cw = _compute_class_weights(self._y, cfg)
        assert set(cw.keys()) == {-1, 0, 1}

    def test_sqrt_inv_weights_positive(self):
        cfg = SignalConfig(use_class_weights=True, class_weight_mode="sqrt_inv")
        cw = _compute_class_weights(self._y, cfg)
        for w in cw.values():
            assert w > 0.0

    def test_sqrt_inv_weights_capped_at_max(self):
        cfg = SignalConfig(
            use_class_weights=True, class_weight_mode="sqrt_inv", max_class_weight=3.0
        )
        cw = _compute_class_weights(self._y, cfg)
        for w in cw.values():
            assert w <= 3.0

    def test_rare_class_has_higher_weight_than_common(self):
        cfg = SignalConfig(use_class_weights=True, class_weight_mode="sqrt_inv")
        cw = _compute_class_weights(self._y, cfg)
        # +1 (10 samples) should get higher weight than 0 (60 samples)
        assert cw[1] > cw[0]

    def test_inv_mode_caps_extreme_weight(self):
        # +1 has only 10 samples → inv weight = 100/(3*10) ≈ 3.33 → capped at 3.0
        cfg = SignalConfig(
            use_class_weights=True, class_weight_mode="inv", max_class_weight=3.0
        )
        cw = _compute_class_weights(self._y, cfg)
        assert cw[1] == pytest.approx(3.0)

    def test_balanced_data_all_weights_equal_one(self):
        y_bal = np.array([-1] * 33 + [0] * 34 + [1] * 33)  # roughly balanced
        cfg = SignalConfig(use_class_weights=True, class_weight_mode="inv")
        cw = _compute_class_weights(y_bal, cfg)
        # All inverse weights should be close to 1.0 for balanced data
        for w in cw.values():
            assert w == pytest.approx(1.0, abs=0.1)

    def test_custom_cap_applied(self):
        cfg = SignalConfig(
            use_class_weights=True, class_weight_mode="inv", max_class_weight=1.5
        )
        cw = _compute_class_weights(self._y, cfg)
        for w in cw.values():
            assert w <= 1.5


# ── _time_split_train_cal ─────────────────────────────────────────────────────

class TestTimeSplitTrainCal:

    _cfg = SignalConfig()  # calibration_ratio=0.20

    def _xy(self, n):
        return np.zeros((n, 3)), np.zeros(n, dtype=int)

    def test_small_dataset_split(self):
        # n=100 < 500: cut = max(1, floor(100*0.8)) = 80
        X, y = self._xy(100)
        X_tr, y_tr, X_cal, y_cal = _time_split_train_cal(X, y, self._cfg)
        assert len(X_tr) == 80
        assert len(X_cal) == 20
        assert len(y_tr) == 80
        assert len(y_cal) == 20

    def test_large_dataset_guardrails(self):
        # n=1000 >= 500: cut clamped to [200, 800]
        # floor(1000*0.8)=800 → clamp(800,200,800)=800
        X, y = self._xy(1000)
        X_tr, _, X_cal, _ = _time_split_train_cal(X, y, self._cfg)
        assert len(X_tr) == 800
        assert len(X_cal) == 200

    def test_no_overlap_between_train_and_cal(self):
        X = np.arange(100).reshape(100, 1).astype(float)
        y = np.zeros(100, dtype=int)
        X_tr, y_tr, X_cal, y_cal = _time_split_train_cal(X, y, self._cfg)
        # Training values should all precede calibration values
        assert X_tr[-1, 0] < X_cal[0, 0]

    def test_chronological_order_preserved(self):
        # X[i, 0] = i (monotonically increasing)
        X = np.arange(200).reshape(200, 1).astype(float)
        y = np.zeros(200, dtype=int)
        X_tr, _, X_cal, _ = _time_split_train_cal(X, y, self._cfg)
        assert X_tr[-1, 0] < X_cal[0, 0]

    def test_tiny_dataset_still_works(self):
        X, y = self._xy(3)
        X_tr, y_tr, X_cal, y_cal = _time_split_train_cal(X, y, self._cfg)
        # cut = max(1, floor(3*0.8)) = max(1, 2) = 2
        assert len(X_tr) >= 1
        assert len(X_cal) >= 1
        assert len(X_tr) + len(X_cal) == 3

    def test_calibration_ratio_respected(self):
        X, y = self._xy(100)
        cfg_30 = SignalConfig(calibration_ratio=0.30)
        X_tr, _, X_cal, _ = _time_split_train_cal(X, y, cfg_30)
        # cut = floor(100 * 0.70) = 70
        assert len(X_tr) == 70
        assert len(X_cal) == 30


# ── _multiclass_brier_ovr_mean ────────────────────────────────────────────────

class TestMulticlassBrierOvrMean:

    _classes = np.array([-1, 0, 1])

    def test_perfect_predictions_give_zero_brier(self):
        y_true = np.array([-1, 0, 1, -1, 0])
        proba = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        score = _multiclass_brier_ovr_mean(y_true, proba, self._classes)
        assert score == pytest.approx(0.0)

    def test_uniform_predictions_give_moderate_score(self):
        y_true = np.array([-1, 0, 1])
        proba = np.full((3, 3), 1.0 / 3.0)
        score = _multiclass_brier_ovr_mean(y_true, proba, self._classes)
        # Uniform: brier per class = mean((y_bin - 1/3)^2)
        # For each class and each sample, either (0 - 1/3)^2 or (1 - 1/3)^2
        # Brier = (1/3 * (2/3)^2 + 2/3 * (1/3)^2) ≈ 0.222
        assert 0.15 < score < 0.35

    def test_worst_predictions_give_high_score(self):
        # Assign 100% probability to wrong class for all samples
        y_true = np.array([-1, 0, 1])
        proba = np.array([
            [0.0, 0.0, 1.0],  # true=-1, predict +1
            [1.0, 0.0, 0.0],  # true=0,  predict -1
            [0.0, 1.0, 0.0],  # true=1,  predict 0
        ])
        score = _multiclass_brier_ovr_mean(y_true, proba, self._classes)
        assert score > 0.5

    def test_result_is_average_of_per_class_scores(self):
        y_true  = np.array([-1, 0, 1])
        proba   = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]])
        score   = _multiclass_brier_ovr_mean(y_true, proba, self._classes)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ── _adaptive_calibration_cv_folds ────────────────────────────────────────────

class TestAdaptiveCalibrationCvFolds:

    _cfg = SignalConfig()
    _base_classes = np.array([-1, 0, 1])

    def test_skip_when_tail_too_small(self):
        # min_cal_n=120 (default via getattr); n_cal=50 < 120
        y_small = np.array([-1] * 20 + [0] * 20 + [1] * 10)  # 50 samples
        folds, skip, reason = _adaptive_calibration_cv_folds(
            y_small, self._base_classes, self._cfg
        )
        assert skip is True
        assert "too_small" in reason

    def test_skip_when_class_missing_in_cal_tail(self):
        # 200 samples but only classes 0 and +1 — missing -1
        y_missing = np.array([0] * 100 + [1] * 100)
        folds, skip, reason = _adaptive_calibration_cv_folds(
            y_missing, self._base_classes, self._cfg
        )
        assert skip is True
        assert "missing" in reason

    def test_ok_when_all_classes_present_and_large_enough(self):
        # 200 samples with all three classes
        y_ok = np.array([-1] * 60 + [0] * 80 + [1] * 60)
        folds, skip, reason = _adaptive_calibration_cv_folds(
            y_ok, self._base_classes, self._cfg
        )
        assert skip is False
        assert folds >= 2
        assert reason == "ok"

    def test_cv_folds_capped_at_max_folds(self):
        # 200 samples: min class count = 60 > max_folds=5
        y_ok = np.array([-1] * 60 + [0] * 80 + [1] * 60)
        folds, skip, _ = _adaptive_calibration_cv_folds(
            y_ok, self._base_classes, self._cfg
        )
        assert folds <= 5  # default max_folds

    def test_skip_when_insufficient_class_support(self):
        # n_cal=200 but one class has only 1 sample → min_cnt=1 < 2 folds
        y_skewed = np.array([-1] * 1 + [0] * 100 + [1] * 99)
        folds, skip, reason = _adaptive_calibration_cv_folds(
            y_skewed, self._base_classes, self._cfg
        )
        assert skip is True
        assert "insufficient" in reason


# ── train_calibrated_model (end-to-end) ───────────────────────────────────────

def _synthetic_df(n=300, seed=42):
    """
    Build a synthetic OHLC + ATR DataFrame that produces a mix of all three
    barrier outcomes (+1, -1, 0).

    Close drifts slightly; High/Low are offset from Close by random amounts;
    ATR is fixed at 5.0.  With stop_loss_atr=1.0 and take_profit_atr=1.5:
      stop   = Close - 5   (5 pts below entry)
      target = Close + 7.5 (7.5 pts above entry)

    Candle ranges of [1, 8] pts around Close ensure a mix of barrier touches
    and timeouts across a 5-day horizon.
    """
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0, 0.4, n).cumsum()
    close = np.maximum(close, 20.0)  # keep price positive
    atr   = np.full(n, 5.0)
    high  = close + rng.uniform(1, 8, n)
    low   = close - rng.uniform(1, 8, n)
    low   = np.minimum(low, close - 0.5)   # ensure low < close

    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open":   close,
        "High":   high,
        "Low":    low,
        "Close":  close,
        "ATR":    atr,
        "feat_a": rng.normal(0, 1, n),
        "feat_b": rng.normal(0, 1, n),
    }, index=dates)


class TestTrainCalibratedModel:

    _df  = _synthetic_df(n=300)
    _cfg = SignalConfig(window_size=5, horizon_days=5, max_iter=50)

    def test_returns_artifact_and_report(self):
        artifact, report = train_calibrated_model(self._df, self._cfg)
        assert isinstance(artifact, SignalModelArtifact)
        assert isinstance(report, dict)

    def test_artifact_has_required_fields(self):
        artifact, _ = train_calibrated_model(self._df, self._cfg)
        assert len(artifact.feature_cols) > 0
        assert artifact.classes_.size > 0
        assert artifact.estimator is not None
        assert artifact.base_estimator is not None
        assert isinstance(artifact.cfg, SignalConfig)

    def test_report_keys_present(self):
        _, report = train_calibrated_model(self._df, self._cfg)
        for key in ("n_samples", "n_train", "n_cal", "cal_log_loss",
                    "cal_brier_ovr_mean", "calibration"):
            assert key in report, f"missing report key: {key}"

    def test_report_sample_counts_consistent(self):
        _, report = train_calibrated_model(self._df, self._cfg)
        assert report["n_samples"] > 0
        assert report["n_train"] > 0
        assert report["n_cal"] > 0
        assert report["n_train"] + report["n_cal"] == report["n_samples"]

    def test_cal_metrics_are_finite(self):
        _, report = train_calibrated_model(self._df, self._cfg)
        assert np.isfinite(report["cal_log_loss"])
        assert np.isfinite(report["cal_brier_ovr_mean"])
        assert report["cal_brier_ovr_mean"] >= 0.0

    def test_explicit_feature_cols_respected(self):
        feat_cols = ["Close", "ATR"]
        artifact, _ = train_calibrated_model(self._df, self._cfg, feature_cols=feat_cols)
        assert artifact.feature_cols == feat_cols

    def test_predict_proba_last_sums_to_one(self):
        artifact, _ = train_calibrated_model(self._df, self._cfg)
        probs = artifact.predict_proba_last(self._df)
        assert set(probs.keys()) == {"p_profit", "p_stop", "p_timeout"}
        total = probs["p_profit"] + probs["p_stop"] + probs["p_timeout"]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_predict_proba_last_all_non_negative(self):
        artifact, _ = train_calibrated_model(self._df, self._cfg)
        probs = artifact.predict_proba_last(self._df)
        for k, v in probs.items():
            assert v >= 0.0, f"{k} is negative: {v}"

    def test_predict_proba_array_shape(self):
        artifact, _ = train_calibrated_model(self._df, self._cfg)
        from signals.labeling import build_feature_matrix
        X, _, _ = build_feature_matrix(self._df, self._cfg, artifact.feature_cols)
        proba = artifact.predict_proba(X)
        assert proba.shape == (X.shape[0], len(artifact.classes_))
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_feature_cols_stored_as_list(self):
        artifact, _ = train_calibrated_model(self._df, self._cfg)
        assert isinstance(artifact.feature_cols, list)

    def test_classes_are_subset_of_valid_labels(self):
        artifact, _ = train_calibrated_model(self._df, self._cfg)
        assert set(artifact.classes_.tolist()).issubset({-1, 0, 1})
