from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss

try:
    # sklearn >= 1.6: replacement for cv="prefit" calibration workflow
    from sklearn.frozen import FrozenEstimator  # type: ignore
except Exception:
    FrozenEstimator = None  # type: ignore

try:
    # Good baseline for tabular data; handles nonlinearities well.
    from sklearn.ensemble import HistGradientBoostingClassifier
except Exception:
    HistGradientBoostingClassifier = None

from signals.config import SignalConfig
from signals.labeling import build_feature_matrix, infer_feature_columns


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
def _model_dir(ticker: str) -> Path:
    return Path("models_signals") / ticker.upper()


def _model_path(ticker: str) -> Path:
    return _model_dir(ticker) / "model.joblib"


def _meta_path(ticker: str) -> Path:
    return _model_dir(ticker) / "metadata.json"


# -----------------------------------------------------------------------------
# Class weights (tempered, capped)
# -----------------------------------------------------------------------------
def _compute_class_weights(y: np.ndarray, cfg: SignalConfig) -> Dict[int, float]:
    """
    Compute class weights for labels in {-1, 0, +1}.

    Why tempered + capped?
      - Inverse-frequency weights can be enormous for rare classes (timeout).
      - If you calibrate with those weights, probabilities get distorted.
      - We therefore:
          1) apply weights ONLY when fitting the base model
          2) temper the weights (sqrt inverse) by default
          3) cap weights at cfg.max_class_weight
    """
    if not getattr(cfg, "use_class_weights", False) or getattr(cfg, "class_weight_mode", "none") == "none":
        return {}

    classes, counts = np.unique(y, return_counts=True)
    total = float(counts.sum())
    k = float(len(classes))

    w: Dict[int, float] = {}
    for c, cnt in zip(classes, counts):
        inv = total / (k * float(cnt))  # classic inverse-frequency

        mode = getattr(cfg, "class_weight_mode", "sqrt_inv")
        if mode == "inv":
            weight = inv
        elif mode == "sqrt_inv":
            weight = float(np.sqrt(inv))
        else:
            weight = 1.0

        cap = float(getattr(cfg, "max_class_weight", 3.0))
        weight = float(min(weight, cap))
        w[int(c)] = weight

    return w


def _sample_weights(y: np.ndarray, cw: Dict[int, float]) -> Optional[np.ndarray]:
    if not cw:
        return None
    return np.asarray([cw[int(v)] for v in y], dtype=float)


# -----------------------------------------------------------------------------
# Artifact
# -----------------------------------------------------------------------------
@dataclass
class SignalModelArtifact:
    """
    Persisted artifact for one ticker.

    estimator:
      - calibrated classifier (CalibratedClassifierCV) OR base estimator if calibration skipped
    base_estimator:
      - underlying fitted estimator (kept for debugging)
    feature_cols:
      - columns used to build windows
    cfg:
      - training config
    classes_:
      - class order used by predict_proba
    """
    estimator: object
    base_estimator: object
    feature_cols: list[str]
    cfg: SignalConfig
    classes_: np.ndarray

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)

    def predict_proba_last(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Return calibrated probabilities for the most recent decision point.

        Output keys match your engine:
          p_profit  = P(label = +1)
          p_stop    = P(label = -1)
          p_timeout = P(label = 0)
        """
        X, _y, _idx = build_feature_matrix(df, self.cfg, self.feature_cols)
        if X.shape[0] == 0:
            raise ValueError("Not enough data to form latest feature window for prediction.")

        proba = self.predict_proba(X[-1:].astype(float))[0]
        class_to_p = {int(c): float(p) for c, p in zip(self.classes_, proba)}

        p_profit = class_to_p.get(+1, 0.0)
        p_stop = class_to_p.get(-1, 0.0)
        p_timeout = class_to_p.get(0, 0.0)

        s = p_profit + p_stop + p_timeout
        if s <= 0:
            return {"p_profit": 0.0, "p_stop": 0.0, "p_timeout": 1.0}

        return {"p_profit": p_profit / s, "p_stop": p_stop / s, "p_timeout": p_timeout / s}


# -----------------------------------------------------------------------------
# Training / calibration
# -----------------------------------------------------------------------------
def _build_base_estimator(cfg: SignalConfig):
    if HistGradientBoostingClassifier is None:
        raise ImportError("HistGradientBoostingClassifier not available; please install scikit-learn>=0.22.")

    # Mild regularization helps with flattened (very wide) feature vectors.
    return HistGradientBoostingClassifier(
        learning_rate=float(getattr(cfg, "learning_rate", 0.05)),
        max_depth=int(getattr(cfg, "max_depth", 3)),
        max_iter=int(getattr(cfg, "max_iter", 300)),
        random_state=int(getattr(cfg, "random_state", 42)),
        l2_regularization=float(getattr(cfg, "l2_regularization", 0.0)),
    )


def _time_split_train_cal(
    X: np.ndarray, y: np.ndarray, cfg: SignalConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological split:
      - Train on first (1 - calibration_ratio)
      - Calibrate on last calibration_ratio
    """
    n = int(X.shape[0])
    cal_ratio = float(getattr(cfg, "calibration_ratio", 0.20))
    cal_ratio = float(np.clip(cal_ratio, 0.05, 0.40))  # guardrails

    cut = int(np.floor(n * (1.0 - cal_ratio)))
    if n >= 500:
        cut = int(np.clip(cut, 200, n - 200))  # ensure minimum sizes
    else:
        cut = max(1, cut)

    X_tr, y_tr = X[:cut], y[:cut]
    X_cal, y_cal = X[cut:], y[cut:]
    return X_tr, y_tr, X_cal, y_cal


def _multiclass_brier_ovr_mean(y_true: np.ndarray, proba: np.ndarray, classes_: np.ndarray) -> float:
    """
    Multiclass Brier proxy: average one-vs-rest Brier across classes.
    """
    briers = []
    for c in classes_:
        y_bin = (y_true == int(c)).astype(int)
        col = int(np.where(classes_ == int(c))[0][0])
        p = proba[:, col]
        briers.append(float(brier_score_loss(y_bin, p)))
    return float(np.mean(briers))


def _counts_dict(arr: np.ndarray) -> Dict[int, int]:
    u, c = np.unique(arr, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}


def _adaptive_calibration_cv_folds(
    y_cal: np.ndarray,
    base_classes: np.ndarray,
    cfg: SignalConfig,
) -> Tuple[int, bool, str]:
    """
    Decide calibration CV folds for CalibratedClassifierCV with FrozenEstimator.

    Returns:
      (cv_folds, should_skip, reason)

    Rules (pragmatic):
      - If calibration tail is very small -> skip
      - If any class that the base model can emit is missing in y_cal -> skip
      - Else choose folds <= min class count in y_cal (capped)
    """
    max_folds = int(getattr(cfg, "calibration_max_folds", 5))
    max_folds = int(np.clip(max_folds, 2, 10))

    min_cal_n = int(getattr(cfg, "calibration_min_samples", 120))
    min_cal_n = int(np.clip(min_cal_n, 50, 1000))

    n_cal = int(len(y_cal))
    if n_cal < min_cal_n:
        return 0, True, f"calibration_tail_too_small(n_cal={n_cal} < {min_cal_n})"

    # Require that y_cal contains all classes the base model was trained to predict.
    # If not, multiclass calibration can become unstable / ill-defined.
    cal_counts = {int(c): int(np.sum(y_cal == int(c))) for c in np.unique(base_classes)}
    missing = [c for c, cnt in cal_counts.items() if cnt <= 0]
    if missing:
        return 0, True, f"missing_classes_in_cal_tail({missing})"

    min_cnt = min(cal_counts.values()) if cal_counts else 0
    cv_folds = int(min(max_folds, min_cnt))

    if cv_folds < 2:
        return 0, True, f"insufficient_class_support(min_class_count={min_cnt})"

    return cv_folds, False, "ok"


def train_calibrated_model(
    df: pd.DataFrame,
    cfg: SignalConfig,
    feature_cols: Optional[list[str]] = None,
) -> Tuple[SignalModelArtifact, Dict]:
    """
    Train + calibrate a signal model.

    Procedure:
      1) Build X,y chronologically.
      2) Split into train segment + calibration tail segment.
      3) Fit base model on train (optionally with tempered weights).
      4) Calibrate on tail:
         - sklearn with FrozenEstimator: use CV calibration with adaptive folds, or skip if too small/imbalanced.
         - older sklearn: fallback to cv="prefit" (may show FutureWarning in sklearn 1.5.x).
      5) Report metrics on calibration tail (for whichever estimator is used).
    """
    if feature_cols is None:
        feature_cols = infer_feature_columns(df, cfg)

    X, y, _idx = build_feature_matrix(df, cfg, feature_cols)

    # Warn, but don't fail: backtests can choose to warm up before calling this.
    warn_n = int(getattr(cfg, "warn_min_train_samples", 800))
    if X.shape[0] < warn_n:
        logging.warning("Signal training has limited samples (%d). Calibration may be noisy.", X.shape[0])

    # Split
    X_tr, y_tr, X_cal, y_cal = _time_split_train_cal(X, y, cfg)

    # Base fit (with tempered + capped class weights)
    base = _build_base_estimator(cfg)
    cw = _compute_class_weights(y_tr, cfg)
    sw = _sample_weights(y_tr, cw)

    used_weights = False
    if sw is not None:
        try:
            base.fit(X_tr, y_tr, sample_weight=sw)
            used_weights = True
        except TypeError:
            base.fit(X_tr, y_tr)
            used_weights = False
    else:
        base.fit(X_tr, y_tr)

    base_classes = np.asarray(getattr(base, "classes_", np.unique(y_tr)), dtype=int)

    # Calibrate
    method = str(getattr(cfg, "calibration_method", "sigmoid"))
    did_calibrate = False
    skip_reason = None
    cv_used = None

    estimator = base  # default: skip calibration
    if FrozenEstimator is not None:
        cv_folds, should_skip, reason = _adaptive_calibration_cv_folds(y_cal, base_classes, cfg)
        if should_skip:
            skip_reason = reason
            estimator = base
            did_calibrate = False
        else:
            frozen = FrozenEstimator(base)
            calibrator = CalibratedClassifierCV(estimator=frozen, method=method, cv=int(cv_folds))
            calibrator.fit(X_cal, y_cal)
            estimator = calibrator
            did_calibrate = True
            cv_used = int(cv_folds)
    else:
        # Backward-compatible fallback: prefit calibration on tail.
        # (In sklearn 1.5.x this can emit a FutureWarning about deprecation in 1.6.)
        try:
            calibrator = CalibratedClassifierCV(estimator=base, cv="prefit", method=method)
        except TypeError:
            calibrator = CalibratedClassifierCV(base_estimator=base, cv="prefit", method=method)
        calibrator.fit(X_cal, y_cal)
        estimator = calibrator
        did_calibrate = True
        cv_used = "prefit"

    classes_ = getattr(estimator, "classes_", None)
    if classes_ is None:
        classes_ = base_classes if base_classes.size else np.array([-1, 0, 1], dtype=int)
    classes_ = np.asarray(classes_, dtype=int)

    artifact = SignalModelArtifact(
        estimator=estimator,
        base_estimator=base,
        feature_cols=list(feature_cols),
        cfg=cfg,
        classes_=classes_,
    )

    # Eval on calibration tail using the chosen estimator (calibrated or base)
    proba_cal = artifact.predict_proba(X_cal)
    ll = float(log_loss(y_cal, proba_cal, labels=list(classes_)))
    brier = _multiclass_brier_ovr_mean(y_cal, proba_cal, classes_)

    report = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_train": int(X_tr.shape[0]),
        "n_cal": int(X_cal.shape[0]),
        "class_counts_train": _counts_dict(y_tr),
        "class_counts_cal": _counts_dict(y_cal),
        "class_weights_train": {int(k): float(v) for k, v in cw.items()},
        "used_sample_weights_base_fit": bool(used_weights),
        "calibration": {
            "method": method,
            "calibration_ratio": float(getattr(cfg, "calibration_ratio", 0.20)),
            "implementation": "FrozenEstimator" if FrozenEstimator is not None else "cv=prefit",
            "did_calibrate": bool(did_calibrate),
            "cv_used": cv_used,
            "skip_reason": skip_reason,
        },
        "cal_log_loss": ll,
        "cal_brier_ovr_mean": brier,
    }

    return artifact, report


def save_artifact(ticker: str, artifact: SignalModelArtifact, report: Dict):
    d = _model_dir(ticker)
    d.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, _model_path(ticker))

    meta = {
        "ticker": ticker.upper(),
        "feature_cols": artifact.feature_cols,
        "cfg": asdict(artifact.cfg),
        "classes_": [int(x) for x in artifact.classes_],
        "report": report,
    }
    _meta_path(ticker).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_artifact(ticker: str) -> SignalModelArtifact:
    path = _model_path(ticker)
    if not path.exists():
        raise FileNotFoundError(f"No signal model artifact at {path}")
    return joblib.load(path)


def ensure_model(
    ticker: str,
    df: pd.DataFrame,
    cfg: SignalConfig,
    *,
    force_retrain: bool = False,
) -> SignalModelArtifact:
    """
    Load existing calibrated model or retrain.

    Retrain triggers:
      - force_retrain=True
      - artifact missing / failed to load
      - key config changed (window/horizon/barriers/calibration/weights)
    """
    ticker = ticker.upper()
    path = _model_path(ticker)

    if path.exists() and not force_retrain:
        try:
            art = load_artifact(ticker)

            key_fields = [
                "window_size", "horizon_days",
                "take_profit_atr", "stop_loss_atr",
                "calibration_ratio", "calibration_method",
                "use_class_weights", "class_weight_mode", "max_class_weight",
                "l2_regularization",
                # optional new knobs (won't break if absent)
                "calibration_min_samples", "calibration_max_folds",
            ]
            for k in key_fields:
                if getattr(art.cfg, k, None) != getattr(cfg, k, None):
                    logging.info("%s: cfg changed (%s). Retraining signal model.", ticker, k)
                    force_retrain = True
                    break

            if not force_retrain:
                return art

        except Exception as e:
            logging.warning("%s: failed to load artifact (%s). Retraining.", ticker, e)
            force_retrain = True

    logging.info("%s: training calibrated signal model (force=%s)", ticker, force_retrain)
    artifact, report = train_calibrated_model(df, cfg)
    save_artifact(ticker, artifact, report)
    return artifact
