from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalConfig:

    """
    Triple-barrier signal configuration.

    Key design:
      - Model predicts probabilities for {-1, 0, +1} labels:
          -1 = stop-first
           0 = timeout
          +1 = profit-first
      - Trade decisions are made via EV in R-units (reward/risk), not p_profit - p_stop.

    IMPORTANT calibration note:
      - For rare classes (like timeout for TSLA), aggressive inverse-frequency weights can
        distort calibration. This config defaults to "tempered" weighting and uses
        a time-based calibration holdout without weights.
    """

    # Feature window length used for each decision sample
    window_size: int = 60

    # Triple-barrier horizon (trading days)
    horizon_days: int = 10

    # Barriers in ATR units
    take_profit_atr: float = 1.5
    stop_loss_atr: float = 1.0

    # ------------------------------------------------------------------
    # EV-based entry / exit gates (in R-units)
    # ------------------------------------------------------------------
    entry_min_ev: float = 0.12
    exit_min_ev: float = -0.05
    exit_min_p_stop: float = 0.55

    # Legacy thresholds (kept for backwards compat; not used by EV logic)
    entry_min_p_profit: float = 0.45
    entry_min_edge: float = 0.10
    exit_min_edge: float = 0.10

    # Risk sizing caps
    max_position_fraction: float = 0.25
    
    # One-way cost: commission + slippage (bid-ask spread + market impact)
    one_way_cost_bps: float = 5.0

    # Feature switches
    use_sentiment: bool = False

    # ------------------------------------------------------------------
    # Signal model training knobs (tabular classifier)
    # ------------------------------------------------------------------
    random_state: int = 42
    max_iter: int = 300
    learning_rate: float = 0.05
    max_depth: int = 3

    # Mild regularization to reduce overfit on flattened windows
    l2_regularization: float = 0.10

    # ------------------------------------------------------------------
    # Probability calibration (time-series correct)
    # ------------------------------------------------------------------
    # Use the most recent calibration_ratio of samples as the calibration set.
    # Example: 0.20 means last 20% of samples are used *only* for calibration.
    calibration_ratio: float = 0.20

    # Calibration method: "sigmoid" is robust for small calibration sets.
    calibration_method: str = "sigmoid"

    # ------------------------------------------------------------------
    # Class imbalance handling (TEMPERED weights)
    # ------------------------------------------------------------------
    # If True, we apply sample weights when fitting the base model (NOT during calibration).
    use_class_weights: bool = True

    # Weighting mode:
    #   "none"      -> no weights
    #   "inv"       -> inverse frequency (often too aggressive)
    #   "sqrt_inv"  -> sqrt inverse frequency (recommended default)
    class_weight_mode: str = "sqrt_inv"

    # Cap any class weight to avoid runaway rare-class inflation.
    max_class_weight: float = 3.0

    def __post_init__(self) -> None:
        errors = []

        # --- Feature window & horizon ---
        if self.window_size < 1:
            errors.append(f"window_size must be >= 1, got {self.window_size}")
        if self.horizon_days < 1:
            errors.append(f"horizon_days must be >= 1, got {self.horizon_days}")

        # --- Barrier sizing ---
        if self.take_profit_atr <= 0:
            errors.append(f"take_profit_atr must be > 0, got {self.take_profit_atr}")
        if self.stop_loss_atr <= 0:
            errors.append(f"stop_loss_atr must be > 0, got {self.stop_loss_atr}")

        # --- EV gates ---
        if self.exit_min_ev >= self.entry_min_ev:
            errors.append(
                f"exit_min_ev ({self.exit_min_ev}) must be < entry_min_ev ({self.entry_min_ev})"
            )
        if not (0.0 < self.exit_min_p_stop <= 1.0):
            errors.append(f"exit_min_p_stop must be in (0, 1], got {self.exit_min_p_stop}")

        # --- Risk sizing ---
        if not (0.0 < self.max_position_fraction <= 1.0):
            errors.append(
                f"max_position_fraction must be in (0, 1], got {self.max_position_fraction}"
            )
        if self.one_way_cost_bps < 0:
            errors.append(f"one_way_cost_bps must be >= 0, got {self.one_way_cost_bps}")

        # --- Model training knobs ---
        if self.max_iter < 1:
            errors.append(f"max_iter must be >= 1, got {self.max_iter}")
        if self.max_depth < 1:
            errors.append(f"max_depth must be >= 1, got {self.max_depth}")
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.l2_regularization < 0:
            errors.append(f"l2_regularization must be >= 0, got {self.l2_regularization}")

        # --- Calibration ---
        if not (0.0 < self.calibration_ratio < 1.0):
            errors.append(
                f"calibration_ratio must be in (0, 1), got {self.calibration_ratio}"
            )
        if self.calibration_method not in {"sigmoid", "isotonic"}:
            errors.append(
                f"calibration_method must be 'sigmoid' or 'isotonic', "
                f"got '{self.calibration_method}'"
            )

        # --- Class weighting ---
        if self.class_weight_mode not in {"none", "inv", "sqrt_inv"}:
            errors.append(
                f"class_weight_mode must be 'none', 'inv', or 'sqrt_inv', "
                f"got '{self.class_weight_mode}'"
            )
        if self.max_class_weight < 1.0:
            errors.append(f"max_class_weight must be >= 1.0, got {self.max_class_weight}")

        if errors:
            raise ValueError(
                "Invalid SignalConfig:\n" + "\n".join(f"  • {e}" for e in errors)
            )
