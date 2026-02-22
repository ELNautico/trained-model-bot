"""
Unit tests for signals/config.py – SignalConfig.__post_init__ validation.
"""
import pytest
from signals.config import SignalConfig


def test_default_config_is_valid():
    """Default constructor must not raise."""
    cfg = SignalConfig()
    assert cfg.window_size == 60
    assert cfg.horizon_days == 10


# ── Individual field violations ───────────────────────────────────────────────

@pytest.mark.parametrize("field,value", [
    ("window_size",          0),
    ("window_size",         -1),
    ("horizon_days",         0),
    ("horizon_days",        -5),
    ("stop_loss_atr",        0.0),
    ("stop_loss_atr",       -1.0),
    ("take_profit_atr",      0.0),
    ("take_profit_atr",     -2.0),
    ("max_position_fraction", 0.0),
    ("max_position_fraction", 1.1),
    ("one_way_cost_bps",    -1.0),
    ("max_iter",             0),
    ("max_depth",            0),
    ("learning_rate",        0.0),
    ("learning_rate",       -0.01),
    ("l2_regularization",   -0.1),
    ("calibration_ratio",    0.0),
    ("calibration_ratio",    1.0),
    ("max_class_weight",     0.9),
])
def test_invalid_field_raises(field, value):
    with pytest.raises(ValueError, match="Invalid SignalConfig"):
        SignalConfig(**{field: value})


@pytest.mark.parametrize("method", ["linear", "platt", "", "SIGMOID"])
def test_invalid_calibration_method(method):
    with pytest.raises(ValueError, match="Invalid SignalConfig"):
        SignalConfig(calibration_method=method)


@pytest.mark.parametrize("mode", ["balanced", "auto", "INV"])
def test_invalid_class_weight_mode(mode):
    with pytest.raises(ValueError, match="Invalid SignalConfig"):
        SignalConfig(class_weight_mode=mode)


def test_exit_ev_must_be_less_than_entry_ev():
    """exit_min_ev must be strictly less than entry_min_ev."""
    with pytest.raises(ValueError, match="Invalid SignalConfig"):
        SignalConfig(entry_min_ev=0.10, exit_min_ev=0.10)
    with pytest.raises(ValueError, match="Invalid SignalConfig"):
        SignalConfig(entry_min_ev=0.10, exit_min_ev=0.20)


def test_exit_min_p_stop_bounds():
    with pytest.raises(ValueError, match="Invalid SignalConfig"):
        SignalConfig(exit_min_p_stop=0.0)
    with pytest.raises(ValueError, match="Invalid SignalConfig"):
        SignalConfig(exit_min_p_stop=1.1)
    # boundary: 1.0 is valid
    cfg = SignalConfig(exit_min_p_stop=1.0)
    assert cfg.exit_min_p_stop == 1.0


def test_multiple_errors_collected():
    """All invalid fields should appear in a single ValueError."""
    with pytest.raises(ValueError) as exc:
        SignalConfig(window_size=0, horizon_days=0, stop_loss_atr=-1.0)
    msg = str(exc.value)
    assert "window_size" in msg
    assert "horizon_days" in msg
    assert "stop_loss_atr" in msg


def test_valid_boundary_values():
    """Values at the boundary of acceptable ranges must not raise."""
    cfg = SignalConfig(
        window_size=1,
        horizon_days=1,
        calibration_ratio=0.05,
        max_class_weight=1.0,
        one_way_cost_bps=0.0,
        l2_regularization=0.0,
        calibration_method="isotonic",
        class_weight_mode="none",
        use_class_weights=False,
    )
    assert cfg.window_size == 1
