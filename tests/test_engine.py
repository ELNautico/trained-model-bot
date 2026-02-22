"""
Unit tests for the pure (stateless) helper functions in signals/engine.py.

Functions tested:
  _risk_size_shares
  _reward_risk_and_cost_in_R
  _ev_net_in_R
  decide_action_ev

The module-level imports of train.pipeline / core.storage are mocked via
conftest.py, so no DB or API access happens.
"""
import pytest

from signals.config import SignalConfig
from signals.engine import (
    _ev_net_in_R,
    _reward_risk_and_cost_in_R,
    _risk_size_shares,
    decide_action_ev,
)

_CFG = SignalConfig()  # uses all validated defaults


# ── _risk_size_shares ──────────────────────────────────────────────────────────

class TestRiskSizeShares:

    def test_basic_sizing(self):
        # account=10_000, risk=1%, entry=100, stop=90
        # risk_dollars = 100, stop_dist = 10
        # shares = floor(100/10) = 10
        # max_dollars = 2500, max_shares = floor(2500/100) = 25 → shares=10
        shares, notional = _risk_size_shares(
            account_balance=10_000,
            risk_per_trade=0.01,
            entry_px=100.0,
            stop_px=90.0,
            cfg=_CFG,
        )
        assert shares == 10
        assert notional == pytest.approx(1_000.0)

    def test_capped_by_max_position_fraction(self):
        # Tiny stop distance → uncapped shares would be huge
        # entry=100, stop=99.99 → stop_dist=max(0.01, 0.01)=0.01
        # risk_dollars=100 → raw shares=10_000
        # max_dollars=10_000*0.25=2500, max_shares=floor(2500/100)=25 → capped
        shares, _ = _risk_size_shares(
            account_balance=10_000,
            risk_per_trade=0.01,
            entry_px=100.0,
            stop_px=99.99,
            cfg=_CFG,
        )
        assert shares == 25

    def test_zero_balance_yields_zero_shares(self):
        shares, notional = _risk_size_shares(
            account_balance=0.0,
            risk_per_trade=0.01,
            entry_px=100.0,
            stop_px=90.0,
            cfg=_CFG,
        )
        assert shares == 0
        assert notional == pytest.approx(0.0)

    def test_entry_equals_stop_uses_minimum_distance(self):
        # stop_dist = max(0.01, 100-100) = 0.01 → does not divide by zero
        shares, _ = _risk_size_shares(
            account_balance=10_000,
            risk_per_trade=0.01,
            entry_px=100.0,
            stop_px=100.0,
            cfg=_CFG,
        )
        assert shares >= 0  # should not crash

    def test_notional_equals_shares_times_entry(self):
        shares, notional = _risk_size_shares(
            account_balance=50_000,
            risk_per_trade=0.02,
            entry_px=250.0,
            stop_px=230.0,
            cfg=_CFG,
        )
        assert notional == pytest.approx(shares * 250.0)

    def test_shares_non_negative(self):
        # Deliberately pathological inputs
        shares, notional = _risk_size_shares(
            account_balance=1.0,
            risk_per_trade=0.001,
            entry_px=500.0,
            stop_px=499.0,
            cfg=_CFG,
        )
        assert shares >= 0
        assert notional >= 0.0


# ── _reward_risk_and_cost_in_R ────────────────────────────────────────────────

class TestRewardRiskAndCostInR:

    def test_known_values(self):
        # entry=100, stop=90, target=115, cost=5bps
        # risk=10, reward=15 → R=1.5
        # roundtrip_frac=2*5/10000=0.001
        # cost_in_R=0.001*100/10=0.01
        R, cost = _reward_risk_and_cost_in_R(
            entry_px=100.0, stop_px=90.0, target_px=115.0, one_way_cost_bps=5.0
        )
        assert R    == pytest.approx(1.5)
        assert cost == pytest.approx(0.01)

    def test_zero_cost_bps(self):
        R, cost = _reward_risk_and_cost_in_R(
            entry_px=100.0, stop_px=90.0, target_px=115.0, one_way_cost_bps=0.0
        )
        assert R    == pytest.approx(1.5)
        assert cost == pytest.approx(0.0)

    def test_entry_equals_stop_does_not_crash(self):
        # stop_dist = max(0.01, 0) = 0.01
        R, cost = _reward_risk_and_cost_in_R(
            entry_px=100.0, stop_px=100.0, target_px=115.0, one_way_cost_bps=5.0
        )
        assert R >= 0.0
        assert cost >= 0.0

    def test_target_below_entry_gives_zero_reward(self):
        # reward = max(0.0, target-entry) = 0
        R, _ = _reward_risk_and_cost_in_R(
            entry_px=100.0, stop_px=90.0, target_px=95.0, one_way_cost_bps=0.0
        )
        assert R == pytest.approx(0.0)

    def test_symmetry_with_r_ratio(self):
        # R should equal take_profit_atr / stop_loss_atr when cost is free
        cfg = SignalConfig(take_profit_atr=2.0, stop_loss_atr=1.0)
        # entry=100, ATR=10 → stop=90, target=120 → R=20/10=2.0
        R, _ = _reward_risk_and_cost_in_R(
            entry_px=100.0, stop_px=90.0, target_px=120.0, one_way_cost_bps=0.0
        )
        assert R == pytest.approx(2.0)


# ── _ev_net_in_R ───────────────────────────────────────────────────────────────

class TestEvNetInR:

    def test_known_value(self):
        # EV = 0.5*1.5 - 0.3 - 0.01 = 0.75 - 0.31 = 0.44
        ev = _ev_net_in_R(p_profit=0.5, p_stop=0.3, R=1.5, cost_in_R=0.01)
        assert ev == pytest.approx(0.44)

    def test_zero_probabilities(self):
        ev = _ev_net_in_R(p_profit=0.0, p_stop=0.0, R=2.0, cost_in_R=0.0)
        assert ev == pytest.approx(0.0)

    def test_negative_ev(self):
        # p_profit low, p_stop high → negative EV
        ev = _ev_net_in_R(p_profit=0.1, p_stop=0.7, R=1.5, cost_in_R=0.05)
        assert ev < 0.0

    def test_cost_reduces_ev(self):
        ev_no_cost  = _ev_net_in_R(p_profit=0.5, p_stop=0.3, R=1.5, cost_in_R=0.00)
        ev_with_cost = _ev_net_in_R(p_profit=0.5, p_stop=0.3, R=1.5, cost_in_R=0.10)
        assert ev_no_cost > ev_with_cost

    def test_higher_r_improves_ev(self):
        ev_low_r  = _ev_net_in_R(p_profit=0.5, p_stop=0.3, R=1.0, cost_in_R=0.01)
        ev_high_r = _ev_net_in_R(p_profit=0.5, p_stop=0.3, R=3.0, cost_in_R=0.01)
        assert ev_high_r > ev_low_r


# ── decide_action_ev ───────────────────────────────────────────────────────────

class TestDecideActionEv:
    """
    Default SignalConfig: entry_min_ev=0.12, exit_min_ev=-0.05,
                         exit_min_p_stop=0.55
    """

    def _probs(self, p_profit, p_stop):
        return {"p_profit": p_profit, "p_stop": p_stop, "p_timeout": 1 - p_profit - p_stop}

    # BUY scenario: no position, EV >= entry_min_ev
    def test_buy_when_ev_above_entry_gate(self):
        # EV = 0.5*1.5 - 0.2 - 0.01 = 0.54 >= 0.12
        action, rationale, ev = decide_action_ev(
            probs=self._probs(0.5, 0.2),
            position=None, R=1.5, cost_in_R=0.01, cfg=_CFG,
        )
        assert action == "BUY"
        assert ev >= _CFG.entry_min_ev
        assert "EV" in rationale

    # WAIT scenario: no position, EV < entry_min_ev
    def test_wait_when_ev_below_entry_gate(self):
        # EV = 0.2*1.5 - 0.3 - 0.01 = -0.01 < 0.12
        action, rationale, ev = decide_action_ev(
            probs=self._probs(0.2, 0.3),
            position=None, R=1.5, cost_in_R=0.01, cfg=_CFG,
        )
        assert action == "WAIT"
        assert ev < _CFG.entry_min_ev

    # SELL (EV gate): position open, EV drops to or below exit_min_ev
    def test_sell_when_ev_triggers_exit_gate(self):
        # EV = 0.1*1.5 - 0.6 - 0.01 = -0.46 <= -0.05
        pos = {"state": "LONG"}
        action, _, ev = decide_action_ev(
            probs=self._probs(0.1, 0.6),
            position=pos, R=1.5, cost_in_R=0.01, cfg=_CFG,
        )
        assert action == "SELL"
        assert ev <= _CFG.exit_min_ev

    # SELL (p_stop gate): position open, EV ok but p_stop too high
    def test_sell_when_p_stop_triggers_exit_gate(self):
        # R=2.0: EV = 0.3*2.0 - 0.56 - 0.01 = 0.03 > -0.05 (EV gate not met)
        # p_stop=0.56 >= 0.55 → p_stop gate fires
        pos = {"state": "LONG"}
        action, rationale, _ = decide_action_ev(
            probs=self._probs(0.3, 0.56),
            position=pos, R=2.0, cost_in_R=0.01, cfg=_CFG,
        )
        assert action == "SELL"
        assert "stop-risk" in rationale

    # HOLD: position open, both gates safe
    def test_hold_when_both_gates_safe(self):
        # EV = 0.5*1.5 - 0.2 - 0.01 = 0.54 > -0.05 ✓, p_stop=0.2 < 0.55 ✓
        pos = {"state": "LONG"}
        action, _, _ = decide_action_ev(
            probs=self._probs(0.5, 0.2),
            position=pos, R=1.5, cost_in_R=0.01, cfg=_CFG,
        )
        assert action == "HOLD"

    def test_ev_exactly_at_entry_gate_triggers_buy(self):
        # EV exactly == entry_min_ev=0.12 → BUY (>= check)
        # 0.12 = p_profit*R - p_stop - cost → pick: R=2.0, p_stop=0.1, cost=0.01
        # p_profit = (0.12 + 0.1 + 0.01) / 2.0 = 0.115
        p_profit = (0.12 + 0.10 + 0.01) / 2.0
        action, _, ev = decide_action_ev(
            probs=self._probs(p_profit, 0.10),
            position=None, R=2.0, cost_in_R=0.01, cfg=_CFG,
        )
        assert action == "BUY"
        assert ev == pytest.approx(_CFG.entry_min_ev, abs=1e-9)

    def test_return_tuple_structure(self):
        action, rationale, ev = decide_action_ev(
            probs=self._probs(0.5, 0.2),
            position=None, R=1.5, cost_in_R=0.01, cfg=_CFG,
        )
        assert isinstance(action, str)
        assert isinstance(rationale, str)
        assert isinstance(ev, float)
        assert action in {"BUY", "WAIT", "SELL", "HOLD"}
