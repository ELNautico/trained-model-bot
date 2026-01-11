from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Optional, Tuple

import numpy as np

from signals.config import SignalConfig
from signals.labeling import compute_barriers_from_row
from signals.model import ensure_model
from storage import (
    get_position,
    upsert_position,
    close_position,
    save_signal,
    save_trade,
)
from train.pipeline import download_data  # your cached OHLCV download
from core.features import enrich_features


def _vienna_now_str() -> str:
    now = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Europe/Vienna"))
    return now.strftime("%d.%m.%Y %H:%M")


def _risk_size_shares(
    *,
    account_balance: float,
    risk_per_trade: float,
    entry_px: float,
    stop_px: float,
    cfg: SignalConfig,
) -> Tuple[int, float]:
    """
    Transparent sizing:
      risk_dollars = account_balance * risk_per_trade
      shares = risk_dollars / (entry_px - stop_px)
      cap by max_position_fraction of account

    Note:
      - This is stop-distance risk sizing (classic and easy to audit).
      - If you want volatility targeting or Kelly, we can add later.
    """
    risk_dollars = float(account_balance * risk_per_trade)
    stop_dist = max(0.01, float(entry_px - stop_px))
    shares = int(max(0, np.floor(risk_dollars / stop_dist)))

    # Cap by max fraction of account
    max_dollars = float(account_balance * cfg.max_position_fraction)
    max_shares = int(max(0, np.floor(max_dollars / max(entry_px, 0.01))))
    shares = int(min(shares, max_shares))

    return shares, float(shares * entry_px)


def _reward_risk_and_cost_in_R(
    *,
    entry_px: float,
    stop_px: float,
    target_px: float,
    one_way_cost_bps: float,
) -> Tuple[float, float]:
    """
    Compute:
      R = reward / risk
      cost_in_R = roundtrip_cost / risk, expressed in "R units"

    Roundtrip cost fraction:
      roundtrip_cost_frac = 2 * one_way_cost_bps / 10000

    Convert to R-units:
      cost_in_R ≈ roundtrip_cost_frac * entry / (entry - stop)

    This is a practical approximation that makes EV thresholds consistent across
    tickers/price levels.
    """
    risk = max(0.01, float(entry_px - stop_px))
    reward = max(0.0, float(target_px - entry_px))
    R = float(reward / risk) if risk > 0 else 0.0

    roundtrip_cost_frac = 2.0 * float(one_way_cost_bps) / 10000.0
    cost_in_R = float(roundtrip_cost_frac * float(entry_px) / risk) if risk > 0 else 0.0
    return R, cost_in_R


def _ev_net_in_R(
    *,
    p_profit: float,
    p_stop: float,
    R: float,
    cost_in_R: float,
) -> float:
    """
    Net expected value in units of "R" (stop-distance multiples).

    EV_net = p_profit * R - p_stop - cost_in_R

    - Timeout payoff is treated as 0 here.
    - This keeps EV conservative and avoids assuming a terminal payoff for timeout.
    """
    return float(p_profit * R - p_stop - cost_in_R)


def decide_action_ev(
    *,
    probs: Dict[str, float],
    position: Optional[Dict],
    R: float,
    cost_in_R: float,
    cfg: SignalConfig,
) -> Tuple[str, str, float]:
    """
    EV-based state machine decision.

    Returns:
      action, rationale, ev_net
    """
    p_profit = float(probs["p_profit"])
    p_stop = float(probs["p_stop"])
    p_timeout = float(probs["p_timeout"])

    ev_net = _ev_net_in_R(p_profit=p_profit, p_stop=p_stop, R=R, cost_in_R=cost_in_R)

    # Break-even condition for EV_net >= 0:
    #   p_profit >= (p_stop + cost_in_R) / R
    if R > 1e-6:
        p_profit_be = float((p_stop + cost_in_R) / R)
    else:
        p_profit_be = 1.0

    if position is None:
        if ev_net >= cfg.entry_min_ev:
            rationale = (
                f"Entry EV gate met: EV_net={ev_net:.2f}R >= {cfg.entry_min_ev:.2f}R | "
                f"R={R:.2f}, cost≈{cost_in_R:.2f}R | "
                f"p_profit={p_profit:.2f} (BE≥{p_profit_be:.2f}), p_stop={p_stop:.2f}, p_timeout={p_timeout:.2f}"
            )
            return "BUY", rationale, ev_net

        rationale = (
            f"No entry: EV_net={ev_net:.2f}R < {cfg.entry_min_ev:.2f}R | "
            f"R={R:.2f}, cost≈{cost_in_R:.2f}R | "
            f"p_profit={p_profit:.2f} (BE≥{p_profit_be:.2f}), p_stop={p_stop:.2f}, p_timeout={p_timeout:.2f}"
        )
        return "WAIT", rationale, ev_net

    # LONG: exit if EV deteriorates or stop-risk becomes too high
    if ev_net <= cfg.exit_min_ev:
        rationale = (
            f"Exit EV gate met: EV_net={ev_net:.2f}R <= {cfg.exit_min_ev:.2f}R | "
            f"R={R:.2f}, cost≈{cost_in_R:.2f}R | "
            f"p_profit={p_profit:.2f}, p_stop={p_stop:.2f}, p_timeout={p_timeout:.2f}"
        )
        return "SELL", rationale, ev_net

    if p_stop >= cfg.exit_min_p_stop:
        rationale = (
            f"Exit stop-risk gate met: p_stop={p_stop:.2f} >= {cfg.exit_min_p_stop:.2f} | "
            f"EV_net={ev_net:.2f}R, R={R:.2f}, cost≈{cost_in_R:.2f}R"
        )
        return "SELL", rationale, ev_net

    rationale = (
        f"Hold: EV_net={ev_net:.2f}R | R={R:.2f}, cost≈{cost_in_R:.2f}R | "
        f"p_profit={p_profit:.2f}, p_stop={p_stop:.2f}, p_timeout={p_timeout:.2f}"
    )
    return "HOLD", rationale, ev_net


def run_signal_cycle_for_ticker(
    ticker: str,
    *,
    cfg: SignalConfig,
    account_balance: float,
    risk_per_trade: float,
    force_retrain: bool = False,
) -> str:
    """
    Main daily loop for one ticker:
      1) Download + feature engineering
      2) Ensure model exists (retrain if forced)
      3) Predict probabilities for latest decision bar
      4) Compute barriers and EV
      5) Apply FLAT/LONG state machine; possibly open/close position
      6) Persist signal and trades
      7) Return a human-readable message
    """
    df = download_data(ticker)
    df.attrs["ticker"] = ticker
    df = enrich_features(df)

    model = ensure_model(ticker, df, cfg, force_retrain=force_retrain)
    probs = model.predict_proba_last(df)

    pos = get_position(ticker)

    # Barriers computed from the latest completed bar, using entry≈last close for messaging/sizing.
    # If you later implement "execute next open" strictly, you'll need an orders table.
    decision_row = df.iloc[-1]
    current_close = float(df["Close"].iloc[-1])
    barriers = compute_barriers_from_row(decision_row, entry_px=current_close, cfg=cfg)

    # Reward-risk and cost normalization
    R, cost_in_R = _reward_risk_and_cost_in_R(
        entry_px=barriers.entry_px,
        stop_px=barriers.stop_px,
        target_px=barriers.target_px,
        one_way_cost_bps=cfg.one_way_cost_bps,
    )

    action, rationale, ev_net = decide_action_ev(
        probs=probs,
        position=pos,
        R=R,
        cost_in_R=cost_in_R,
        cfg=cfg,
    )

    # Persist daily signal record
    ts = df.index[-1].date().isoformat()
    save_signal({
        "ts": ts,
        "ticker": ticker.upper(),
        "action": action,
        "p_profit": probs["p_profit"],
        "p_timeout": probs["p_timeout"],
        "p_stop": probs["p_stop"],
        "edge": probs["p_profit"] - probs["p_stop"],  # retained for debugging only
        "entry_px": barriers.entry_px,
        "stop_px": barriers.stop_px,
        "target_px": barriers.target_px,
        "horizon_days": barriers.horizon_days,
        "meta_json": {
            "rationale": rationale,
            "cfg": asdict(cfg),
            "R": R,
            "cost_in_R": cost_in_R,
            "ev_net_R": ev_net,
        },
    })

    # Apply state transitions
    if action == "BUY" and pos is None:
        shares, _notional = _risk_size_shares(
            account_balance=account_balance,
            risk_per_trade=risk_per_trade,
            entry_px=barriers.entry_px,
            stop_px=barriers.stop_px,
            cfg=cfg,
        )

        if shares <= 0:
            action = "WAIT"
            rationale = "Sizing yielded 0 shares (risk too small vs stop distance)."
        else:
            upsert_position({
                "ticker": ticker.upper(),
                "state": "LONG",
                "entry_ts": ts,
                "entry_px": barriers.entry_px,
                "shares": shares,
                "stop_px": barriers.stop_px,
                "target_px": barriers.target_px,
                "horizon_days": barriers.horizon_days,
                "hold_days": 0,
                "last_update_ts": ts,
            })

    elif action == "SELL" and pos is not None and pos.get("state") == "LONG":
        # Soft exit is executed at current close in DB for simplicity/auditability.
        exit_px = current_close
        shares = int(pos.get("shares", 0))
        entry_px = float(pos.get("entry_px", current_close))
        pnl = float((exit_px - entry_px) * shares)
        ret = float((exit_px / entry_px - 1.0) if entry_px > 0 else 0.0)

        save_trade({
            "ts": ts,
            "ticker": ticker.upper(),
            "side": "LONG",
            "entry_ts": pos.get("entry_ts"),
            "entry_px": entry_px,
            "exit_ts": ts,
            "exit_px": exit_px,
            "shares": shares,
            "reason": "model_exit_ev",
            "pnl": pnl,
            "return_pct": ret * 100.0,
        })
        close_position(ticker)

    # Build message
    ts_local = df.index[-1].tz_convert(ZoneInfo("Europe/Vienna")) if getattr(df.index[-1], "tz", None) else df.index[-1]
    msg = (
        f"{ticker.upper()} Signal ({ts_local.strftime('%Y-%m-%d')})\n"
        f"Time: {_vienna_now_str()} (Vienna)\n\n"
        f"Action: {action}\n\n"
        f"Probabilities (next {cfg.horizon_days}d after entry):\n"
        f"  Profit-first: {probs['p_profit']:.2f}\n"
        f"  Stop-first:   {probs['p_stop']:.2f}\n"
        f"  Timeout:      {probs['p_timeout']:.2f}\n\n"
        f"Levels (ATR-based, entry≈last close):\n"
        f"  Entry:  {barriers.entry_px:.2f}\n"
        f"  Stop:   {barriers.stop_px:.2f}\n"
        f"  Target: {barriers.target_px:.2f}\n\n"
        f"EV framing:\n"
        f"  R (reward/risk): {R:.2f}\n"
        f"  Cost (approx):   {cost_in_R:.2f}R\n"
        f"  EV_net:          {ev_net:.2f}R\n\n"
        f"Notes: {rationale}"
    )

    # Position summary
    pos2 = get_position(ticker)
    if pos2 is None:
        msg += "\n\nPosition: FLAT"
    else:
        msg += (
            f"\n\nPosition: {pos2.get('state')}\n"
            f"  Shares: {pos2.get('shares')}\n"
            f"  Entry:  {float(pos2.get('entry_px', 0.0)):.2f}\n"
            f"  Stop:   {float(pos2.get('stop_px', 0.0)):.2f}\n"
            f"  Target: {float(pos2.get('target_px', 0.0)):.2f}\n"
            f"  Hold days: {int(pos2.get('hold_days', 0))}/{int(pos2.get('horizon_days', cfg.horizon_days))}"
        )

    return msg


def run_eod_position_checks(ticker: str) -> Optional[str]:
    """
    End-of-day "hard exit" checks for open positions:
      - stop-loss hit
      - take-profit hit
      - time stop (horizon exceeded)

    Conservative OHLC handling:
      - If both stop and target touched in same day, treat it as stop first.
    """
    pos = get_position(ticker)
    if pos is None or pos.get("state") != "LONG":
        return None

    df = download_data(ticker)
    df.attrs["ticker"] = ticker
    df = enrich_features(df)

    ts = df.index[-1].date().isoformat()
    hi = float(df["High"].iloc[-1])
    lo = float(df["Low"].iloc[-1])
    close = float(df["Close"].iloc[-1])

    stop_px = float(pos.get("stop_px"))
    target_px = float(pos.get("target_px"))
    entry_px = float(pos.get("entry_px"))
    shares = int(pos.get("shares", 0))

    # Update hold days
    hold_days = int(pos.get("hold_days", 0)) + 1
    horizon = int(pos.get("horizon_days", 0))

    exit_reason = None
    exit_px = None

    if (lo <= stop_px) and (hi >= target_px):
        exit_reason, exit_px = "stop_loss", stop_px
    elif lo <= stop_px:
        exit_reason, exit_px = "stop_loss", stop_px
    elif hi >= target_px:
        exit_reason, exit_px = "take_profit", target_px
    elif horizon > 0 and hold_days >= horizon:
        exit_reason, exit_px = "time_stop", close

    if exit_reason is not None:
        pnl = float((exit_px - entry_px) * shares)
        ret = float((exit_px / entry_px - 1.0) if entry_px > 0 else 0.0)

        save_trade({
            "ts": ts,
            "ticker": ticker.upper(),
            "side": "LONG",
            "entry_ts": pos.get("entry_ts"),
            "entry_px": entry_px,
            "exit_ts": ts,
            "exit_px": float(exit_px),
            "shares": shares,
            "reason": exit_reason,
            "pnl": pnl,
            "return_pct": ret * 100.0,
        })
        close_position(ticker)
        return (
            f"{ticker.upper()} EXIT ({ts})\n"
            f"Reason: {exit_reason}\n"
            f"Exit px: {exit_px:.2f}\n"
            f"PnL: {pnl:.2f} ({ret*100.0:.2f}%)"
        )

    # Persist hold-day update (still LONG)
    upsert_position({
        "ticker": ticker.upper(),
        "state": "LONG",
        "entry_ts": pos.get("entry_ts"),
        "entry_px": entry_px,
        "shares": shares,
        "stop_px": stop_px,
        "target_px": target_px,
        "horizon_days": horizon,
        "hold_days": hold_days,
        "last_update_ts": ts,
    })

    return None
