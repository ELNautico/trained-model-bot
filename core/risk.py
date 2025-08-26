"""
Risk & position-sizing utilities.

Key upgrades
------------
1. Leverage scaling targets a *desired portfolio vol* instead of a fixed threshold.
2. Position size can be computed either via classic vol-targeting *or*
   fractional Kelly (f = k * μ / σ²).
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Leverage helpers
# ─────────────────────────────────────────────────────────────────────────────
def adjust_leverage_for_volatility(
    base_leverage: float,
    volatility: float,
    target_vol: float = 0.02,          # 2 % daily vol target
    min_leverage: float = 2.0,
) -> float:
    """
    Scale leverage so that  (leverage × σ)  ≈ target_vol.

    Parameters
    ----------
    base_leverage : float
        Raw leverage suggestion before risk control.
    volatility : float
        Realised or forecast daily σ (e.g. std of log-returns).
    target_vol : float
        Desired portfolio volatility per day (default 2 %).
    min_leverage : float
        Never go below this leverage floor.

    Returns
    -------
    float
        Safe leverage rounded to 2 decimals.
    """
    if volatility <= 0:
        return float(base_leverage)

    scaled = target_vol / volatility
    safe_lev = max(min_leverage, base_leverage * scaled)
    return float(np.round(safe_lev, 2))


def calculate_optimal_leverage(
    current_price: float,
    predicted_price: float,
    cert_type: str,
    knockout_buffer: float = 0.10,
) -> float:
    """
    Continuous leverage heuristic based on distance to knockout barrier.
    Kept unchanged except for minor clarity tweaks.
    """
    pct_move = abs(predicted_price / current_price - 1) * 100.0
    if pct_move < 0.01:
        return 2.0

    candidate = float(np.clip(50.0 / pct_move, 2.0, 40.0))

    if cert_type == "Call":
        barrier = current_price * (1 - knockout_buffer)
        profit = predicted_price - current_price
        gap = current_price - barrier
    elif cert_type == "Put":
        barrier = current_price * (1 + knockout_buffer)
        profit = current_price - predicted_price
        gap = barrier - current_price
    else:
        return candidate

    risk_ratio = profit / gap if gap else 0.0
    if risk_ratio <= 1.0:
        return 2.0
    if risk_ratio < 2.0:
        return float(np.interp(risk_ratio, [1.0, 2.0], [2.0, candidate]))
    return candidate


def generate_certificate(
    current_price: float,
    predicted_price: float,
    volatility: float,
) -> str:
    """
    Human-readable certificate suggestion using safe leverage.
    """
    if predicted_price > current_price:
        ctype, desc = "Call", "Call Certificate (Bullish)"
    elif predicted_price < current_price:
        ctype, desc = "Put", "Put Certificate (Bearish)"
    else:
        return "Neutral Certificate (No strong signal)"

    raw_lev = calculate_optimal_leverage(current_price, predicted_price, ctype)
    safe_lev = adjust_leverage_for_volatility(raw_lev, volatility)
    return f"{desc} with recommended leverage {safe_lev:.2f}x"


# ─────────────────────────────────────────────────────────────────────────────
#  Trade / sizing helpers
# ─────────────────────────────────────────────────────────────────────────────
def determine_trade_signal(
    predicted_pct_change: float,
    trend: str,
    rsi: float,
    volume: float,
    min_threshold: float = 0.5,
) -> tuple[str, float]:
    """
    Simple heuristics for buy / sell / no-trade.
    """
    if abs(predicted_pct_change) < min_threshold:
        return "No Trade", 0.0

    confidence = float(min(100.0, abs(predicted_pct_change) * 10.0))
    if predicted_pct_change > 0 and trend == "Bullish" and rsi < 70:
        return "Buy", confidence
    if predicted_pct_change < 0 and trend == "Bearish" and rsi > 30:
        return "Sell", confidence
    return "No Trade", confidence


def determine_position_size(
    account_balance: float,
    risk_per_trade: float,
    volatility: float,
    *,
    expected_return: float | None = None,
    method: str = "vol_target",
    kelly_fraction: float = 0.5,
    max_fraction: float = 0.25,
) -> float:
    """
    Position sizing with two methods:

    1. **Volatility target** (default, backward-compatible)
       size = (account_balance × risk_per_trade) / σ

    2. **Fractional Kelly** if `method="kelly"` *and* `expected_return` supplied
       f* = k * μ / σ²      (capped at `max_fraction`)
       position = account_balance × f*

    Parameters
    ----------
    account_balance : float
        Total capital.
    risk_per_trade : float
        Capital fraction you are willing to risk (still used as a ceiling for Kelly).
    volatility : float
        Forecast or realised σ.
    expected_return : float, optional
        Forecast μ (decimal, not %). Required for Kelly sizing.
    method : {"vol_target", "kelly"}
        Sizing formula to apply.
    kelly_fraction : float
        Usually 0.25–0.5 to damp raw Kelly.
    max_fraction : float
        Hard cap on allocation regardless of formula.

    Returns
    -------
    float
        Dollar position size.
    """
    safe_vol = max(volatility, 1e-4)

    if method == "kelly" and expected_return is not None:
        raw_kelly = expected_return / (safe_vol ** 2)
        frac = kelly_fraction * raw_kelly
        frac = np.clip(frac, 0.0, min(max_fraction, risk_per_trade))
        return float(account_balance * frac)

    # fallback → classic vol target
    position = (account_balance * risk_per_trade) / safe_vol
    return float(position)
