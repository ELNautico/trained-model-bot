import numpy as np

def adjust_leverage_for_volatility(base_leverage: float, volatility: float, threshold: float = 0.02) -> float:
    """
    Smoothly reduces leverage if volatility exceeds a safe threshold.

    Args:
        base_leverage: initial leverage recommendation.
        volatility: observed volatility (e.g. std of returns).
        threshold: volatility threshold above which leverage reduces.

    Returns:
        Adjusted leverage as a float, not dropping below the minimum of 2.
    """
    # If volatility is too high, scale leverage down proportionally
    if volatility > threshold and volatility > 0:
        adjusted = base_leverage * (threshold / volatility)
        # Ensure minimum leverage of 2
        return float(max(2.0, adjusted))
    # Otherwise keep original leverage
    return float(base_leverage)


def calculate_optimal_leverage(
    current_price: float,
    predicted_price: float,
    cert_type: str,
    knockout_buffer: float = 0.10
) -> float:
    """
    Determines a smooth, continuous leverage recommendation based on predicted movement,
    risk margin (knockout buffer), and desired risk-return ratio.

    Uses a continuous clamp rather than snapping to discrete levels.
    """
    # Compute percentage change
    pct_change = (predicted_price / current_price - 1) * 100.0
    # If movement too small, return minimum leverage
    if abs(pct_change) < 0.01:
        return 2.0

    # Base candidate leverage (continuous), clamped between 2x and 40x
    candidate = 50.0 / abs(pct_change)
    candidate = float(np.clip(candidate, 2.0, 40.0))

    # Compute risk margin parameters
    if cert_type == "Call":
        barrier = current_price * (1 - knockout_buffer)
        profit = predicted_price - current_price
        gap = current_price - barrier
    elif cert_type == "Put":
        barrier = current_price * (1 + knockout_buffer)
        profit = current_price - predicted_price
        gap = barrier - current_price
    else:
        # If not a Call/Put, just use candidate
        return candidate

    # Risk ratio: reward per unit gap
    risk_ratio = profit / gap if gap != 0 else 0.0

    # Smooth adjustment: ramps from 2x at ratio=1 to candidate at ratio>=2
    if risk_ratio <= 1.0:
        adjusted = 2.0
    elif risk_ratio < 2.0:
        adjusted = float(np.interp(risk_ratio, [1.0, 2.0], [2.0, candidate]))
    else:
        adjusted = candidate

    return adjusted


def generate_certificate(current_price: float, predicted_price: float, volatility: float) -> str:
    """
    Generates a human-readable recommendation for a leverage product.
    """
    if predicted_price > current_price:
        cert_type = "Call"
        desc = "Call Certificate (Bullish)"
    elif predicted_price < current_price:
        cert_type = "Put"
        desc = "Put Certificate (Bearish)"
    else:
        return "Neutral Certificate (No strong signal)"

    base_leverage = calculate_optimal_leverage(current_price, predicted_price, cert_type)
    safe_leverage = adjust_leverage_for_volatility(base_leverage, volatility)
    return f"{desc} with recommended leverage {safe_leverage:.2f}x"


def determine_trade_signal(
    predicted_pct_change: float,
    trend: str,
    rsi: float,
    volume: float,
    min_threshold: float = 0.5
) -> tuple[str, float]:
    """
    Returns a simple trade recommendation based on predicted move, trend, and RSI.
    """
    abs_change = abs(predicted_pct_change)
    if abs_change < min_threshold:
        return "No Trade", 0.0

    confidence = float(min(100.0, abs_change * 10.0))
    if predicted_pct_change > 0 and trend == "Bullish" and rsi < 70:
        return "Buy", confidence
    elif predicted_pct_change < 0 and trend == "Bearish" and rsi > 30:
        return "Sell", confidence
    else:
        return "No Trade", confidence


def determine_position_size(
    account_balance: float,
    risk_per_trade: float,
    volatility: float
) -> float:
    """
    Computes a position size based on risk exposure and volatility.
    """
    base = account_balance * risk_per_trade
    # Avoid division by zero
    safe_vol = max(volatility, 0.01)
    return float(base / safe_vol)
