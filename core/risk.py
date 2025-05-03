
def adjust_leverage_for_volatility(base_leverage: float, volatility: float, threshold: float = 0.02) -> float:
    """
    Reduces leverage if volatility is higher than a safe threshold.
    """
    if volatility > threshold:
        adjusted = base_leverage * (threshold / volatility)
        return max(2, round(adjusted))
    return base_leverage


def calculate_optimal_leverage(current_price: float, predicted_price: float, cert_type: str, knockout_buffer=0.10) -> int:
    """
    Determines leverage based on predicted movement and risk margin (knockout buffer).
    """
    pct_change = (predicted_price / current_price - 1) * 100
    if abs(pct_change) < 0.01:
        return 2  # Too small to justify leverage

    candidate = round(50 / abs(pct_change))
    candidate = min(40, max(2, candidate))

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

    risk_ratio = profit / gap if gap else 0
    if risk_ratio < 1:
        adjusted = 2
    elif risk_ratio < 2:
        adjusted = 2 + (risk_ratio - 1) * (candidate - 2)
    else:
        adjusted = candidate

    allowed_levels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40]
    return min(allowed_levels, key=lambda x: abs(x - adjusted))


def generate_certificate(current_price, predicted_price, volatility):
    """
    Generates a human-readable recommendation for leverage product.
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
    return f"{desc} with recommended leverage {safe_leverage}x"


def determine_trade_signal(predicted_pct_change, trend, rsi, volume, min_threshold=0.5):
    """
    Returns a simple trade recommendation based on predicted price move, trend, and RSI.
    """
    abs_change = abs(predicted_pct_change)
    if abs_change < min_threshold:
        return "No Trade", 0

    confidence = min(100, abs_change * 10)
    if predicted_pct_change > 0 and trend == "Bullish" and rsi < 70:
        return "Buy", confidence
    elif predicted_pct_change < 0 and trend == "Bearish" and rsi > 30:
        return "Sell", confidence
    else:
        return "No Trade", confidence


def determine_position_size(account_balance, risk_per_trade, volatility):
    """
    Computes a position size based on risk exposure and volatility.
    """
    base = account_balance * risk_per_trade
    safe_vol = max(volatility, 0.01)
    return base / safe_vol
