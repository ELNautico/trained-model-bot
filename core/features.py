import requests
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import logging
import toml
import pathlib


def fetch_sentiment_alpha_vantage(ticker: str, api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch news sentiment scores from Alpha Vantage for a given ticker and date range.
    Returns a DataFrame with date and sentiment score.

    NOTE:
      - AlphaVantage NEWS_SENTIMENT timestamps are intraday.
      - Aligning intraday news to daily bars can introduce subtle leakage unless you
        enforce an "available-at" time. For now we keep this feature, but the
        signal-model excludes it by default (cfg.use_sentiment=False).
    """
    url = "https://www.alphavantage.co/query"

    import datetime
    try:
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    except Exception:
        end_dt = datetime.datetime.utcnow()

    # Expand range for more news coverage
    start_dt = end_dt - datetime.timedelta(days=30)

    # Alpha Vantage expects YYYYMMDDT0000-ish formats
    start_date_fmt = start_dt.strftime("%Y%m%dT0000")
    end_date_fmt = end_dt.strftime("%Y%m%dT0000")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": api_key,
        "time_from": start_date_fmt,
        "time_to": end_date_fmt,
        "sort": "LATEST",
        "limit": 100
    }

    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        logging.warning("Alpha Vantage sentiment API error: %s", r.status_code)
        return pd.DataFrame()

    data = r.json()
    if "feed" not in data:
        logging.warning("No sentiment feed returned from Alpha Vantage.")
        return pd.DataFrame()

    rows = []
    for item in data["feed"]:
        date_str = item.get("time_published", "")
        try:
            date = pd.to_datetime(date_str, format="%Y%m%dT%H%M%S", utc=True)
        except Exception:
            date = pd.to_datetime(date_str, errors="coerce", utc=True)

        sentiment = item.get("overall_sentiment_score", 0)
        rows.append({"date": date, "sentiment": sentiment})

    df_sent = pd.DataFrame(rows)
    return df_sent


def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df["BB_mid"] = df["Close"].rolling(window).mean()
    df["BB_std"] = df["Close"].rolling(window).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]
    return df


def add_stochastic_oscillator(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    low_min = df["Low"].rolling(k_window).min()
    high_max = df["High"].rolling(k_window).max()
    df["Stoch_%K"] = (df["Close"] - low_min) / (high_max - low_min + 1e-10)
    df["Stoch_%D"] = df["Stoch_%K"].rolling(d_window).mean()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (typical_price * df["Volume"]).cumsum() / (df["Volume"].cumsum() + 1e-10)
    return df


def add_cmf(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    mf_mult = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + 1e-10)
    mf_volume = mf_mult * df["Volume"]
    df["CMF"] = mf_volume.rolling(window).sum() / (df["Volume"].rolling(window).sum() + 1e-10)
    return df


def add_volatility_regime(df: pd.DataFrame, window: int = 10, n_regimes: int = 3) -> pd.DataFrame:
    returns = df["Close"].pct_change().fillna(0)
    vol = returns.rolling(window).std().fillna(0).values.reshape(-1, 1)

    if np.count_nonzero(vol > 0) > 50:
        gm = GaussianMixture(n_components=n_regimes, random_state=42)
        df["Volatility_Regime"] = gm.fit_predict(vol)
    else:
        df["Volatility_Regime"] = 0
    return df


def add_basic_indicators(
    df: pd.DataFrame,
    sma_windows: tuple[int, int] = (20, 50),
    rsi_window: int = 14,
    macd_spans: tuple[int, int] = (12, 26),
    atr_window: int = 14,
    roc_period: int = 10
) -> pd.DataFrame:
    """
    Adds basic technical indicators: SMA, RSI, MACD, ATR, ROC.

    IMPORTANT FIX:
      - ATR is now computed as a standard Average True Range (Wilder-style),
        not as a rolling max-min range (which can be dramatically larger).
    """
    # Simple Moving Averages
    df[f"SMA_{sma_windows[0]}"] = df["Close"].rolling(sma_windows[0]).mean()
    df[f"SMA_{sma_windows[1]}"] = df["Close"].rolling(sma_windows[1]).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_window).mean()
    avg_loss = loss.rolling(rsi_window).mean()
    df[f"RSI_{rsi_window}"] = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-10))

    # MACD (difference of EMAs)
    df["MACD"] = (
        df["Close"].ewm(span=macd_spans[0], adjust=False).mean()
        - df["Close"].ewm(span=macd_spans[1], adjust=False).mean()
    )

    # True Range and ATR (Wilder's ATR approximation with rolling mean)
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(atr_window).mean()

    # ROC
    df["ROC"] = df["Close"].pct_change(roc_period)

    return df


def get_feature_columns() -> list[str]:
    """
    Returns a stable core list used by your older transformer pipeline.

    NOTE:
      - enrich_features() adds additional lag/rolling columns beyond this list.
      - The new signal model selects numeric columns directly and does not depend
        on this list.
    """
    return [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_20", "SMA_50", "RSI_14", "MACD", "ATR", "ROC",
        "BB_mid", "BB_upper", "BB_lower",
        "Stoch_%K", "Stoch_%D", "OBV", "VWAP", "CMF", "Volatility_Regime",
        "sentiment"
    ]


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature suite.

    IMPORTANT FIX:
      - Removed bfill() to prevent lookahead leakage. We only forward-fill and
        then drop remaining NaNs (which typically occur at the start due to
        rolling windows and lags).
    """
    df = df.copy()

    # Ensure volume exists and is positive
    if "Volume" not in df.columns:
        df["Volume"] = 1e3
    df["Volume"] = df["Volume"].replace(0, np.nan)
    valid_vol = df["Volume"][df["Volume"] > 0]
    fallback = float(valid_vol.min()) if not valid_vol.empty else 1e3
    df["Volume"] = df["Volume"].fillna(fallback)

    # Core indicators
    df = add_basic_indicators(df)
    df = add_bollinger_bands(df)
    df = add_stochastic_oscillator(df)
    df = add_obv(df)
    df = add_vwap(df)
    df = add_cmf(df)
    df = add_volatility_regime(df)

    # Lagged features
    for col in ["Close", "Volume"]:
        for lag in [1, 3, 7]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Rolling statistics
    for col in ["Close", "Volume"]:
        for win in [7, 14, 30]:
            df[f"{col}_rollmean{win}"] = df[col].rolling(win).mean()
            df[f"{col}_rollstd{win}"] = df[col].rolling(win).std()
            df[f"{col}_rollmin{win}"] = df[col].rolling(win).min()
            df[f"{col}_rollmax{win}"] = df[col].rolling(win).max()
            df[f"{col}_rollskew{win}"] = df[col].rolling(win).skew()
            df[f"{col}_rollkurt{win}"] = df[col].rolling(win).kurt()

    # Sentiment feature (kept, but the new signal model excludes by default)
    try:
        cfg_path = pathlib.Path(__file__).resolve().parent.parent / "config.toml"
        _cfg = toml.load(cfg_path)
        api_key = _cfg.get("alphavantage", {}).get("alva_api_key", None)
    except Exception:
        api_key = None

    ticker = df.attrs.get("ticker")
    df["sentiment"] = 0.0

    if ticker and api_key:
        try:
            start_date = str(df.index.min())[:10]
            end_date = str(df.index.max())[:10]
            df_sent = fetch_sentiment_alpha_vantage(ticker, api_key, start_date, end_date)
            if not df_sent.empty:
                df_sent = df_sent.dropna(subset=["date"]).set_index("date").sort_index()
                df_sent = df_sent[~df_sent.index.duplicated(keep="first")]

                # Ensure df index is UTC-aware if possible
                if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
                    try:
                        df.index = df.index.tz_localize("UTC")
                    except Exception:
                        pass

                # Reindex with ffill: uses past sentiment only
                # (still imperfect w.r.t. EOD availability, but avoids future-fill)
                df["sentiment"] = df_sent["sentiment"].reindex(df.index, method="ffill").fillna(0.0)
        except Exception as e:
            logging.warning("Sentiment feature failed for %s: %s", ticker, e)
            df["sentiment"] = 0.0

    # Clean-up (NO backfill to avoid leakage)
    before = len(df)
    df = df.ffill()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    dropped = before - len(df)
    logging.info("[Feature Engineering] Dropped %d rows due to NaNs. Remaining: %d", dropped, len(df))

    return df
