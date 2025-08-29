import requests
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import logging
import toml, pathlib

def fetch_sentiment_alpha_vantage(ticker: str, api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches news sentiment scores from Alpha Vantage for a given ticker and date range.
    Returns a DataFrame with date and sentiment score.
    """
    url = f"https://www.alphavantage.co/query"
    # Expand date range to last 30 days for more news coverage
    import datetime
    try:
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    except Exception:
        end_dt = datetime.datetime.utcnow()
    start_dt = end_dt - datetime.timedelta(days=30)
    # Format as YYYYMMDDT0000 (midnight)
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
    r = requests.get(url, params=params)
    if r.status_code != 200:
        logging.warning(f"Alpha Vantage sentiment API error: {r.status_code}")
        return pd.DataFrame()
    data = r.json()
    if "feed" not in data:
        logging.warning("No sentiment feed returned from Alpha Vantage.")
        return pd.DataFrame()
    rows = []
    for item in data["feed"]:
        # Parse full time_published string (YYYYMMDDTHHMMSS)
        date_str = item.get("time_published", "")
        try:
            date = pd.to_datetime(date_str, format="%Y%m%dT%H%M%S")
        except Exception:
            date = pd.to_datetime(date_str, errors="coerce")
        sentiment = item.get("overall_sentiment_score", 0)
        rows.append({"date": date, "sentiment": sentiment})
    df_sent = pd.DataFrame(rows)
    return df_sent

def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Adds Bollinger Bands (middle, upper, lower) to the DataFrame.
    """
    df['BB_mid'] = df['Close'].rolling(window).mean()
    df['BB_std'] = df['Close'].rolling(window).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
    return df


def add_stochastic_oscillator(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    """
    Adds the Stochastic Oscillator %K and %D lines to the DataFrame.
    """
    low_min = df['Low'].rolling(k_window).min()
    high_max = df['High'].rolling(k_window).max()
    df['Stoch_%K'] = (df['Close'] - low_min) / (high_max - low_min + 1e-10)
    df['Stoch_%D'] = df['Stoch_%K'].rolling(d_window).mean()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds On-Balance Volume (OBV) to the DataFrame.
    """
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Volume-Weighted Average Price (VWAP) to the DataFrame.
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df


def add_cmf(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Adds the Chaikin Money Flow (CMF) to the DataFrame.
    """
    mf_mult = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
    mf_volume = mf_mult * df['Volume']
    df['CMF'] = mf_volume.rolling(window).sum() / df['Volume'].rolling(window).sum()
    return df


def add_volatility_regime(df: pd.DataFrame, window: int = 10, n_regimes: int = 3) -> pd.DataFrame:
    """
    Adds a volatility regime label (0,1,2,...) via Gaussian Mixture Model clustering on rolling volatility.
    """
    returns = df['Close'].pct_change().fillna(0)
    vol = returns.rolling(window).std().fillna(0).values.reshape(-1, 1)
    if np.count_nonzero(vol > 0) > 50:
        gm = GaussianMixture(n_components=n_regimes, random_state=42)
        df['Volatility_Regime'] = gm.fit_predict(vol)
    else:
        df['Volatility_Regime'] = 0
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
    """
    # Simple Moving Averages
    df[f'SMA_{sma_windows[0]}'] = df['Close'].rolling(sma_windows[0]).mean()
    df[f'SMA_{sma_windows[1]}'] = df['Close'].rolling(sma_windows[1]).mean()

    # Relative Strength Index
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_window).mean()
    avg_loss = loss.rolling(rsi_window).mean()
    df[f'RSI_{rsi_window}'] = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-10))

    # Moving Average Convergence Divergence
    df['MACD'] = (
        df['Close'].ewm(span=macd_spans[0], adjust=False).mean()
        - df['Close'].ewm(span=macd_spans[1], adjust=False).mean()
    )

    # Average True Range
    high_roll = df['High'].rolling(atr_window).max()
    low_roll = df['Low'].rolling(atr_window).min()
    df['ATR'] = high_roll - low_roll

    # Rate of Change
    df['ROC'] = df['Close'].pct_change(roc_period)

    return df


def get_feature_columns() -> list[str]:
    """
    Returns the list of feature column names for modeling.
    """
    return [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'ATR', 'ROC',
        'BB_mid', 'BB_upper', 'BB_lower',
        'Stoch_%K', 'Stoch_%D', 'OBV', 'VWAP', 'CMF', 'Volatility_Regime'
    ]


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the full suite of feature engineering:
      - Safe volume handling
      - Basic indicators
      - Rich indicators (BB, Stoch, OBV, VWAP, CMF, Vol regimes)
      - Forward/backward fill and drop nulls
    """
    df = df.copy()

    # Safe volume handling
    if 'Volume' not in df.columns:
        df['Volume'] = 1e3
    df['Volume'] = df['Volume'].replace(0, np.nan)
    valid_vol = df['Volume'][df['Volume'] > 0]
    fallback = float(valid_vol.min()) if not valid_vol.empty else 1e3
    df['Volume'] = df['Volume'].fillna(fallback)

    # Compute features
    df = add_basic_indicators(df)
    df = add_bollinger_bands(df)
    df = add_stochastic_oscillator(df)
    df = add_obv(df)
    df = add_vwap(df)
    df = add_cmf(df)
    df = add_volatility_regime(df)

    # Add lagged features (1, 3, 7 days)
    for col in ['Close', 'Volume']:
        for lag in [1, 3, 7]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Add rolling statistics (window: 7, 14, 30 days)
    for col in ['Close', 'Volume']:
        for win in [7, 14, 30]:
            df[f'{col}_rollmean{win}'] = df[col].rolling(win).mean()
            df[f'{col}_rollstd{win}'] = df[col].rolling(win).std()
            df[f'{col}_rollmin{win}'] = df[col].rolling(win).min()
            df[f'{col}_rollmax{win}'] = df[col].rolling(win).max()
            df[f'{col}_rollskew{win}'] = df[col].rolling(win).skew()
            df[f'{col}_rollkurt{win}'] = df[col].rolling(win).kurt()

    # Add sentiment feature from Alpha Vantage
    # Requires 'ticker' and date range; fallback to index if not present
    import os
    _cfg = toml.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.toml"))
    api_key = _cfg["alphavantage"]["alva_api_key"]

    if 'ticker' in df.attrs:
        ticker = df.attrs['ticker']
    else:
        ticker = None
    if ticker:
        start_date = str(df.index.min())[:10]
        end_date = str(df.index.max())[:10]
        df_sent = fetch_sentiment_alpha_vantage(ticker, api_key, start_date, end_date)
        if not df_sent.empty:
            df_sent = df_sent.set_index('date')
            # Remove duplicate index labels in both DataFrames before reindexing
            df_sent = df_sent[~df_sent.index.duplicated(keep='first')]
            df = df[~df.index.duplicated(keep='first')]
            # Ensure both indexes are timezone-naive, then localize to UTC if required
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if hasattr(df_sent.index, 'tz') and df_sent.index.tz is not None:
                df_sent.index = df_sent.index.tz_localize(None)
            # Localize to UTC if tz-naive and required downstream
            if not hasattr(df.index, 'tz') or df.index.tz is None:
                try:
                    df.index = df.index.tz_localize('UTC')
                except Exception:
                    pass
            if not hasattr(df_sent.index, 'tz') or df_sent.index.tz is None:
                try:
                    df_sent.index = df_sent.index.tz_localize('UTC')
                except Exception:
                    pass
            df['sentiment'] = df_sent['sentiment'].reindex(df.index, method='ffill').fillna(0)
        else:
            df['sentiment'] = 0
    else:
        df['sentiment'] = 0

    # Clean up
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    dropped = before - len(df)
    logging.info(f"[Feature Engineering] Dropped {dropped} rows due to NaNs. Remaining: {len(df)}")

    return df