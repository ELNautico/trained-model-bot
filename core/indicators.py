import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import logging

def add_basic_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    df['RSI_14'] = 100 - 100 / (1 + gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-10))

    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['ATR'] = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
    df['ROC'] = df['Close'].pct_change(10)

    df.dropna(inplace=True)
    return df

def add_richer_features(df: pd.DataFrame, n_regimes: int = 3) -> pd.DataFrame:
    df = df.copy()

    # Safe volume handling
    if 'Volume' not in df.columns:
        df['Volume'] = 1e3
    df['Volume'] = df['Volume'].replace(0, np.nan)

    valid_vals = df['Volume'][df['Volume'] > 0].values
    if len(valid_vals) == 0:
        fallback = 1e3
    else:
        fallback = float(np.min(valid_vals))

    df['Volume'] = df['Volume'].fillna(fallback)

    # Bollinger Bands
    df['BB_mid'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']

    # Stochastic Oscillator
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['Stoch_%K'] = (df['Close'] - low_14) / (high_14 - low_14 + 1e-10)
    df['Stoch_%D'] = df['Stoch_%K'].rolling(3).mean()

    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # VWAP
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

    # CMF
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
    mf_volume = mf_multiplier * df['Volume']
    df['CMF'] = mf_volume.rolling(20).sum() / df['Volume'].rolling(20).sum()

    # Volatility regime
    returns = df['Close'].pct_change().fillna(0)
    vol_series = returns.rolling(10).std().fillna(0).values.reshape(-1, 1)
    if np.count_nonzero(vol_series > 0) > 50:
        gm = GaussianMixture(n_components=n_regimes, random_state=42)
        df['Volatility_Regime'] = gm.fit_predict(vol_series)
    else:
        df['Volatility_Regime'] = 0

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    dropped = before - len(df)
    logging.info(f"[FeatureCleaner] Dropped {dropped} rows due to NaNs.")

    return df

def get_feature_columns():
    return [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'ATR', 'ROC',
        'BB_mid', 'BB_upper', 'BB_lower',
        'Stoch_%K', 'Stoch_%D',
        'OBV', 'VWAP', 'CMF', 'Volatility_Regime'
    ]
