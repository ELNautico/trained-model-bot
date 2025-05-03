import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

def add_bollinger_bands(df: pd.DataFrame, window: int = 20):
    df['BB_mid'] = df['Close'].rolling(window=window).mean()
    df['BB_std'] = df['Close'].rolling(window=window).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
    return df

def add_stochastic_oscillator(df: pd.DataFrame, k_window=14, d_window=3):
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    df['Stoch_%K'] = (df['Close'] - low_min) / (high_max - low_min + 1e-10)
    df['Stoch_%D'] = df['Stoch_%K'].rolling(window=d_window).mean()
    return df

def add_obv(df: pd.DataFrame):
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return df

def add_vwap(df: pd.DataFrame):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df

def add_cmf(df: pd.DataFrame, window: int = 20):
    mf_mult = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
    mf_volume = mf_mult * df['Volume']
    df['CMF'] = mf_volume.rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()
    return df

def add_volatility_regime(df: pd.DataFrame, n_regimes=3):
    returns = df['Close'].pct_change().fillna(0)
    vol = returns.rolling(window=10).std().fillna(0).values.reshape(-1, 1)

    if np.count_nonzero(vol > 0) > 50:
        gm = GaussianMixture(n_components=n_regimes, random_state=42)
        df['Volatility_Regime'] = gm.fit_predict(vol)
    else:
        df['Volatility_Regime'] = 0
    return df

def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_bollinger_bands(df)
    df = add_stochastic_oscillator(df)
    df = add_obv(df)
    df = add_vwap(df)
    df = add_cmf(df)
    df = add_volatility_regime(df)
    df = df.ffill().bfill().dropna()
    return df
