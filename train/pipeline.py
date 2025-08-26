"""
train/pipeline.py
End-to-end pipeline with automatic *feature pruning*.

Flow
----
1. Download & cache OHLCV
2. Feature-engineer full set
3. If a pruned-feature file exists → use it
4. Hyper-parameter tuning (walk-forward, via transformer_utils)
5. Sensitivity analysis → drop weakest P% features
6. Retrain ensemble on pruned features
7. Forecast price & σ̂ and forward them to risk module
"""
from __future__ import annotations
from core.features import enrich_features, get_feature_columns

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import toml
from sklearn.preprocessing import MinMaxScaler
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from twelvedata import TDClient

from core.features import enrich_features, get_feature_columns
from core.risk import generate_certificate
from core.sensitivity import compute_feature_sensitivity
from train.core import load_model, predict_price, train_and_save_model
from train.evaluate import backtest_strategy

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
_cfg_path = Path(__file__).resolve().parent.parent / "config.toml"
api_key = toml.load(_cfg_path)["twelvedata"]["api_key"]
td = TDClient(apikey=api_key)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

CACHE_TTL = timedelta(hours=24)
MAX_RETURN_THRESHOLD = 0.10
MAX_STALE_DAYS = 2
PRUNE_PCT = 30                      # bottom X % features to drop

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers ­– I/O
# ─────────────────────────────────────────────────────────────────────────────
def _pruned_path(ticker: str) -> Path:
    """models/TKR/LATEST_pruned_features.json"""
    return Path("models") / ticker / "LATEST_pruned_features.json"


def load_pruned_features(ticker: str) -> list[str] | None:
    path = _pruned_path(ticker)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            logging.warning("⚠️  Failed to read pruned-feature list for %s", ticker)
    return None


def save_pruned_features(ticker: str, pruned: list[str], version_dir: Path):
    latest = _pruned_path(ticker)
    try:
        latest.write_text(json.dumps(pruned, indent=2))
        (version_dir / "pruned_features.json").write_text(json.dumps(pruned, indent=2))
    except Exception as e:
        logging.warning("⚠️  Could not persist pruned features: %s", e)

# ─────────────────────────────────────────────────────────────────────────────
#  Raw data download + clean
# ─────────────────────────────────────────────────────────────────────────────
@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
)
def _download_raw_ohlcv(ticker: str) -> pd.DataFrame:
    df = td.time_series(symbol=ticker, interval="1day",
                        outputsize=5000, timezone="UTC").as_pandas()
    if df.empty:
        raise ValueError(f"Empty OHLCV for {ticker}")
    df.columns = [c.capitalize() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def cached_download(ticker: str) -> pd.DataFrame:
    key = hashlib.md5(ticker.encode()).hexdigest()
    path = CACHE_DIR / f"{key}.pkl"

    if path.exists() and datetime.utcnow() - datetime.utcfromtimestamp(path.stat().st_mtime) < CACHE_TTL:
        df = joblib.load(path)
    else:
        df = _download_raw_ohlcv(ticker)
        joblib.dump(df, path)

    df["ret"] = df["Close"].pct_change()
    df = df.loc[df["ret"].abs() <= MAX_RETURN_THRESHOLD].drop(columns="ret")

    gaps = df.index.to_series().diff() > pd.Timedelta(days=MAX_STALE_DAYS)
    if gaps.any():
        df = df.loc[~gaps]
    return df

# ─────────────────────────────────────────────────────────────────────────────
#  Sequence builder (d1 & d5 only)         << UPDATED
# ─────────────────────────────────────────────────────────────────────────────
def build_sequences(
    df: pd.DataFrame,
    feats: list[str],
    *,
    window_size: int,
    test_ratio: float = 0.2,
):
    close = df["Close"].values

    df = df[feats].ffill().bfill()

    split = int(len(df) * (1 - test_ratio))
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    scaler = MinMaxScaler()
    train_sc = scaler.fit_transform(train_df)
    test_sc  = scaler.transform(test_df)
    sc_all   = np.vstack([train_sc, test_sc])

    X_tr, X_te = [], []
    y1_tr, y5_tr = [], []
    y1_te, y5_te = [], []

    last_idx = len(df) - 5

    def _append(X, y1, y5, i):
        X.append(sc_all[i - window_size : i])
        y1.append(np.log(close[i] / close[i - 1]))        # 1-day return
        y5.append(np.log(close[i + 4] / close[i - 1]))    # 5-day return

    for i in range(window_size, split - 5):
        _append(X_tr, y1_tr, y5_tr, i)
    for i in range(split, last_idx):
        _append(X_te, y1_te, y5_te, i)

    y_train = {"d1": np.array(y1_tr), "d5": np.array(y5_tr)}
    y_test  = {"d1": np.array(y1_te), "d5": np.array(y5_te)}

    return (
        np.array(X_tr),
        y_train,
        np.array(X_te),
        y_test,
        scaler,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def train_predict_for_ticker(
    ticker: str,
    use_ensemble: bool = True,      # ← works both positionally *and* as keyword
    account_balance: float = 100_000.0,
    risk_per_trade: float = 0.01,
):
    logging.info("🚀 %s – pipeline start", ticker)
    df = cached_download(ticker)

    df = enrich_features(df)

    # ── feature subset (may be pruned)
    all_feats = get_feature_columns()
    pruned = load_pruned_features(ticker)
    feats_in_use = [f for f in all_feats if f not in (pruned or [])]

    win = 60 if len(df) >= 600 else 30
    X_tr, y_tr, X_te, y_te, scaler = build_sequences(df, feats_in_use, window_size=win)

    # ── model load / train
    try:
        model = load_model(ticker)
        logging.info("📦 model loaded")
    except FileNotFoundError:
        logging.info("🛠️ training from scratch (full feature set)")
        model, _, best_hp = train_and_save_model(X_tr, y_tr, X_tr.shape[1:], ticker)

    # ── sensitivity → prune bottom PRUNE_PCT %
    if pruned is None:
        last_seq_full = X_te[-1:][...]
        sens = compute_feature_sensitivity(model, last_seq_full, feats_in_use)
        thresh = np.percentile(list(sens.values()), PRUNE_PCT)
        weak = [f for f, v in sens.items() if v <= thresh]

        if weak and len(weak) < len(all_feats) * 0.5:   # keep at least 50 %
            logging.info("✂️  Pruning %d weak features: %s", len(weak), weak)
            # rebuild sequences WITHOUT the weak features
            feats_pruned = [f for f in feats_in_use if f not in weak]
            X_tr_p, y_tr_p, X_te_p, y_te_p, _ = build_sequences(
                df, feats_pruned, window_size=win
            )
            # reuse same HPs → fast retrain
            model, _, _ = train_and_save_model(X_tr_p, y_tr_p,
                                               X_tr_p.shape[1:], ticker)
            version_dir = Path("models") / ticker / sorted(
                (Path("models") / ticker).iterdir(), reverse=True
            )[0].name
            save_pruned_features(ticker, weak, version_dir)
            feats_in_use = feats_pruned
            X_te = X_te_p  # update inference seq

    # ── forecast
    current_px = float(df["Close"].iloc[-1])
    last_seq = X_te[-1:][...]
    pred_px, pred_vol = predict_price(model, last_seq, current_px)
    final_px = 0.9 * pred_px + 0.1 * current_px if use_ensemble else pred_px

    hist_vol = float(df["Close"].pct_change().std())
    vol_used = float(np.clip(pred_vol, 0.5 * hist_vol, 2.0 * hist_vol))

    pct_change = (final_px / current_px - 1) * 100
    confidence = round(min(abs(pct_change) * 10, 100), 1)
    direction = "Buy" if pct_change > 0 else "Sell" if pct_change < 0 else "Hold"
    acc, cum_ret = backtest_strategy(model, X_te, df, win)

    ts_local = df.index[-1].tz_convert(ZoneInfo("Europe/Vienna"))
    cert = generate_certificate(current_px, final_px, vol_used)
    sens_out = compute_feature_sensitivity(model, last_seq, feats_in_use)

    return (
        {
            "Ticker": ticker,
            "Timestamp": ts_local.strftime("%Y-%m-%d %H:%M %Z"),
            "Current Price": round(current_px, 2),
            "Predicted Price": round(final_px, 2),
            "Predicted % Change": round(pct_change, 2),
            "Confidence": confidence,
            "Volatility": round(vol_used, 4),
            "Directional Accuracy": round(acc, 4),
            "Cumulative Return": round(cum_ret, 4),
            "Recommended Certificate": cert,
            "Feature Sensitivity": sens_out,
            "Feature Count": len(feats_in_use),
        },
        df,
    )

# --------------------------------------------------------------------------
# Back-compat aliases (used by jobs.py  and Telegram-Bot)
# --------------------------------------------------------------------------

def prepare_data_and_split(
    df: pd.DataFrame,
    *,
    window_size: int = 60,
    test_ratio: float = 0.2,
):
    """
    Legacy-Wrapper für alten Code (retrain, Bot).
    Rechnet Features, ruft build_sequences() und liefert
    dieselbe 6-teilige Tuple-Struktur wie früher zurück.
    """
    df = enrich_features(df)

    feats = [f for f in get_feature_columns() if f in df.columns]
    X_tr, y_tr, X_te, y_te, scaler = build_sequences(
        df,
        feats,
        window_size=window_size,
        test_ratio=test_ratio,
    )
    train_size = len(X_tr)
    return X_tr, y_tr, X_te, y_te, scaler, train_size


# keep old alias
download_data = cached_download
