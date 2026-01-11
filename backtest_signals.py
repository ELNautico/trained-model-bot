"""
backtest_signals.py

Walk-forward backtest for your calibrated triple-barrier signal model with a
simple FLAT/LONG state machine.

Key properties:
- Does NOT use bot.db (no DB reset needed).
- Uses time-series-safe walk-forward training:
    * retrain every N bars (default 20)
    * training uses only data available up to the retrain date
    * calibration uses a chronological tail split inside the train window
- Simulates:
    * Entry at Close[t] (consistent with your labeling assumption)
    * Barrier hits using next-day OHLC (conservative rule: if stop & target hit same bar -> stop)
    * Time stop at horizon (exit at Close)
    * Optional model-based exit at Close (EV/p_stop gates), matching your bot logic

Outputs:
- Printed summary metrics (FULL) and optionally HOLDOUT metrics via --metrics-start/--metrics-end
- CSVs:
    backtests/<TICKER>_trades[_<TAG>].csv
    backtests/<TICKER>_equity[_<TAG>].csv

Usage examples:
  python backtest_signals.py TSLA
  python backtest_signals.py TSLA --start 2018-01-01 --retrain-every 20 --lookback 2000
  python backtest_signals.py TSLA --no-model-exit

Holdout evaluation example (simulate from 2018, report metrics from 2023 onward):
  python backtest_signals.py TSLA --start 2018-01-01 --end 2025-12-31 --metrics-start 2023-01-01 --tag holdout
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd

# --- your project imports ---
from train.pipeline import download_data  # alias to cached_download in your code
from signals.config import SignalConfig
from signals.model import train_calibrated_model
from signals.labeling import infer_feature_columns


# -----------------------------------------------------------------------------
# Timezone-safe slicing helpers
# -----------------------------------------------------------------------------
def _coerce_slice_ts(ts: pd.Timestamp, idx: pd.Index) -> pd.Timestamp:
    """
    Make ts timezone-compatible with idx (DatetimeIndex).
    - If idx is tz-aware and ts is naive -> localize ts to idx.tz
    - If idx is naive and ts is tz-aware -> drop tz from ts
    """
    if not isinstance(idx, pd.DatetimeIndex):
        return ts

    idx_tz = idx.tz
    if idx_tz is not None and ts.tzinfo is None:
        return ts.tz_localize(idx_tz)

    if idx_tz is None and ts.tzinfo is not None:
        return ts.tz_convert(None)

    return ts


def _parse_ts(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    return pd.Timestamp(s)


def _to_utc(ts: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
    """
    Normalize timestamps for internal comparisons. If tz-naive, interpret as UTC.
    """
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


# -----------------------------------------------------------------------------
# Feature engineering (SAFE VERSION)  [kept for reference; not required if core.features is already safe]
# -----------------------------------------------------------------------------
def enrich_features_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backtest-safe feature enrichment: forward-fill only, then dropna.

    If core.features.enrich_features is already leakage-safe (no bfill),
    you can keep using it. This function is retained as a "belt & suspenders"
    option if you ever reintroduce bfill elsewhere.
    """
    from core.features import (
        add_basic_indicators,
        add_bollinger_bands,
        add_stochastic_oscillator,
        add_obv,
        add_vwap,
        add_cmf,
        add_volatility_regime,
    )

    df = df.copy().sort_index()

    # Ensure required base columns exist
    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            raise ValueError(f"Missing required OHLC column: {c}")

    if "Volume" not in df.columns:
        df["Volume"] = 1e3

    # Safe volume handling (same intent as your production)
    df["Volume"] = df["Volume"].replace(0, np.nan)
    valid_vol = df["Volume"][df["Volume"] > 0]
    fallback = float(valid_vol.min()) if not valid_vol.empty else 1e3
    df["Volume"] = df["Volume"].fillna(fallback)

    # Indicators
    df = add_basic_indicators(df)
    df = add_bollinger_bands(df)  # includes BB_mid, BB_std, BB_upper, BB_lower
    df = add_stochastic_oscillator(df)
    df = add_obv(df)
    df = add_vwap(df)
    df = add_cmf(df)
    df = add_volatility_regime(df)

    # Lags
    for col in ["Close", "Volume"]:
        for lag in [1, 3, 7]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Rolling stats
    for col in ["Close", "Volume"]:
        for win in [7, 14, 30]:
            df[f"{col}_rollmean{win}"] = df[col].rolling(win).mean()
            df[f"{col}_rollstd{win}"] = df[col].rolling(win).std()
            df[f"{col}_rollmin{win}"] = df[col].rolling(win).min()
            df[f"{col}_rollmax{win}"] = df[col].rolling(win).max()
            df[f"{col}_rollskew{win}"] = df[col].rolling(win).skew()
            df[f"{col}_rollkurt{win}"] = df[col].rolling(win).kurt()

    # Forward fill only (no lookahead)
    df = df.ffill()

    # Drop remaining NaNs (early rows from rolling/lag windows)
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped > 0:
        print(f"[enrich_features_safe] Dropped {dropped} warmup rows (no leakage). Remaining={len(df)}")

    return df


# -----------------------------------------------------------------------------
# Signal math helpers
# -----------------------------------------------------------------------------
def compute_levels(entry_px: float, atr: float, cfg: SignalConfig) -> Tuple[float, float]:
    """
    Long-only levels:
      stop   = entry - stop_loss_atr * ATR
      target = entry + take_profit_atr * ATR
    """
    atr = float(atr)
    if not np.isfinite(atr) or atr <= 0:
        atr = max(0.005 * float(entry_px), 1e-6)  # fallback ~0.5%

    stop = float(entry_px - cfg.stop_loss_atr * atr)
    target = float(entry_px + cfg.take_profit_atr * atr)
    return stop, target


def compute_R(cfg: SignalConfig) -> float:
    """
    With ATR-scaled symmetric barriers, reward/risk in price space simplifies to:
      R = take_profit_atr / stop_loss_atr
    """
    if cfg.stop_loss_atr <= 0:
        return 0.0
    return float(cfg.take_profit_atr / cfg.stop_loss_atr)


def cost_in_R(entry_px: float, stop_px: float, cfg: SignalConfig) -> float:
    """
    Convert two-way transaction cost from dollars/share into R units:
      costR = (2 * entry_px * cost_bps/10000) / stop_distance
    """
    stop_dist = float(entry_px - stop_px)
    if stop_dist <= 0:
        return 0.0

    one_way = float(cfg.one_way_cost_bps) / 10000.0
    two_way_cost_per_share = 2.0 * float(entry_px) * one_way
    return float(two_way_cost_per_share / stop_dist)


def proba_map(classes_: np.ndarray, proba: np.ndarray) -> Dict[int, float]:
    return {int(c): float(p) for c, p in zip(classes_, proba)}


def compute_ev_net(p_profit: float, p_stop: float, entry_px: float, stop_px: float, cfg: SignalConfig) -> float:
    R = compute_R(cfg)
    cR = cost_in_R(entry_px, stop_px, cfg)
    return float(p_profit * R - p_stop - cR)


# -----------------------------------------------------------------------------
# Backtest helpers
# -----------------------------------------------------------------------------
def extract_X_at(df_feat: pd.DataFrame, t: int, feature_cols: List[str], window_size: int) -> Optional[np.ndarray]:
    """
    Build flattened window features for decision index t.
    Uses [t-window_size, t) window (excludes the decision bar), consistent with your training code.

    Returns shape (1, window_size*n_features) or None.
    """
    if t - window_size < 0:
        return None

    window = df_feat[feature_cols].iloc[t - window_size : t]
    if len(window) != window_size:
        return None

    arr = window.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        return None

    return arr.reshape(1, -1)


def position_size_shares(
    equity: float,
    entry_px: float,
    stop_px: float,
    cfg: SignalConfig,
    risk_per_trade: float,
) -> int:
    """
    Shares sized by:
      - risk budget (equity * risk_per_trade) divided by stop distance
      - and capped by max_position_fraction of equity
    """
    stop_dist = float(entry_px - stop_px)
    if stop_dist <= 0:
        return 0

    risk_budget = float(equity * risk_per_trade)
    shares_by_risk = int(math.floor(risk_budget / stop_dist))

    cap_budget = float(equity * cfg.max_position_fraction)
    shares_by_cap = int(math.floor(cap_budget / float(entry_px)))

    return int(max(0, min(shares_by_risk, shares_by_cap)))


def apply_cost(notional: float, one_way_bps: float) -> float:
    """
    Trading cost in dollars for a notional value.
    """
    return float(notional * (float(one_way_bps) / 10000.0))


# -----------------------------------------------------------------------------
# Main backtest
# -----------------------------------------------------------------------------
def run_backtest(
    ticker: str,
    cfg: SignalConfig,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    initial_cash: float = 100_000.0,
    risk_per_trade: float = 0.01,
    retrain_every: int = 20,
    train_lookback_days: int = 2000,
    model_exit: bool = True,
    cooldown_days: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      trades_df, equity_df
    """
    ticker = ticker.upper()

    # 1) Load and feature-engineer
    raw = download_data(ticker).sort_index()

    if start:
        s = _coerce_slice_ts(pd.Timestamp(start), raw.index)
        raw = raw.loc[s:]

    if end:
        e = _coerce_slice_ts(pd.Timestamp(end), raw.index)
        raw = raw.loc[:e]

    if len(raw) < 500:
        raise ValueError(f"Not enough data for backtest: rows={len(raw)}")

    from core.features import enrich_features

    df = enrich_features(raw)

    # 2) Freeze feature columns once (consistent across walk-forward retrains)
    feature_cols = infer_feature_columns(df, cfg)
    if not feature_cols:
        raise ValueError("No feature columns inferred.")

    # Arrays for speed
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    atrs = df["ATR"].to_numpy(dtype=float)
    dates = df.index

    ws = int(cfg.window_size)
    h = int(cfg.horizon_days)

    # --- stricter warm-up control ---
    min_train_samples = int(getattr(cfg, "min_train_samples", 800))
    min_train_samples = int(np.clip(min_train_samples, 200, 5000))

    # Start index must allow feature window and at least one forward bar for exits
    i0 = max(ws + 1, 1)

    # Portfolio state
    cash = float(initial_cash)
    shares = 0
    entry_px = 0.0
    stop_px = 0.0
    target_px = 0.0
    entry_i = -1
    hold_days = 0
    cooldown_left = 0

    # Capture entry diagnostics (optional but useful)
    entry_ev_net = np.nan
    entry_p_profit = np.nan
    entry_p_stop = np.nan

    # Walk-forward model state
    model = None
    classes_ = None
    last_retrain_i = None

    trades: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []

    def equity_at(i: int) -> float:
        return float(cash + shares * closes[i])

    # 3) Walk-forward loop
    for i in range(i0, len(df) - 1):
        # --- Step A: manage open position using today's bar (i) ---
        exit_reason = None
        exit_px = None

        if shares > 0:
            # Only start checking barriers from the day AFTER entry (entry is at Close[entry_i])
            if i > entry_i:
                lo = float(lows[i])
                hi = float(highs[i])

                hit_stop = lo <= stop_px
                hit_target = hi >= target_px

                # Conservative: if both hit on same bar -> stop
                if hit_stop and hit_target:
                    exit_reason = "STOP_SAME_BAR"
                    exit_px = float(stop_px)
                elif hit_stop:
                    exit_reason = "STOP"
                    exit_px = float(stop_px)
                elif hit_target:
                    exit_reason = "TARGET"
                    exit_px = float(target_px)
                else:
                    hold_days += 1
                    if hold_days >= h:
                        exit_reason = "TIME"
                        exit_px = float(closes[i])

            # Model-based exit at close (only if not already exited via barriers/time)
            if shares > 0 and exit_reason is None and model_exit:
                X_now = extract_X_at(df, i, feature_cols, ws)
                if X_now is not None and model is not None:
                    proba = model.predict_proba(X_now)[0]
                    pm = proba_map(classes_, proba)  # type: ignore
                    p_profit = pm.get(+1, 0.0)
                    p_stop = pm.get(-1, 0.0)

                    stop_today, _target_today = compute_levels(float(closes[i]), float(atrs[i]), cfg)
                    ev_net = compute_ev_net(p_profit, p_stop, float(closes[i]), float(stop_today), cfg)

                    if ev_net <= cfg.exit_min_ev:
                        exit_reason = "MODEL_EV"
                        exit_px = float(closes[i])
                    elif p_stop >= cfg.exit_min_p_stop:
                        exit_reason = "MODEL_PSTOP"
                        exit_px = float(closes[i])

        # Execute exit if signaled
        if shares > 0 and exit_reason is not None and exit_px is not None:
            exit_notional = float(exit_px * shares)
            exit_cost = apply_cost(exit_notional, cfg.one_way_cost_bps)
            cash += exit_notional - exit_cost

            pnl = (exit_px - entry_px) * shares
            entry_notional = float(entry_px * shares)
            entry_cost = apply_cost(entry_notional, cfg.one_way_cost_bps)
            pnl_net = float(pnl - entry_cost - exit_cost)

            stop_dist = float(entry_px - stop_px)
            r_mult = float(pnl_net / (shares * stop_dist)) if stop_dist > 0 else 0.0

            trades.append(
                {
                    "ticker": ticker,
                    "entry_date": str(dates[entry_i]),
                    "exit_date": str(dates[i]),
                    "entry_px": float(entry_px),
                    "exit_px": float(exit_px),
                    "shares": int(shares),
                    "hold_days": int(hold_days),
                    "exit_reason": str(exit_reason),
                    "pnl_net": float(pnl_net),
                    "r_mult": float(r_mult),
                    # entry diagnostics
                    "entry_ev_net": float(entry_ev_net) if np.isfinite(entry_ev_net) else np.nan,
                    "entry_p_profit": float(entry_p_profit) if np.isfinite(entry_p_profit) else np.nan,
                    "entry_p_stop": float(entry_p_stop) if np.isfinite(entry_p_stop) else np.nan,
                }
            )

            shares = 0
            entry_px = stop_px = target_px = 0.0
            entry_i = -1
            hold_days = 0
            cooldown_left = int(cooldown_days)

            entry_ev_net = np.nan
            entry_p_profit = np.nan
            entry_p_stop = np.nan

        # --- Step B: retrain model periodically (time-safe, stricter warm-up, skip failures) ---
        need_retrain = model is None
        if (last_retrain_i is not None) and (i - last_retrain_i >= int(retrain_every)):
            need_retrain = True

        if need_retrain:
            # Train window ends at i (today's close known at decision time)
            start_i = max(0, i - int(train_lookback_days))
            df_train = df.iloc[start_i : i + 1]

            effective_samples = int(len(df_train) - ws - h)
            if effective_samples >= min_train_samples:
                try:
                    artifact, report = train_calibrated_model(df_train, cfg, feature_cols=feature_cols)
                    model = artifact.estimator
                    classes_ = artifact.classes_

                    cal = report.get("calibration", {})
                    logging.info(
                        "Retrain %s %s | eff=%d | calibrated=%s | cv=%s | skip=%s",
                        ticker,
                        str(dates[i]),
                        effective_samples,
                        cal.get("did_calibrate"),
                        cal.get("cv_used"),
                        cal.get("skip_reason"),
                    )
                except Exception as e:
                    logging.warning(
                        "Retrain skipped at %s for %s (effective_samples=%d): %s",
                        str(dates[i]),
                        ticker,
                        effective_samples,
                        e,
                    )
                finally:
                    # Advance stride even if skipped/failed so we don't hammer every bar
                    last_retrain_i = i
            else:
                # Not enough data yet; still advance stride to check again on schedule
                if last_retrain_i is None or (i - last_retrain_i >= int(retrain_every)):
                    last_retrain_i = i

        # --- Step C: entry decision at Close[i] (if FLAT) ---
        action = "WAIT" if shares == 0 else "HOLD"

        if cooldown_left > 0:
            cooldown_left -= 1
        else:
            if shares == 0 and model is not None:
                X_now = extract_X_at(df, i, feature_cols, ws)
                if X_now is not None:
                    proba = model.predict_proba(X_now)[0]
                    pm = proba_map(classes_, proba)  # type: ignore
                    p_profit = pm.get(+1, 0.0)
                    p_stop = pm.get(-1, 0.0)

                    ep = float(closes[i])
                    sp, tp = compute_levels(ep, float(atrs[i]), cfg)
                    ev_net = compute_ev_net(p_profit, p_stop, ep, sp, cfg)

                    if ev_net >= cfg.entry_min_ev:
                        eq = equity_at(i)
                        sh = position_size_shares(eq, ep, sp, cfg, float(risk_per_trade))
                        if sh > 0:
                            entry_notional = float(ep * sh)
                            entry_cost = apply_cost(entry_notional, cfg.one_way_cost_bps)

                            if cash >= entry_notional + entry_cost:
                                cash -= entry_notional + entry_cost

                                shares = int(sh)
                                entry_px = float(ep)
                                stop_px = float(sp)
                                target_px = float(tp)
                                entry_i = int(i)
                                hold_days = 0

                                entry_ev_net = float(ev_net)
                                entry_p_profit = float(p_profit)
                                entry_p_stop = float(p_stop)

                                action = "BUY"

        # --- Step D: record equity curve at close ---
        eq = equity_at(i)
        equity_rows.append(
            {
                "date": str(dates[i]),
                "equity": float(eq),
                "cash": float(cash),
                "shares": int(shares),
                "close": float(closes[i]),
                "action": str(action),
            }
        )

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return trades_df, equity_df


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def _slice_equity_df(equity_df: pd.DataFrame, metrics_start: Optional[pd.Timestamp], metrics_end: Optional[pd.Timestamp]) -> pd.DataFrame:
    edf = equity_df.copy()
    edf["date_ts"] = pd.to_datetime(edf["date"], utc=True, errors="coerce")
    ms = _to_utc(metrics_start)
    me = _to_utc(metrics_end)

    m = edf["date_ts"].notna()
    if ms is not None:
        m &= edf["date_ts"] >= ms
    if me is not None:
        m &= edf["date_ts"] <= me
    return edf.loc[m].reset_index(drop=True)


def _slice_trades_df(trades_df: pd.DataFrame, metrics_start: Optional[pd.Timestamp], metrics_end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df.copy()

    tdf = trades_df.copy()
    tdf["entry_ts"] = pd.to_datetime(tdf["entry_date"], utc=True, errors="coerce")
    tdf["exit_ts"] = pd.to_datetime(tdf["exit_date"], utc=True, errors="coerce")

    ms = _to_utc(metrics_start)
    me = _to_utc(metrics_end)

    # For holdout, it's typically most consistent to count trades by EXIT time (realized PnL),
    # while equity stats are computed from the equity curve slice.
    m = tdf["exit_ts"].notna()
    if ms is not None:
        m &= tdf["exit_ts"] >= ms
    if me is not None:
        m &= tdf["exit_ts"] <= me
    return tdf.loc[m].reset_index(drop=True)


def summarize(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    *,
    initial_cash: float,
    label: str = "FULL",
    metrics_start: Optional[pd.Timestamp] = None,
    metrics_end: Optional[pd.Timestamp] = None,
) -> None:
    edf = _slice_equity_df(equity_df, metrics_start, metrics_end)
    if len(edf) < 2:
        print(f"Not enough equity points to summarize for {label}.")
        return

    eq = edf["equity"].to_numpy(dtype=float)
    start_equity = float(eq[0])
    end_equity = float(eq[-1])
    total_return = float(end_equity / max(start_equity, 1e-9) - 1.0)

    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-9)
    mean = float(np.mean(rets))
    std = float(np.std(rets))
    sharpe = float((mean / std) * math.sqrt(252)) if std > 1e-12 else 0.0

    mdd = max_drawdown(eq)

    tdf = _slice_trades_df(trades_df, metrics_start, metrics_end)
    n_trades = int(len(tdf))
    win_rate = float((tdf["pnl_net"] > 0).mean()) if n_trades else 0.0
    avg_r = float(tdf["r_mult"].mean()) if n_trades else 0.0
    med_r = float(tdf["r_mult"].median()) if n_trades else 0.0

    gains = float(tdf.loc[tdf["pnl_net"] > 0, "pnl_net"].sum()) if n_trades else 0.0
    losses = float(tdf.loc[tdf["pnl_net"] < 0, "pnl_net"].sum()) if n_trades else 0.0
    profit_factor = float(gains / abs(losses)) if losses < 0 else float("inf") if gains > 0 else 0.0

    avg_hold = float(tdf["hold_days"].mean()) if n_trades else 0.0

    w_start_s = str(metrics_start) if metrics_start is not None else ""
    w_end_s = str(metrics_end) if metrics_end is not None else ""

    print(f"\n==================== BACKTEST SUMMARY ({label}) ====================")
    if label != "FULL":
        print(f"Window: {w_start_s} .. {w_end_s}".strip())
    print(f"Equity start: {start_equity:,.2f}")
    print(f"Equity end  : {end_equity:,.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"Sharpe (naive, daily): {sharpe:.2f}")
    print(f"Max drawdown: {mdd:.2%}")
    print("----------------------------------------------------------")
    print(f"Trades (by exit in window): {n_trades}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Avg R: {avg_r:.3f} | Median R: {med_r:.3f}")
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"Avg hold days: {avg_hold:.2f}")
    if n_trades and "exit_reason" in tdf.columns:
        print("\nExit reason counts:")
        print(tdf["exit_reason"].value_counts().to_string())
    print("==========================================================\n")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("ticker", type=str, help="Ticker, e.g. TSLA")
    p.add_argument("--start", type=str, default=None, help="Start date, e.g. 2018-01-01")
    p.add_argument("--end", type=str, default=None, help="End date, e.g. 2025-12-31")

    p.add_argument("--initial-cash", type=float, default=100_000.0)
    p.add_argument("--risk", type=float, default=0.01, help="Risk per trade (fraction of equity)")
    p.add_argument("--retrain-every", type=int, default=20, help="Retrain stride in bars")
    p.add_argument("--lookback", type=int, default=2000, help="Training lookback bars")
    p.add_argument("--cooldown", type=int, default=0, help="Cooldown days after exiting a trade")
    p.add_argument("--no-model-exit", action="store_true", help="Disable model-based exits (EV/p_stop), use barriers only")

    p.add_argument("--entry-min-ev", type=float, default=None, help="Override cfg.entry_min_ev")
    p.add_argument("--exit-min-ev", type=float, default=None, help="Override cfg.exit_min_ev (model-based exit EV gate)")
    p.add_argument("--exit-min-p-stop", type=float, default=None, help="Override cfg.exit_min_p_stop (model-based exit p_stop gate)")

    # Step C support: compute metrics on a holdout slice of the equity/trade stream
    p.add_argument("--metrics-start", type=str, default=None, help="Report metrics starting at this date (e.g. 2023-01-01)")
    p.add_argument("--metrics-end", type=str, default=None, help="Report metrics ending at this date (e.g. 2025-12-31)")

    # Optional: keep outputs from multiple runs
    p.add_argument("--tag", type=str, default=None, help="Optional suffix tag for output CSV filenames")

    args = p.parse_args()

    cfg = SignalConfig()

    # Optional overrides (kept lightweight so backtests/sweeps can tune gates without editing config files)
    overrides = {}
    if args.entry_min_ev is not None:
        overrides["entry_min_ev"] = float(args.entry_min_ev)
    if args.exit_min_ev is not None:
        overrides["exit_min_ev"] = float(args.exit_min_ev)
    if args.exit_min_p_stop is not None:
        overrides["exit_min_p_stop"] = float(args.exit_min_p_stop)

    if overrides:
        d = asdict(cfg)
        d.update(overrides)
        cfg = SignalConfig(**d)

    trades_df, equity_df = run_backtest(
        args.ticker,
        cfg,
        start=args.start,
        end=args.end,
        initial_cash=float(args.initial_cash),
        risk_per_trade=float(args.risk),
        retrain_every=int(args.retrain_every),
        train_lookback_days=int(args.lookback),
        model_exit=not bool(args.no_model_exit),
        cooldown_days=int(args.cooldown),
    )

    out_dir = Path("backtests")
    out_dir.mkdir(exist_ok=True)

    tkr = args.ticker.upper()
    tag = f"_{args.tag}" if args.tag else ""
    trades_path = out_dir / f"{tkr}_trades{tag}.csv"
    equity_path = out_dir / f"{tkr}_equity{tag}.csv"

    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)

    print(f"Saved trades: {trades_path}")
    print(f"Saved equity: {equity_path}")

    # Always show full-period summary
    summarize(trades_df, equity_df, initial_cash=float(args.initial_cash), label="FULL")

    # Optional holdout summary
    ms = _parse_ts(args.metrics_start)
    me = _parse_ts(args.metrics_end)
    if ms is not None or me is not None:
        summarize(
            trades_df,
            equity_df,
            initial_cash=float(args.initial_cash),
            label="HOLDOUT",
            metrics_start=ms,
            metrics_end=me,
        )


if __name__ == "__main__":
    main()
