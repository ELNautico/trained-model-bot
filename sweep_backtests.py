"""
sweep_backtests.py

Parameter sweep (grid search) harness for backtest_signals.py.

Why this exists
---------------
Your backtest is expensive because each run performs walk-forward retraining and
calibration. Manually re-running dozens of configs is error-prone and slow.

This script automates the experiment loop:
- Runs a grid of (entry_min_ev, retrain_every, lookback) combinations
- Computes BOTH:
    * FULL metrics over the entire backtest span
    * HOLDOUT metrics over a user-specified evaluation window
- Writes one consolidated CSV you can sort/filter in Excel / pandas
- Optionally re-runs the TOP-K configs and saves their trades/equity CSVs

Design notes
------------
- Uses your existing engine: backtest_signals.run_backtest()
- Does NOT modify your production logic; this is a research harness
- Robust to failures: exceptions are captured and recorded per config
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Import your backtest engine (must be in same folder or on PYTHONPATH)
import backtest_signals
from signals.config import SignalConfig


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def parse_value_list(spec: str, *, cast=float) -> List[Any]:
    """
    Parse a comma list or range spec.

    Supported:
      - Comma list: "0.16,0.18,0.20"
      - Range:      "0.16:0.26:0.02"   => start:stop:step (inclusive-ish)

    Notes:
      - We avoid float drift by stepping with an integer counter.
      - The range includes the stop if it lands on the grid within a tolerance.
    """
    spec = (spec or "").strip()
    if not spec:
        return []

    # Range form: a:b:c
    if ":" in spec and "," not in spec:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Invalid range spec '{spec}'. Use start:stop:step")
        start, stop, step = (float(parts[0]), float(parts[1]), float(parts[2]))
        if step == 0:
            raise ValueError(f"Invalid range spec '{spec}': step cannot be 0")

        vals: List[Any] = []
        n = 0

        def done(x: float) -> bool:
            # step can be positive or negative
            return x > stop + 1e-12 if step > 0 else x < stop - 1e-12

        x = start
        while not done(x):
            vals.append(cast(x))
            n += 1
            x = start + n * step

        # include stop if it's one step away due to rounding
        if vals:
            last = float(vals[-1])
            if abs(last - stop) > 1e-9 and abs((last + step) - stop) < 1e-9:
                vals.append(cast(stop))
        return vals

    # Comma list
    return [cast(x.strip()) for x in spec.split(",") if x.strip()]


def set_cfg_field(cfg: SignalConfig, field: str, value: Any) -> SignalConfig:
    """
    Safely set a config field whether cfg is mutable or "frozen-ish".

    - If attribute assignment works, set in-place and return cfg.
    - Otherwise, reconstruct via dataclasses.asdict.
    """
    try:
        setattr(cfg, field, value)
        return cfg
    except Exception:
        if not is_dataclass(cfg):
            raise
        d = asdict(cfg)
        d[field] = value
        return SignalConfig(**d)


def to_ts(x: Optional[str]) -> Optional[pd.Timestamp]:
    """Parse a date string to UTC Timestamp (handles tz-aware strings too)."""
    if x is None:
        return None
    return pd.to_datetime(x, utc=True)


def slice_window(df: pd.DataFrame, col: str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    """Filter df where datetime(df[col]) is within [start, end]."""
    if df.empty:
        return df
    dt = pd.to_datetime(df[col], utc=True, errors="coerce")
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= dt >= start
    if end is not None:
        mask &= dt <= end
    return df.loc[mask.values]


# -----------------------------------------------------------------------------
# Metrics computation (mirrors backtest_signals.summarize semantics)
# -----------------------------------------------------------------------------
def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peak, 1e-12) - 1.0
    return float(np.min(dd))


def summarize_window(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    *,
    initial_equity: Optional[float] = None,
    metrics_start: Optional[pd.Timestamp] = None,
    metrics_end: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """
    Compute summary stats for a specified window.

    - Equity metrics use equity_df['date'] in [metrics_start, metrics_end]
    - Trade stats count trades whose exit_date is in [metrics_start, metrics_end]
      (Matches your "Trades (by exit in window)" convention.)
    """
    out: Dict[str, Any] = {}

    # --- Equity metrics over window ---
    eqw = slice_window(equity_df, "date", metrics_start, metrics_end) if not equity_df.empty else equity_df
    eq = eqw["equity"].to_numpy(dtype=float) if not eqw.empty else np.array([], dtype=float)

    out["equity_points"] = int(eq.size)

    if eq.size < 2:
        out.update(
            {
                "equity_start": float(eq[0]) if eq.size == 1 else np.nan,
                "equity_end": float(eq[-1]) if eq.size == 1 else np.nan,
                "total_return": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
            }
        )
    else:
        equity_start = float(eq[0]) if initial_equity is None else float(initial_equity)
        equity_end = float(eq[-1])

        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
        mean = float(np.mean(rets))
        std = float(np.std(rets))
        sharpe = float((mean / std) * math.sqrt(252)) if std > 1e-12 else 0.0

        out.update(
            {
                "equity_start": equity_start,
                "equity_end": equity_end,
                "total_return": float(equity_end / equity_start - 1.0) if equity_start > 0 else np.nan,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown(eq),
            }
        )

    # --- Trade stats (by exit in window) ---
    tw = slice_window(trades_df, "exit_date", metrics_start, metrics_end) if not trades_df.empty else trades_df
    n = int(len(tw))
    out["trades"] = n

    if n == 0:
        out.update(
            {
                "win_rate": np.nan,
                "avg_r": np.nan,
                "med_r": np.nan,
                "profit_factor": np.nan,
                "avg_hold_days": np.nan,
            }
        )
    else:
        pnl = tw["pnl_net"].to_numpy(dtype=float)
        r = tw["r_mult"].to_numpy(dtype=float)
        win_rate = float(np.mean(pnl > 0))
        gains = float(np.sum(pnl[pnl > 0]))
        losses = float(np.sum(pnl[pnl < 0]))
        pf = float(gains / abs(losses)) if losses < 0 else (float("inf") if gains > 0 else 0.0)

        out.update(
            {
                "win_rate": win_rate,
                "avg_r": float(np.mean(r)),
                "med_r": float(np.median(r)),
                "profit_factor": pf,
                "avg_hold_days": float(np.mean(tw["hold_days"].to_numpy(dtype=float))),
            }
        )

    return out


def objective_value(row: Dict[str, Any], objective: str) -> float:
    """
    Convert a result row into a numeric objective for ranking.
    Higher is better.
    """
    def g(k: str) -> float:
        v = row.get(k, np.nan)
        try:
            return float(v)
        except Exception:
            return float("nan")

    if objective == "holdout_sharpe":
        return g("holdout_sharpe")
    if objective == "holdout_return":
        return g("holdout_total_return")
    if objective == "holdout_profit_factor":
        return g("holdout_profit_factor")
    if objective == "holdout_avg_r":
        return g("holdout_avg_r")
    if objective == "full_sharpe":
        return g("full_sharpe")
    if objective == "full_return":
        return g("full_total_return")

    raise ValueError(f"Unknown objective '{objective}'.")


# -----------------------------------------------------------------------------
# Sweep runner
# -----------------------------------------------------------------------------
def run_sweep(
    ticker: str,
    *,
    start: Optional[str],
    end: Optional[str],
    metrics_start: Optional[str],
    metrics_end: Optional[str],
    entry_min_evs: Sequence[float],
    retrain_everys: Sequence[int],
    lookbacks: Sequence[int],
    exit_min_evs: Sequence[float],
    exit_min_p_stops: Sequence[float],
    initial_cash: float,
    risk_per_trade: float,
    cooldown_days: int,
    model_exit: bool,
    objective: str,
    min_holdout_trades: int,
) -> pd.DataFrame:
    """
    Execute the grid search and return a results DataFrame.
    """
    ticker = ticker.upper()
    ms = to_ts(metrics_start)
    me = to_ts(metrics_end)

    cfg0 = SignalConfig()

    combos = list(itertools.product(entry_min_evs, retrain_everys, lookbacks, exit_min_evs, exit_min_p_stops))
    if not combos:
        raise ValueError("No parameter combinations to run. Check your sweep lists/ranges.")

    rows: List[Dict[str, Any]] = []
    total = len(combos)

    for k, (ev, r_every, lb, exit_ev, exit_pstop) in enumerate(combos, start=1):
        print(f"[{k}/{total}] entry_ev={ev} | exit_ev={exit_ev} | exit_pstop={exit_pstop} | retrain_every={r_every} | lookback={lb}")

        row: Dict[str, Any] = {
            "ticker": ticker,
            "entry_min_ev": float(ev),
            "exit_min_ev": float(exit_ev),
            "exit_min_p_stop": float(exit_pstop),
            "retrain_every": int(r_every),
            "lookback": int(lb),
            "status": "ok",
            "error": None,
        }

        try:
            cfg = SignalConfig(**asdict(cfg0))
            cfg = set_cfg_field(cfg, "entry_min_ev", float(ev))
            cfg = set_cfg_field(cfg, "exit_min_ev", float(exit_ev))
            cfg = set_cfg_field(cfg, "exit_min_p_stop", float(exit_pstop))

            trades_df, equity_df = backtest_signals.run_backtest(
                ticker,
                cfg,
                start=start,
                end=end,
                initial_cash=float(initial_cash),
                risk_per_trade=float(risk_per_trade),
                retrain_every=int(r_every),
                train_lookback_days=int(lb),
                model_exit=bool(model_exit),
                cooldown_days=int(cooldown_days),
            )

            # FULL metrics (entire run)
            full = summarize_window(trades_df, equity_df, initial_equity=float(initial_cash))
            # HOLDOUT metrics (windowed)
            hold = summarize_window(trades_df, equity_df, metrics_start=ms, metrics_end=me)

            for kk, vv in full.items():
                row[f"full_{kk}"] = vv
            for kk, vv in hold.items():
                row[f"holdout_{kk}"] = vv

            # Filter small holdout samples (prevents overfitting on tiny #trades)
            hold_trades = row.get("holdout_trades", 0) or 0
            try:
                hold_trades = int(hold_trades)
            except Exception:
                hold_trades = 0

            if hold_trades < int(min_holdout_trades):
                row["status"] = "filtered_min_trades"

            # Objective for ranking
            obj = objective_value(row, objective)
            if not np.isfinite(obj):
                obj = -1e9
            row["objective"] = float(obj)

        except Exception as e:
            row["status"] = "error"
            row["error"] = repr(e)
            row["objective"] = -1e12

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sorting: ok first, then filtered, then error; within each by objective desc
    status_order = {"ok": 0, "filtered_min_trades": 1, "error": 2}
    df["_status_rank"] = df["status"].map(lambda x: status_order.get(str(x), 9))
    df = df.sort_values(by=["_status_rank", "objective"], ascending=[True, False]).drop(columns=["_status_rank"])

    return df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Grid-search sweep for backtest_signals.py")

    # Backtest span
    p.add_argument("ticker", type=str, help="Ticker, e.g. TSLA")
    p.add_argument("--start", type=str, default=None, help="Backtest start date, e.g. 2018-01-01")
    p.add_argument("--end", type=str, default=None, help="Backtest end date, e.g. 2025-12-31")

    # Holdout evaluation window
    p.add_argument("--metrics-start", type=str, default=None, help="Holdout window start, e.g. 2024-01-01")
    p.add_argument("--metrics-end", type=str, default=None, help="Holdout window end, e.g. 2025-12-31")
    p.add_argument("--min-holdout-trades", type=int, default=20, help="Minimum trades by exit in holdout window")

    # Sweep parameters (required so you don't accidentally run an empty sweep)
    p.add_argument(
        "--entry-min-ev",
        type=str,
        required=True,
        help="Comma list or range start:stop:step, e.g. 0.16,0.18,0.20 or 0.16:0.26:0.02",
    )
    p.add_argument("--retrain-every", type=str, required=True, help="Comma list or range, e.g. 60,80,100 or 40:120:20")
    p.add_argument("--lookback", type=str, required=True, help="Comma list or range, e.g. 1200,1500,2000")

    # Optional: sweep model-based exit gates as well (if omitted, use cfg defaults)
    p.add_argument("--exit-min-ev", type=str, default=None, help="Comma list or range for cfg.exit_min_ev (model exit EV gate)")
    p.add_argument("--exit-min-p-stop", type=str, default=None, help="Comma list or range for cfg.exit_min_p_stop (model exit p_stop gate)")

    # Portfolio assumptions (keep consistent while tuning model gates)
    p.add_argument("--initial-cash", type=float, default=100_000.0)
    p.add_argument("--risk", type=float, default=0.01, help="Risk per trade (fraction of equity)")
    p.add_argument("--cooldown", type=int, default=0, help="Cooldown days after exiting a trade")
    p.add_argument("--no-model-exit", action="store_true", help="Disable model-based exits (EV/p_stop), use barriers only")

    # Ranking / output
    p.add_argument(
        "--objective",
        type=str,
        default="holdout_sharpe",
        choices=[
            "holdout_sharpe",
            "holdout_return",
            "holdout_profit_factor",
            "holdout_avg_r",
            "full_sharpe",
            "full_return",
        ],
        help="Metric to rank configs by (higher is better)",
    )
    p.add_argument("--top", type=int, default=10, help="How many top configs to print to console")
    p.add_argument("--tag", type=str, default=None, help="Tag appended to output filenames")
    p.add_argument("--out-dir", type=str, default="backtests", help="Output directory for results CSV")
    p.add_argument(
        "--save-top-k",
        type=int,
        default=0,
        help="If >0, re-run the top K configs and save detailed trades/equity CSVs for each.",
    )

    args = p.parse_args()

    entry_min_evs = parse_value_list(args.entry_min_ev, cast=float)
    retrain_everys = parse_value_list(args.retrain_every, cast=int)
    lookbacks = parse_value_list(args.lookback, cast=int)

    # Optional exit-gate sweeps. If omitted, use the defaults from SignalConfig.
    cfg_default = SignalConfig()
    exit_min_evs = (
        parse_value_list(args.exit_min_ev, cast=float)
        if args.exit_min_ev is not None
        else [float(getattr(cfg_default, "exit_min_ev"))]
    )
    exit_min_p_stops = (
        parse_value_list(args.exit_min_p_stop, cast=float)
        if args.exit_min_p_stop is not None
        else [float(getattr(cfg_default, "exit_min_p_stop"))]
    )


    df = run_sweep(
        args.ticker,
        start=args.start,
        end=args.end,
        metrics_start=args.metrics_start,
        metrics_end=args.metrics_end,
        entry_min_evs=entry_min_evs,
        retrain_everys=retrain_everys,
        lookbacks=lookbacks,
        exit_min_evs=exit_min_evs,
        exit_min_p_stops=exit_min_p_stops,
        initial_cash=float(args.initial_cash),
        risk_per_trade=float(args.risk),
        cooldown_days=int(args.cooldown),
        model_exit=not bool(args.no_model_exit),
        objective=str(args.objective),
        min_holdout_trades=int(args.min_holdout_trades),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = (args.tag or "").strip()
    tag_part = f"_{tag}" if tag else ""
    out_path = out_dir / f"{args.ticker.upper()}_sweep_results{tag_part}.csv"
    df.to_csv(out_path, index=False)

    print("\nSaved sweep results:")
    print(str(out_path))

    # Print the top N rows with key columns
    top_n = int(max(1, args.top))
    cols = [
        "entry_min_ev",
        "exit_min_ev",
        "exit_min_p_stop",
        "retrain_every",
        "lookback",
        "status",
        "objective",
        "holdout_total_return",
        "holdout_sharpe",
        "holdout_max_drawdown",
        "holdout_trades",
        "holdout_profit_factor",
        "full_total_return",
        "full_sharpe",
        "full_max_drawdown",
        "full_trades",
    ]
    cols = [c for c in cols if c in df.columns]

    print(f"\nTop {top_n} configs by '{args.objective}' (higher is better):")
    print(df.head(top_n)[cols].to_string(index=False))

    # Optionally re-run top K and save details (trades/equity CSVs per config)
    save_k = int(args.save_top_k)
    if save_k > 0:
        print(f"\nRe-running top {save_k} configs to save detailed trades/equity CSVs...")
        cfg0 = SignalConfig()

        top_records = df.head(save_k).to_dict(orient="records")
        for rank, rec in enumerate(top_records, start=1):
            if rec.get("status") != "ok":
                continue

            ev = float(rec["entry_min_ev"])
            exit_ev = float(rec.get("exit_min_ev", getattr(cfg0, "exit_min_ev")))
            exit_pstop = float(rec.get("exit_min_p_stop", getattr(cfg0, "exit_min_p_stop")))
            r_every = int(rec["retrain_every"])
            lb = int(rec["lookback"])

            cfg = SignalConfig(**asdict(cfg0))
            cfg = set_cfg_field(cfg, "entry_min_ev", ev)
            cfg = set_cfg_field(cfg, "exit_min_ev", exit_ev)
            cfg = set_cfg_field(cfg, "exit_min_p_stop", exit_pstop)

            trades_df, equity_df = backtest_signals.run_backtest(
                args.ticker,
                cfg,
                start=args.start,
                end=args.end,
                initial_cash=float(args.initial_cash),
                risk_per_trade=float(args.risk),
                retrain_every=int(r_every),
                train_lookback_days=int(lb),
                model_exit=not bool(args.no_model_exit),
                cooldown_days=int(args.cooldown),
            )

            # Deterministic file tag per config (Windows-safe)
            cfg_tag = f"ev{ev:.3f}_xev{exit_ev:.3f}_xp{exit_pstop:.3f}_r{r_every}_lb{lb}".replace(".", "p")
            if tag:
                cfg_tag = f"{tag}_{cfg_tag}"

            trades_path = out_dir / f"{args.ticker.upper()}_trades_{cfg_tag}.csv"
            equity_path = out_dir / f"{args.ticker.upper()}_equity_{cfg_tag}.csv"

            trades_df.to_csv(trades_path, index=False)
            equity_df.to_csv(equity_path, index=False)

            print(f"  [{rank}] Saved: {trades_path.name} | {equity_path.name}")


if __name__ == "__main__":
    main()
