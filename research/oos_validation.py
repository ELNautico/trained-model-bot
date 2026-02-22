"""
oos_validation.py

Multi-ticker out-of-sample validation for the triple-barrier signal model.

Purpose
-------
Runs backtest_signals.run_backtest() on a list of diverse tickers under a
SINGLE fixed config and evaluates all results on a shared holdout window.
This tests whether the edge is general, not ticker-specific.

Usage examples
--------------
# Fixed config, multiple tickers
python research/oos_validation.py \
    TSLA AAPL NVDA SPY GLD AMD MSFT AMZN \
    --start 2018-01-01 --end 2025-12-31 \
    --holdout-start 2024-01-01 --holdout-end 2025-12-31 \
    --entry-min-ev 0.20 --retrain-every 80 --lookback 1500 \
    --tag oos_grid_best

# Load best config automatically from a prior sweep CSV
python research/oos_validation.py \
    TSLA AAPL NVDA SPY GLD AMD \
    --start 2018-01-01 --end 2025-12-31 \
    --holdout-start 2024-01-01 --holdout-end 2025-12-31 \
    --from-sweep backtests/runs/<run_id>/sweep_results.csv \
    --tag oos_from_sweep

Output
------
backtests/
  runs/
    <run_id>/
      oos_results.csv      per-ticker metrics
      oos_summary.json     aggregate stats + config used
      meta.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Robust import: works both as "python research/oos_validation.py" from repo
# root, and via "from research import oos_validation" as a package.
# ---------------------------------------------------------------------------
try:
    from . import backtest_signals  # type: ignore
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import backtest_signals  # type: ignore

from signals.config import SignalConfig


# ---------------------------------------------------------------------------
# Helpers shared with sweep_backtests (duplicated here to keep this standalone)
# ---------------------------------------------------------------------------

def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _slug(s: Optional[str]) -> str:
    s = (s or "").strip()
    if not s:
        return "untagged"
    out = []
    for ch in s:
        out.append(ch if (ch.isalnum() or ch in "._-") else "_")
    res = "".join(out)
    while "__" in res:
        res = res.replace("__", "_")
    return res.strip("_") or "untagged"


def _safe_git_sha() -> Optional[str]:
    try:
        import subprocess
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return sha or None
    except Exception:
        return None


def write_meta(path: Path, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / np.maximum(peak, 1e-12) - 1.0))


def _to_utc(ts: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
    if ts is None:
        return None
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _slice_equity(equity_df: pd.DataFrame,
                  start: Optional[pd.Timestamp],
                  end: Optional[pd.Timestamp]) -> np.ndarray:
    edf = equity_df.copy()
    edf["_dt"] = pd.to_datetime(edf["date"], utc=True, errors="coerce")
    mask = edf["_dt"].notna()
    if start:
        mask &= edf["_dt"] >= start
    if end:
        mask &= edf["_dt"] <= end
    return edf.loc[mask, "equity"].to_numpy(dtype=float)


def _slice_trades(trades_df: pd.DataFrame,
                  start: Optional[pd.Timestamp],
                  end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df
    tdf = trades_df.copy()
    tdf["_dt"] = pd.to_datetime(tdf["exit_date"], utc=True, errors="coerce")
    mask = tdf["_dt"].notna()
    if start:
        mask &= tdf["_dt"] >= start
    if end:
        mask &= tdf["_dt"] <= end
    return tdf.loc[mask].reset_index(drop=True)


def compute_metrics(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    initial_cash: float,
    window_start: Optional[pd.Timestamp],
    window_end: Optional[pd.Timestamp],
) -> Dict[str, Any]:
    """Compute all metrics for one ticker over the specified window."""
    eq = _slice_equity(equity_df, window_start, window_end)
    tw = _slice_trades(trades_df, window_start, window_end)

    out: Dict[str, Any] = {"equity_points": int(eq.size), "trades": int(len(tw))}

    if eq.size >= 2:
        eq_start = float(eq[0])
        eq_end = float(eq[-1])
        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
        mean_r = float(np.mean(rets))
        std_r = float(np.std(rets))
        sharpe = float((mean_r / std_r) * math.sqrt(252)) if std_r > 1e-12 else 0.0
        total_return = float(eq_end / max(eq_start, 1e-9) - 1.0)
        mdd = _max_drawdown(eq)
    else:
        eq_start = eq_end = float(initial_cash)
        sharpe = total_return = mdd = float("nan")

    out.update({
        "equity_start": eq_start,
        "equity_end": eq_end,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": mdd,
    })

    n = int(len(tw))
    if n > 0:
        pnl = tw["pnl_net"].to_numpy(dtype=float)
        r = tw["r_mult"].to_numpy(dtype=float)
        gains = float(pnl[pnl > 0].sum())
        losses = float(pnl[pnl < 0].sum())
        pf = float(gains / abs(losses)) if losses < 0 else (float("inf") if gains > 0 else 0.0)
        exit_counts: Dict[str, int] = {}
        if "exit_reason" in tw.columns:
            exit_counts = tw["exit_reason"].value_counts().to_dict()
        out.update({
            "win_rate": float(np.mean(pnl > 0)),
            "avg_r": float(np.mean(r)),
            "med_r": float(np.median(r)),
            "profit_factor": pf,
            "avg_hold_days": float(tw["hold_days"].mean()),
            "exit_counts": exit_counts,
        })
    else:
        out.update({
            "win_rate": float("nan"),
            "avg_r": float("nan"),
            "med_r": float("nan"),
            "profit_factor": float("nan"),
            "avg_hold_days": float("nan"),
            "exit_counts": {},
        })

    return out


# ---------------------------------------------------------------------------
# Load best config from a sweep CSV
# ---------------------------------------------------------------------------

def load_best_config_from_sweep(sweep_csv: str) -> Dict[str, Any]:
    """
    Read a sweep_results.csv and return the parameter dict from the top 'ok' row.
    Raises if no valid row is found.
    """
    path = Path(sweep_csv)
    if not path.exists():
        raise FileNotFoundError(f"Sweep CSV not found: {path}")

    df = pd.read_csv(path)
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    if ok.empty:
        raise ValueError(f"No 'ok' rows found in sweep CSV: {path}")

    # Sort by objective descending if present
    if "objective" in ok.columns:
        ok = ok.sort_values("objective", ascending=False)

    best = ok.iloc[0].to_dict()
    params = {}
    for field in ["entry_min_ev", "exit_min_ev", "exit_min_p_stop", "retrain_every", "lookback"]:
        if field in best and not (isinstance(best[field], float) and math.isnan(best[field])):
            params[field] = best[field]

    return params


# ---------------------------------------------------------------------------
# Core: run validation across all tickers
# ---------------------------------------------------------------------------

def run_oos_validation(
    tickers: List[str],
    cfg: SignalConfig,
    *,
    start: Optional[str],
    end: Optional[str],
    holdout_start: Optional[str],
    holdout_end: Optional[str],
    initial_cash: float,
    risk_per_trade: float,
    retrain_every: int,
    lookback: int,
    model_exit: bool,
    cooldown_days: int,
    min_trades_warn: int = 10,
) -> pd.DataFrame:
    """
    Run one backtest per ticker, compute holdout metrics, return summary DataFrame.
    """
    hs = _to_utc(pd.Timestamp(holdout_start)) if holdout_start else None
    he = _to_utc(pd.Timestamp(holdout_end)) if holdout_end else None

    rows: List[Dict[str, Any]] = []

    for ticker in tickers:
        ticker = ticker.upper()
        print(f"\n{'─'*60}")
        print(f"  Running: {ticker}")
        print(f"{'─'*60}")

        row: Dict[str, Any] = {
            "ticker": ticker,
            "status": "ok",
            "error": None,
        }

        try:
            trades_df, equity_df = backtest_signals.run_backtest(
                ticker,
                cfg,
                start=start,
                end=end,
                initial_cash=initial_cash,
                risk_per_trade=risk_per_trade,
                retrain_every=retrain_every,
                train_lookback_days=lookback,
                model_exit=model_exit,
                cooldown_days=cooldown_days,
            )

            # Full-span metrics
            full = compute_metrics(trades_df, equity_df, initial_cash, None, None)
            for k, v in full.items():
                if k != "exit_counts":
                    row[f"full_{k}"] = v

            # Holdout-window metrics
            hold = compute_metrics(trades_df, equity_df, initial_cash, hs, he)
            for k, v in hold.items():
                if k != "exit_counts":
                    row[f"oos_{k}"] = v
            row["oos_exit_counts"] = json.dumps(hold.get("exit_counts", {}))

            n_trades = int(hold.get("trades", 0))
            if n_trades < min_trades_warn:
                row["status"] = "low_trades"
                print(f"  WARNING: only {n_trades} OOS trades — results may not be reliable")

            # Per-ticker console summary
            print(f"  OOS trades   : {n_trades}")
            print(f"  OOS return   : {hold.get('total_return', float('nan')):.2%}")
            print(f"  OOS Sharpe   : {hold.get('sharpe', float('nan')):.3f}")
            print(f"  OOS MaxDD    : {hold.get('max_drawdown', float('nan')):.2%}")
            print(f"  OOS win rate : {hold.get('win_rate', float('nan')):.2%}")
            print(f"  OOS avg R    : {hold.get('avg_r', float('nan')):.3f}")
            print(f"  OOS pf       : {hold.get('profit_factor', float('nan')):.2f}")

        except Exception as exc:
            logging.exception("Backtest failed for %s", ticker)
            row["status"] = "error"
            row["error"] = repr(exc)
            print(f"  ERROR: {exc}")

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregate stats across tickers
# ---------------------------------------------------------------------------

def compute_aggregate(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute cross-ticker aggregate statistics from the OOS metrics.
    Only 'ok' and 'low_trades' rows are included (errors are excluded).
    """
    valid = results_df[results_df["status"].isin(["ok", "low_trades"])].copy()
    n_tickers = int(len(valid))

    if n_tickers == 0:
        return {"error": "No valid tickers to aggregate."}

    def _mean(col: str) -> float:
        if col not in valid.columns:
            return float("nan")
        return float(valid[col].dropna().mean())

    def _median(col: str) -> float:
        if col not in valid.columns:
            return float("nan")
        return float(valid[col].dropna().median())

    def _pct_positive(col: str) -> float:
        if col not in valid.columns:
            return float("nan")
        s = valid[col].dropna()
        return float((s > 0).mean()) if len(s) > 0 else float("nan")

    total_oos_trades = int(valid["oos_trades"].dropna().sum()) if "oos_trades" in valid.columns else 0

    return {
        "n_tickers_total": int(len(results_df)),
        "n_tickers_valid": n_tickers,
        "n_tickers_error": int((results_df["status"] == "error").sum()),
        "n_tickers_low_trades": int((results_df["status"] == "low_trades").sum()),
        "total_oos_trades": total_oos_trades,
        "mean_oos_return": _mean("oos_total_return"),
        "median_oos_return": _median("oos_total_return"),
        "pct_tickers_positive_return": _pct_positive("oos_total_return"),
        "mean_oos_sharpe": _mean("oos_sharpe"),
        "median_oos_sharpe": _median("oos_sharpe"),
        "pct_tickers_positive_sharpe": _pct_positive("oos_sharpe"),
        "mean_oos_max_drawdown": _mean("oos_max_drawdown"),
        "mean_oos_win_rate": _mean("oos_win_rate"),
        "mean_oos_avg_r": _mean("oos_avg_r"),
        "mean_oos_profit_factor": _mean("oos_profit_factor"),
        "mean_oos_trades": _mean("oos_trades"),
    }


# ---------------------------------------------------------------------------
# Formatted console report
# ---------------------------------------------------------------------------

def _pct(v: Any) -> str:
    try:
        return f"{float(v):.2%}"
    except Exception:
        return "n/a"


def _f2(v: Any) -> str:
    try:
        return f"{float(v):.3f}"
    except Exception:
        return "n/a"


def _i(v: Any) -> str:
    try:
        return str(int(v))
    except Exception:
        return "n/a"


def print_summary_table(results_df: pd.DataFrame, aggregate: Dict[str, Any]) -> None:
    cols = [
        ("ticker", 6, str),
        ("status", 11, str),
        ("oos_trades", 6, _i),
        ("oos_total_return", 9, _pct),
        ("oos_sharpe", 9, _f2),
        ("oos_max_drawdown", 9, _pct),
        ("oos_win_rate", 9, _pct),
        ("oos_avg_r", 7, _f2),
        ("oos_profit_factor", 5, _f2),
    ]

    header = "  ".join(h.ljust(w) for h, w, _ in cols)
    sep = "  ".join("-" * w for h, w, _ in cols)

    print("\n" + "=" * 80)
    print("OOS VALIDATION — PER-TICKER RESULTS")
    print("=" * 80)
    print(header)
    print(sep)

    for _, row in results_df.iterrows():
        line_parts = []
        for col, width, fmt in cols:
            val = row.get(col, "n/a")
            try:
                cell = fmt(val)
            except Exception:
                cell = str(val)
            line_parts.append(cell.ljust(width))
        print("  ".join(line_parts))

    print(sep)
    print()
    print("=" * 80)
    print("AGGREGATE STATISTICS (valid tickers only)")
    print("=" * 80)

    n_v = aggregate.get("n_tickers_valid", 0)
    n_t = aggregate.get("n_tickers_total", 0)
    n_e = aggregate.get("n_tickers_error", 0)
    n_l = aggregate.get("n_tickers_low_trades", 0)

    print(f"  Tickers  : {n_v}/{n_t} valid  ({n_e} errors, {n_l} low-trade warnings)")
    print(f"  OOS trades total : {_i(aggregate.get('total_oos_trades'))}")
    print()
    print(f"  Return     mean={_pct(aggregate.get('mean_oos_return'))}  "
          f"median={_pct(aggregate.get('median_oos_return'))}  "
          f"pct>0={_pct(aggregate.get('pct_tickers_positive_return'))}")
    print(f"  Sharpe     mean={_f2(aggregate.get('mean_oos_sharpe'))}  "
          f"median={_f2(aggregate.get('median_oos_sharpe'))}  "
          f"pct>0={_pct(aggregate.get('pct_tickers_positive_sharpe'))}")
    print(f"  MaxDD      mean={_pct(aggregate.get('mean_oos_max_drawdown'))}")
    print(f"  Win rate   mean={_pct(aggregate.get('mean_oos_win_rate'))}")
    print(f"  Avg R      mean={_f2(aggregate.get('mean_oos_avg_r'))}")
    print(f"  Profit fac mean={_f2(aggregate.get('mean_oos_profit_factor'))}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Multi-ticker OOS validation for the triple-barrier signal model."
    )

    p.add_argument(
        "tickers",
        nargs="+",
        help="Tickers to validate, e.g. TSLA AAPL NVDA SPY GLD AMD",
    )

    # Backtest span
    p.add_argument("--start", type=str, default=None, help="Backtest start date, e.g. 2018-01-01")
    p.add_argument("--end", type=str, default=None, help="Backtest end date, e.g. 2025-12-31")

    # OOS evaluation window (required)
    p.add_argument("--holdout-start", type=str, required=True, help="OOS window start, e.g. 2024-01-01")
    p.add_argument("--holdout-end", type=str, required=True, help="OOS window end, e.g. 2025-12-31")
    p.add_argument("--min-trades-warn", type=int, default=10,
                   help="Warn if OOS trade count is below this threshold (default: 10)")

    # Config — either specify directly or load from a sweep CSV
    g = p.add_mutually_exclusive_group()
    g.add_argument("--from-sweep", type=str, default=None,
                   help="Path to a sweep_results.csv; loads the best config automatically.")

    p.add_argument("--entry-min-ev", type=float, default=None)
    p.add_argument("--exit-min-ev", type=float, default=None)
    p.add_argument("--exit-min-p-stop", type=float, default=None)
    p.add_argument("--retrain-every", type=int, default=80)
    p.add_argument("--lookback", type=int, default=1500)

    # Portfolio params
    p.add_argument("--initial-cash", type=float, default=100_000.0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--cooldown", type=int, default=0)
    p.add_argument("--no-model-exit", action="store_true")

    # Output
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="backtests")

    args = p.parse_args()

    # -----------------------------------------------------------------------
    # Build SignalConfig
    # -----------------------------------------------------------------------
    cfg = SignalConfig()
    cfg_source = "default"
    retrain_every = int(args.retrain_every)
    lookback = int(args.lookback)

    if args.from_sweep:
        print(f"\nLoading best config from sweep: {args.from_sweep}")
        sweep_params = load_best_config_from_sweep(args.from_sweep)
        print(f"  Loaded params: {sweep_params}")

        d = asdict(cfg)
        for field, val in sweep_params.items():
            if field in ("retrain_every", "lookback"):
                continue  # handled separately below
            if field in d:
                d[field] = val
        cfg = SignalConfig(**d)

        retrain_every = int(sweep_params.get("retrain_every", retrain_every))
        lookback = int(sweep_params.get("lookback", lookback))
        cfg_source = f"sweep:{args.from_sweep}"
    else:
        # Apply CLI overrides
        d = asdict(cfg)
        if args.entry_min_ev is not None:
            d["entry_min_ev"] = float(args.entry_min_ev)
        if args.exit_min_ev is not None:
            d["exit_min_ev"] = float(args.exit_min_ev)
        if args.exit_min_p_stop is not None:
            d["exit_min_p_stop"] = float(args.exit_min_p_stop)
        cfg = SignalConfig(**d)
        cfg_source = "cli"

    print("\nConfig being validated:")
    print(f"  entry_min_ev   : {cfg.entry_min_ev}")
    print(f"  exit_min_ev    : {cfg.exit_min_ev}")
    print(f"  exit_min_p_stop: {cfg.exit_min_p_stop}")
    print(f"  retrain_every  : {retrain_every}")
    print(f"  lookback       : {lookback}")
    print(f"  source         : {cfg_source}")
    print(f"\nTickers ({len(args.tickers)}): {', '.join(t.upper() for t in args.tickers)}")
    print(f"OOS window      : {args.holdout_start} → {args.holdout_end}")

    # -----------------------------------------------------------------------
    # Run validation
    # -----------------------------------------------------------------------
    results_df = run_oos_validation(
        tickers=args.tickers,
        cfg=cfg,
        start=args.start,
        end=args.end,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
        initial_cash=float(args.initial_cash),
        risk_per_trade=float(args.risk),
        retrain_every=retrain_every,
        lookback=lookback,
        model_exit=not bool(args.no_model_exit),
        cooldown_days=int(args.cooldown),
        min_trades_warn=int(args.min_trades_warn),
    )

    aggregate = compute_aggregate(results_df)

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    ts = _utc_ts()
    run_id = f"{ts}__MULTI__{_slug(args.tag)}__oos"
    backtests_dir = Path(args.out_dir)
    rdir = backtests_dir / "runs" / run_id
    rdir.mkdir(parents=True, exist_ok=True)

    results_path = rdir / "oos_results.csv"

    # Drop the exit_counts JSON column before saving CSV for readability
    csv_df = results_df.drop(columns=["oos_exit_counts"], errors="ignore")
    csv_df.to_csv(results_path, index=False)

    summary_path = rdir / "oos_summary.json"
    write_meta(summary_path, {
        "runId": run_id,
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "gitSha": _safe_git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "cfg_source": cfg_source,
        "config": {
            "entry_min_ev": cfg.entry_min_ev,
            "exit_min_ev": cfg.exit_min_ev,
            "exit_min_p_stop": cfg.exit_min_p_stop,
            "retrain_every": retrain_every,
            "lookback": lookback,
            "model_exit": not bool(args.no_model_exit),
            "cooldown_days": int(args.cooldown),
            "initial_cash": float(args.initial_cash),
            "risk_per_trade": float(args.risk),
        },
        "backtest_span": {"start": args.start, "end": args.end},
        "oos_window": {"start": args.holdout_start, "end": args.holdout_end},
        "tickers": [t.upper() for t in args.tickers],
        "aggregate": aggregate,
    })

    write_meta(rdir / "meta.json", {
        "kind": "oos_validation",
        "runId": run_id,
        "tag": args.tag,
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "tickers": [t.upper() for t in args.tickers],
        "cli": vars(args),
    })

    # -----------------------------------------------------------------------
    # Console report
    # -----------------------------------------------------------------------
    print_summary_table(results_df, aggregate)

    print(f"\nSaved results : {results_path}")
    print(f"Saved summary : {summary_path}")
    print()

    # Interpretation guidance
    n_pos = aggregate.get("pct_tickers_positive_return", float("nan"))
    mean_sharpe = aggregate.get("mean_oos_sharpe", float("nan"))

    print("Interpretation guide:")
    if isinstance(n_pos, float) and not math.isnan(n_pos):
        if n_pos >= 0.70:
            print("  STRONG  — edge appears cross-ticker (>70% of tickers profitable OOS)")
        elif n_pos >= 0.50:
            print("  MODERATE — edge holds on majority of tickers, but not all")
        else:
            print("  WEAK    — fewer than half of tickers are profitable OOS; "
                  "edge may be ticker-specific or config over-fit")
    if isinstance(mean_sharpe, float) and not math.isnan(mean_sharpe):
        if mean_sharpe >= 0.5:
            print("  Mean OOS Sharpe is acceptable (>=0.5)")
        elif mean_sharpe >= 0.0:
            print("  Mean OOS Sharpe is positive but modest (<0.5)")
        else:
            print("  Mean OOS Sharpe is negative — consider re-evaluating config")
    print()


if __name__ == "__main__":
    main()
