"""
bootstrap_ci.py

Bootstrap-Konfidenzintervalle für Backtest-Metriken.

Liest trades.csv (Pflicht) und optional equity.csv, um per Resampling
95%-Konfidenzintervalle für folgende Metriken zu berechnen:

  Aus trades.csv (Trade-Bootstrap):
    - Win Rate
    - Avg R-Multiple
    - Profit Factor
    - Total Return (näherungsweise: sum(pnl_net) / initial_equity)
    - Avg Hold Days

  Aus equity.csv (Block-Bootstrap auf Tagesrenditen):
    - Sharpe Ratio (annualisiert, tagesgenau)
    - Max Drawdown

  p-Wert: Anteil der Bootstrap-Samples, bei denen die Metrik <= 0 liegt
          (bzw. Profit Factor <= 1). Kleine p-Werte stützen die Hypothese,
          dass die Edge real ist und kein Zufall.

Verwendung
----------
  # Nur trades.csv (Sharpe aus Trade-Renditen geschätzt)
  python research/bootstrap_ci.py backtests/runs/<run_id>/trades.csv

  # Mit equity.csv (präziser Sharpe + Drawdown-CI)
  python research/bootstrap_ci.py backtests/runs/<run_id>/trades.csv \\
    --equity backtests/runs/<run_id>/equity.csv \\
    --n-samples 2000 --ci 0.95

  # Auf Holdout-Fenster beschränken
  python research/bootstrap_ci.py backtests/runs/<run_id>/trades.csv \\
    --equity backtests/runs/<run_id>/equity.csv \\
    --holdout-start 2024-01-01 --holdout-end 2025-12-31

  # Ergebnis als JSON speichern
  python research/bootstrap_ci.py backtests/runs/<run_id>/trades.csv \\
    --equity backtests/runs/<run_id>/equity.csv \\
    --out backtests/runs/<run_id>/bootstrap_ci.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_trades(path: str,
                 holdout_start: Optional[str],
                 holdout_end: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"trades.csv is empty: {path}")

    required = {"pnl_net", "r_mult", "hold_days"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"trades.csv missing columns: {missing}")

    df["pnl_net"]   = pd.to_numeric(df["pnl_net"],   errors="coerce")
    df["r_mult"]    = pd.to_numeric(df["r_mult"],     errors="coerce")
    df["hold_days"] = pd.to_numeric(df["hold_days"],  errors="coerce")
    df = df.dropna(subset=["pnl_net", "r_mult"])

    # Optional: filter to holdout window (by exit_date)
    if (holdout_start or holdout_end) and "exit_date" in df.columns:
        df["exit_dt"] = pd.to_datetime(df["exit_date"], utc=True, errors="coerce")
        if holdout_start:
            df = df[df["exit_dt"] >= pd.Timestamp(holdout_start, tz="UTC")]
        if holdout_end:
            df = df[df["exit_dt"] <= pd.Timestamp(holdout_end, tz="UTC")]
        df = df.drop(columns=["exit_dt"])

    return df.reset_index(drop=True)


def _load_equity(path: str,
                 holdout_start: Optional[str],
                 holdout_end: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"equity.csv is empty: {path}")

    if "equity" not in df.columns:
        raise ValueError("equity.csv must contain an 'equity' column")

    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    df = df.dropna(subset=["equity"])

    if "date" in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.sort_values("date_dt")

        if holdout_start:
            df = df[df["date_dt"] >= pd.Timestamp(holdout_start, tz="UTC")]
        if holdout_end:
            df = df[df["date_dt"] <= pd.Timestamp(holdout_end, tz="UTC")]

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Metric functions
# ─────────────────────────────────────────────────────────────────────────────

def _sharpe(daily_rets: np.ndarray) -> float:
    if len(daily_rets) < 2:
        return float("nan")
    mu  = float(np.mean(daily_rets))
    std = float(np.std(daily_rets, ddof=1))
    return float((mu / std) * math.sqrt(252)) if std > 1e-12 else 0.0


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    dd   = equity / np.maximum(peak, 1e-12) - 1.0
    return float(np.min(dd))


def _profit_factor(pnl: np.ndarray) -> float:
    gains  = float(pnl[pnl > 0].sum())
    losses = float(pnl[pnl < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / abs(losses))


# ─────────────────────────────────────────────────────────────────────────────
# Trade-level bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def _trade_metrics(pnl: np.ndarray,
                   r:   np.ndarray,
                   hd:  np.ndarray,
                   initial_equity: float) -> Dict[str, float]:
    return {
        "win_rate":      float(np.mean(pnl > 0)),
        "avg_r":         float(np.mean(r)),
        "profit_factor": _profit_factor(pnl),
        "total_return":  float(pnl.sum() / initial_equity),
        "avg_hold_days": float(np.mean(hd)),
    }


def bootstrap_trades(
    pnl: np.ndarray,
    r:   np.ndarray,
    hd:  np.ndarray,
    *,
    initial_equity: float,
    n_samples: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Resample trades with replacement N times; return arrays of metric values."""
    n = len(pnl)
    results: Dict[str, List[float]] = {
        k: [] for k in ("win_rate", "avg_r", "profit_factor", "total_return", "avg_hold_days")
    }

    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        m   = _trade_metrics(pnl[idx], r[idx], hd[idx], initial_equity)
        for k, v in m.items():
            results[k].append(v)

    return {k: np.array(v) for k, v in results.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Block bootstrap on equity-curve daily returns
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_equity(
    equity: np.ndarray,
    *,
    n_samples: int,
    block_size: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Block bootstrap on daily log-returns from the equity curve.

    Block size ~20 bars (≈1 trading month) preserves short-range autocorrelation
    that is present in equity curves (momentum, mean-reversion periods).
    """
    daily_rets = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    T = len(daily_rets)

    if T < 2:
        return {"sharpe": np.array([float("nan")] * n_samples),
                "max_drawdown": np.array([float("nan")] * n_samples)}

    # Build block start indices
    block_starts = np.arange(T - block_size + 1)
    n_blocks_needed = math.ceil(T / block_size)

    sharpes: List[float] = []
    drawdowns: List[float] = []

    for _ in range(n_samples):
        chosen = rng.choice(block_starts, size=n_blocks_needed, replace=True)
        boot_rets = np.concatenate([daily_rets[s: s + block_size] for s in chosen])[:T]

        sharpes.append(_sharpe(boot_rets))

        # Reconstruct equity path for drawdown
        eq = np.cumprod(1.0 + boot_rets) * equity[0]
        drawdowns.append(_max_drawdown(eq))

    return {
        "sharpe":       np.array(sharpes),
        "max_drawdown": np.array(drawdowns),
    }


# ─────────────────────────────────────────────────────────────────────────────
# p-value computation
# ─────────────────────────────────────────────────────────────────────────────

def _pvalue(samples: np.ndarray, null: float = 0.0, direction: str = "above") -> float:
    """
    Fraction of bootstrap samples at or below the null hypothesis value.
    direction="above" → H1: metric > null (e.g. return > 0, Sharpe > 0)
    direction="below" → H1: metric < null (e.g. drawdown < 0)
    """
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0:
        return float("nan")
    if direction == "above":
        return float(np.mean(finite <= null))
    return float(np.mean(finite >= null))


# ─────────────────────────────────────────────────────────────────────────────
# Summary building
# ─────────────────────────────────────────────────────────────────────────────

def _ci(samples: np.ndarray, level: float) -> Tuple[float, float]:
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0:
        return (float("nan"), float("nan"))
    alpha = 1.0 - level
    lo = float(np.percentile(finite, 100 * alpha / 2))
    hi = float(np.percentile(finite, 100 * (1 - alpha / 2)))
    return lo, hi


def build_summary(
    trades_df: pd.DataFrame,
    equity_df: Optional[pd.DataFrame],
    *,
    initial_equity: float,
    n_samples: int,
    ci_level: float,
    block_size: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    pnl = trades_df["pnl_net"].to_numpy(dtype=float)
    r   = trades_df["r_mult"].to_numpy(dtype=float)
    hd  = trades_df["hold_days"].fillna(0).to_numpy(dtype=float)

    n_trades = int(len(trades_df))

    # ── Observed values ────────────────────────────────────────────────────
    observed_trades = _trade_metrics(pnl, r, hd, initial_equity)

    observed_sharpe   = float("nan")
    observed_drawdown = float("nan")

    if equity_df is not None and len(equity_df) >= 2:
        eq = equity_df["equity"].to_numpy(dtype=float)
        daily_rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
        observed_sharpe   = _sharpe(daily_rets)
        observed_drawdown = _max_drawdown(eq)

    # ── Bootstrap ──────────────────────────────────────────────────────────
    boot_trades = bootstrap_trades(
        pnl, r, hd,
        initial_equity=initial_equity,
        n_samples=n_samples,
        rng=rng,
    )

    boot_equity: Dict[str, np.ndarray] = {}
    if equity_df is not None and len(equity_df) >= 2:
        eq = equity_df["equity"].to_numpy(dtype=float)
        boot_equity = bootstrap_equity(
            eq,
            n_samples=n_samples,
            block_size=block_size,
            rng=rng,
        )

    # ── Assemble rows ──────────────────────────────────────────────────────
    rows: List[Dict[str, Any]] = []

    def _row(name: str,
             observed: float,
             samples: np.ndarray,
             fmt: str = "pct",
             null: float = 0.0,
             p_direction: str = "above") -> Dict[str, Any]:
        lo, hi = _ci(samples, ci_level)
        pval   = _pvalue(samples, null=null, direction=p_direction)
        return dict(
            metric=name,
            observed=observed,
            ci_lower=lo,
            ci_upper=hi,
            p_value=pval,
            n_bootstrap=n_samples,
            fmt=fmt,
        )

    rows.append(_row("Win Rate",       observed_trades["win_rate"],
                     boot_trades["win_rate"],       fmt="pct"))
    rows.append(_row("Avg R-Multiple", observed_trades["avg_r"],
                     boot_trades["avg_r"],           fmt="num3"))
    rows.append(_row("Profit Factor",  observed_trades["profit_factor"],
                     boot_trades["profit_factor"],   fmt="num2", null=1.0))
    rows.append(_row("Total Return",   observed_trades["total_return"],
                     boot_trades["total_return"],    fmt="pct"))
    rows.append(_row("Avg Hold Days",  observed_trades["avg_hold_days"],
                     boot_trades["avg_hold_days"],   fmt="num1",
                     p_direction="none"))

    if "sharpe" in boot_equity:
        rows.append(_row("Sharpe Ratio",  observed_sharpe,
                         boot_equity["sharpe"],       fmt="num2"))
    if "max_drawdown" in boot_equity:
        rows.append(_row("Max Drawdown",  observed_drawdown,
                         boot_equity["max_drawdown"], fmt="pct",
                         null=0.0, p_direction="below"))

    return {
        "n_trades":        n_trades,
        "initial_equity":  initial_equity,
        "n_samples":       n_samples,
        "ci_level":        ci_level,
        "block_size":      block_size,
        "seed":            seed,
        "has_equity_curve": equity_df is not None,
        "rows":            rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Formatting + printing
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_val(v: float, fmt: str) -> str:
    if not math.isfinite(v):
        return "n/a"
    if fmt == "pct":
        return f"{v * 100:+.1f}%"
    if fmt == "num1":
        return f"{v:.1f}"
    if fmt == "num2":
        return f"{v:.2f}"
    if fmt == "num3":
        return f"{v:.3f}"
    return str(v)


def _stars(p: float) -> str:
    if not math.isfinite(p):
        return "   "
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "** "
    if p < 0.10:
        return "*  "
    return "   "


def print_report(summary: Dict[str, Any]) -> None:
    rows        = summary["rows"]
    n_trades    = summary["n_trades"]
    n_samples   = summary["n_samples"]
    ci_pct      = int(summary["ci_level"] * 100)
    has_eq      = summary["has_equity_curve"]
    block_size  = summary["block_size"]

    sep  = "─" * 72
    sep2 = "=" * 72

    print()
    print(sep2)
    print(f"  BOOTSTRAP KONFIDENZINTERVALLE  "
          f"(n_samples={n_samples:,}, CI={ci_pct}%)")
    print(sep2)
    print(f"  Trades: {n_trades}   |   "
          f"Equity-Kurve: {'ja (Block-BS, block=' + str(block_size) + ')' if has_eq else 'nein (Sharpe geschätzt)'}")
    print(sep)

    col_w = (22, 10, 10, 10, 9, 4)
    header = (
        f"{'Metrik':<{col_w[0]}}"
        f"{'Beobachtet':>{col_w[1]}}"
        f"{'CI untere':>{col_w[2]}}"
        f"{'CI obere':>{col_w[3]}}"
        f"{'p-Wert':>{col_w[4]}}"
        f"{'':>{col_w[5]}}"
    )
    print(f"  {header}")
    print(f"  {sep}")

    for row in rows:
        fmt  = row["fmt"]
        pdir = row.get("p_direction", "above")

        obs_s = _fmt_val(row["observed"],  fmt)
        lo_s  = _fmt_val(row["ci_lower"],  fmt)
        hi_s  = _fmt_val(row["ci_upper"],  fmt)

        pval  = row["p_value"]
        if pdir == "none" or not math.isfinite(pval):
            pval_s = "  —  "
            stars  = "   "
        else:
            pval_s = f"{pval:.3f}"
            stars  = _stars(pval)

        line = (
            f"  {row['metric']:<{col_w[0]}}"
            f"{obs_s:>{col_w[1]}}"
            f"{lo_s:>{col_w[2]}}"
            f"{hi_s:>{col_w[3]}}"
            f"{pval_s:>{col_w[4]}}"
            f"  {stars}"
        )
        print(line)

    print(f"  {sep}")
    print()
    print("  Signifikanzniveaus: *** p<0.01  ** p<0.05  * p<0.10")
    print("  p-Wert = Anteil Bootstrap-Samples, bei denen die Metrik die")
    print("           Nullhypothese (Edge=0) nicht übertrifft.")
    print()
    print("  Breite CIs deuten auf zu wenige Trades hin — Ergebnis statistisch")
    print("  schwach. Enge CIs mit kleinem p-Wert stützen eine reale Edge.")
    print(sep2)
    print()

    # Interpretation
    ret_row = next((r for r in rows if r["metric"] == "Total Return"), None)
    sharpe_row = next((r for r in rows if r["metric"] == "Sharpe Ratio"), None)

    print("  Interpretation:")

    if ret_row:
        lo = ret_row["ci_lower"]
        hi = ret_row["ci_upper"]
        pv = ret_row["p_value"]
        width = hi - lo if math.isfinite(lo) and math.isfinite(hi) else float("nan")

        if math.isfinite(pv) and pv < 0.05 and math.isfinite(width) and width < 0.40:
            print("  STARK  — Return-CI ist eng und signifikant (p<0.05).")
        elif math.isfinite(pv) and pv < 0.10:
            print("  MODERAT — Return tendenziell positiv, aber CI noch breit.")
        else:
            print("  SCHWACH — Return nicht signifikant von Null verschieden.")

    if sharpe_row:
        lo_sh = sharpe_row["ci_lower"]
        if math.isfinite(lo_sh):
            if lo_sh > 0.5:
                print("  Sharpe-CI untere Grenze > 0.5: robuste risikobereinigte Rendite.")
            elif lo_sh > 0.0:
                print("  Sharpe-CI untere Grenze > 0: Edge vorhanden, aber schwankend.")
            else:
                print("  Sharpe-CI schließt 0 ein: risikobereinigte Rendite unsicher.")

    if n_trades < 30:
        print(f"\n  WARNUNG: Nur {n_trades} Trades — CIs sind sehr breit.")
        print("           Mehr Daten oder weniger restriktives entry_min_ev empfohlen.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Bootstrap-Konfidenzintervalle für Backtest-Metriken."
    )

    p.add_argument("trades", help="Pfad zu trades.csv")
    p.add_argument(
        "--equity", default=None,
        help="Pfad zu equity.csv (optional; ermöglicht präzisen Sharpe + Drawdown-CI)",
    )

    p.add_argument("--holdout-start", default=None,
                   help="Auswertungsfenster Start (z. B. 2024-01-01)")
    p.add_argument("--holdout-end",   default=None,
                   help="Auswertungsfenster Ende (z. B. 2025-12-31)")

    p.add_argument("--n-samples",      type=int,   default=2000,
                   help="Anzahl Bootstrap-Iterationen (Standard: 2000)")
    p.add_argument("--ci",             type=float, default=0.95,
                   help="Konfidenzniveau (Standard: 0.95)")
    p.add_argument("--block-size",     type=int,   default=20,
                   help="Blockgröße für Equity-Block-Bootstrap (Standard: 20 Tage)")
    p.add_argument("--initial-equity", type=float, default=100_000.0,
                   help="Startkapital für Return-Berechnung (Standard: 100000)")
    p.add_argument("--seed",           type=int,   default=42,
                   help="Zufalls-Seed für Reproduzierbarkeit (Standard: 42)")
    p.add_argument("--out",            default=None,
                   help="Optionaler Ausgabepfad für JSON-Ergebnis")

    args = p.parse_args()

    if not (0 < args.ci < 1):
        p.error("--ci muss zwischen 0 und 1 liegen (z. B. 0.95)")
    if args.n_samples < 100:
        p.error("--n-samples muss mindestens 100 sein")
    if args.block_size < 1:
        p.error("--block-size muss mindestens 1 sein")

    # Load data
    print(f"\nLade trades.csv: {args.trades}")
    trades_df = _load_trades(args.trades, args.holdout_start, args.holdout_end)
    print(f"  → {len(trades_df)} Trades geladen")

    equity_df = None
    if args.equity:
        print(f"Lade equity.csv: {args.equity}")
        equity_df = _load_equity(args.equity, args.holdout_start, args.holdout_end)
        print(f"  → {len(equity_df)} Equity-Punkte geladen")

    print(f"\nStarte Bootstrap (n={args.n_samples:,}, CI={int(args.ci*100)}%)…")

    summary = build_summary(
        trades_df,
        equity_df,
        initial_equity=float(args.initial_equity),
        n_samples=int(args.n_samples),
        ci_level=float(args.ci),
        block_size=int(args.block_size),
        seed=int(args.seed),
    )

    print_report(summary)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Make JSON-serialisierbar
        def _clean(obj: Any) -> Any:
            if isinstance(obj, float):
                return None if not math.isfinite(obj) else obj
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            return obj

        out_path.write_text(
            json.dumps(_clean(summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Ergebnis gespeichert: {out_path}")


if __name__ == "__main__":
    main()
