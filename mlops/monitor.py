"""
mlops/monitor.py
Drift monitor using KS-test on prediction residuals.

• check_ticker()    – returns (p_value, is_drift)
• drift_job()       – loops over watchlist; triggers retrain_job(force=True)
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Tuple

from scipy.stats import ks_2samp

from storage import DB, get_watchlist
from mlops.utils import retrain_job   # re-use your existing function


# ─────────────────────────────────────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_residuals(ticker: str, days: int = 60):
    """Return list of residuals (actual - predicted) for `ticker`, newest first."""
    sql = """
        SELECT error
        FROM   evaluation
        WHERE  ticker = ?
        ORDER BY ts DESC
        LIMIT ?
    """
    with sqlite3.connect(DB) as conn:
        rows = conn.execute(sql, (ticker, days)).fetchall()
    return [r[0] for r in rows if r[0] is not None]


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────
def check_ticker(
    ticker: str,
    *,
    lookback: int = 5,
    baseline: int = 30,
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """
    KS-test between the most-recent `lookback` residuals and the prior
    `baseline` residuals.  Returns (p_value, is_drift).
    """
    resids = _fetch_residuals(ticker, days=lookback + baseline)
    if len(resids) < lookback + baseline:
        logging.info("%s: not enough residuals for drift test", ticker)
        return 1.0, False

    recent = resids[:lookback]                 # newest
    hist   = resids[lookback:]                 # baseline
    p_val  = ks_2samp(recent, hist).pvalue
    return p_val, p_val < alpha


def drift_job( 
    lookback: int = 5,
    baseline: int = 30,
    alpha: float = 0.05,
):
    """
    Loop over all tickers in the watchlist; retrain those that drifted.
    """
    for tkr in get_watchlist():
        p, drift = check_ticker(tkr, lookback=lookback, baseline=baseline, alpha=alpha)
        if drift:
            logging.warning("⚠️  Drift detected for %s (p=%.4f) – retraining…", tkr, p)
            retrain_job(force=True, tickers=[tkr])
        else:
            logging.info("✅ No drift for %s (p=%.4f)", tkr, p)
