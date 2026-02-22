import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

from .alert import send
from core.storage import (
    init_db,
    get_watchlist,
    list_positions,
    list_paper_positions,
)
from signals.config import SignalConfig
from signals.engine import (
    run_signal_cycle_for_ticker,
    run_eod_position_checks,
)

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
ACCT_BAL = 100_000
RISK_PER_TRADE = 0.01

# Trading calendar: for US equities, NYSE calendar is sufficient for "is today a trading day?"
# (NASDAQ/NYSE share holiday schedule; close times differ but day-open is aligned.)
DEFAULT_CALENDAR = "NYSE"


def is_trading_day(calendar_name: str = DEFAULT_CALENDAR) -> bool:
    today = datetime.utcnow().date()
    cal = mcal.get_calendar(calendar_name)
    sched = cal.schedule(start_date=today, end_date=today)
    return not sched.empty


def signal_job(force_retrain: bool = False, dry_run: bool = False):
    """
    Generate BUY/WAIT/HOLD/SELL signals for every ticker in the watchlist.

    dry_run=True activates paper-trading mode: all analysis runs as normal,
    but positions and trades are written to the paper_* tables only.
    No real capital is committed; messages are prefixed with [PAPER].
    """
    if not is_trading_day(DEFAULT_CALENDAR):
        logging.info("Market closed today – skipping signal job.")
        send("🛑 Market closed today – skipping signal job.")
        return

    init_db()
    cfg = SignalConfig()

    tickers = get_watchlist()
    if not tickers:
        send("📭 Watchlist is empty. Add tickers with /add TICKER.")
        return

    for ticker in tickers:
        # Be polite to rate limits / APIs
        time.sleep(5)

        try:
            msg = run_signal_cycle_for_ticker(
                ticker,
                cfg=cfg,
                account_balance=ACCT_BAL,
                risk_per_trade=RISK_PER_TRADE,
                force_retrain=force_retrain,
                dry_run=dry_run,
            )
            send(msg)

        except Exception as e:
            logging.exception("%s: signal job failed", ticker)
            send(f"❌ {ticker}: signal job failed – {e}")


def evaluate_job(dry_run: bool = False):
    """
    End-of-day position checks: stop/target/time exits.

    dry_run=True checks paper_positions instead of real positions.
    """
    if not is_trading_day(DEFAULT_CALENDAR):
        logging.info("Market closed today – skipping evaluation job.")
        send("🛑 Market closed today – skipping evaluation job.")
        return

    init_db()
    tickers = get_watchlist()
    if not tickers:
        send("📭 Watchlist is empty.")
        return

    exited_any = False
    for ticker in tickers:
        time.sleep(2)
        try:
            exit_msg = run_eod_position_checks(ticker, dry_run=dry_run)
            if exit_msg:
                exited_any = True
                send(exit_msg)
        except Exception as e:
            logging.exception("%s: evaluation failed", ticker)
            send(f"❌ {ticker}: evaluation failed – {e}")

    if not exited_any:
        pos = list_paper_positions() if dry_run else list_positions()
        label = "paper " if dry_run else ""
        if not pos:
            send(f"✅ EOD checks complete. No open {label}positions.")
        else:
            send(f"✅ EOD checks complete. Open {label}positions: {len(pos)}")


def positions_job():
    """Convenience job: prints current real open positions."""
    init_db()
    pos = list_positions()
    if not pos:
        send("Position report: FLAT (no open positions).")
        return

    lines = ["Open positions:"]
    for p in pos:
        lines.append(
            f"- {p['ticker']}: {p['state']} | shares={p['shares']} | "
            f"entry={float(p['entry_px']):.2f} | stop={float(p['stop_px']):.2f} | "
            f"target={float(p['target_px']):.2f} | hold={int(p['hold_days'])}/{int(p['horizon_days'])}"
        )
    send("\n".join(lines))


def paper_positions_job():
    """Convenience job: prints current paper (simulated) open positions."""
    init_db()
    pos = list_paper_positions()
    if not pos:
        send("Paper position report: FLAT (no open paper positions).")
        return

    lines = ["[PAPER] Open positions:"]
    for p in pos:
        lines.append(
            f"- {p['ticker']}: {p['state']} | shares={p['shares']} | "
            f"entry={float(p['entry_px']):.2f} | stop={float(p['stop_px']):.2f} | "
            f"target={float(p['target_px']):.2f} | hold={int(p['hold_days'])}/{int(p['horizon_days'])}"
        )
    send("\n".join(lines))


def help_job():
    help_text = (
        "Available Commands:\n\n"
        "/forecast        – Run the signal engine for all tickers (BUY/WAIT/HOLD/SELL).\n"
        "/paper           – Paper-trading run (same signals, no real positions).\n"
        "/evaluate        – Run EOD exit checks (stop/target/time-stop).\n"
        "/evaluate_paper  – EOD checks for paper positions.\n"
        "/positions       – Show open real positions.\n"
        "/paper_positions – Show open paper positions.\n"
        "/add TICKER      – Add a stock ticker to your watchlist.\n"
        "/remove TICKER   – Remove a stock from your watchlist.\n"
        "/watchlist       – Show your current watchlist.\n"
        "/help            – Show this message.\n\n"
        "Note: This bot provides model-driven signals, not investment advice."
    )
    send(help_text)


def _cli():
    import sys

    init_db()
    job = sys.argv[1] if len(sys.argv) > 1 else None

    if job == "forecast":
        signal_job(force_retrain=False)
    elif job == "forecast_force":
        signal_job(force_retrain=True)
    elif job == "forecast_paper":
        signal_job(dry_run=True)
    elif job == "evaluate":
        evaluate_job()
    elif job == "evaluate_paper":
        evaluate_job(dry_run=True)
    elif job == "positions":
        positions_job()
    elif job == "paper_positions":
        paper_positions_job()
    elif job == "help":
        help_job()
    else:
        print(
            "Usage: jobs.py "
            "[forecast|forecast_force|forecast_paper|"
            "evaluate|evaluate_paper|"
            "positions|paper_positions|help]"
        )


if __name__ == "__main__":
    _cli()
