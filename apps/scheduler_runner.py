"""
APScheduler entrypoint for the scheduler container.

Schedules:
  Mon–Fri 14:45 UTC  →  signal_job(dry_run=True)   (15:45 CET / 10:45 ET)
  Mon–Fri 21:15 UTC  →  evaluate_job(dry_run=True)  (22:15 CET / 16:15 ET)

is_trading_day() is already called inside each job, so no extra guard is needed here.
"""
import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from apps.jobs import signal_job, evaluate_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)

scheduler = BlockingScheduler(timezone="UTC")

scheduler.add_job(
    lambda: signal_job(dry_run=True),
    CronTrigger(day_of_week="mon-fri", hour=14, minute=45, timezone="UTC"),
    id="signal_job",
    name="Daily signal job (paper)",
)

scheduler.add_job(
    lambda: evaluate_job(dry_run=True),
    CronTrigger(day_of_week="mon-fri", hour=21, minute=15, timezone="UTC"),
    id="evaluate_job",
    name="EOD evaluation job (paper)",
)

if __name__ == "__main__":
    logging.info("Scheduler starting – signal@14:45 UTC, evaluate@21:15 UTC (Mon–Fri)")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Scheduler stopped.")
