"""Compatibility wrapper.

The sweep harness was moved to research/sweep_backtests.py.
This wrapper keeps existing commands working:
  python sweep_backtests.py ...
"""

from research.sweep_backtests import main


if __name__ == "__main__":
    main()
