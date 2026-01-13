"""Compatibility wrapper.

The backtest engine was moved to research/backtest_signals.py.
This wrapper keeps existing commands working:
  python backtest_signals.py ...
"""

from research.backtest_signals import main


if __name__ == "__main__":
    main()
