"""
signals package

Implements a decision-focused trading signal engine:
- Triple-barrier labeling
- Probabilistic classifier (profit / timeout / stop)
- Per-ticker state machine (FLAT/LONG) with DB-backed positions
"""
