import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

DB = Path(__file__).with_name("bot.db")

# --------------------------------------------------------------------------------------
# SCHEMA
# --------------------------------------------------------------------------------------
#
# We keep your existing forecast/evaluation tables for backward compatibility,
# but the new signal engine uses:
#
#   signals   : daily model outputs per ticker (probabilities + recommended action)
#   positions : current open position per ticker (state machine: FLAT/LONG)
#   trades    : closed trades with realized PnL
#
# This design supports:
#   - "wait until buy" behavior
#   - "tell me when to sell"
#   - auditable history (signals + trades)
#
_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecast (
  ts TEXT,
  ticker TEXT,
  current_px REAL,
  predicted_px REAL,
  direction TEXT,
  confidence REAL,
  model_tag TEXT,
  PRIMARY KEY (ts, ticker)
);

CREATE TABLE IF NOT EXISTS evaluation (
  ts TEXT,
  ticker TEXT,
  predicted_px REAL,
  actual_px REAL,
  error REAL,
  pct_error REAL,
  model_tag TEXT,
  PRIMARY KEY (ts, ticker)
);

CREATE TABLE IF NOT EXISTS watchlist (
  ticker TEXT PRIMARY KEY,
  added_at TEXT DEFAULT CURRENT_TIMESTAMP,
  last_forecast TEXT,
  last_trained TEXT
);

-- NEW: probabilistic signals (triple-barrier model outputs)
CREATE TABLE IF NOT EXISTS signals (
  ts TEXT,
  ticker TEXT,
  action TEXT,
  p_profit REAL,
  p_timeout REAL,
  p_stop REAL,
  edge REAL,
  entry_px REAL,
  stop_px REAL,
  target_px REAL,
  horizon_days INTEGER,
  meta_json TEXT,
  PRIMARY KEY (ts, ticker)
);

-- NEW: single open position per ticker (state machine)
CREATE TABLE IF NOT EXISTS positions (
  ticker TEXT PRIMARY KEY,
  state TEXT,
  entry_ts TEXT,
  entry_px REAL,
  shares INTEGER,
  stop_px REAL,
  target_px REAL,
  horizon_days INTEGER,
  hold_days INTEGER,
  last_update_ts TEXT
);

-- NEW: executed trades log (closed positions)
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT,
  ticker TEXT,
  side TEXT,
  entry_ts TEXT,
  entry_px REAL,
  exit_ts TEXT,
  exit_px REAL,
  shares INTEGER,
  reason TEXT,
  pnl REAL,
  return_pct REAL
);
"""


def _conn():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    with _conn() as c:
        c.executescript(_SCHEMA)


# --------------------------------------------------------------------------------------
# Legacy forecast/evaluation helpers (kept)
# --------------------------------------------------------------------------------------
def save_forecast(row: dict):
    with _conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO forecast
               VALUES(:ts, :ticker, :current_px, :predicted_px,
                      :direction, :confidence, :model_tag)""",
            row
        )


def save_evaluation(row: dict):
    with _conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO evaluation
               VALUES(:ts, :ticker, :predicted_px,
                      :actual_px, :error, :pct_error,
                      :model_tag)""",
            row
        )


def get_recent_errors(n: int) -> List[float]:
    with _conn() as c:
        rows = c.execute(
            "SELECT pct_error FROM evaluation ORDER BY ts DESC LIMIT ?", (n,)
        ).fetchall()
    return [r[0] for r in rows]


# --------------------------------------------------------------------------------------
# Watchlist
# --------------------------------------------------------------------------------------
def add_to_watchlist(ticker: str):
    with _conn() as c:
        c.execute(
            """
            INSERT OR IGNORE INTO watchlist (ticker, added_at)
            VALUES (?, ?)
            """,
            (ticker.upper(), datetime.utcnow().isoformat())
        )


def remove_from_watchlist(ticker: str):
    with _conn() as c:
        c.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))


def get_watchlist() -> List[str]:
    with _conn() as c:
        rows = c.execute("SELECT ticker FROM watchlist").fetchall()
    return [r[0] for r in rows]


def update_watchlist_timestamp(ticker: str, column: str):
    if column not in ("last_trained", "last_forecast"):
        raise ValueError("Invalid column for timestamp update.")
    with _conn() as c:
        c.execute(
            f"UPDATE watchlist SET {column} = ? WHERE ticker = ?",
            (datetime.utcnow().isoformat(), ticker.upper())
        )


# --------------------------------------------------------------------------------------
# NEW: signals / positions / trades
# --------------------------------------------------------------------------------------
def save_signal(row: Dict[str, Any]) -> None:
    """
    Expected keys:
      ts, ticker, action, p_profit, p_timeout, p_stop, edge,
      entry_px, stop_px, target_px, horizon_days, meta_json (dict)
    """
    payload = dict(row)
    payload["ticker"] = payload["ticker"].upper()
    payload["meta_json"] = json.dumps(payload.get("meta_json", {}), ensure_ascii=False)

    with _conn() as c:
        c.execute(
            """
            INSERT OR REPLACE INTO signals
            (ts, ticker, action, p_profit, p_timeout, p_stop, edge,
             entry_px, stop_px, target_px, horizon_days, meta_json)
            VALUES
            (:ts, :ticker, :action, :p_profit, :p_timeout, :p_stop, :edge,
             :entry_px, :stop_px, :target_px, :horizon_days, :meta_json)
            """,
            payload
        )


def get_position(ticker: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute(
            """
            SELECT ticker, state, entry_ts, entry_px, shares, stop_px, target_px,
                   horizon_days, hold_days, last_update_ts
            FROM positions
            WHERE ticker = ?
            """,
            (ticker.upper(),)
        ).fetchone()

    if not row:
        return None

    keys = [
        "ticker", "state", "entry_ts", "entry_px", "shares", "stop_px", "target_px",
        "horizon_days", "hold_days", "last_update_ts"
    ]
    return dict(zip(keys, row))


def upsert_position(pos: Dict[str, Any]) -> None:
    payload = dict(pos)
    payload["ticker"] = payload["ticker"].upper()

    with _conn() as c:
        c.execute(
            """
            INSERT OR REPLACE INTO positions
            (ticker, state, entry_ts, entry_px, shares, stop_px, target_px,
             horizon_days, hold_days, last_update_ts)
            VALUES
            (:ticker, :state, :entry_ts, :entry_px, :shares, :stop_px, :target_px,
             :horizon_days, :hold_days, :last_update_ts)
            """,
            payload
        )


def close_position(ticker: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM positions WHERE ticker = ?", (ticker.upper(),))


def list_positions() -> List[Dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            """
            SELECT ticker, state, entry_ts, entry_px, shares, stop_px, target_px,
                   horizon_days, hold_days, last_update_ts
            FROM positions
            ORDER BY ticker ASC
            """
        ).fetchall()

    keys = [
        "ticker", "state", "entry_ts", "entry_px", "shares", "stop_px", "target_px",
        "horizon_days", "hold_days", "last_update_ts"
    ]
    return [dict(zip(keys, r)) for r in rows]


def save_trade(row: Dict[str, Any]) -> None:
    """
    Expected keys:
      ts, ticker, side, entry_ts, entry_px, exit_ts, exit_px, shares, reason, pnl, return_pct
    """
    payload = dict(row)
    payload["ticker"] = payload["ticker"].upper()

    with _conn() as c:
        c.execute(
            """
            INSERT INTO trades
            (ts, ticker, side, entry_ts, entry_px, exit_ts, exit_px, shares, reason, pnl, return_pct)
            VALUES
            (:ts, :ticker, :side, :entry_ts, :entry_px, :exit_ts, :exit_px, :shares, :reason, :pnl, :return_pct)
            """,
            payload
        )
