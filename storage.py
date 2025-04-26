# storage.py
import sqlite3, json
from pathlib import Path
from datetime import datetime
from typing import List

DB = Path(__file__).with_name("bot.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecast(
  ts            TEXT,    -- YYYY-MM-DD
  ticker        TEXT,
  current_px    REAL,
  predicted_px  REAL,
  direction     TEXT,
  confidence    REAL,
  model_tag     TEXT,
  PRIMARY KEY(ts, ticker)
);

CREATE TABLE IF NOT EXISTS evaluation(
  ts            TEXT,    -- YYYY-MM-DD
  ticker        TEXT,
  predicted_px  REAL,
  actual_px     REAL,
  error         REAL,
  pct_error     REAL,
  model_tag     TEXT,
  PRIMARY KEY(ts, ticker)
);
"""

def _conn():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    with _conn() as c:
        c.executescript(_SCHEMA)


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
    """
    Return the last `n` pct_error values from the evaluation table, ordered by date descending.
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT pct_error FROM evaluation ORDER BY ts DESC LIMIT ?", (n,)
        ).fetchall()
    return [r[0] for r in rows]
