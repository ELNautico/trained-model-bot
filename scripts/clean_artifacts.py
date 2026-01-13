"""scripts/clean_artifacts.py

Removes locally-generated artifacts that should not be committed.

What it deletes (by default)
----------------------------
- Python bytecode caches: __pycache__/ and *.pyc
- Backtest outputs: backtests/*.csv
- SQLite DB: bot.db

This script is intentionally conservative; it only touches well-known
generated artifacts. Extend patterns as needed for your workflow.

Usage
-----
  python scripts/clean_artifacts.py
  python scripts/clean_artifacts.py --dry-run
  python scripts/clean_artifacts.py --backtests-glob "backtests/*.csv"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def rm(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] rm {path}")
        return

    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
        # remove empty dirs bottom-up
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass
        try:
            path.rmdir()
        except OSError:
            pass
    else:
        path.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Delete locally generated artifacts")
    p.add_argument("--dry-run", action="store_true", help="Print what would be deleted")
    p.add_argument(
        "--backtests-glob",
        type=str,
        default="backtests/*.csv",
        help="Glob for backtest outputs to delete",
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    # 1) Python caches
    for d in repo_root.rglob("__pycache__"):
        rm(d, dry_run=bool(args.dry_run))

    for f in repo_root.rglob("*.pyc"):
        rm(f, dry_run=bool(args.dry_run))

    # 2) Backtest outputs
    for f in repo_root.glob(args.backtests_glob):
        rm(f, dry_run=bool(args.dry_run))

    # 3) Local DB
    db = repo_root / "bot.db"
    if db.exists():
        rm(db, dry_run=bool(args.dry_run))

    if not args.dry_run:
        print("Done.")


if __name__ == "__main__":
    main()
