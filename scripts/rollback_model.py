"""
Rollback utility: point ACTIVE symlink to a previous model version.

Usage:
    python scripts/rollback_model.py TSLA --version 20250115_143022
    python scripts/rollback_model.py TSLA --list  # show available versions
"""

import argparse
import logging
from pathlib import Path
import re


def list_versions(ticker: str) -> None:
    """List all available model versions for a ticker."""
    model_base = Path("models") / ticker.upper()
    
    if not model_base.exists():
        print(f"❌ No model directory for {ticker}")
        return
    
    ts_pattern = re.compile(r"^\d{8}_\d{6}$")
    versions = [d for d in model_base.iterdir() if d.is_dir() and ts_pattern.match(d.name)]
    
    if not versions:
        print(f"No versioned models found for {ticker}")
        return
    
    versions = sorted(versions, reverse=True)
    
    # Identify current ACTIVE
    active_link = model_base / "ACTIVE"
    current_active = None
    
    if active_link.exists() or active_link.is_symlink():
        if active_link.is_symlink():
            try:
                current_active = active_link.resolve().name
            except Exception:
                pass
        else:
            try:
                current_active = active_link.read_text(encoding="utf-8").strip()
            except Exception:
                pass
    
    print(f"\n📦 Available model versions for {ticker}:\n")
    for v in versions:
        marker = " ← ACTIVE" if v.name == current_active else ""
        # Check if model.h5 exists
        model_file = v / "model.h5"
        status = "✓" if model_file.exists() else "✗"
        print(f"  {status} {v.name}{marker}")
    
    print()


def rollback(ticker: str, version: str) -> None:
    """Rollback ACTIVE to a specific version."""
    model_base = Path("models") / ticker.upper()
    version_dir = model_base / version
    
    if not version_dir.exists():
        print(f"❌ Version {version} does not exist for {ticker}")
        print(f"Run with --list to see available versions")
        return
    
    # Check if model.h5 exists
    model_file = version_dir / "model.h5"
    if not model_file.exists():
        print(f"⚠️  Warning: model.h5 not found in {version}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Rollback cancelled.")
            return
    
    active_link = model_base / "ACTIVE"
    
    # Remove existing ACTIVE
    if active_link.exists() or active_link.is_symlink():
        active_link.unlink()
    
    # Create new symlink or marker file
    try:
        active_link.symlink_to(version, target_is_directory=True)
        print(f"✅ Rolled back ACTIVE → {version} (symlink)")
    except OSError:
        # Fallback for Windows
        active_link.write_text(version, encoding="utf-8")
        print(f"✅ Rolled back ACTIVE → {version} (marker file)")
    
    print(f"\n🔄 {ticker} now using model: {version}")


def main():
    logging.basicConfig(level=logging.INFO)
    
    p = argparse.ArgumentParser(description="Rollback model to a previous version")
    p.add_argument("ticker", type=str, help="Ticker symbol")
    p.add_argument("--list", action="store_true", help="List available versions")
    p.add_argument("--version", type=str, help="Version timestamp to roll back to (e.g., 20250115_143022)")
    
    args = p.parse_args()
    
    if args.list:
        list_versions(args.ticker)
    elif args.version:
        rollback(args.ticker, args.version)
    else:
        print("❌ Must specify either --list or --version")
        p.print_help()


if __name__ == "__main__":
    main()
