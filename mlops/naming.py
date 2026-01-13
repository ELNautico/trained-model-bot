from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_run_ts(dt: Optional[datetime] = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return dt.strftime("%Y%m%d_%H%M%SZ")


def slug(s: Optional[str]) -> str:
    s = (s or "").strip()
    if not s:
        return "untagged"
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "untagged"


def dec_token(x: float, decimals: int = 3) -> str:
    # 0.2 -> "0p200" (fixed decimals for stable naming)
    return f"{x:.{decimals}f}".replace(".", "p")


def build_run_id(*, ticker: str, tag: Optional[str], kind: str, ts: Optional[str] = None) -> str:
    ts = ts or utc_run_ts()
    return f"{ts}__{ticker.upper()}__{slug(tag)}__{kind}"


def run_dir(backtests_dir: Path, run_id: str) -> Path:
    return backtests_dir / "runs" / run_id


def write_meta(path: Path, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON without importing json here keeps this file dependency-light
    import json
    path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    return dict(obj)
