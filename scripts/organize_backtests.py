from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

SWEEP_RE = re.compile(r"^(?P<ticker>[A-Z0-9._-]+)_sweep_results(?:_(?P<tag>.+))?\.csv$", re.I)
ART_RE = re.compile(r"^(?P<ticker>[A-Z0-9._-]+)_(?P<kind>equity|trades)_(?P<tag>.+)\.csv$", re.I)

# tries to parse ev/r/lb embedded in tags like "..._ev0p220_r80_lb1500"
PARAM_RE = re.compile(r"ev(?P<ev>\d+p\d+).*?_r(?P<r>\d+).*?_lb(?P<lb>\d+)", re.I)


def ts_from_mtime(p: Path) -> str:
    dt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    return dt.strftime("%Y%m%d_%H%M%SZ")


def slug(s: Optional[str]) -> str:
    s = (s or "").strip()
    if not s:
        return "untagged"
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "untagged"


def run_id_for_sweep(ticker: str, tag: Optional[str], ts: str) -> str:
    return f"{ts}__{ticker.upper()}__{slug(tag)}__sweep"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtests", default="backtests", help="Backtests directory")
    ap.add_argument("--apply", action="store_true", help="Actually move files (default is dry-run)")
    args = ap.parse_args()

    bt = Path(args.backtests).resolve()
    runs_dir = bt / "runs"
    legacy_dir = bt / "legacy_flat_csv"
    ensure_dir(runs_dir)
    ensure_dir(legacy_dir)

    files = [p for p in bt.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]

    # 1) detect sweeps and create run folders
    sweeps: dict[Tuple[str, str], Path] = {}  # (ticker, tagSlug) -> runDir
    for p in files:
        m = SWEEP_RE.match(p.name)
        if not m:
            continue
        ticker = m.group("ticker").upper()
        tag = m.group("tag")
        ts = ts_from_mtime(p)
        run_id = run_id_for_sweep(ticker, tag, ts)
        rdir = runs_dir / run_id
        ensure_dir(rdir)

        dest = rdir / "sweep_results.csv"
        print(f"[SWEEP] {p.name} -> {dest.relative_to(bt)}")
        if args.apply:
            shutil.move(str(p), str(dest))

        sweeps[(ticker, slug(tag))] = rdir

        # optional meta stub
        meta = rdir / "meta.json"
        if args.apply and not meta.exists():
            meta.write_text(
                f'{{\n  "kind": "sweep",\n  "runId": "{run_id}",\n  "ticker": "{ticker}",\n  "tag": "{tag or ""}"\n}}\n',
                encoding="utf-8",
            )

    # refresh file list after moving sweeps
    files = [p for p in bt.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]

    # 2) move artifacts
    for p in files:
        m = ART_RE.match(p.name)
        if not m:
            # unknown csv -> legacy
            dest = legacy_dir / p.name
            print(f"[LEGACY] {p.name} -> {dest.relative_to(bt)}")
            if args.apply:
                shutil.move(str(p), str(dest))
            continue

        ticker = m.group("ticker").upper()
        kind = m.group("kind").lower()  # equity|trades
        tag = m.group("tag")
        tag_slug = slug(tag)

        # attach to matching sweep if tag starts with sweep tag
        # Example: sweep tag "SMOKE", artifact tag "SMOKE_ev0p200_r80_lb1500"
        attach_dir = None
        for (tkr, sweep_tag_slug), rdir in sweeps.items():
            if tkr != ticker:
                continue
            if tag_slug == sweep_tag_slug or tag_slug.startswith(sweep_tag_slug + "_"):
                attach_dir = rdir
                break

        if attach_dir:
            topk = attach_dir / "topk"
            ensure_dir(topk)

            pm = PARAM_RE.search(tag)
            if pm:
                ev = pm.group("ev")
                r = pm.group("r")
                lb = pm.group("lb")
                cfg_dir = topk / f"ev{ev}__r{r}__lb{lb}"
                ensure_dir(cfg_dir)
                dest = cfg_dir / f"{kind}.csv"
            else:
                # not parseable -> artifacts bucket
                cfg_dir = topk / "artifacts_unparsed"
                ensure_dir(cfg_dir)
                dest = cfg_dir / p.name

            print(f"[ART] {p.name} -> {dest.relative_to(bt)}")
            if args.apply:
                shutil.move(str(p), str(dest))
        else:
            # standalone backtest run folder
            ts = ts_from_mtime(p)
            run_id = f"{ts}__{ticker}__{tag_slug}__backtest"
            rdir = runs_dir / run_id
            ensure_dir(rdir)
            dest = rdir / f"{kind}.csv"
            print(f"[BT] {p.name} -> {dest.relative_to(bt)}")
            if args.apply:
                shutil.move(str(p), str(dest))

            meta = rdir / "meta.json"
            if args.apply and not meta.exists():
                meta.write_text(
                    f'{{\n  "kind": "backtest",\n  "runId": "{run_id}",\n  "ticker": "{ticker}",\n  "tag": "{tag}"\n}}\n',
                    encoding="utf-8",
                )

    if not args.apply:
        print("\nDry-run complete. Re-run with --apply to actually move files.")


if __name__ == "__main__":
    main()
