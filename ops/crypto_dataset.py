"""T5 — assemble the crypto event dataset: headlines joined to the first tradeable 1m bar.

Usage:
    python ops/crypto_dataset.py
    python ops/crypto_dataset.py --news "data/crypto/news/*.parquet" --since 2026-01-01
    python ops/crypto_dataset.py --out data/crypto/dataset/events.parquet

``--news`` is repeatable and takes globs (default: ``wayback.parquet`` plus
``recorder.parquet``). Each file is read **once** into memory — the harvest rewrites
``wayback.parquet`` atomically while this runs, so the builder never re-reads it. An
absent or empty harvest produces the empty schema with a zeroed funnel, exit code 0.

The funnel, the leakage checks (PASS/FAIL per check) and the per-horizon counts
(``n``, mean excess in bps) are printed as plain text; the JSON report lands in
``results/crypto_dataset.json``. Exit codes: 0 on success, 2 when any check FAILS.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from spp.crypto.dataset import (
    CANONICAL_ASSETS,
    FUNNEL_KEYS,
    HORIZONS,
    build_dataset,
    load_bars,
    load_dataset_config,
    run_checks,
)
from spp.data.store import write_parquet_atomic
from spp.logging import get_logger

ROOT = Path(__file__).resolve().parents[1]
LOG = get_logger("ops.crypto_dataset")

DEFAULT_NEWS: tuple[str, ...] = (
    str(ROOT / "data" / "crypto" / "news" / "wayback.parquet"),
    str(ROOT / "data" / "crypto" / "news" / "recorder.parquet"),
)
DEFAULT_BARS_DIR = ROOT / "data" / "crypto" / "bars" / "1m"
DEFAULT_OUT = ROOT / "data" / "crypto" / "dataset" / "events.parquet"
DEFAULT_REPORT = ROOT / "results" / "crypto_dataset.json"

_NEWS_COLUMNS: tuple[str, ...] = (
    "source",
    "title",
    "url",
    "guid",
    "published_at",
    "snapshot_ts",
    "tickers",
)


def read_news(patterns: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Read every matched parquet once; return (combined frame, files, unmatched globs)."""
    frames: list[pd.DataFrame] = []
    found: list[str] = []
    missing: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            missing.append(pattern)
            continue
        for path in matches:
            try:
                frame = pd.read_parquet(path)
            except Exception as exc:  # the harvest may swap the file between glob and read
                LOG.warning("read_news: cannot read %s: %s", path, exc)
                missing.append(path)
                continue
            found.append(path)
            print(f"news: {path}: {len(frame)} rows")
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=list(_NEWS_COLUMNS)), found, missing
    return pd.concat(frames, ignore_index=True, sort=False), found, missing


def _horizon_stats(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per-horizon non-NaN excess count and mean excess in basis points."""
    stats: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS:
        excess = frame[f"excess_{horizon}m"].dropna()
        n = int(excess.size)
        mean_bps = float(excess.mean()) * 1e4 if n else None
        stats[f"{horizon}m"] = {"n": n, "mean_excess_bps": mean_bps}
    return stats


def _jsonable(value: Any) -> Any:
    """numpy scalars and NaN -> JSON-safe values (non-finite floats become null)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, bool | int | str) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the crypto news event dataset")
    parser.add_argument(
        "--news",
        action="append",
        default=None,
        metavar="GLOB",
        help="News parquet glob, repeatable (default: wayback.parquet + recorder.parquet)",
    )
    parser.add_argument("--bars", default=str(DEFAULT_BARS_DIR), help="1m parquet directory")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output parquet path")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="JSON report path")
    parser.add_argument("--since", default=None, help="First published_at (inclusive)")
    parser.add_argument("--until", default=None, help="Last published_at (inclusive)")
    args = parser.parse_args(argv)

    patterns = args.news or list(DEFAULT_NEWS)
    config = load_dataset_config()
    symbols = [str(s) for s in (config.get("symbols") or [])] or list(CANONICAL_ASSETS.values())
    print(
        f"config: labels.horizons={config['labels']['horizons']} "
        f"news_lag_minutes={config['labels']['news_lag_minutes']} "
        f"excess_vs={config['labels']['excess_vs']} "
        f"round_trip_perp_bps={config['costs']['round_trip_perp_bps']}"
    )

    news, found, unmatched = read_news(patterns)
    for pattern in unmatched:
        print(f"news: no file matched {pattern}")
    print(f"news: {len(found)} file(s), {len(news)} row(s) in memory")
    if len(news) == 0:
        print("news is absent or empty: building the empty schema with a zeroed funnel")

    bars = load_bars(args.bars, symbols, since=args.since, until=args.until)
    print(f"bars: {args.bars} ({len(symbols)} symbols declared, read lazily)")
    print(f"window: since={args.since} until={args.until}")

    frame, funnel = build_dataset(news, bars, config, since=args.since, until=args.until)
    checks = run_checks(frame, bars)

    print("funnel:")
    for key in FUNNEL_KEYS:
        print(f"  {key:<22} {funnel[key]}")

    print("checks:")
    failed: list[str] = []
    for key, value in checks.items():
        if isinstance(value, bool):
            status = "PASS" if value else "FAIL"
            if not value:
                failed.append(key)
            print(f"  {status} {key:<30} violations={checks.get(f'{key}_count', 0)}")

    if len(frame):
        counts = frame["split"].value_counts().to_dict()
        print("splits: " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    else:
        print("splits: (no rows)")
    stats = _horizon_stats(frame)
    for key, entry in stats.items():
        mean = entry["mean_excess_bps"]
        mean_text = "nan" if mean is None else f"{mean:.4f}"
        print(f"horizon {key:>4}: n={entry['n']} mean_excess_bps={mean_text}")

    out_path = Path(args.out)
    write_parquet_atomic(frame, out_path)
    print(f"dataset: {len(frame)} rows -> {out_path}")

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "news_files": found,
        "news_unmatched": unmatched,
        "news_rows": int(len(news)),
        "window": {"since": args.since, "until": args.until},
        "bars_dir": str(args.bars),
        "bars_loaded": sorted(getattr(bars, "loaded", set())),
        "funnel": funnel,
        "checks": checks,
        "splits": {str(k): int(v) for k, v in frame["split"].value_counts().items()}
        if len(frame)
        else {},
        "horizons": stats,
        "rows": int(len(frame)),
        "out": str(out_path),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(report), indent=2, ensure_ascii=False) + "\n"
    report_path.write_text(payload, encoding="utf-8")
    print(f"report: {report_path}")

    if failed:
        print(f"FAIL: {len(failed)} check(s) failed: {', '.join(failed)}")
        return 2
    print("OK: all leakage checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
