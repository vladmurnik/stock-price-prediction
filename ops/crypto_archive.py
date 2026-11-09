"""T2 — resumable GDELT DOC 2.0 crawler CLI for the historical crypto news archive.

Walks two-hour windows x assets, appends new rows to a parquet store and checkpoints a JSON
state file after every window, so the crawl can be killed and resumed without losing progress
or re-issuing completed requests. Rate limiting (HTTP 429) is the expected failure mode: the
run stops cleanly on either budget and reports what it actually got.

Usage:
    python ops/crypto_archive.py --assets bitcoin,ethereum --start 2026-09-01 --end 2026-09-08
    python ops/crypto_archive.py --start 2026-09-01 --end 2026-09-04 --max-minutes 4
    python ops/crypto_archive.py --start 2026-09-01 --end 2026-09-04 --max-requests 40

Exit codes: 0 when the run finished its budget or the crawl completed; 2 on a hard block
(zero rows kept while every request failed); 1 otherwise (bad arguments or an operational
error).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from spp.crypto.archive import crawl, load_archive, load_archive_config

DEFAULT_OUT = "data/crypto/news/archive_gdelt.parquet"
DEFAULT_STATE = "data/crypto/news/archive_state.json"


def _print_details(out_path: Path) -> None:
    """Per-asset counts and the per-day p50 of ``published_at`` when the store has rows.

    The archive carries no detection timestamps, so the per-day p50 is over ``published_at``
    itself (minute precision) — the distribution of when GDELT saw the day's articles.
    """
    if not out_path.is_file():
        print("(no rows yet: archive store does not exist)")
        return
    frame = load_archive(out_path)
    if frame.empty:
        print("(archive store is empty)")
        return
    counts = frame["asset"].value_counts().sort_index()
    print("rows per asset: " + ", ".join(f"{name}={count}" for name, count in counts.items()))
    print(f"{'day':<12} {'rows':>6}  p50_published_at")
    day = frame["published_at"].dt.date
    for day_value, group in frame.groupby(day, sort=True):
        print(f"{str(day_value):<12} {len(group):>6}  {group['published_at'].median()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets",
        default="bitcoin,ethereum",
        metavar="A,A",
        help="comma-separated asset keys from archive.query_terms (default: bitcoin,ethereum)",
    )
    parser.add_argument(
        "--start", required=True, metavar="YYYY-MM-DD",
        help="first window start, inclusive",
    )
    parser.add_argument(
        "--end", required=True, metavar="YYYY-MM-DD",
        help="crawl boundary, exclusive",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT, metavar="PATH",
        help=f"parquet archive (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--state", default=DEFAULT_STATE, metavar="PATH",
        help=f"JSON resume state (default: {DEFAULT_STATE})",
    )
    parser.add_argument(
        "--max-requests", type=int, default=None, metavar="N",
        help="stop after N window fetches",
    )
    parser.add_argument(
        "--max-minutes", type=float, default=None, metavar="M",
        help="stop after M minutes of wall time",
    )
    parser.add_argument(
        "--config", default=None, metavar="PATH",
        help="configs/crypto_dataset.yaml (default: from the package root)",
    )
    args = parser.parse_args(argv)

    try:
        config = load_archive_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: cannot load archive config: {exc}", file=sys.stderr)
        return 1

    assets = [name.strip() for name in args.assets.split(",") if name.strip()]
    if not assets:
        print("error: --assets needs at least one asset", file=sys.stderr)
        return 1
    unknown = [name for name in assets if name not in config.terms]
    if unknown:
        known = ", ".join(config.terms)
        print(f"error: unknown asset(s): {', '.join(unknown)}; known: {known}", file=sys.stderr)
        return 1

    try:
        start = pd.Timestamp(args.start)
        end = pd.Timestamp(args.end)
    except ValueError as exc:
        print(f"error: bad date: {exc}", file=sys.stderr)
        return 1
    if start >= end:
        print("error: --start must be before --end", file=sys.stderr)
        return 1

    try:
        summary = crawl(
            config,
            assets,
            start,
            end,
            Path(args.out),
            Path(args.state),
            max_requests=args.max_requests,
            max_minutes=args.max_minutes,
        )
    except (OSError, ValueError) as exc:
        print(f"error: crawl failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False))
    _print_details(Path(args.out))

    blocked = (
        summary["kept"] == 0
        and summary["requests"] > 0
        and summary["failed"] == summary["requests"]
    )
    if blocked:
        print("hard block: every request failed and no rows were kept", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
