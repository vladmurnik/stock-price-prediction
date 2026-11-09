"""Resumable Wayback Machine RSS snapshot harvester CLI (historical crypto headlines).

Rebuilds the 2023-01-01..2026-11-09 headline corpus from Internet Archive snapshots of the
configured RSS feeds — live feeds only expose the last few hours and GDELT answers HTTP 429.
One CDX listing per feed-month plus one capture fetch per archived day; completed days are
checkpointed to the state file, so a killed run resumes without refetching anything.

Usage:
    python ops/crypto_wayback.py --list-feeds
    python ops/crypto_wayback.py --feeds coindesk --start 2025-06-01 --end 2025-06-10
    python ops/crypto_wayback.py --start 2023-01-01 --end 2026-11-09 --max-requests 400

The harvest summary is printed as JSON on one line at the end. Exit codes: 0 when rows
were kept or the requested range completed; 2 when every request failed and nothing was
kept; 1 otherwise (bad arguments or an operational error).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spp.crypto.wayback import HarvestConfig, feed_map, harvest

DEFAULT_OUT = "data/crypto/news/wayback.parquet"
DEFAULT_STATE = "data/crypto/news/wayback_state.json"


def main(argv: list[str] | None = None) -> int:
    defaults = HarvestConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feeds",
        default=None,
        metavar="N,N",
        help="comma-separated feed names from the news config (default: all)",
    )
    parser.add_argument(
        "--start",
        default=None,
        metavar="YYYY-MM-DD",
        help="first capture day, inclusive (YYYY-MM-DD or YYYYMM)",
    )
    parser.add_argument(
        "--end",
        default=None,
        metavar="YYYY-MM-DD",
        help="last capture day, inclusive (YYYY-MM-DD or YYYYMM)",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=defaults.max_requests,
        metavar="N",
        help=f"stop after N HTTP requests (default: {defaults.max_requests})",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=defaults.max_minutes,
        metavar="M",
        help=f"stop after M minutes of wall time (default: {defaults.max_minutes})",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=defaults.pause_seconds,
        metavar="S",
        help=f"seconds to sleep between requests (default: {defaults.pause_seconds})",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        metavar="PATH",
        help=f"parquet corpus (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--state",
        default=DEFAULT_STATE,
        metavar="PATH",
        help=f"JSON resume state (default: {DEFAULT_STATE})",
    )
    parser.add_argument(
        "--list-feeds",
        action="store_true",
        help="print the configured feed names and exit",
    )
    args = parser.parse_args(argv)

    try:
        feeds = feed_map()
    except (OSError, ValueError) as exc:
        print(f"error: cannot load news config: {exc}", file=sys.stderr)
        return 1

    if args.list_feeds:
        for name, url in feeds.items():
            print(f"{name} {url}")
        return 0

    if not args.start or not args.end:
        print("error: --start and --end are required", file=sys.stderr)
        return 1

    if args.feeds is None:
        selected = feeds
    else:
        wanted = [name.strip() for name in args.feeds.split(",") if name.strip()]
        if not wanted:
            print("error: --feeds needs at least one name", file=sys.stderr)
            return 1
        unknown = [name for name in wanted if name not in feeds]
        if unknown:
            known = ", ".join(feeds)
            print(f"error: unknown feed(s): {', '.join(unknown)}; known: {known}", file=sys.stderr)
            return 1
        selected = {name: feeds[name] for name in wanted}
    if not selected:
        print("error: no feeds configured", file=sys.stderr)
        return 1

    try:
        summary = harvest(
            selected,
            args.start,
            args.end,
            Path(args.out),
            Path(args.state),
            max_requests=args.max_requests,
            max_minutes=args.max_minutes,
            pause=args.pause,
        )
    except (OSError, ValueError) as exc:
        print(f"error: harvest failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False))

    blocked = (
        summary["kept"] == 0
        and summary["requests"] > 0
        and summary["failed"] == summary["requests"]
    )
    if blocked:
        print("hard block: every request failed and no rows were kept", file=sys.stderr)
        return 2
    if summary["kept"] > 0 or summary["stopped"] == "done":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
