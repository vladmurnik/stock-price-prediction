"""T0 — keyless RSS recorder CLI for the crypto news archive.

Polls the feeds from ``configs/crypto_dataset.yaml:news`` and appends every unseen item to a
parquet store (append-only, ``first_seen_at`` stamped per poll), so a later historical archive
can be compared against it and publication-to-detection latency measured.

Usage:
    python ops/crypto_news.py --once [--feeds coindesk,bitcoinist] [--store PATH]
    python ops/crypto_news.py --duration 240 [--config PATH]

Exit codes: 0 on success, 2 when zero items were fetched from every feed (or on bad arguments
/ a missing config).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

from spp.crypto.news import RECORD_COLUMNS, Recorder, load_news_config

DEFAULT_STORE = "data/crypto/news/recorder.parquet"


def _print_table(frame: pd.DataFrame) -> None:
    """Compact per-source table: source, count, min latency of the new rows."""
    print(f"{'source':<16} {'count':>5} {'min_latency_s':>13}")
    if frame.empty:
        print("(no new items)")
        return
    for source, group in frame.groupby("source", sort=True):
        latencies = group["latency_seconds"].dropna()
        minimum = f"{latencies.min():.1f}" if len(latencies) else "-"
        print(f"{str(source):<16} {len(group):>5} {minimum:>13}")


def _print_summary(store: Path) -> None:
    """Final line: total stored rows and p50/p90 latency over the whole store."""
    if store.is_file():
        frame = pd.read_parquet(store)
    else:
        frame = pd.DataFrame(columns=RECORD_COLUMNS)
    latencies = (
        frame["latency_seconds"].dropna()
        if "latency_seconds" in frame.columns
        else pd.Series(dtype="float64")
    )
    if len(latencies):
        p50 = f"{latencies.quantile(0.5):.1f} s"
        p90 = f"{latencies.quantile(0.9):.1f} s"
    else:
        p50 = p90 = "n/a"
    print(f"stored rows: {len(frame)} | latency p50 {p50} | p90 {p90}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="one poll, print a table, exit")
    mode.add_argument(
        "--duration",
        type=float,
        metavar="MINUTES",
        help="keep polling for MINUTES minutes",
    )
    parser.add_argument(
        "--feeds",
        default=None,
        metavar="NAME,NAME",
        help="subset of the configured feeds",
    )
    parser.add_argument(
        "--store",
        default=DEFAULT_STORE,
        metavar="PATH",
        help=f"parquet store (default: {DEFAULT_STORE})",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="configs/crypto_dataset.yaml (default: from the package root)",
    )
    args = parser.parse_args(argv)

    try:
        config = load_news_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: cannot load news config: {exc}", file=sys.stderr)
        return 2

    if args.feeds:
        names = [name.strip() for name in args.feeds.split(",") if name.strip()]
        if not names:
            parser.error("--feeds needs at least one feed name")
        unknown = [name for name in names if name not in config.feeds]
        if unknown:
            parser.error(
                f"unknown feed(s): {', '.join(unknown)}; known: {', '.join(config.feeds)}"
            )
        config = replace(config, feeds={name: config.feeds[name] for name in names})

    recorder = Recorder(config, Path(args.store))
    if args.once:
        _print_table(recorder.poll_once())
    else:
        stored = recorder.run_recorder(args.duration * 60.0)
        print(f"run complete: {stored} new items over {args.duration} min")
    _print_summary(Path(args.store))

    if recorder.fetched_total == 0:
        print("warning: zero items fetched from every feed")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
