"""Fetch historical GDELT news for the intraday window.

Writes ``data/interim/news_gdelt.parquet``. Usage:
    python ops/fetch_gdelt.py --start 2024-11-11 --end 2026-11-07 --tickers AAPL,MSFT,JPM
    python ops/fetch_gdelt.py --start 2026-09-01 --end 2026-09-15 --all

The runner prints the funnel with every rejection reason. A run that keeps 3 000 of 90 000 articles
is a different experiment from one that keeps 40 000, so the numbers are printed before anything is
used downstream.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from spp.data.news_gdelt_hist import build_corpus, default_terms, save_corpus
from spp.data.normalize import load_alias_index
from spp.logging import get_logger

ROOT = Path(__file__).resolve().parents[1]
LOG = get_logger("ops.gdelt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="First publication date, YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="Last publication date, YYYY-MM-DD.")
    parser.add_argument("--tickers", default="", help="Comma-separated symbols.")
    parser.add_argument("--all", action="store_true", help="Every ticker in the universe file.")
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--pause", type=float, default=5.0, help="Seconds between API calls.")
    parser.add_argument("--out", default="data/interim/news_gdelt.parquet")
    args = parser.parse_args()

    index = load_alias_index()
    if args.all:
        tickers = list(index.tickers())
    else:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise SystemExit("no tickers: pass --tickers or --all")

    terms = default_terms(tickers, index=index)
    LOG.info("crawling %d tickers %s..%s", len(terms), args.start, args.end)
    frame, funnel = build_corpus(
        terms,
        datetime.fromisoformat(args.start),
        datetime.fromisoformat(args.end),
        index=index,
        chunk_days=args.chunk_days,
        pause=args.pause,
    )
    print(json.dumps(funnel.as_dict(), indent=2, ensure_ascii=False))
    if frame.empty:
        print("no articles: nothing written")
        return 2
    path = save_corpus(frame, ROOT / args.out)
    print(f"rows={len(frame)} tickers={frame['ticker'].nunique()} -> {path}")
    print(
        "window:",
        frame["published_at"].min().isoformat(),
        "..",
        frame["published_at"].max().isoformat(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
