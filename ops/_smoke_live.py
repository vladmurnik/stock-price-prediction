"""Smoke the live-feed parsers on real recorded payloads (no network)."""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime, timedelta

import pandas as pd

from spp.config import load_universe
from spp.data.normalize import SignalPolicy, alias_index_from_universe
from spp.data.news_live import dedupe_live, parse_gdelt_payload, parse_rss_payload

FIX = pathlib.Path("tests/fixtures/live")
RSS = {
    "cnbc.xml": "cnbc",
    "marketwatch.xml": "marketwatch",
    "yahoo_aapl.xml": "yahoo",
    "investing.xml": "investing",
}


def main() -> int:
    universe = load_universe()
    index = alias_index_from_universe(universe)
    now = datetime.now(UTC)
    since = now - timedelta(days=30)

    frames = []
    for name, source in RSS.items():
        payload = (FIX / f"rss_{name}").read_bytes() if name != "marketwatch.xml" else (FIX / f"rss_{name}").read_bytes()
        frame = parse_rss_payload(payload, source=source, index=index, since=since, now=now)
        print(f"rss {source:12s}: {len(frame):3d} rows")
        frames.append(frame)

    gdelt = parse_gdelt_payload(
        (FIX / "gdelt_artlist.json").read_bytes(), index=index, since=since.date()
    )
    print(f"gdelt        : {len(gdelt):3d} rows")
    frames.append(gdelt)

    combined = pd.concat(frames, ignore_index=True)
    print("columns:", list(combined.columns))
    print("before dedupe:", len(combined))
    if not combined.empty:
        print("by source:", combined["source"].value_counts().to_dict())
        print("tickers:", combined["ticker"].nunique(), sorted(combined["ticker"].unique())[:8])
    deduped, stats = dedupe_live(combined)
    print("after dedupe:", len(deduped), "stats:", stats)
    if not deduped.empty:
        bad = (pd.to_datetime(deduped["signal_date"]) < pd.to_datetime(deduped["date"])).sum()
        print("signal_date < date violations:", int(bad))
        print("policy:", SignalPolicy.from_config("16:00", strict_dates=False))
        print(deduped[["date", "signal_date", "ticker", "source", "title"]].head(5).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
