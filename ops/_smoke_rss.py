"""Why do the RSS fixtures yield so few rows? Count the drop reasons."""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime, timedelta

from spp.config import load_universe
from spp.data.normalize import DropStats, alias_index_from_universe
from spp.data.news_live import parse_rss_payload

FIX = pathlib.Path("tests/fixtures/live")
index = alias_index_from_universe(load_universe())
now = datetime.now(UTC)
since = now - timedelta(days=30)

for name in ("rss_cnbc.xml", "rss_marketwatch.xml", "rss_yahoo_aapl.xml", "rss_investing.xml"):
    payload = (FIX / name).read_bytes()
    stats = DropStats()
    frame = parse_rss_payload(payload, source=name, index=index, since=since, now=now, stats=stats)
    print(f"--- {name}: kept {len(frame)}")
    print("   stats:", stats.as_dict())
    if len(frame):
        print("   dates:", frame["date"].min(), "..", frame["date"].max())

    # unfiltered pass: no `since` cutoff, to separate age from alias drops
    stats2 = DropStats()
    wide = parse_rss_payload(
        payload,
        source=name,
        index=index,
        since=datetime(2000, 1, 1, tzinfo=UTC),
        now=now,
        stats=stats2,
    )
    print("   no-since kept:", len(wide), "out_of_range:", stats2.out_of_range)
