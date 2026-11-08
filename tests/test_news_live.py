"""Per-ticker RSS feeds: an unlinked headline still belongs to the feed's ticker."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from spp.config import load_universe
from spp.data.news_live import parse_rss_payload
from spp.data.normalize import DropStats, alias_index_from_universe

NOW = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
SINCE = datetime(2026, 10, 1, tzinfo=UTC)

#: One headline that never names a company, one that names a different one.
PAYLOAD = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Example</title>
<item>
  <title>A new factory opens in Ohio next spring</title>
  <link>https://example.com/a</link>
  <pubDate>Fri, 06 Nov 2026 14:00:00 GMT</pubDate>
</item>
<item>
  <title>Microsoft beats quarterly revenue estimates</title>
  <link>https://example.com/b</link>
  <pubDate>Fri, 06 Nov 2026 14:05:00 GMT</pubDate>
</item>
</channel></rss>
"""


@pytest.fixture(scope="module")
def index():
    return alias_index_from_universe(load_universe())


def _parse(index, **kwargs):
    stats = DropStats()
    frame = parse_rss_payload(
        PAYLOAD,
        source="rss_example.xml",
        index=index,
        since=SINCE,
        now=NOW,
        stats=stats,
        **kwargs,
    )
    return frame, stats


def test_generic_feed_drops_headline_without_alias(index) -> None:
    frame, stats = _parse(index)
    assert list(frame["ticker"]) == ["MSFT"]
    assert stats.no_alias == 1


def test_hint_ticker_keeps_unlinked_headline(index) -> None:
    frame, stats = _parse(index, hint_ticker="AAPL")
    assert sorted(frame["ticker"].tolist()) == ["AAPL", "MSFT"]
    # The linked alias wins: the headline about Microsoft is not relabelled as Apple.
    assert stats.alias_mismatch == 1
    assert stats.no_alias == 0


def test_hint_ticker_does_not_override_a_linked_alias(index) -> None:
    frame, _ = _parse(index, hint_ticker="AAPL")
    by_url = dict(zip(frame["url"], frame["ticker"], strict=True))
    assert by_url["https://example.com/b"] == "MSFT"
    assert by_url["https://example.com/a"] == "AAPL"
