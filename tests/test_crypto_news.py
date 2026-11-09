"""T0 + T3 — crypto RSS recorder: parsing, dedupe, latency clamping and attribution.

All tests are hermetic: no network (``fetch_feed`` is monkeypatched or made to fail),
no real sleeps (retry sleep patched to 0, recorder clock/sleep injected), and every
store lives under ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import requests

from spp.crypto import news
from spp.crypto.news import (
    RECORD_COLUMNS,
    NewsConfig,
    Recorder,
    fetch_feed,
    link_tickers,
    load_alias_index,
    load_news_config,
    parse_feed,
)

CLOCK_AT = pd.Timestamp("2026-11-09 12:00:00", tz="UTC")

RSS_PAYLOAD = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
<channel><title>Example</title>
<item>
  <title>Bitcoin  &lt;b&gt;rises&lt;/b&gt;  again</title>
  <link>https://example.com/btc-rises</link>
  <guid isPermaLink="false">guid-btc-1</guid>
  <dc:date>2026-11-09T11:55:00+00:00</dc:date>
  <description>Markets &lt;p&gt;move&lt;/p&gt; higher</description>
</item>
<item>
  <title>Second headline about ethereum network</title>
  <link>https://example.com/eth</link>
  <guid>guid-eth-1</guid>
  <pubDate>Mon, 09 Nov 2026 11:58:00 GMT</pubDate>
</item>
</channel></rss>
"""

ATOM_PAYLOAD = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Example Atom</title>
  <entry>
    <title>Solana upgrade completes &amp; network recovers</title>
    <link href="https://example.com/sol-upgrade" rel="alternate"/>
    <id>urn:uuid:sol-1</id>
    <updated>2026-11-09T09:30:00Z</updated>
    <summary>All &lt;em&gt;validators&lt;/em&gt; agree</summary>
  </entry>
</feed>
"""

NO_DATE_PAYLOAD = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item>
  <title>Undated headline about ethereum markets</title>
  <link>https://example.com/undated</link>
  <guid>guid-undated-1</guid>
</item>
</channel></rss>
"""

MIXED_LATENCY_PAYLOAD = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item>
  <title>Bitcoin breaks above a new record high today</title>
  <link>https://example.com/future</link>
  <guid>guid-future</guid>
  <pubDate>Mon, 09 Nov 2026 12:05:00 GMT</pubDate>
</item>
<item>
  <title>Ethereum staking yields compress further now</title>
  <link>https://example.com/past</link>
  <guid>guid-past</guid>
  <pubDate>Mon, 09 Nov 2026 11:50:00 GMT</pubDate>
</item>
</channel></rss>
"""

SECOND_PAYLOAD = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item>
  <title>Chainlink partners with a new exchange venue</title>
  <link>https://example.com/chain</link>
  <guid>guid-chain-1</guid>
  <pubDate>Mon, 09 Nov 2026 11:59:00 GMT</pubDate>
</item>
</channel></rss>
"""


def make_config(**overrides: object) -> NewsConfig:
    """Small deterministic config; tests override individual sections."""
    base: dict[str, object] = {
        "feeds": {"example": "https://example.invalid/rss"},
        "poll_seconds": 30.0,
        "request_timeout_seconds": 1.0,
        "user_agent": "spp-crypto-test/0.1",
        "dedupe_minutes": 60.0,
        "aliases": {
            "BTCUSDT": ["bitcoin", "btc"],
            "ETHUSDT": ["ethereum"],
            "LINKUSDT": ["chainlink"],
        },
        "weak_aliases": {"SOLUSDT": ["sol"]},
        "crypto_context": ["crypto", "coin", "token"],
    }
    base.update(overrides)
    return NewsConfig(**base)  # type: ignore[arg-type]


def make_recorder(tmp_path: Path, config: NewsConfig | None = None) -> Recorder:
    return Recorder(
        config or make_config(),
        tmp_path / "nested" / "recorder.parquet",
        clock=lambda: CLOCK_AT,
    )


def stub_fetch(monkeypatch: pytest.MonkeyPatch, *payloads: str) -> None:
    """Replace ``news.fetch_feed`` with a scripted sequence of parse results."""
    calls = {"i": 0}

    def fake_fetch(url: str, config: NewsConfig) -> list[dict[str, object]]:
        index = min(calls["i"], len(payloads) - 1)
        calls["i"] += 1
        return parse_feed(payloads[index], "stub")

    monkeypatch.setattr(news, "fetch_feed", fake_fetch)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_parse_feed_rss_with_dc_date() -> None:
    rows = parse_feed(RSS_PAYLOAD, "coindesk")
    assert len(rows) == 2
    first = rows[0]
    assert first["source"] == "coindesk"
    # HTML tags stripped, whitespace collapsed.
    assert first["title"] == "Bitcoin rises again"
    assert first["url"] == "https://example.com/btc-rises"
    assert first["guid"] == "guid-btc-1"
    published = first["published_at"]
    assert isinstance(published, pd.Timestamp)
    assert published.tz is not None
    assert published == pd.Timestamp("2026-11-09 11:55:00", tz="UTC")
    assert first["summary"] == "Markets move higher"
    # pubDate (RFC 822) is handled alongside dc:date.
    assert rows[1]["published_at"] == pd.Timestamp("2026-11-09 11:58:00", tz="UTC")


def test_parse_feed_atom_entry() -> None:
    rows = parse_feed(ATOM_PAYLOAD, "atom_source")
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "atom_source"
    assert row["title"] == "Solana upgrade completes & network recovers"
    assert row["url"] == "https://example.com/sol-upgrade"
    assert row["guid"] == "urn:uuid:sol-1"
    assert row["published_at"] == pd.Timestamp("2026-11-09 09:30:00", tz="UTC")
    assert row["summary"] == "All validators agree"


def test_parse_feed_missing_date_keeps_item_with_nat() -> None:
    rows = parse_feed(NO_DATE_PAYLOAD, "no_date")
    assert len(rows) == 1
    assert rows[0]["title"] == "Undated headline about ethereum markets"
    assert pd.isna(rows[0]["published_at"])
    assert rows[0]["guid"] == "guid-undated-1"


def test_parse_feed_unparseable_payload_returns_empty() -> None:
    assert parse_feed("<rss><channel><item>", "broken") == []


# --------------------------------------------------------------------------- #
# Recorder: dedupe, latency, schema
# --------------------------------------------------------------------------- #
def test_poll_once_dedupes_across_polls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = make_recorder(tmp_path)
    stub_fetch(monkeypatch, RSS_PAYLOAD)
    first = recorder.poll_once()
    second = recorder.poll_once()
    assert len(first) == 2
    assert second.empty
    stored = pd.read_parquet(tmp_path / "nested" / "recorder.parquet")
    assert len(stored) == 2
    # Both polls saw the same two guids; the second added nothing.
    assert recorder.fetched_total == 4


def test_missing_date_item_survives_freshness_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = make_recorder(tmp_path)
    stub_fetch(monkeypatch, NO_DATE_PAYLOAD)
    frame = recorder.poll_once()
    assert len(frame) == 1
    assert pd.isna(frame["published_at"].iloc[0])
    assert frame["latency_seconds"].iloc[0] != frame["latency_seconds"].iloc[0]  # NaN
    assert not bool(frame["negative_latency"].iloc[0])


def test_latency_clamped_at_zero_with_negative_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = make_recorder(tmp_path)
    stub_fetch(monkeypatch, MIXED_LATENCY_PAYLOAD)
    frame = recorder.poll_once()
    assert len(frame) == 2
    by_guid = {row["guid"]: row for row in frame.to_dict(orient="records")}
    # Published in the future relative to the clock: clamped to 0, flagged.
    assert by_guid["guid-future"]["latency_seconds"] == 0.0
    assert bool(by_guid["guid-future"]["negative_latency"]) is True
    # Ten minutes before first_seen: exact latency, no flag.
    assert by_guid["guid-past"]["latency_seconds"] == 600.0
    assert bool(by_guid["guid-past"]["negative_latency"]) is False


def test_poll_once_exact_column_order_and_appends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "deep" / "nested" / "recorder.parquet"
    recorder = Recorder(make_config(), store, clock=lambda: CLOCK_AT)
    stub_fetch(monkeypatch, RSS_PAYLOAD, SECOND_PAYLOAD)

    new_first = recorder.poll_once()
    assert list(new_first.columns) == list(RECORD_COLUMNS)
    assert store.is_file()  # parent dirs created, file created

    new_second = recorder.poll_once()
    assert list(new_second.columns) == list(RECORD_COLUMNS)
    assert len(new_second) == 1

    stored = pd.read_parquet(store)
    assert list(stored.columns) == list(RECORD_COLUMNS)
    assert len(stored) == 3  # prior rows kept, only the new one appended
    assert stored["title"].iloc[0] == "Bitcoin rises again"
    assert stored["guid"].tolist() == ["guid-btc-1", "guid-eth-1", "guid-chain-1"]
    assert stored["event_kind"].tolist() == ["", "", ""]
    assert stored["tickers"].iloc[0] == ["BTCUSDT"]
    assert stored["tickers"].iloc[2] == ["LINKUSDT"]


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #
def test_strong_alias_word_boundary() -> None:
    config = make_config(aliases={"SOLUSDT": ["sol"]}, weak_aliases={})
    index = load_alias_index(config)
    assert link_tickers("A solution for markets", index, config) == []
    assert link_tickers("Sol reaches a new high", index, config) == ["SOLUSDT"]
    assert link_tickers("SOL is up today", index, config) == ["SOLUSDT"]


def test_weak_alias_requires_crypto_context() -> None:
    config = make_config(
        aliases={"BTCUSDT": ["bitcoin"]},
        weak_aliases={"SOLUSDT": ["sol"], "LINKUSDT": ["link"]},
        crypto_context=["crypto", "coin"],
    )
    index = load_alias_index(config)
    # Index keeps config order: strong aliases first, then weak ones.
    assert list(index) == ["bitcoin", "sol", "link"]
    # Weak alias without a context word in the same text: rejected.
    assert link_tickers("Sol jumps 10 percent today", index, config) == []
    assert link_tickers("This link goes nowhere", index, config) == []
    # Same text plus a crypto context word: accepted.
    assert link_tickers("Crypto: sol jumps 10 percent", index, config) == ["SOLUSDT"]
    # Strong aliases never need context.
    assert link_tickers("Bitcoin rises again", index, config) == ["BTCUSDT"]


# --------------------------------------------------------------------------- #
# Fetching (network failure) and config loading
# --------------------------------------------------------------------------- #
def test_fetch_feed_returns_empty_on_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        raise requests.ConnectionError("network is down")

    monkeypatch.setattr(news.requests, "get", boom)
    monkeypatch.setattr(news, "_RETRY_SLEEP_SECONDS", 0.0)
    assert fetch_feed("https://example.invalid/rss", make_config()) == []


def test_load_news_config_from_package_root(monkeypatch: pytest.MonkeyPatch) -> None:
    # Resolved from the package root, not the cwd.
    monkeypatch.chdir(Path.home())
    config = load_news_config()
    assert {"coindesk", "bitcoinist", "theblock"} <= set(config.feeds)
    assert config.poll_seconds == 30.0
    assert config.request_timeout_seconds == 20.0
    assert config.dedupe_minutes == 60.0
    assert config.aliases["BTCUSDT"] == ["bitcoin", "btc", "xbt"]
    assert config.weak_aliases["SOLUSDT"] == ["sol"]
    assert "crypto" in config.crypto_context


# --------------------------------------------------------------------------- #
# run_recorder: injected clock and sleep
# --------------------------------------------------------------------------- #
class FakeClock:
    """Deterministic clock advanced only by the injected sleep."""

    def __init__(self) -> None:
        self.now = CLOCK_AT
        self.sleeps: list[float] = []

    def __call__(self) -> pd.Timestamp:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now = self.now + pd.Timedelta(seconds=seconds)


def test_run_recorder_uses_injected_clock_and_sleep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    seen: list[str] = []

    def fake_fetch(url: str, config: NewsConfig) -> list[dict[str, object]]:
        guid = f"guid-{len(seen)}"
        seen.append(guid)
        return [
            {
                "source": "example",
                "title": f"Bitcoin headline number {len(seen)} about markets",
                "url": f"https://example.com/{guid}",
                "guid": guid,
                "published_at": pd.Timestamp("2026-11-09 11:59:00", tz="UTC"),
                "summary": "",
            }
        ]

    monkeypatch.setattr(news, "fetch_feed", fake_fetch)
    recorder = Recorder(make_config(), tmp_path / "run.parquet", clock=clock)
    # 90 s duration, 30 s poll: polls at T+0, T+30, T+60, T+90, sleeps only between them.
    total = recorder.run_recorder(90.0, sleep=clock.sleep)
    assert total == 4
    assert len(seen) == 4
    assert clock.sleeps == [30.0, 30.0, 30.0]
    stored = pd.read_parquet(tmp_path / "run.parquet")
    assert len(stored) == 4
