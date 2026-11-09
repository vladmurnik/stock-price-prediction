"""Wayback snapshot harvester: CDX parsing, capture fetch, resume, budgets and schema.

All tests are hermetic: no network (an injected ``fake_get`` stands in for ``requests.get``,
one CLI test monkeypatches ``requests.get`` itself), no real sleeping (``time.sleep`` is
patched to a collector wherever a retry can fire), and every store/state file lives under
``tmp_path``. Throttling is simulated with scripted 500/404 responses.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from spp.crypto import wayback
from spp.crypto.news import NewsConfig, load_news_config
from spp.crypto.wayback import (
    WAYBACK_COLUMNS,
    HarvestConfig,
    feed_map,
    fetch_capture,
    fetch_captures,
    harvest,
    load_wayback,
    parse_capture,
)

FEED_URL = "https://example.invalid/feed/rss"
DAY1_TS = "20250601090000"
DAY2_TS = "20250602090000"
DAY3_TS = "20250603090000"
SNAPSHOT_TS = "20250601120000"

CDX_HEADER = ["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"]


def build_rss(items: list[tuple[str, str, str, str | None]]) -> str:
    """Minimal RSS 2.0 body: ``(title, url, guid, pubDate-or-None)`` per item."""
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<rss version=\"2.0\"><channel>"]
    for title, url, guid, pub_date in items:
        parts.append("<item>")
        parts.append(f"<title>{title}</title>")
        parts.append(f"<link>{url}</link>")
        parts.append(f"<guid>{guid}</guid>")
        if pub_date:
            parts.append(f"<pubDate>{pub_date}</pubDate>")
        parts.append("</item>")
    parts.append("</channel></rss>")
    return "".join(parts)


ONE_ITEM_RSS = build_rss(
    [
        (
            "Bitcoin ETF approval clears regulator",
            "https://example.com/etf-approval",
            "guid-etf",
            "Sun, 01 Jun 2025 11:55:00 GMT",
        )
    ]
)

TWO_ITEM_RSS = build_rss(
    [
        (
            "Bitcoin ETF approval boosts markets",
            "https://example.com/etf",
            "guid-etf",
            "Sun, 01 Jun 2025 10:00:00 GMT",
        ),
        (
            "Ethereum network upgrade completes",
            "https://example.com/eth-upgrade",
            "guid-eth",
            "Sun, 01 Jun 2025 11:00:00 GMT",
        ),
    ]
)


class _Response:
    """Scripted HTTP response: ``json()`` raises when there is no payload, like requests does."""

    def __init__(
        self,
        status_code: int = 200,
        *,
        text: str = "",
        payload: Any = None,
        content: bytes | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.content = text.encode("utf-8") if content is None else content
        self._payload = payload

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def cdx_payload(*pairs: tuple[str, str]) -> list[list[str]]:
    """CDX ``output=json`` body: header row plus one row per ``(timestamp, original)``."""
    rows: list[list[str]] = [list(CDX_HEADER)]
    for timestamp, original in pairs:
        rows.append(["com,example)/feed", timestamp, original, "application/rss+xml",
                     "200", "ABCDEF", "1234"])
    return rows


def harvest_get(
    listing: list[list[str]], payloads: dict[str, str]
) -> tuple[Callable[..., Any], list[str]]:
    """Scripted ``get``: CDX listing first, then one RSS body per capture timestamp."""
    calls: list[str] = []

    def fake_get(
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> _Response:
        calls.append(url)
        if "cdx/search/cdx" in url:
            return _Response(payload=listing)
        timestamp = url.split("/web/")[1].split("id_/")[0]
        return _Response(text=payloads[timestamp])

    return fake_get, calls


def _load_cli() -> Any:
    """Import ``ops/crypto_wayback.py`` by path (ops is not a package)."""
    path = Path(__file__).resolve().parents[1] / "ops" / "crypto_wayback.py"
    spec = importlib.util.spec_from_file_location("crypto_wayback_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# CDX listing
# --------------------------------------------------------------------------- #
def test_fetch_captures_parses_header_and_rows() -> None:
    seen: dict[str, Any] = {}
    payload = cdx_payload((SNAPSHOT_TS, FEED_URL), (DAY2_TS, FEED_URL))

    def fake_get(
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> _Response:
        seen.update(url=url, params=params, headers=headers)
        return _Response(payload=payload)

    captures = fetch_captures(FEED_URL, "2025-06-01", "2025-06-10", get=fake_get)
    assert [capture["timestamp"] for capture in captures] == [SNAPSHOT_TS, DAY2_TS]
    assert all(capture["original"] == FEED_URL for capture in captures)
    assert seen["url"] == wayback.CDX_ENDPOINT
    assert seen["params"] == {
        "url": FEED_URL,
        "from": "20250601",
        "to": "20250610",
        "output": "json",
        "filter": "statuscode:200",
        "collapse": "timestamp:8",
        "limit": 500,
    }
    assert seen["headers"]["User-Agent"] == "spp-crypto-research/0.1 (+research)"


def test_fetch_captures_empty_header_only_and_error_page() -> None:
    bodies = [
        _Response(payload=[]),
        _Response(payload=[CDX_HEADER]),
        _Response(text="<html>temporarily blocked</html>"),
    ]
    calls = {"n": 0}

    def fake_get(
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> _Response:
        body = bodies[calls["n"]]
        calls["n"] += 1
        return body

    for _ in range(3):
        assert fetch_captures(FEED_URL, "2025-06-01", "2025-06-02", get=fake_get) == []
    assert calls["n"] == 3  # all three were HTTP 200, so none was retried


def test_fetch_captures_http_error_returns_empty() -> None:
    def fake_get(
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> _Response:
        return _Response(404, text="no captures")

    # A deterministic 404 is not retried and never raises out of the public API.
    assert fetch_captures(FEED_URL, "2025-06-01", "2025-06-02", get=fake_get) == []


def test_fetch_captures_yyyymm_bounds_expand_to_month() -> None:
    seen: dict[str, Any] = {}

    def fake_get(
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> _Response:
        seen.update(params or {})
        return _Response(payload=[])

    fetch_captures(FEED_URL, "202506", "202506", get=fake_get)
    assert seen["from"] == "20250601"
    assert seen["to"] == "20250630"


def test_fetch_captures_retries_500_then_200(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(wayback.time, "sleep", sleeps.append)
    calls = {"n": 0}

    def fake_get(
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> _Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return _Response(500, text="boom")
        return _Response(payload=cdx_payload((SNAPSHOT_TS, FEED_URL)))

    captures = fetch_captures(FEED_URL, "2025-06-01", "2025-06-01", get=fake_get)
    assert calls["n"] == 2
    assert sleeps == [wayback._RETRY_SLEEP_SECONDS]
    assert [capture["timestamp"] for capture in captures] == [SNAPSHOT_TS]


# --------------------------------------------------------------------------- #
# Capture fetch + parse
# --------------------------------------------------------------------------- #
def test_fetch_capture_builds_id_url_and_returns_body() -> None:
    seen: dict[str, Any] = {}
    body = "<?xml version='1.0'?><rss version='2.0'></rss>"

    def fake_get(
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> _Response:
        seen.update(url=url, headers=headers, params=params)
        return _Response(text=body)

    text = fetch_capture(SNAPSHOT_TS, FEED_URL, get=fake_get)
    assert seen["url"] == f"http://web.archive.org/web/{SNAPSHOT_TS}id_/{FEED_URL}"
    assert seen["headers"]["User-Agent"] == "spp-crypto-research/0.1 (+research)"
    assert seen["params"] is None
    assert text == body


def test_fetch_capture_raises_on_persistent_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wayback.time, "sleep", lambda _seconds: None)

    def fake_get(
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> _Response:
        return _Response(404, text="not found")

    with pytest.raises(RuntimeError, match="404"):
        fetch_capture(SNAPSHOT_TS, FEED_URL, get=fake_get)


def test_parse_capture_two_items_sets_source() -> None:
    rows = parse_capture(TWO_ITEM_RSS, "coindesk")
    assert len(rows) == 2
    assert [row["source"] for row in rows] == ["coindesk", "coindesk"]
    assert rows[0]["title"] == "Bitcoin ETF approval boosts markets"
    assert rows[1]["url"] == "https://example.com/eth-upgrade"
    # snapshot_ts is a placeholder here; harvest stamps the capture time.
    assert all("snapshot_ts" in row for row in rows)
    assert all(pd.isna(row["snapshot_ts"]) for row in rows)


def test_parse_capture_garbage_returns_empty() -> None:
    assert parse_capture("definitely not xml", "coindesk") == []
    assert parse_capture("<html><body>blocked</body></html>", "coindesk") == []
    assert parse_capture(None, "coindesk") == []


# --------------------------------------------------------------------------- #
# harvest: dedupe, filtering, resume, budgets
# --------------------------------------------------------------------------- #
def test_harvest_dedupes_url_across_snapshots(tmp_path: Path) -> None:
    out = tmp_path / "wb.parquet"
    state = tmp_path / "state.json"
    day1 = build_rss(
        [("First sighting", "https://example.com/a", "g-a", "Sun, 01 Jun 2025 09:00:00 GMT")]
    )
    day2 = build_rss(
        [
            ("Second sighting", "https://example.com/a", "g-a", "Mon, 02 Jun 2025 09:00:00 GMT"),
            ("Fresh headline", "https://example.com/b", "g-b", "Mon, 02 Jun 2025 10:00:00 GMT"),
        ]
    )
    listing = cdx_payload((DAY1_TS, FEED_URL), (DAY2_TS, FEED_URL))
    fake_get, calls = harvest_get(listing, {DAY1_TS: day1, DAY2_TS: day2})

    summary = harvest(
        {"coindesk": FEED_URL}, "2025-06-01", "2025-06-30", out, state,
        pause=0.0, get=fake_get,
    )
    assert summary["requests"] == 3  # 1 CDX listing + 2 capture fetches
    assert summary["captures"] == 2
    assert summary["kept"] == 3  # all three parsed rows survived the filters
    assert summary["new"] == 2  # url /a arrived twice; only the first snapshot was added
    assert summary["stopped"] == "done"
    assert len(calls) == 3

    stored = pd.read_parquet(out)
    assert stored["url"].tolist() == ["https://example.com/a", "https://example.com/b"]
    # First wins: the day-1 snapshot of url /a is the row that stays.
    assert stored["title"].tolist() == ["First sighting", "Fresh headline"]
    done = json.loads(state.read_text(encoding="utf-8"))["done"]
    assert done == ["coindesk|20250601", "coindesk|20250602"]


def test_harvest_drops_nat_rows(tmp_path: Path) -> None:
    out = tmp_path / "wb.parquet"
    state = tmp_path / "state.json"
    payload = build_rss(
        [
            ("Dated coin headline", "https://example.com/dated", "g-d",
             "Sun, 01 Jun 2025 09:00:00 GMT"),
            ("Undated coin headline", "https://example.com/undated", "g-u", None),
        ]
    )
    fake_get, _calls = harvest_get(cdx_payload((DAY1_TS, FEED_URL)), {DAY1_TS: payload})

    summary = harvest(
        {"coindesk": FEED_URL}, "2025-06-01", "2025-06-30", out, state,
        pause=0.0, get=fake_get,
    )
    assert summary["kept"] == 1
    assert summary["new"] == 1
    stored = pd.read_parquet(out)
    assert stored["url"].tolist() == ["https://example.com/dated"]
    assert not stored["published_at"].isna().any()


def test_harvest_resumes_and_skips_done_day(tmp_path: Path) -> None:
    out = tmp_path / "wb.parquet"
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({"done": ["coindesk|20250601"], "requests": 1, "failed": 0}),
        encoding="utf-8",
    )
    day2 = build_rss(
        [("Day two headline", "https://example.com/d2", "g-d2",
          "Mon, 02 Jun 2025 09:00:00 GMT")]
    )
    # Day 1 has no payload on purpose: fetching it would raise a KeyError.
    listing = cdx_payload((DAY1_TS, FEED_URL), (DAY2_TS, FEED_URL))
    fake_get, calls = harvest_get(listing, {DAY2_TS: day2})

    summary = harvest(
        {"coindesk": FEED_URL}, "2025-06-01", "2025-06-30", out, state,
        pause=0.0, get=fake_get,
    )
    capture_calls = [url for url in calls if "/web/" in url]
    assert len(capture_calls) == 1
    assert DAY2_TS in capture_calls[0]
    assert summary["requests"] == 2  # 1 CDX + 1 capture; day 1 was skipped for free
    assert summary["failed"] == 0
    assert summary["kept"] == 1
    loaded = json.loads(state.read_text(encoding="utf-8"))
    assert loaded["done"] == ["coindesk|20250601", "coindesk|20250602"]
    assert loaded["requests"] == 3  # cumulative: prior run 1 + this run 2


def test_harvest_budget_stops_at_max_requests(tmp_path: Path) -> None:
    out = tmp_path / "wb.parquet"
    state = tmp_path / "state.json"
    payloads = {
        day: build_rss(
            [(f"Headline {day}", f"https://example.com/{day}", f"g-{day}",
              "Sun, 01 Jun 2025 09:00:00 GMT")]
        )
        for day in (DAY1_TS, DAY2_TS, DAY3_TS)
    }
    listing = cdx_payload((DAY1_TS, FEED_URL), (DAY2_TS, FEED_URL), (DAY3_TS, FEED_URL))
    fake_get, calls = harvest_get(listing, payloads)

    summary = harvest(
        {"coindesk": FEED_URL}, "2025-06-01", "2025-06-30", out, state,
        max_requests=3, pause=0.0, get=fake_get,
    )
    assert len(calls) == 3  # 1 CDX + 2 captures; the third capture never started
    assert summary["requests"] == 3
    assert summary["failed"] == 0
    assert summary["captures"] == 2
    assert summary["stopped"] == "budget"
    assert summary["kept"] == 2
    stored = pd.read_parquet(out)
    assert len(stored) == 2
    loaded = json.loads(state.read_text(encoding="utf-8"))
    # Day 3 is not done, so the next run resumes exactly there.
    assert loaded["done"] == ["coindesk|20250601", "coindesk|20250602"]
    assert loaded["requests"] == 3


def test_harvest_all_failed_keeps_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wayback.time, "sleep", lambda _seconds: None)
    out = tmp_path / "wb.parquet"
    state = tmp_path / "state.json"

    def fake_get(
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> _Response:
        return _Response(500, text="throttled")

    summary = harvest(
        {"coindesk": FEED_URL}, "2025-06-01", "2025-06-30", out, state,
        pause=0.0, get=fake_get,
    )
    assert summary["requests"] == 2  # the CDX listing was retried once
    assert summary["failed"] == 2
    assert summary["failed"] == summary["requests"]  # the CLI's hard-block condition
    assert summary["kept"] == 0
    assert summary["captures"] == 0
    assert not out.exists()


def test_harvest_column_order(tmp_path: Path) -> None:
    out = tmp_path / "wb.parquet"
    state = tmp_path / "state.json"
    listing = cdx_payload((SNAPSHOT_TS, FEED_URL))
    fake_get, _calls = harvest_get(listing, {SNAPSHOT_TS: ONE_ITEM_RSS})

    harvest({"coindesk": FEED_URL}, "2025-06-01", "2025-06-30", out, state,
            pause=0.0, get=fake_get)
    assert list(WAYBACK_COLUMNS) == [
        "source", "title", "url", "guid", "published_at", "snapshot_ts", "tickers",
    ]
    assert list(pd.read_parquet(out).columns) == list(WAYBACK_COLUMNS)
    assert list(load_wayback(out).columns) == list(WAYBACK_COLUMNS)


def test_harvest_stamps_snapshot_ts_and_attributes_tickers(tmp_path: Path) -> None:
    out = tmp_path / "wb.parquet"
    state = tmp_path / "state.json"
    listing = cdx_payload((SNAPSHOT_TS, FEED_URL))
    fake_get, _calls = harvest_get(listing, {SNAPSHOT_TS: ONE_ITEM_RSS})

    summary = harvest(
        {"coindesk": FEED_URL}, "2025-06-01", "2025-06-30", out, state,
        pause=0.0, get=fake_get,
    )
    assert summary["kept"] == 1
    stored = pd.read_parquet(out)
    row = stored.iloc[0]
    assert row["source"] == "coindesk"
    assert row["snapshot_ts"] == pd.Timestamp("2025-06-01 12:00:00", tz="UTC")
    assert stored["snapshot_ts"].dt.tz is not None
    assert row["published_at"] == pd.Timestamp("2025-06-01 11:55:00", tz="UTC")
    assert row["tickers"] == ["BTCUSDT"]  # "Bitcoin ETF approval" -> bitcoin


# --------------------------------------------------------------------------- #
# Config, schema helpers, CLI
# --------------------------------------------------------------------------- #
def test_feed_map_supports_dict_and_list_shapes() -> None:
    config = load_news_config()
    feeds = feed_map(config)
    assert feeds == config.feeds
    assert feeds["coindesk"] == "https://www.coindesk.com/arc/outboundfeeds/rss/"

    listed = NewsConfig(
        feeds=[
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://decrypt.co/feed",
        ],
        poll_seconds=30.0,
        request_timeout_seconds=20.0,
        user_agent="spp-crypto-test/0.1",
        dedupe_minutes=60.0,
        aliases={},
        weak_aliases={},
        crypto_context=[],
    )
    assert feed_map(listed) == {
        "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "decrypt": "https://decrypt.co/feed",
    }


def test_harvest_config_is_frozen() -> None:
    config = HarvestConfig()
    assert config.max_requests == 1000
    assert config.max_minutes == 30.0
    assert config.pause_seconds == 0.6
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_requests = 5  # type: ignore[misc]


def test_load_wayback_missing_file_returns_empty(tmp_path: Path) -> None:
    frame = load_wayback(tmp_path / "absent.parquet")
    assert frame.empty
    assert list(frame.columns) == list(WAYBACK_COLUMNS)


def test_cli_list_feeds(capsys: pytest.CaptureFixture[str]) -> None:
    cli = _load_cli()
    code = cli.main(["--list-feeds"])
    out = capsys.readouterr().out
    assert code == 0
    assert "coindesk https://www.coindesk.com/arc/outboundfeeds/rss/" in out


def test_cli_all_failed_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli()
    monkeypatch.setattr(wayback.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        requests, "get", lambda _url, **_kwargs: _Response(500, text="throttled")
    )

    code = cli.main(
        [
            "--feeds", "coindesk",
            "--start", "2025-06-01",
            "--end", "2025-06-02",
            "--out", str(tmp_path / "cli.parquet"),
            "--state", str(tmp_path / "cli_state.json"),
            "--pause", "0",
        ]
    )
    out = capsys.readouterr().out
    summary = json.loads(out.strip().splitlines()[-1])  # the summary JSON is the last line
    assert summary["kept"] == 0
    assert summary["requests"] > 0
    assert summary["failed"] == summary["requests"]
    assert code == 2
