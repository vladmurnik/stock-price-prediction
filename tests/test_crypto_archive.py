"""T2 — GDELT archive crawler contracts: windows, parsing, retries, resume, budgets.

All tests are hermetic: no network (an injected ``fake_get`` stands in for ``requests.get``),
no real sleeping (``time.sleep`` is monkeypatched to a collector), and every store/state file
lives under ``tmp_path``. Rate limiting is simulated with scripted 429 responses.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from pathlib import Path

import pandas as pd
import pytest

from spp.crypto import archive
from spp.crypto.archive import (
    ARCHIVE_COLUMNS,
    ArchiveConfig,
    crawl,
    fetch_window,
    iter_windows,
    load_archive,
    load_archive_config,
    parse_articles,
)

WIN_START = pd.Timestamp("2026-09-01 00:00", tz="UTC")
WIN_END = pd.Timestamp("2026-09-01 02:00", tz="UTC")
#: Two full windows (00-02, 02-04) — the range tests that need more than one window use this.
END_2 = pd.Timestamp("2026-09-01 04:00", tz="UTC")

GDELT_PAYLOAD: dict = {
    "articles": [
        {
            "url": "https://www.coindesk.com/markets/2024/05/01/bitcoin-holds-57k/",
            "title": "Bitcoin Holds $57K as ETF Inflows Accelerate",
            "seendate": "20240501T123000Z",
            "domain": "coindesk.com",
            "language": "English",
            "sourcecountry": "United States",
            "socialimage": "https://www.coindesk.com/wp-content/uploads/2024/05/btc.jpg",
        },
        {
            "url": "https://www.theblock.co/post/289100/ethereum-rolls-ahead",
            "title": "Ethereum rolls ahead of the market",
            "seendate": "20240501T123400Z",
            "domain": "theblock.co",
            "language": "English",
            "sourcecountry": "United States",
            "socialimage": "",
        },
    ]
}


def make_config(**overrides: object) -> ArchiveConfig:
    """Small deterministic config; tests override individual sections."""
    base: dict[str, object] = {
        "endpoint": "https://gdelt.invalid/api/v2/doc/doc",
        "window_hours": 2,
        "max_records": 250,
        "max_retries": 5,
        "backoff_seconds": (15, 30, 60, 120, 300),
        "terms": {"bitcoin": "bitcoin", "ethereum": "ethereum"},
    }
    base.update(overrides)
    return ArchiveConfig(**base)  # type: ignore[arg-type]


class _Response:
    """Scripted HTTP response: ``json()`` raises when there is no payload, like requests does."""

    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _article(url: str, seendate: str) -> dict[str, str]:
    return {
        "url": url,
        "title": f"Headline for {url}",
        "seendate": seendate,
        "domain": "example.com",
        "language": "English",
        "sourcecountry": "USA",
        "socialimage": "",
    }


@pytest.fixture()
def log_messages() -> Generator[list[str], None, None]:
    """Messages logged by ``spp.crypto.archive`` (the spp root logger does not propagate)."""
    messages: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = _Collector(level=logging.DEBUG)
    previous_level = archive.LOG.level
    archive.LOG.addHandler(handler)
    archive.LOG.setLevel(logging.DEBUG)
    try:
        yield messages
    finally:
        archive.LOG.removeHandler(handler)
        archive.LOG.setLevel(previous_level)


# --------------------------------------------------------------------------- #
# Window iteration
# --------------------------------------------------------------------------- #
def test_iter_windows_no_gaps_and_clips_end() -> None:
    start = pd.Timestamp("2026-09-01 00:00", tz="UTC")
    end = pd.Timestamp("2026-09-01 05:30", tz="UTC")
    windows = iter_windows(start, end, 2)
    assert len(windows) == 3
    assert windows[0] == (start, pd.Timestamp("2026-09-01 02:00", tz="UTC"))
    assert windows[1][0] == windows[0][1]
    assert windows[2][0] == windows[1][1]
    # The last window is clipped to end, and together the windows cover exactly [start, end).
    assert windows[-1][1] == end
    assert sum(((stop - cursor) for cursor, stop in windows), pd.Timedelta(0)) == end - start
    assert all(cursor.tz is not None and cursor.utcoffset() == pd.Timedelta(0)
               for cursor, _ in windows)
    # Naive bounds are interpreted as UTC, and an empty range yields no windows.
    assert iter_windows(pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-01 02:00"), 2) == [
        (WIN_START, WIN_END)
    ]
    assert iter_windows(end, start, 2) == []


def test_iter_windows_rejects_non_positive_hours() -> None:
    with pytest.raises(ValueError, match="positive"):
        iter_windows(WIN_START, WIN_END, 0)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_parse_articles_real_payload() -> None:
    rows = parse_articles(GDELT_PAYLOAD, asset="bitcoin", query="bitcoin")
    assert len(rows) == 2
    first = rows[0]
    assert first["asset"] == "bitcoin"
    assert first["query"] == "bitcoin"
    assert first["title"] == "Bitcoin Holds $57K as ETF Inflows Accelerate"
    assert first["url"] == "https://www.coindesk.com/markets/2024/05/01/bitcoin-holds-57k/"
    assert first["domain"] == "coindesk.com"
    assert first["source"] == "gdelt"
    assert [row["published_at"] for row in rows] == [
        pd.Timestamp("2024-05-01 12:30:00", tz="UTC"),
        pd.Timestamp("2024-05-01 12:34:00", tz="UTC"),
    ]


def test_parse_articles_seendate_has_minute_precision() -> None:
    rows = parse_articles(
        {"articles": [_article("https://example.com/m", "20240501T123000Z")]},
        asset="ethereum",
        query="ethereum",
    )
    published = rows[0]["published_at"]
    assert isinstance(published, pd.Timestamp)
    assert published.tz is not None
    assert (published.year, published.month, published.day) == (2024, 5, 1)
    assert (published.hour, published.minute, published.second) == (12, 30, 0)


def test_parse_articles_missing_articles_key_and_string_bodies() -> None:
    assert parse_articles({"something": "else"}, asset="bitcoin", query="bitcoin") == []
    assert parse_articles("not json at all", asset="bitcoin", query="bitcoin") == []
    as_string = json.dumps(GDELT_PAYLOAD)
    assert len(parse_articles(as_string, asset="bitcoin", query="bitcoin")) == 2


def test_parse_articles_drops_bad_date_and_counts_it(
    log_messages: list[str],
) -> None:
    payload = {
        "articles": [
            _article("https://example.com/good", "20240501T123000Z"),
            _article("https://example.com/bad", "not-a-date"),
            {"url": "https://example.com/no-seendate", "title": "no stamp"},
        ]
    }
    rows = parse_articles(payload, asset="bitcoin", query="bitcoin")
    assert [row["url"] for row in rows] == ["https://example.com/good"]
    assert any("dropped 2 article(s)" in message for message in log_messages)


# --------------------------------------------------------------------------- #
# fetch_window: retries, backoff, bad bodies
# --------------------------------------------------------------------------- #
def test_fetch_window_retries_429_then_success_with_configured_sleeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(max_retries=3, backoff_seconds=(7, 11, 13))
    sleeps: list[float] = []
    monkeypatch.setattr(archive.time, "sleep", sleeps.append)
    calls = {"n": 0}

    def fake_get(url: str, params: dict | None = None, timeout: float | None = None) -> _Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return _Response(429, text="rate limited")
        return _Response(200, {"articles": [_article("https://example.com/a", "20260901T010000Z")]})

    rows, saturated = fetch_window(config, "bitcoin", WIN_START, WIN_END, get=fake_get)
    assert calls["n"] == 2
    assert sleeps == [7]  # the configured backoff, before the retry
    assert len(rows) == 1
    assert saturated is False
    assert rows[0]["published_at"] == pd.Timestamp("2026-09-01 01:00:00", tz="UTC")
    assert rows[0]["query"] == "bitcoin"


def test_fetch_window_five_failures_return_empty_and_log(
    monkeypatch: pytest.MonkeyPatch, log_messages: list[str]
) -> None:
    config = make_config(max_retries=5, backoff_seconds=(1, 2, 3, 4, 5))
    sleeps: list[float] = []
    monkeypatch.setattr(archive.time, "sleep", sleeps.append)
    calls = {"n": 0}

    def fake_get(url: str, params: dict | None = None, timeout: float | None = None) -> _Response:
        calls["n"] += 1
        return _Response(429, text="still limited")

    rows, saturated = fetch_window(config, "bitcoin", WIN_START, WIN_END, get=fake_get)
    assert (rows, saturated) == ([], False)
    assert calls["n"] == 5
    assert sleeps == [1, 2, 3, 4]  # one sleep per retry, backoff exhausted in order
    assert any("failed after 5 attempt(s)" in message for message in log_messages)


def test_fetch_window_non_json_body_returns_empty_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(max_retries=2, backoff_seconds=(1, 2))
    sleeps: list[float] = []
    monkeypatch.setattr(archive.time, "sleep", sleeps.append)
    calls = {"n": 0}

    def fake_get(url: str, params: dict | None = None, timeout: float | None = None) -> _Response:
        calls["n"] += 1
        return _Response(200, payload=None, text="<html>temporarily unavailable</html>")

    rows, saturated = fetch_window(config, "bitcoin", WIN_START, WIN_END, get=fake_get)
    assert (rows, saturated) == ([], False)
    assert calls["n"] == 1  # a 200 body problem is reported, not retried
    assert sleeps == []


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def test_load_archive_config_defaults_from_package_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Resolved from the package root, not the cwd.
    monkeypatch.chdir(tmp_path)
    config = load_archive_config()
    assert config.endpoint == "https://api.gdeltproject.org/api/v2/doc/doc"
    assert config.window_hours == 2
    assert config.max_records == 250
    assert config.max_retries == 5
    assert config.backoff_seconds == (15, 30, 60, 120, 300)
    assert config.terms["bitcoin"] == "bitcoin"
    assert config.terms["ethereum"] == "ethereum"


# --------------------------------------------------------------------------- #
# crawl: resume, dedupe, saturation, budgets
# --------------------------------------------------------------------------- #
def test_crawl_resumes_and_skips_windows_in_state(tmp_path: Path) -> None:
    config = make_config()
    out = tmp_path / "archive.parquet"
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "done": ["bitcoin|2026-09-01T00:00:00+00:00"],
                "requests": 1,
                "failed": 0,
                "saturated": 0,
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_get(url: str, params: dict | None = None, timeout: float | None = None) -> _Response:
        calls.append(str(params["startdatetime"]))
        article = _article("https://example.com/r1", "20260901T030000Z")
        return _Response(200, {"articles": [article]})

    # Two windows (00-02, 02-04); the first is already recorded as done.
    summary = crawl(
        config, ["bitcoin"], WIN_START, END_2, out, state, get=fake_get
    )
    assert calls == ["20260901020000"]
    assert summary["requests"] == 1
    assert summary["kept"] == 1
    assert summary["windows_done"] == 2  # window 1 was already complete
    loaded = json.loads(state.read_text(encoding="utf-8"))
    # Cumulative counters: prior state (1) + this run (1).
    assert loaded["requests"] == 2
    assert "bitcoin|2026-09-01T02:00:00+00:00" in loaded["done"]
    assert out.is_file()


def test_crawl_dedupes_url_across_windows(tmp_path: Path) -> None:
    config = make_config()
    out = tmp_path / "archive.parquet"
    state = tmp_path / "state.json"

    def fake_get(url: str, params: dict | None = None, timeout: float | None = None) -> _Response:
        if params["startdatetime"] == "20260901000000":
            articles = [
                _article("https://example.com/a", "20260901T010000Z"),
                _article("https://example.com/b", "20260901T013000Z"),
            ]
        else:
            articles = [
                _article("https://example.com/b", "20260901T030000Z"),
                _article("https://example.com/c", "20260901T033000Z"),
            ]
        return _Response(200, {"articles": articles})

    summary = crawl(config, ["bitcoin"], WIN_START, END_2, out, state, get=fake_get)
    assert summary["requests"] == 2
    # url-b arrives in both windows; only the first occurrence is kept.
    assert summary["kept"] == 3
    stored = pd.read_parquet(out)
    assert stored["url"].tolist() == [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ]


def test_crawl_saturation_halves_window_once(tmp_path: Path) -> None:
    config = make_config(max_records=250)
    out = tmp_path / "archive.parquet"
    state = tmp_path / "state.json"
    calls = {"n": 0}

    def fake_get(url: str, params: dict | None = None, timeout: float | None = None) -> _Response:
        index = calls["n"]
        calls["n"] += 1
        articles = [
            _article(f"https://example.com/{index}-{i}", "20260901T003000Z")
            for i in range(250)
        ]
        return _Response(200, {"articles": articles})

    summary = crawl(config, ["bitcoin"], WIN_START, WIN_END, out, state, get=fake_get)
    # Main fetch hit the cap -> exactly two halves are fetched, and they are not split again.
    assert calls["n"] == 3
    assert summary["requests"] == 3
    assert summary["saturated"] == 3
    assert summary["kept"] == 750
    loaded = json.loads(state.read_text(encoding="utf-8"))
    assert loaded["saturated"] == 3
    assert loaded["done"] == ["bitcoin|2026-09-01T00:00:00+00:00"]


def test_crawl_stops_at_max_requests(tmp_path: Path) -> None:
    config = make_config()
    out = tmp_path / "archive.parquet"
    state = tmp_path / "state.json"
    calls = {"n": 0}

    def fake_get(url: str, params: dict | None = None, timeout: float | None = None) -> _Response:
        index = calls["n"]
        calls["n"] += 1
        return _Response(
            200, {"articles": [_article(f"https://example.com/b{index}", "20260901T010000Z")]}
        )

    # Four windows (00..08h), but the budget allows two fetches only.
    summary = crawl(
        config,
        ["bitcoin"],
        WIN_START,
        pd.Timestamp("2026-09-01 08:00", tz="UTC"),
        out,
        state,
        max_requests=2,
        get=fake_get,
    )
    assert calls["n"] == 2
    assert summary["requests"] == 2
    assert summary["failed"] == 0
    # Both fetched windows completed; the third stopped at the budget check.
    assert summary["windows_done"] == 2
    assert summary["kept"] == 2
    loaded = json.loads(state.read_text(encoding="utf-8"))
    assert loaded["requests"] == 2
    assert len(loaded["done"]) == 2


# --------------------------------------------------------------------------- #
# Parquet round trip
# --------------------------------------------------------------------------- #
def test_parquet_round_trip_sorted_by_published_at(tmp_path: Path) -> None:
    config = make_config()
    out = tmp_path / "nested" / "archive.parquet"
    state = tmp_path / "state.json"

    def fake_get(url: str, params: dict | None = None, timeout: float | None = None) -> _Response:
        # datedesc: each page lists its newest article first, so arrival order is not sorted.
        if params["startdatetime"] == "20260901000000":
            articles = [
                _article("https://example.com/w1-new", "20260901T015000Z"),
                _article("https://example.com/w1-old", "20260901T011000Z"),
            ]
        else:
            articles = [
                _article("https://example.com/w2-new", "20260901T033000Z"),
                _article("https://example.com/w2-old", "20260901T021500Z"),
            ]
        return _Response(200, {"articles": articles})

    summary = crawl(config, ["bitcoin"], WIN_START, END_2, out, state, get=fake_get)
    assert summary["kept"] == 4

    loaded = load_archive(out)
    assert list(loaded.columns) == list(ARCHIVE_COLUMNS)
    assert loaded["published_at"].tolist() == [
        pd.Timestamp("2026-09-01 01:10:00", tz="UTC"),
        pd.Timestamp("2026-09-01 01:50:00", tz="UTC"),
        pd.Timestamp("2026-09-01 02:15:00", tz="UTC"),
        pd.Timestamp("2026-09-01 03:30:00", tz="UTC"),
    ]
    assert loaded["published_at"].is_monotonic_increasing
