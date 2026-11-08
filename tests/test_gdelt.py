"""GDELT crawler contracts that must hold without touching the network.

The crawler earned these tests the hard way: the first live smoke fetched four articles with a
query term of ``app store`` and linked none of them to Apple, and a two-request burst was enough to
trigger HTTP 429. Both failures were silent in the happy-path code, so they are pinned here:
the search term must be the company name, and a rate limit must be retried rather than converted
into an empty result.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from spp.data import news_gdelt_hist as gdelt


class _Response:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_company_name_is_chosen_over_the_most_specific_product() -> None:
    terms = gdelt.default_terms(["AAPL", "MSFT", "JPM", "XOM"])
    assert terms["AAPL"] == "apple"
    assert terms["MSFT"] == "microsoft"
    assert terms["JPM"] == "jpmorgan"
    assert terms["XOM"] == "exxon"


def test_universe_aliases_keep_the_declaration_order() -> None:
    ordered = gdelt._first_aliases_from_universe()
    assert ordered["AAPL"] == "apple"
    assert ordered["JNJ"] == "johnson & johnson"


def test_rate_limit_is_retried_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_get(url: str, params: dict, timeout: float) -> _Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return _Response(429, text="slow down")
        return _Response(
            200,
            {"articles": [{"title": "Apple beats", "seendate": "20261026T143000Z"}]},
            text='{"articles": [{"title": "Apple beats"}]}',
        )

    monkeypatch.setattr(gdelt.requests, "get", fake_get)
    monkeypatch.setattr(gdelt.time, "sleep", lambda _seconds: None)
    articles = gdelt._request({"query": "apple"}, retries=3, backoff=0.0)
    assert calls["n"] == 2
    assert len(articles) == 1


def test_persistent_rate_limit_raises_instead_of_returning_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gdelt.requests, "get", lambda *a, **k: _Response(429, text="no"))
    monkeypatch.setattr(gdelt.time, "sleep", lambda _seconds: None)
    with pytest.raises(gdelt.GdeltError, match="429"):
        gdelt._request({"query": "apple"}, retries=2, backoff=0.0)


def test_empty_body_is_an_empty_result_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gdelt.requests, "get", lambda *a, **k: _Response(200, text="   "))
    assert gdelt._request({"query": "apple"}, retries=1, backoff=0.0) == []


def test_fetch_range_accepts_iso_strings_and_slices_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[datetime, datetime]] = []

    def fake_articles(query, start, end, *, max_records):
        seen.append((start, end))
        return pd.DataFrame(columns=["title", "url", "seendate"])

    monkeypatch.setattr(gdelt, "fetch_articles", fake_articles)
    funnel = gdelt.GdeltFunnel()
    frame = gdelt.fetch_range("apple", "2026-10-01", "2026-10-15", chunk_days=7, funnel=funnel)
    assert frame.empty
    assert funnel.requests == 2
    assert [tuple(x.date() for x in pair) for pair in seen] == [
        (datetime(2026, 10, 1).date(), datetime(2026, 10, 8).date()),
        (datetime(2026, 10, 8).date(), datetime(2026, 10, 15).date()),
    ]


def test_failed_slice_is_counted_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise gdelt.GdeltError("HTTP 429 rate limited")

    monkeypatch.setattr(gdelt, "fetch_articles", boom)
    funnel = gdelt.GdeltFunnel()
    frame = gdelt.fetch_range("apple", "2026-10-01", "2026-10-08", chunk_days=7, funnel=funnel)
    assert frame.empty
    assert funnel.requests == 1
    assert funnel.failed_requests == 1


def test_corpus_columns_match_the_daily_pipeline() -> None:
    assert set(gdelt._COLUMNS) >= {
        "article_id",
        "ticker",
        "published_at",
        "title",
        "url",
        "text_hash",
    }
