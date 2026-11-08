"""Tests for T3: title cleaning, entity linking, dedupe, signal_date policy, ingest pipeline."""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from spp.config import load_universe
from spp.data import normalize as nz
from spp.data.news_fnspid import (
    fetch_news,
    fixture_source,
    load_news_raw,
    news_coverage,
    news_path,
    parse_published_at,
)
from spp.data.store import read_manifest, read_parquet

SUITE_DIR = Path(__file__).resolve().parent
FIXTURE = SUITE_DIR / "fixtures" / "fnspid_sample.csv"
UNIVERSE = SUITE_DIR.parent / "configs" / "universe_us_largecap.yaml"

START = date(2019, 1, 1)
END = date(2023, 12, 31)
POLICY = nz.SignalPolicy.from_config("16:00", strict_dates=False)


@pytest.fixture(scope="module")
def index() -> nz.AliasIndex:
    return nz.alias_index_from_universe(load_universe(UNIVERSE))


# --------------------------------------------------------------------------- #
# Title cleaning (plan: publisher suffix removed, entities unescaped)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "Apple shares rise after iPhone sales beat estimates - Reuters",
            "Apple shares rise after iPhone sales beat estimates",
        ),
        (
            "Apple shares rise after iPhone sales beat estimates | MarketWatch",
            "Apple shares rise after iPhone sales beat estimates",
        ),
        ("AT&T beats estimates -- CNBC.com", "AT&T beats estimates"),
        ("What happens next for the market? – Bloomberg", "What happens next for the market?"),
    ],
)
def test_publisher_suffix_is_stripped(raw: str, expected: str) -> None:
    assert nz.clean_title(raw) == expected


def test_html_entities_and_quotes_are_folded() -> None:
    cleaned = nz.clean_title("AT&amp;T says \u201cgood news\u201d for   customers")
    assert cleaned == 'AT&T says "good news" for customers'


def test_short_and_boilerplate_titles_are_rejected() -> None:
    assert nz.clean_title("Too short") is None
    assert nz.clean_title("Stock Market News") is None
    assert nz.clean_title("Top Stories") is None
    assert nz.clean_title(None) is None
    assert nz.clean_title("   ") is None


def test_title_is_truncated_to_limit() -> None:
    long_title = "Apple " * 100
    cleaned = nz.clean_title(long_title)
    assert cleaned is not None
    assert len(cleaned) <= nz.MAX_TITLE_CHARS


def test_normalize_title_and_hash() -> None:
    norm = nz.normalize_title("AT&T's Earnings: Beat!")
    assert norm == "att s earnings beat" or norm == "atts earnings beat"
    assert nz.text_hash(norm) == hashlib.sha1(norm.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Entity linking (plan: Apple Hospitality -> AAPL must NOT match)
# --------------------------------------------------------------------------- #
def test_apple_hospitality_is_not_linked_to_aapl(index: nz.AliasIndex) -> None:
    ticker, status = nz.link_ticker("Apple Hospitality REIT cuts dividend guidance", index)
    assert ticker is None
    assert status == nz.ENTITY_AMBIGUOUS


def test_longest_alias_wins_for_unitedhealth(index: nz.AliasIndex) -> None:
    # "unitedhealth group" and "unitedhealth" both hit; both belong to UNH -> ok.
    assert nz.link_ticker("UnitedHealth Group lifts outlook", index) == ("UNH", nz.ENTITY_OK)


def test_headline_matching_two_tickers_is_dropped(index: nz.AliasIndex) -> None:
    # "Tesla" (TSLA) and "Apple" (AAPL) both match distinct tickers.
    assert nz.link_ticker("Tesla and Apple both report earnings", index) == (
        None,
        nz.ENTITY_AMBIGUOUS,
    )


def test_ticker_token_is_case_sensitive(index: nz.AliasIndex) -> None:
    # bare ticker in upper case matches; "aapl" inside a sentence is not a token match
    assert nz.link_ticker("AAPL climbs after earnings", index) == ("AAPL", nz.ENTITY_OK)
    ticker, status = nz.link_ticker("Analysts praise aapl into the print", index)
    assert ticker is None and status == nz.ENTITY_NONE


def test_no_alias_returns_none(index: nz.AliasIndex) -> None:
    assert nz.link_ticker("Regional bank raises deposit rates", index) == (None, nz.ENTITY_NONE)


def test_alias_respects_word_boundaries(index: nz.AliasIndex) -> None:
    # "mac" must not fire inside "macroeconomics"
    assert nz.link_ticker("Macroeconomics data surprises analysts", index) == (
        None,
        nz.ENTITY_NONE,
    )


# --------------------------------------------------------------------------- #
# Dedupe
# --------------------------------------------------------------------------- #
def _record(
    ticker: str,
    title: str,
    publisher: str,
    when: datetime,
    url: str = "https://x.y/z",
) -> dict:
    norm = nz.normalize_title(title)
    return {
        "article_id": nz.article_id(publisher, url, title),
        "ticker": ticker,
        "published_at": pd.Timestamp(when),
        "date": pd.Timestamp(when.date()),
        "title": title,
        "title_norm": norm,
        "url": url,
        "publisher": publisher,
        "source": "fnspid",
        "text_hash": nz.text_hash(norm),
    }


def test_exact_dupe_with_entities_is_collapsed() -> None:
    a = _record("AAPL", "AT&T beats estimates", "Reuters", datetime(2020, 6, 1, 13, tzinfo=UTC))
    b = _record(
        "AAPL", "AT&T beats estimates", "CNBC", datetime(2020, 6, 1, 20, tzinfo=UTC), "https://c/n"
    )
    out, stats = nz.dedupe_news(pd.DataFrame([b, a]))
    assert len(out) == 1
    assert stats.exact_dupe == 1
    assert out.iloc[0]["published_at"] == pd.Timestamp("2020-06-01 13:00", tz="UTC")


def test_near_dupe_keeps_higher_priority_publisher() -> None:
    t1 = "Apple shares rise after iPhone sales beat street estimates"
    t2 = "Apple shares rise after iPhone sales beat street estimate"  # one char differs
    reuters = _record("AAPL", t1, "CNBC", datetime(2020, 6, 1, 14, tzinfo=UTC))
    lower = _record("AAPL", t2, "Bloomberg", datetime(2020, 6, 1, 13, tzinfo=UTC), "https://b/1")
    grams1 = nz.trigrams(nz.normalize_title(t1))
    grams2 = nz.trigrams(nz.normalize_title(t2))
    assert nz.jaccard(grams1, grams2) > 0.9
    out, stats = nz.dedupe_news(pd.DataFrame([reuters, lower]))
    assert stats.near_dupe == 1
    assert out.iloc[0]["publisher"] == "Bloomberg"


def test_near_dupe_is_scoped_to_ticker_day() -> None:
    title = "Company reports record quarterly revenue growth"
    a = _record("AAPL", title, "Reuters", datetime(2020, 6, 1, 14, tzinfo=UTC))
    b = _record("MSFT", title, "Reuters", datetime(2020, 6, 1, 14, tzinfo=UTC))
    c = _record("AAPL", title, "Reuters", datetime(2020, 6, 2, 14, tzinfo=UTC))
    out, stats = nz.dedupe_news(pd.DataFrame([a, b, c]))
    assert stats.near_dupe == 0
    assert len(out) == 3


# --------------------------------------------------------------------------- #
# signal_date policy (plan: 18:00 ET -> +1, weekend -> next trading day)
# --------------------------------------------------------------------------- #
def test_after_cutoff_shifts_one_day() -> None:
    # 18:00 ET on a Friday -> Monday (next trading day)
    stamp = datetime(2020, 6, 5, 18, 0, tzinfo=nz.ET)
    days = nz.nyse_trading_days(date(2020, 6, 1), date(2020, 6, 20))
    [signal] = nz.assign_signal_date([stamp.date()], [stamp], POLICY, days)
    assert signal == date(2020, 6, 8)


def test_before_cutoff_stays_same_day() -> None:
    stamp = datetime(2020, 6, 5, 10, 0, tzinfo=nz.ET)
    days = nz.nyse_trading_days(date(2020, 6, 1), date(2020, 6, 20))
    [signal] = nz.assign_signal_date([stamp.date()], [stamp], POLICY, days)
    assert signal == date(2020, 6, 5)


def test_weekend_date_only_rolls_to_next_trading_day() -> None:
    saturday = date(2020, 6, 6)
    days = nz.nyse_trading_days(date(2020, 6, 1), date(2020, 6, 20))
    [signal] = nz.assign_signal_date([saturday], [None], POLICY, days)
    assert signal == date(2020, 6, 8)


def test_date_only_strict_flag_shifts_plus_one() -> None:
    strict = nz.SignalPolicy.from_config("16:00", strict_dates=True)
    monday = date(2020, 6, 1)
    assert nz.assign_signal_date([monday], [None], strict) == [date(2020, 6, 2)]
    assert nz.assign_signal_date([monday], [None], POLICY) == [monday]


def test_signal_date_never_precedes_date() -> None:
    # UTC timestamp that is still the previous calendar day in ET must not go backwards
    stamp = datetime(2020, 6, 1, 2, 0, tzinfo=UTC)  # 22:00 ET on May 31
    [signal] = nz.assign_signal_date([stamp.date()], [stamp], POLICY)
    assert signal >= stamp.date()


def test_parse_published_at_branches() -> None:
    stamp, day = parse_published_at("2020-06-05 06:30:54 UTC")
    assert stamp == datetime(2020, 6, 5, 6, 30, 54)
    assert day == date(2020, 6, 5)
    stamp2, day2 = parse_published_at("2020-06-05")
    assert stamp2 is None and day2 == date(2020, 6, 5)
    assert parse_published_at("not-a-date") == (None, None)
    assert parse_published_at(None) == (None, None)


# --------------------------------------------------------------------------- #
# Ingest pipeline (offline fixture)
# --------------------------------------------------------------------------- #
def _run(tmp_path: Path, index: nz.AliasIndex, **kwargs):
    return fetch_news(
        START,
        END,
        tmp_path / "data",
        tmp_path / "reports",
        index,
        POLICY,
        source=fixture_source(FIXTURE),
        **kwargs,
    )


def test_pipeline_schema_and_dtypes(tmp_path: Path, index: nz.AliasIndex) -> None:
    result = _run(tmp_path, index)
    stored = read_parquet(result.path)
    assert list(stored.columns) == list(nz.NEWS_COLUMNS)
    assert str(stored["ticker"].dtype) in {"string", "object"}
    assert pd.api.types.is_datetime64_any_dtype(stored["date"])
    assert pd.api.types.is_datetime64_any_dtype(stored["signal_date"])
    assert result.violations == [], result.violations


def test_pipeline_filters(tmp_path: Path, index: nz.AliasIndex) -> None:
    result = _run(tmp_path, index)
    stats = result.stats
    assert stats.bad_date == 1
    assert stats.out_of_range == 1  # 2024 row
    assert stats.non_universe == 1  # symbol "A"
    assert stats.short_or_boilerplate >= 1  # "Stock Market News"
    assert stats.exact_dupe >= 1
    titles = set(result.frame["title"])
    assert not any("Apple Hospitality" in t for t in titles)
    assert not any("Tesla and Apple supplier" in t for t in titles)
    assert (result.frame["ticker"] == "AAPL").any()


def test_signal_date_invariant_holds_for_every_row(tmp_path: Path, index: nz.AliasIndex) -> None:
    result = _run(tmp_path, index)
    frame = result.frame
    assert len(frame) > 0
    assert (frame["signal_date"] >= frame["date"]).all()
    assert (frame["signal_date"].dt.day_of_week < 5).all()
    # 2020-06-01 21:30 UTC = 17:30 ET (after cutoff) -> Tuesday 2020-06-02
    late = frame.loc[frame["title"].str.startswith("Apple unveils new Mac")]
    assert late.iloc[0]["signal_date"] == pd.Timestamp("2020-06-02")


def test_rerun_is_byte_identical(tmp_path: Path, index: nz.AliasIndex) -> None:
    first = _run(tmp_path, index)
    digest_a = hashlib.sha256(first.path.read_bytes()).hexdigest()
    second = _run(tmp_path, index, refresh=True)
    digest_b = hashlib.sha256(second.path.read_bytes()).hexdigest()
    assert digest_a == digest_b


def test_cache_hit_skips_source(tmp_path: Path, index: nz.AliasIndex) -> None:
    _run(tmp_path, index)
    calls: list[str] = []

    def failing_source(shard: str):
        calls.append(shard)
        raise AssertionError("cache hit must not touch the source")

    cached = fetch_news(
        START, END, tmp_path / "data", tmp_path / "reports", index, POLICY, source=failing_source
    )
    assert cached.cached is True
    assert calls == []


def test_no_near_dupes_survive(tmp_path: Path, index: nz.AliasIndex) -> None:
    result = _run(tmp_path, index)
    assert nz.near_dupe_pairs(result.frame) == 0


def test_coverage_report_is_written(tmp_path: Path, index: nz.AliasIndex) -> None:
    result = _run(tmp_path, index)
    coverage_file = tmp_path / "reports" / "news_coverage.csv"
    bad_file = tmp_path / "reports" / "news_bad_dates.csv"
    assert coverage_file.is_file()
    assert bad_file.is_file()
    coverage = pd.read_csv(coverage_file)
    assert {"ticker", "median_per_day", "empty_share"} <= set(coverage.columns)
    manifest_file = news_path(tmp_path / "data").with_name("news_raw.parquet.manifest.json")
    manifest = read_manifest(manifest_file)
    assert manifest["rows"] == len(result.frame)
    assert manifest["shards"] == ["Stock_news/All_external.csv"]


def test_validate_detects_signal_date_violation(tmp_path: Path, index: nz.AliasIndex) -> None:
    result = _run(tmp_path, index)
    broken = result.frame.copy()
    broken["signal_date"] = broken["signal_date"] - pd.Timedelta(days=10)
    violations = nz.validate_news(broken, index.tickers)
    assert any("signal_date < date" in v for v in violations)


def test_load_and_coverage_helpers(tmp_path: Path, index: nz.AliasIndex) -> None:
    result = _run(tmp_path, index)
    loaded = load_news_raw(tmp_path / "data")
    assert len(loaded) == len(result.frame)
    coverage = news_coverage(loaded, START, END, list(index.tickers))
    assert len(coverage) == len(index.tickers)
    assert (coverage["empty_share"] >= 0).all() and (coverage["empty_share"] <= 1).all()
