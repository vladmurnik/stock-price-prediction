from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from spp.data.extract import _parse_date, extract_from_html, extract_from_url

FIXTURE = Path(__file__).parent / "fixtures" / "article.html"


def test_extracts_title_body_and_timestamp():
    article = extract_from_html(FIXTURE.read_text(encoding="utf-8"), "https://example.com/a")
    assert article is not None
    assert article.method.startswith("trafilatura")
    assert "Apple beats estimates" in (article.title or "")
    assert "surge in services" in article.text
    # trafilatura normalises metadata dates to the calendar day; the policy
    # treats such rows conservatively, so the precision flag must say "day".
    assert article.published_at == datetime(2020, 6, 1, tzinfo=UTC)
    assert article.date_precision == "day"
    assert article.word_count >= 30


def test_navigation_and_footer_are_not_part_of_the_body():
    article = extract_from_html(FIXTURE.read_text(encoding="utf-8"), "https://example.com/a")
    assert article is not None
    assert "Copyright Reuters" not in article.text
    assert "Related: Microsoft" not in article.text


def test_empty_html_returns_none():
    assert extract_from_html("") is None
    assert extract_from_html("   \n ") is None
    assert extract_from_url("") is None
    assert extract_from_url("example.com/no-scheme") is None


def test_missing_date_is_reported_as_none_not_guessed():
    body = "word " * 40
    html = f"<html><body><article><h1>Headline</h1><p>{body}</p></article></body></html>"
    article = extract_from_html(html, None)
    assert article is not None
    assert article.published_at is None
    assert article.date_precision == "none"


def test_parse_date_keeps_minute_precision_when_available():
    stamped, precision = _parse_date("2020-06-01T21:30:00Z")
    assert precision == "minute"
    assert stamped == datetime(2020, 6, 1, 21, 30, tzinfo=UTC)


def test_parse_date_marks_day_precision():
    stamped, precision = _parse_date("2020-06-01")
    assert precision == "day"
    assert stamped == datetime(2020, 6, 1, tzinfo=UTC)
