"""Historical news from the GDELT DOC 2.0 API, for the period the FNSPID corpus does not cover.

The intraday experiment needs two things the existing corpus cannot give: publication timestamps
that are accurate to the minute, and coverage that overlaps the intraday bar history. FNSPID ends
on 2023-12-18; yfinance 60-minute bars begin around 2024-11-11. Nothing overlaps, so a new news
source is not a convenience - it is the precondition for the experiment to exist at all.

GDELT is used because it is free, keyless and queryable by time window. Four properties matter and
none of them is assumed here:

* ``seendate`` is parsed as UTC in the compact ``YYYYMMDDTHHMMSSZ`` form GDELT emits;
* a query window is honoured by the server, so a chunked crawl reconstructs the timeline instead of
  returning "the last 250 articles" regardless of the dates asked for;
* results are capped per call (``MAXRECORDS``), which is why the crawler slices by ticker and by
  week and reports whether any slice hit the cap - a truncated slice is a coverage hole, not a
  smaller sample;
* tickers are attributed by **headline alias**, not by the query that found the article. A story
  mentioning three companies appears once per ticker it names, and a story that names none is
  dropped with a counted reason instead of being filed under whoever searched for it.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from spp.data.normalize import (
    ENTITY_AMBIGUOUS,
    ENTITY_OK,
    AliasIndex,
    article_id,
    clean_title,
    link_ticker,
    load_alias_index,
    normalize_title,
    text_hash,
)
from spp.logging import get_logger

__all__ = [
    "GDELT_ENDPOINT",
    "GdeltError",
    "GdeltFunnel",
    "build_corpus",
    "fetch_articles",
    "fetch_range",
]

_LOG = get_logger("data.gdelt")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
#: GDELT refuses anything above 250 per call. The crawler checks this and says so when a slice is
#: full, because a full slice means "there was more news than I asked for".
MAXRECORDS = 250
_COLUMNS = (
    "article_id",
    "ticker",
    "published_at",
    "date",
    "signal_date",
    "title",
    "title_norm",
    "url",
    "publisher",
    "source",
    "text_hash",
)


class GdeltError(RuntimeError):
    """Raised when the API cannot be used; never downgraded to an empty frame silently."""


@dataclass
class GdeltFunnel:
    """Counters for every decision the corpus builder makes."""

    requests: int = 0
    failed_requests: int = 0
    saturated_slices: int = 0
    raw_articles: int = 0
    duplicate_url: int = 0
    blank_title: int = 0
    no_alias: int = 0
    ambiguous_alias: int = 0
    kept_rows: int = 0
    extra: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, int]:
        payload = {
            "requests": self.requests,
            "failed_requests": self.failed_requests,
            "saturated_slices": self.saturated_slices,
            "raw_articles": self.raw_articles,
            "duplicate_url": self.duplicate_url,
            "blank_title": self.blank_title,
            "no_alias": self.no_alias,
            "ambiguous_alias": self.ambiguous_alias,
            "kept_rows": self.kept_rows,
        }
        payload.update(self.extra)
        return payload


def _parse_seendate(values: pd.Series) -> pd.Series:
    """GDELT stamps are UTC; anything that does not parse is a data error, not a missing value."""
    parsed = pd.to_datetime(values, format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
    return parsed


def _request(
    params: dict[str, Any],
    *,
    timeout: float = 60.0,
    retries: int = 5,
    backoff: float = 15.0,
) -> list[dict[str, Any]]:
    """One API call, with explicit failure. An empty article list is legitimate; a 429 is not.

    GDELT throttles aggressively: measured, three requests with a one-second pause lost two to
    HTTP 429. The default budget is therefore four attempts starting at five seconds, which is
    slow on purpose - a run that finishes is worth more than a run that gives up quietly.
    """
    delay = backoff
    last_error: str | None = None
    for _attempt in range(1, retries + 1):
        try:
            response = requests.get(GDELT_ENDPOINT, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_error = f"network error: {exc}"
            time.sleep(delay)
            delay = min(delay * 2, 120.0)
            continue
        if response.status_code == 429:
            last_error = "HTTP 429 rate limited"
            time.sleep(delay)
            delay = min(delay * 2, 120.0)
            continue
        if response.status_code >= 500:
            last_error = f"HTTP {response.status_code}"
            time.sleep(delay)
            delay = min(delay * 2, 120.0)
            continue
        if response.status_code != 200:
            raise GdeltError(f"GDELT returned HTTP {response.status_code}: {response.text[:200]}")
        text = response.text.strip()
        if not text:
            return []
        try:
            payload = response.json()
        except ValueError as exc:
            raise GdeltError(f"GDELT returned non-JSON: {text[:200]}") from exc
        if isinstance(payload, dict) and "error" in payload:
            raise GdeltError(f"GDELT error: {payload['error']}")
        articles = payload.get("articles") if isinstance(payload, dict) else None
        return list(articles or [])
    raise GdeltError(f"GDELT unavailable after {retries} attempts: {last_error}")


def fetch_articles(
    query: str,
    start: datetime | date,
    end: datetime | date,
    *,
    max_records: int = MAXRECORDS,
    sort: str = "dateasc",
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Articles for one query inside one window. Returns an empty frame when GDELT has none."""
    if max_records > MAXRECORDS:
        raise ValueError(f"max_records must be <= {MAXRECORDS}; GDELT silently caps above that")
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": int(max_records),
        "sort": sort,
        "startdatetime": _gdelt_stamp(start),
        "enddatetime": _gdelt_stamp(end),
    }
    articles = _request(params, timeout=timeout)
    if not articles:
        return pd.DataFrame(columns=["title", "seendate", "url", "domain", "language"])
    frame = pd.DataFrame(articles)
    for column in ("title", "seendate", "url", "domain", "language"):
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame.loc[:, ["title", "seendate", "url", "domain", "language"]]
    frame["seendate"] = _parse_seendate(frame["seendate"])
    return frame


def fetch_range(
    query: str,
    start: datetime,
    end: datetime,
    *,
    chunk_days: int = 7,
    max_records: int = MAXRECORDS,
    pause: float = 5.0,
    funnel: GdeltFunnel | None = None,
) -> pd.DataFrame:
    """Walk a window in slices so the per-call record cap cannot truncate the timeline."""
    stats = funnel if funnel is not None else GdeltFunnel()
    # Accept ISO strings and pandas timestamps: a caller typing "2026-10-26" should not have to
    # know that the sliding window is built by datetime arithmetic.
    start = pd.Timestamp(start).to_pydatetime().replace(tzinfo=None)
    end = pd.Timestamp(end).to_pydatetime().replace(tzinfo=None)
    window = timedelta(days=int(chunk_days))
    frames: list[pd.DataFrame] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + window, end)
        stats.requests += 1
        try:
            part = fetch_articles(query, cursor, stop, max_records=max_records)
        except GdeltError as exc:
            stats.failed_requests += 1
            _LOG.warning("slice %s..%s failed: %s", cursor.date(), stop.date(), exc)
            cursor = stop
            continue
        if len(part) >= max_records:
            stats.saturated_slices += 1
        stats.raw_articles += int(len(part))
        if not part.empty:
            frames.append(part)
        cursor = stop
        if pause and cursor < end:
            time.sleep(pause)
    if not frames:
        return pd.DataFrame(columns=["title", "seendate", "url", "domain", "language"])
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["url"], keep="first")
    return merged.sort_values("seendate").reset_index(drop=True)


def _gdelt_stamp(value: datetime | date | str) -> str:
    """GDELT wants ``YYYYMMDDHHMMSS``. Accepts strings too, because every caller that passes a
    plain date from the CLI hands over a string and the old signature only pretended otherwise."""
    if isinstance(value, str):
        value = pd.Timestamp(value).to_pydatetime()
    if isinstance(value, datetime):
        stamp = value if value.tzinfo is None else value.astimezone(tz=None)
        return stamp.strftime("%Y%m%d%H%M%S")
    return datetime(value.year, value.month, value.day).strftime("%Y%m%d%H%M%S")


def build_corpus(
    terms: dict[str, str],
    start: datetime,
    end: datetime,
    *,
    index: AliasIndex | None = None,
    chunk_days: int = 7,
    pause: float = 5.0,
    max_records: int = MAXRECORDS,
    extra_filter: str = "sourcelang:english",
) -> tuple[pd.DataFrame, GdeltFunnel]:
    """Crawl one query per ticker, then attribute every article by its headline.

    ``terms`` maps ticker -> query term. The query only decides *what to look for*; the ticker on
    each row comes from :func:`link_ticker`, so a story about Apple found by the Microsoft query is
    filed under AAPL and a story naming nobody is dropped and counted.
    """
    alias_index = index or load_alias_index()
    stats = GdeltFunnel()
    seen_urls: set[str] = set()
    rows: list[dict[str, Any]] = []
    for ticker, term in terms.items():
        query = f'"{term}" {extra_filter}'.strip()
        _LOG.info("crawling %s: %s", ticker, query)
        part = fetch_range(
            query,
            start,
            end,
            chunk_days=chunk_days,
            max_records=max_records,
            pause=pause,
            funnel=stats,
        )
        if part.empty:
            continue
        for record in part.itertuples(index=False):
            url = str(record.url) if pd.notna(record.url) else ""
            if not url:
                stats.blank_title += 1
                continue
            if url in seen_urls:
                stats.duplicate_url += 1
                continue
            seen_urls.add(url)
            title = clean_title(record.title)
            if title is None:
                stats.blank_title += 1
                continue
            linked, status = link_ticker(title, alias_index)
            if status == ENTITY_AMBIGUOUS:
                stats.ambiguous_alias += 1
                continue
            if status != ENTITY_OK or linked is None:
                stats.no_alias += 1
                continue
            published = record.seendate
            if pd.isna(published):
                stats.extra["bad_timestamp"] = stats.extra.get("bad_timestamp", 0) + 1
                continue
            publisher = str(record.domain) if pd.notna(record.domain) else "gdelt"
            title_norm = normalize_title(title)
            rows.append(
                {
                    "article_id": article_id(publisher, url, title),
                    "ticker": linked,
                    "published_at": published,
                    "date": pd.Timestamp(published).tz_convert("UTC").date(),
                    "signal_date": pd.NaT,
                    "title": title,
                    "title_norm": title_norm,
                    "url": url,
                    "publisher": publisher,
                    "source": "gdelt",
                    "text_hash": text_hash(title_norm),
                }
            )
    frame = pd.DataFrame(rows, columns=list(_COLUMNS)) if rows else pd.DataFrame(
        columns=list(_COLUMNS)
    )
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["article_id", "ticker"], keep="first")
        frame = frame.sort_values(["published_at", "ticker", "article_id"]).reset_index(drop=True)
    stats.kept_rows = int(len(frame))
    return frame, stats


def save_corpus(frame: pd.DataFrame, path: Path | str) -> Path:
    """Write the corpus with the same column set as ``news_raw.parquet`` so downstream code and
    the signal-date policy can be reused unchanged."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["signal_date"] = pd.NaT
    out.to_parquet(target, index=False)
    return target


def default_terms(tickers: Iterable[str], index: AliasIndex | None = None) -> dict[str, str]:
    """Search term per ticker: the company name, not its most specific product.

    ``aliases_for`` returns aliases sorted alphabetically, so any "longest alias" rule picks
    ``app store`` for Apple and ``office 365`` for Microsoft - product names that name the company
    only by accident, and the first smoke run confirmed the damage: four articles fetched, zero
    linked to the ticker. The universe file keeps aliases ordered with the company name first, and
    that order is the only one that means anything, so it is read from the source rather than
    reconstructed.
    """
    alias_index = index or load_alias_index()
    ordered = _first_aliases_from_universe()
    terms: dict[str, str] = {}
    for ticker in tickers:
        symbol = str(ticker)
        chosen = ordered.get(symbol)
        if not chosen:
            aliases: Sequence[str] = alias_index.aliases_for(symbol)
            readable = [a for a in aliases if not any(ch.isdigit() for ch in a)]
            chosen = max(readable, key=len) if readable else (aliases[0] if aliases else symbol)
        terms[symbol] = str(chosen)
    return terms


def _first_aliases_from_universe(path: Path | None = None) -> dict[str, str]:
    """Ticker -> the first alias declared in the universe file (the company name)."""
    target = path or _PROJECT_ROOT / "configs" / "universe_us_largecap.yaml"
    if not target.is_file():
        return {}
    try:
        payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # noqa: BLE001 - a broken universe file must be visible
        _LOG.warning("cannot read universe aliases from %s: %s", target, exc)
        return {}
    out: dict[str, str] = {}
    for ticker, spec in (payload.get("tickers") or {}).items():
        aliases = (spec or {}).get("aliases") or []
        if aliases:
            out[str(ticker)] = str(aliases[0])
    return out
