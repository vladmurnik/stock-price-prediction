"""T12 — live news feeds: RSS, Finnhub company-news, GDELT DOC 2.0.

No HTML scraping. Every source below is an XML/RSS feed or a JSON API:

===========================  ==================================================  ==============
source                       endpoint                                            paid / limits
===========================  ==================================================  ==============
``rss:cnbc``                 ``www.cnbc.com/id/100003114/device/rss/rss.html``    free
``rss:marketwatch``          ``feeds.content.dowjones.io/public/rss/mw_topstories`` free
``rss:yahoo``                ``finance.yahoo.com/rss/headline?s=<TICKER>``       free
``rss:investing``            ``www.investing.com/rss/news_25.rss``               free
``finnhub``                  ``/api/v1/company-news?symbol=X&from=..&to=..``      env key, 60 rpm
``gdelt``                    ``/api/v2/doc/doc?mode=artlist&format=json``         free, ≥5 s
``alpha_vantage``            ``NEWS_SENTIMENT?tickers=X``                         25/day, opt-in
===========================  ==================================================  ==============

Every ``fetch_*`` function returns the ``news_raw`` schema of T3
(:data:`spp.data.normalize.NEWS_COLUMNS`). Rows carry ``source`` = feed id
(``rss:cnbc``, ``finnhub``, ``gdelt``, …) so the health report can separate
"the feed broke" from "there was no news".

Policy highlights (``12-live-feeds.md``)
    * RSS without a timestamp → ``published_at = now`` (UTC), ``signal_date`` by the T3 policy.
    * RSS timestamps in the future (publisher bug) are clamped to ``now`` and counted.
    * Finnhub: one request per ticker per range, ``1.1 s`` pause (≤60 req/min).
    * GDELT: ``seendate`` (``YYYYMMDDTHHMMSSZ``) is a *crawl* time, so ``published_at`` is
      left empty and ``date`` = ``signal_date`` = ``seendate`` day (conservative: the article
      is known to be public by then). ``sourcelang:english`` is appended to every query —
      without it the bare alias ``apple`` pulls in the Chinese fruit trade press.
    * Entity linking is :func:`spp.data.normalize.link_ticker`; headlines with no alias or
      an ambiguous alias are dropped.
    * Dedupe: exact by ``text_hash``, near-duplicates by trigram Jaccard ``> 0.9`` inside
      ``(ticker, 6-hour bucket)``, canonical URLs (``utm_*``, ``ref=``, ``amp`` removed).
    * ``signal_date >= date`` is asserted, never clamped.

Idempotency: ``data/live/news/{YYYY-MM-DD}.parquet`` is rebuilt from the union of its
existing rows and the fresh fetch, deduped, sorted and written atomically — a rerun with
the same feed payloads produces a byte-identical file.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd

from spp.data.news_fnspid import _empty_news, _finalize_news
from spp.data.normalize import (
    ENTITY_AMBIGUOUS,
    ENTITY_OK,
    AliasIndex,
    DropStats,
    SignalPolicy,
    article_id,
    clean_title,
    jaccard,
    link_ticker,
    normalize_title,
    text_hash,
    trigrams,
    validate_news,
)
from spp.data.store import read_parquet, write_parquet_atomic
from spp.logging import get_logger

__all__ = [
    "ALPHA_VANTAGE_ENV",
    "DEFAULT_FEEDS",
    "FETCH_LIVE_ORDER",
    "FINNHUB_ENV",
    "FEED_SOURCE_IDS",
    "GDELT_MIN_INTERVAL_S",
    "HealthRow",
    "LiveCollectResult",
    "backfill",
    "build_feed_url",
    "collect_live",
    "feed_health",
    "fetch_alpha_vantage",
    "fetch_finnhub",
    "fetch_gdelt",
    "fetch_rss",
    "health_path",
    "live_news_path",
    "parse_gdelt_payload",
    "parse_rss_payload",
    "write_health",
]

LOG = get_logger("data.news_live")

FINNHUB_ENV: Final[str] = "FINNHUB_API_KEY"
ALPHA_VANTAGE_ENV: Final[str] = "ALPHA_VANTAGE_API_KEY"

FINNHUB_URL: Final[str] = "https://finnhub.io/api/v1/company-news"
GDELT_URL: Final[str] = "https://api.gdeltproject.org/api/v2/doc/doc"
ALPHA_VANTAGE_URL: Final[str] = "https://www.alphavantage.co/query"

FINNHUB_MIN_INTERVAL_S: Final[float] = 1.1
GDELT_MIN_INTERVAL_S: Final[float] = 5.0
GDELT_MAX_RETRIES: Final[int] = 2
HTTP_TIMEOUT_S: Final[float] = 30.0
USER_AGENT: Final[str] = "spp-live-feeds/0.1 (+research; feed reading only)"

#: Source id prefixes. RSS feeds get ``rss:<name>``; APIs use the bare id.
RSS_PREFIX: Final[str] = "rss:"
SOURCE_FINNHUB: Final[str] = "finnhub"
SOURCE_GDELT: Final[str] = "gdelt"
SOURCE_AV: Final[str] = "alpha_vantage"

#: Default RSS feeds. Yahoo's ``rssindex`` endpoint returns 404 since 2024; the
#: per-ticker headline feed works and is therefore the Yahoo entry (see probe in
#: ``ops/_t12_probe2.py``). ``build_feed_url`` expands ``{TICKER}``.
DEFAULT_FEEDS: Final[dict[str, str]] = {
    "cnbc": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "marketwatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "yahoo": "https://finance.yahoo.com/rss/headline?s={TICKER}",
    "investing": "https://www.investing.com/rss/news_25.rss",
}

FEED_SOURCE_IDS: Final[tuple[str, ...]] = tuple(f"{RSS_PREFIX}{k}" for k in DEFAULT_FEEDS) + (
    SOURCE_FINNHUB,
    SOURCE_GDELT,
    SOURCE_AV,
)

#: Order in which sources are attempted (health rows keep this order).
FETCH_LIVE_ORDER: Final[tuple[str, ...]] = FEED_SOURCE_IDS


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def live_news_path(live_dir: str | Path, day: date) -> Path:
    """``data/live/news/{YYYY-MM-DD}.parquet``."""
    return Path(live_dir) / "news" / f"{day.isoformat()}.parquet"


def health_path(reports_dir: str | Path) -> Path:
    return Path(reports_dir) / "live_feed_health.csv"


def build_feed_url(template: str, ticker: str | None = None) -> str:
    """Expand ``{TICKER}`` in a feed template (Yahoo's per-ticker headline feed)."""
    if "{TICKER}" not in template:
        return template
    if ticker is None:
        raise ValueError(f"feed template {template!r} needs a ticker")
    return template.replace("{TICKER}", ticker.upper())


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _source_of(feed: Mapping[str, str] | str) -> str:
    """``{'id': 'cnbc', ...}`` / ``'rss:cnbc'`` / ``'https://…'`` → source id."""
    ident = feed.get("id") if isinstance(feed, Mapping) else feed
    if ident is None:
        raise ValueError("feed entry without an id")
    ident = str(ident)
    return (
        ident
        if ident.startswith(RSS_PREFIX)
        or ident
        in {
            SOURCE_FINNHUB,
            SOURCE_GDELT,
            SOURCE_AV,
        }
        else f"{RSS_PREFIX}{ident}"
    )


def _record(
    *,
    ticker: str,
    title: str,
    title_norm: str,
    published_at: datetime | None,
    source_date: date,
    url: str,
    publisher: str,
    source: str,
) -> dict[str, Any]:
    return {
        "article_id": article_id(publisher, url, title),
        "ticker": ticker,
        "published_at": published_at,
        "date": source_date,
        "title": title,
        "title_norm": title_norm,
        "url": url,
        "publisher": publisher,
        "source": source,
        "text_hash": text_hash(title_norm),
    }


def _link_and_clean(
    title_raw: object,
    index: AliasIndex,
    stats: DropStats,
    hint_ticker: str | None = None,
    fallback_to_hint: bool = False,
) -> tuple[str, str, str] | None:
    """Clean the headline and link a ticker. Returns ``(ticker, title, title_norm)``.

    ``hint_ticker`` is the symbol the row was fetched for (Finnhub company-news). When the
    headline links to that symbol the row is kept; when it links to *another* universe
    ticker the row is kept under the linked ticker (the headline is the truth); no alias or
    an ambiguous alias drops the row.
    """
    title = clean_title(title_raw)
    if title is None:
        stats.short_or_boilerplate += 1
        return None
    title_norm = normalize_title(title)
    if len(title_norm) < 5:
        stats.short_or_boilerplate += 1
        return None
    ticker, status = link_ticker(title, index)
    if status == ENTITY_AMBIGUOUS:
        stats.ambiguous_alias += 1
        return None
    if status != ENTITY_OK:
        if fallback_to_hint and hint_ticker is not None:
            # A per-ticker feed (``?s=AAPL``) is authoritative about the symbol: the headline
            # still belongs to that ticker, it just says "the iPhone maker" instead of "Apple".
            # Without this fallback a per-ticker RSS feed loses ~90 % of its rows.
            stats.kept += 1
            return hint_ticker, title, title_norm
        stats.no_alias += 1
        return None
    assert ticker is not None
    if hint_ticker is not None and ticker != hint_ticker:
        stats.alias_mismatch += 1
    return ticker, title, title_norm


def _rss_publisher(feed_title: str | None, url: str, source: str) -> str:
    """Publisher name: the feed's own title, else the URL host, else the source id."""
    if feed_title and str(feed_title).strip():
        return str(feed_title).strip()
    host = urlsplit(url).netloc.lower()
    if host:
        return host.removeprefix("www.")
    return source.removeprefix(RSS_PREFIX)


def canonical_url(raw: str | None) -> str:
    """Canonical URL: trackers (``utm_*``, ``ref``, ``amp``…) removed, host lower-cased.

    Builds on :func:`spp.data.news_fnspid._canonical_url` (which drops the fragment and the
    common trackers) and additionally removes ``amp``/``amp_*`` markers and sorts the query
    parameters so two spellings of the same link collapse to one string.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    try:
        parts = urlsplit(text)
    except ValueError:
        return text
    if not parts.scheme or not parts.netloc:
        return text
    pairs = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not re.match(r"^(amp|amp_\w+|outputType|guccounter)$", k)
        and not k.startswith("utm_")
        and k not in {"ref", "fbclid", "gclid", "cmp", "mod", "rss", "soc_src", "source"}
    ]
    query = urlencode(sorted(pairs))
    netloc = parts.netloc.lower()
    path = re.sub(r"/amp/?$", "/", parts.path)
    return urlunsplit((parts.scheme.lower(), netloc, path, query, ""))


# --------------------------------------------------------------------------- #
# RSS
# --------------------------------------------------------------------------- #
def parse_rss_payload(
    payload: bytes | str,
    *,
    source: str,
    index: AliasIndex,
    since: datetime,
    now: datetime | None = None,
    stats: DropStats | None = None,
    hint_ticker: str | None = None,
) -> pd.DataFrame:
    """Parse one RSS/Atom document into ``news_raw`` rows (pure function, no I/O).

    * ``entry.title`` → cleaned + linked; unlinked headlines are dropped.
    * ``published_parsed`` / ``updated_parsed`` → UTC; missing → ``now`` (counted in
      ``DropStats.bad_date``? no — in the caller's ``missing_timestamp`` counter).
    * timestamps strictly after ``now`` are clamped to ``now`` and counted.
    * entries older than ``since`` are skipped.
    """
    import feedparser

    own_stats = stats if stats is not None else DropStats()
    now = now or _now_utc()
    feed = feedparser.parse(payload if isinstance(payload, bytes) else payload.encode("utf-8"))
    feed_title = (feed.feed or {}).get("title") if hasattr(feed, "feed") else None
    records: list[dict[str, Any]] = []
    clamped = 0
    missing = 0
    for entry in getattr(feed, "entries", []) or []:
        own_stats.scanned += 1
        raw_title = entry.get("title")
        linked = _link_and_clean(
            raw_title,
            index,
            own_stats,
            hint_ticker=hint_ticker,
            fallback_to_hint=hint_ticker is not None,
        )
        if linked is None:
            continue
        ticker, title, title_norm = linked
        url = canonical_url(entry.get("link"))
        published_at: datetime | None = None
        for attr in ("published_parsed", "updated_parsed"):
            stamp = entry.get(attr)
            if stamp:
                try:
                    published_at = datetime(*stamp[:6], tzinfo=UTC)
                except (TypeError, ValueError):
                    published_at = None
                if published_at is not None:
                    break
        if published_at is None:
            missing += 1
            published_at = now
        elif published_at > now:
            clamped += 1
            published_at = now
        if published_at < since:
            own_stats.out_of_range += 1
            continue
        publisher = _rss_publisher(feed_title, url, source)
        records.append(
            _record(
                ticker=ticker,
                title=title,
                title_norm=title_norm,
                published_at=published_at,
                source_date=published_at.date(),
                url=url,
                publisher=publisher,
                source=source,
            )
        )
    if clamped:
        LOG.warning("%s: clamped %d future timestamp(s) to now", source, clamped)
    if records:
        own_stats.kept += len(records)
    frame = pd.DataFrame.from_records(records) if records else _empty_news()
    frame.attrs["missing_timestamp"] = missing
    frame.attrs["clamped"] = clamped
    return frame


def fetch_rss(
    feeds: Sequence[str | Mapping[str, str]],
    since: datetime,
    *,
    index: AliasIndex,
    http_get: Callable[[str], bytes] | None = None,
    now: datetime | None = None,
    tickers: Sequence[str] | None = None,
    stats: DropStats | None = None,
) -> pd.DataFrame:
    """Fetch and parse ``feeds``; one network document per feed (Yahoo per ticker).

    ``feeds`` accepts ``"rss:cnbc"`` / ``"cnbc"`` / ``"https://…"`` or mappings
    ``{"id": "cnbc", "url": "https://…"}``. Templates containing ``{TICKER}`` are expanded
    for every ticker in ``tickers``. A feed that fails to download is logged as a WARN and
    skipped; the other feeds still run.
    """
    get = http_get or _http_get
    now = now or _now_utc()
    frames: list[pd.DataFrame] = []
    for feed in feeds:
        source = _source_of(feed)
        template = feed.get("url") if isinstance(feed, Mapping) else _template_for(feed)
        targets = (
            [(build_feed_url(template, t), t) for t in (tickers or [])]
            if "{TICKER}" in template
            else [(template, None)]
        )
        if not targets:
            LOG.warning("%s: template needs a ticker but none given; skipped", source)
            continue
        for url, url_ticker in targets:
            try:
                payload = get(url)
            except Exception as exc:  # noqa: BLE001 - network boundary, log+skip
                LOG.warning("%s: fetch failed (%s); skipped", source, type(exc).__name__)
                continue
            frame = parse_rss_payload(
                payload,
                source=source,
                index=index,
                since=since,
                now=now,
                stats=stats,
                hint_ticker=url_ticker,
            )
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return _empty_news()
    return pd.concat(frames, ignore_index=True)


def _template_for(feed: str | Mapping[str, str]) -> str:
    if isinstance(feed, Mapping):
        return str(feed["url"])
    if "://" in feed:
        return feed
    name = feed.removeprefix(RSS_PREFIX)
    if name not in DEFAULT_FEEDS:
        raise ValueError(f"unknown feed id {feed!r}; pass a URL or one of {sorted(DEFAULT_FEEDS)}")
    return DEFAULT_FEEDS[name]


def feed_templates(feeds: Iterable[str | Mapping[str, str]]) -> dict[str, str]:
    return {_source_of(f): _template_for(f) for f in feeds}


def _http_get(url: str, *, timeout: float = HTTP_TIMEOUT_S) -> bytes:
    import requests

    response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response.content


# --------------------------------------------------------------------------- #
# Finnhub
# --------------------------------------------------------------------------- #
def _parse_finnhub_items(
    items: Sequence[Mapping[str, Any]],
    *,
    index: AliasIndex,
    since: datetime,
    until: datetime,
    stats: DropStats,
    hint_ticker: str | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for item in items:
        stats.scanned += 1
        epoch = item.get("datetime")
        try:
            published_at = datetime.fromtimestamp(float(epoch), tz=UTC)
        except (TypeError, ValueError, OSError):
            stats.bad_date += 1
            continue
        if published_at < since or published_at > until + timedelta(days=1):
            stats.out_of_range += 1
            continue
        linked = _link_and_clean(item.get("headline"), index, stats, hint_ticker=hint_ticker)
        if linked is None:
            continue
        ticker, title, title_norm = linked
        url = canonical_url(item.get("url"))
        publisher = str(item.get("source") or "").strip() or "finnhub"
        records.append(
            _record(
                ticker=ticker,
                title=title,
                title_norm=title_norm,
                published_at=published_at,
                source_date=published_at.date(),
                url=url,
                publisher=publisher,
                source=SOURCE_FINNHUB,
            )
        )
    stats.kept += len(records)
    return pd.DataFrame.from_records(records) if records else _empty_news()


def fetch_finnhub(
    tickers: list[str],
    since: date,
    until: date,
    api_key: str | None,
    *,
    index: AliasIndex,
    http_get_json: Callable[[str, dict[str, Any]], Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: datetime | None = None,
    stats: DropStats | None = None,
) -> pd.DataFrame:
    """``GET /api/v1/company-news`` per ticker; 1.1 s pause keeps ≤60 req/min.

    Missing key → :class:`MissingCredentialError` (the CLI turns it into a WARN and the
    other sources keep running). The key never appears in logs or in raised messages.
    """
    if not api_key:
        raise MissingCredentialError(FINNHUB_ENV)
    get_json = http_get_json or _http_get_json
    own_stats = stats if stats is not None else DropStats()
    now = now or _now_utc()
    since_dt = datetime.combine(since, datetime.min.time(), tzinfo=UTC)
    until_dt = datetime.combine(until, datetime.max.time(), tzinfo=UTC)
    frames: list[pd.DataFrame] = []
    for position, ticker in enumerate(tickers):
        if position:
            sleep(FINNHUB_MIN_INTERVAL_S)
        params = {
            "symbol": ticker.upper(),
            "from": since.isoformat(),
            "to": until.isoformat(),
            "token": api_key,
        }
        try:
            payload = get_json(FINNHUB_URL, params)
        except Exception as exc:  # noqa: BLE001 - network boundary
            LOG.warning("finnhub %s: request failed (%s)", ticker, type(exc).__name__)
            continue
        if not isinstance(payload, list):
            LOG.warning("finnhub %s: unexpected payload %s", ticker, type(payload).__name__)
            continue
        frame = _parse_finnhub_items(
            payload,
            index=index,
            since=since_dt,
            until=until_dt,
            stats=own_stats,
            hint_ticker=ticker,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return _empty_news()
    return pd.concat(frames, ignore_index=True)


def _http_get_json(url: str, params: dict[str, Any]) -> Any:
    import requests

    response = requests.get(
        url, params=params, timeout=HTTP_TIMEOUT_S, headers={"User-Agent": USER_AGENT}
    )
    if response.status_code == 429:
        raise RateLimitedError(f"HTTP 429 from {urlsplit(url).netloc}")
    if response.status_code in {401, 403}:
        raise MissingCredentialError(f"HTTP {response.status_code} from {urlsplit(url).netloc}")
    response.raise_for_status()
    return response.json()


class MissingCredentialError(RuntimeError):
    """A source needs a credential that is not configured; the name is safe to log."""


class RateLimitedError(RuntimeError):
    """Upstream returned 429."""


# --------------------------------------------------------------------------- #
# GDELT
# --------------------------------------------------------------------------- #
def _gdelt_timestamp(stamp: datetime) -> str:
    return stamp.astimezone(UTC).strftime("%Y%m%d%H%M%S")


def build_gdelt_query(aliases: Sequence[str]) -> str:
    """``("alias one" OR "alias two") sourcelang:english``."""
    terms = [f'"{a}"' for a in (str(x).strip() for x in aliases) if a]
    if not terms:
        return ""
    return "(" + " OR ".join(terms) + ") sourcelang:english"


def parse_gdelt_payload(
    payload: bytes | str | Mapping[str, Any],
    *,
    index: AliasIndex,
    stats: DropStats | None = None,
    hint_ticker: str | None = None,
    since: date | None = None,
    until: date | None = None,
) -> pd.DataFrame:
    """Parse a GDELT ``artlist`` JSON document.

    ``seendate`` (``YYYYMMDDTHHMMSSZ``) is a crawl timestamp, not a publication time, so
    ``published_at`` stays empty and ``date = signal_date = seendate.date()`` (the
    conservative T3 branch for date-only rows).
    """
    own_stats = stats if stats is not None else DropStats()
    if isinstance(payload, (bytes, str)):
        text = payload.decode("utf-8", "replace") if isinstance(payload, bytes) else payload
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            LOG.warning("gdelt: non-JSON payload (%d bytes)", len(text))
            return _empty_news()
    else:
        data = payload
    articles = (data or {}).get("articles") or [] if isinstance(data, Mapping) else []
    records: list[dict[str, Any]] = []
    for item in articles:
        own_stats.scanned += 1
        seen = str(item.get("seendate") or "").strip()
        try:
            seen_at = datetime.strptime(seen, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except ValueError:
            own_stats.bad_date += 1
            continue
        day = seen_at.date()
        if (since is not None and day < since) or (until is not None and day > until):
            own_stats.out_of_range += 1
            continue
        linked = _link_and_clean(item.get("title"), index, own_stats, hint_ticker=hint_ticker)
        if linked is None:
            continue
        ticker, title, title_norm = linked
        url = canonical_url(item.get("url"))
        publisher = str(item.get("domain") or "").strip().lower() or "gdelt"
        records.append(
            _record(
                ticker=ticker,
                title=title,
                title_norm=title_norm,
                published_at=None,
                source_date=day,
                url=url,
                publisher=publisher,
                source=SOURCE_GDELT,
            )
        )
    own_stats.kept += len(records)
    return pd.DataFrame.from_records(records) if records else _empty_news()


def fetch_gdelt(
    query: str,
    since: date,
    until: date,
    *,
    index: AliasIndex,
    http_get_json: Callable[[str, dict[str, Any]], Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    max_records: int = 250,
    hint_ticker: str | None = None,
    stats: DropStats | None = None,
) -> pd.DataFrame:
    """One GDELT ``artlist`` request; 429 → sleep 5 s, 2 retries, then skip (WARN)."""
    if not query.strip():
        return _empty_news()
    get_json = http_get_json or _http_get_json
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": int(max_records),
        "startdatetime": _gdelt_timestamp(datetime.combine(since, datetime.min.time(), tzinfo=UTC)),
        "enddatetime": _gdelt_timestamp(datetime.combine(until, datetime.max.time(), tzinfo=UTC)),
    }
    payload: Any = None
    for attempt in range(GDELT_MAX_RETRIES + 1):
        if attempt:
            sleep(GDELT_MIN_INTERVAL_S)
        try:
            payload = get_json(GDELT_URL, params)
            break
        except RateLimitedError:
            LOG.warning("gdelt: HTTP 429 (attempt %d/%d)", attempt + 1, GDELT_MAX_RETRIES + 1)
            continue
        except Exception as exc:  # noqa: BLE001 - network boundary
            # GDELT answers bad queries with HTTP 200 + plain text; JSON parse errors land here.
            LOG.warning("gdelt: request failed (%s); skipped", type(exc).__name__)
            return _empty_news()
    if payload is None:
        LOG.warning("gdelt: still rate-limited after %d retries; skipped", GDELT_MAX_RETRIES)
        return _empty_news()
    return parse_gdelt_payload(
        payload, index=index, stats=stats, hint_ticker=hint_ticker, since=since, until=until
    )


# --------------------------------------------------------------------------- #
# Alpha Vantage (cross-check only, never in prod)
# --------------------------------------------------------------------------- #
def fetch_alpha_vantage(
    tickers: Sequence[str],
    *,
    index: AliasIndex,
    api_key: str | None,
    limit: int = 200,
    http_get_json: Callable[[str, dict[str, Any]], Any] | None = None,
    stats: DropStats | None = None,
) -> pd.DataFrame:
    """``NEWS_SENTIMENT?tickers=…`` — 25 req/day, so it is opt-in (``--cross-check``)."""
    if not api_key:
        raise MissingCredentialError(ALPHA_VANTAGE_ENV)
    get_json = http_get_json or _http_get_json
    own_stats = stats if stats is not None else DropStats()
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ",".join(t.upper() for t in tickers),
        "limit": int(limit),
        "apikey": api_key,
    }
    try:
        data = get_json(ALPHA_VANTAGE_URL, params)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("alpha_vantage: request failed (%s); skipped", type(exc).__name__)
        return _empty_news()
    feed = (data or {}).get("feed") or [] if isinstance(data, Mapping) else []
    records: list[dict[str, Any]] = []
    for item in feed:
        own_stats.scanned += 1
        raw = str(item.get("time_published") or "")
        try:
            published_at = datetime.strptime(raw, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        except ValueError:
            own_stats.bad_date += 1
            continue
        linked = _link_and_clean(item.get("title"), index, own_stats)
        if linked is None:
            continue
        ticker, title, title_norm = linked
        records.append(
            _record(
                ticker=ticker,
                title=title,
                title_norm=title_norm,
                published_at=published_at,
                source_date=published_at.date(),
                url=canonical_url(item.get("url")),
                publisher=str(item.get("source") or "alpha_vantage").strip(),
                source=SOURCE_AV,
            )
        )
    own_stats.kept += len(records)
    return pd.DataFrame.from_records(records) if records else _empty_news()


# --------------------------------------------------------------------------- #
# Dedupe across feeds
# --------------------------------------------------------------------------- #
#: 6-hour time bucket for near-duplicate detection (plan §5: ``(ticker, 6 hours)``).
DEDUPE_BUCKET_HOURS: Final[int] = 6


def _bucket(published_at: pd.Series, day: pd.Series) -> pd.Series:
    """6-hour dedupe bucket label: ``YYYY-MM-DD-b<n>``; ``-na`` when no timestamp exists.

    Uses the UTC hour (``-b0`` = 00–05, ``-b1`` = 06–11, …). The *date* component is the
    source date, so a GDELT row without a timestamp lands in its own ``-na`` bucket.
    """
    utc = pd.to_datetime(published_at, utc=True)
    hours = utc.dt.hour
    bucket = (hours // DEDUPE_BUCKET_HOURS).astype("Int64")
    days = pd.to_datetime(day, errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    labels = [
        f"{d}-b{int(b)}" if pd.notna(b) else f"{d}-na" for d, b in zip(days, bucket, strict=True)
    ]
    return pd.Series(labels, index=day.index, dtype="string")


def dedupe_live(df: pd.DataFrame, threshold: float = 0.9) -> tuple[pd.DataFrame, dict[str, int]]:
    """Exact (``text_hash``) + near-duplicate (Jaccard > ``threshold``) removal.

    Two duplicate classes, both inside ``(ticker, 6-hour bucket)``:

    1. **Canonical URL** — the same link syndicated by several feeds. Survivor: best
       publisher priority (Reuters > Bloomberg > WSJ > CNBC > others), then the earliest
       ``published_at``, then ``article_id``.
    2. **Headline** — ``text_hash`` exact, then trigram Jaccard above ``threshold``.
       Survivor: same ordering. The *class* is by group size so the counters are stable
       (``by_url``, ``by_title``, ``near``).
    """
    from spp.data.normalize import publisher_priority

    if df.empty:
        return df.copy(), {"by_url": 0, "by_title": 0, "near": 0}
    work = df.copy()
    work["_priority"] = work["publisher"].map(lambda p: publisher_priority(str(p))).astype("int32")
    work["_bucket"] = _bucket(work["published_at"], work["date"])
    work["_pub_missing"] = work["published_at"].isna()
    order = ["_priority", "_pub_missing", "published_at", "article_id"]
    counts = {"by_url": 0, "by_title": 0, "near": 0}

    # 1. canonical URL — empty urls never merge (a missing link is not evidence).
    with_url = work["url"].astype(str) != ""
    if with_url.any():
        subset = work.loc[with_url].sort_values(["ticker", "_bucket", "url", *order], kind="stable")
        dup = subset.duplicated(subset=["ticker", "_bucket", "url"], keep="first")
        counts["by_url"] = int(dup.sum())
        work = pd.concat([work.loc[~with_url], subset.loc[~dup]])

    # 2. exact text_hash, then near-duplicate clustering.
    work = work.sort_values(["ticker", "_bucket", *order], kind="stable")
    before = len(work)
    work = work.drop_duplicates(subset=["ticker", "_bucket", "text_hash"], keep="first")
    counts["by_title"] = before - len(work)
    keep: dict[Any, bool] = {}
    for (_ticker, _bucket_id), group in work.groupby(["ticker", "_bucket"], sort=True):
        survivors: list[frozenset[str]] = []
        for idx, _row in group.iterrows():
            grams = trigrams(str(group.at[idx, "title_norm"]))
            if any(jaccard(grams, kept) > threshold for kept in survivors):
                counts["near"] += 1
                keep[idx] = False
                continue
            survivors.append(grams)
            keep[idx] = True
    work = work[[keep[idx] for idx in work.index]]

    work = work.drop(columns=["_priority", "_bucket", "_pub_missing"])
    work = work.sort_values(
        ["date", "ticker", "published_at", "article_id"], kind="stable", na_position="last"
    )
    return work.reset_index(drop=True), counts


# --------------------------------------------------------------------------- #
# Health report
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HealthRow:
    source: str
    n_items: int
    n_after_dedup: int
    max_published_at: str  # ISO-8601 string; empty when the source has no timestamp
    status: str


@dataclass
class LiveCollectResult:
    frame: pd.DataFrame
    per_source: dict[str, pd.DataFrame]
    health: list[HealthRow]
    dedupe: dict[str, int]
    violations: list[str] = field(default_factory=list)
    artifacts: list[Path] = field(default_factory=list)
    dry_run: bool = False
    day: date | None = None

    @property
    def by_source_counts(self) -> dict[str, int]:
        return {row.source: row.n_after_dedup for row in self.health}


def _status_for(n_items: int, error: str | None) -> str:
    if error is not None:
        return "ERROR" if error.startswith("auth") else "WARN"
    return "OK" if n_items else "EMPTY"


def feed_health(
    per_source: Mapping[str, pd.DataFrame],
    survivors: pd.DataFrame,
    *,
    errors: Mapping[str, str] | None = None,
    expected: Sequence[str] = (),
) -> list[HealthRow]:
    """One row per expected source — empty sources included (``EMPTY`` = WARN, not ERROR)."""
    errors = dict(errors or {})
    rows: list[HealthRow] = []
    seen = list(dict.fromkeys([*expected, *per_source.keys()]))
    for source in seen:
        frame = per_source.get(source)
        n_items = 0 if frame is None else len(frame)
        subset = survivors[survivors["source"] == source] if not survivors.empty else survivors
        n_after = 0 if subset is None or subset.empty else len(subset)
        stamp = ""
        if n_items and "published_at" in frame.columns:
            stamps = pd.to_datetime(frame["published_at"], utc=True, errors="coerce").dropna()
            if len(stamps):
                stamp = max(stamps).isoformat()
            elif "date" in frame.columns:
                days = pd.to_datetime(frame["date"], errors="coerce").dropna()
                if len(days):
                    stamp = max(days).date().isoformat()
        rows.append(
            HealthRow(
                source=source,
                n_items=n_items,
                n_after_dedup=n_after,
                max_published_at=stamp,
                status=_status_for(n_items, errors.get(source)),
            )
        )
    return rows


def write_health(rows: Sequence[HealthRow], path: str | Path) -> Path:
    """Atomic write of ``reports/live_feed_health.csv`` (no timestamp column → idempotent)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "source": r.source,
                "n_items": r.n_items,
                "n_after_dedup": r.n_after_dedup,
                "max_published_at": r.max_published_at,
                "status": r.status,
            }
            for r in rows
        ],
        columns=["source", "n_items", "n_after_dedup", "max_published_at", "status"],
    )
    tmp = target.with_name(target.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(tmp, target)
    return target


# --------------------------------------------------------------------------- #
# collect_live / backfill
# --------------------------------------------------------------------------- #
def _source_errors(exc: BaseException) -> str:
    if isinstance(exc, MissingCredentialError):
        return f"auth: {exc}"
    if isinstance(exc, RateLimitedError):
        return "rate-limited"
    return f"error: {type(exc).__name__}"


def _default_globals_month(day: date) -> date:
    return day.replace(day=1) - timedelta(days=1)


def collect_live(
    tickers: Sequence[str],
    aliases: Mapping[str, Sequence[str]] | AliasIndex,
    since: datetime,
    *,
    until: datetime | None = None,
    feeds: Sequence[str | Mapping[str, str]] | None = None,
    index: AliasIndex | None = None,
    policy: SignalPolicy | None = None,
    include_finnhub: bool = True,
    include_gdelt: bool = True,
    include_alpha_vantage: bool = False,
    finnhub_key: str | None = None,
    alpha_vantage_key: str | None = None,
    max_records: int = 250,
    stats: DropStats | None = None,
) -> LiveCollectResult:
    """Fetch every enabled source, link, dedupe across feeds and build the health table.

    ``aliases`` may be a ready :class:`AliasIndex` or a ``{ticker: [alias, …]}`` mapping.
    Sources that fail (missing key, 429 after retries, network) land in the health file
    with ``WARN``/``ERROR``; the run continues. Returns data only — writing is the
    caller's job (:func:`write_health`, :func:`write_daily`).
    """
    from spp.data.normalize import build_alias_index

    if index is None:
        if isinstance(aliases, AliasIndex):
            index = aliases
        else:
            key = tuple(
                sorted(
                    (str(t), tuple(str(a) for a in (names or ()))) for t, names in aliases.items()
                )
            )
            index = build_alias_index(key)
    policy = policy or SignalPolicy()
    own_stats = stats if stats is not None else DropStats()
    now = until or _now_utc()
    since = since.astimezone(UTC)
    errors: dict[str, str] = {}
    per_source: dict[str, pd.DataFrame] = {}

    template_feeds = (
        list(feeds)
        if feeds is not None
        else [{"id": name, "url": url} for name, url in DEFAULT_FEEDS.items()]
    )
    template_feeds = [f for f in template_feeds if _source_of(f).startswith(RSS_PREFIX)]
    fetched = fetch_rss(
        template_feeds,
        since,
        index=index,
        now=now,
        tickers=[t.upper() for t in tickers],
        stats=own_stats,
    )
    frames_to_watch = list(template_feeds)
    if fetched.empty:
        for feed in frames_to_watch:
            per_source[_source_of(feed)] = _empty_news()
    else:
        for feed in frames_to_watch:
            source = _source_of(feed)
            per_source[source] = fetched[fetched["source"] == source].reset_index(drop=True)
    new_frames: list[pd.DataFrame] = []
    if not fetched.empty:
        new_frames.append(fetched)
    new_frames: list[pd.DataFrame] = []
    if not fetched.empty:
        new_frames.append(fetched)

    if include_finnhub:
        finnhub_key = finnhub_key if finnhub_key is not None else os.environ.get(FINNHUB_ENV)
        try:
            frame = fetch_finnhub(
                list(tickers),
                since.date(),
                now.date(),
                finnhub_key,
                index=index,
                now=now,
                stats=own_stats,
            )
            per_source[SOURCE_FINNHUB] = frame
        except MissingCredentialError as exc:
            errors[SOURCE_FINNHUB] = _source_errors(exc)
            LOG.warning("finnhub: %s not set; source skipped", FINNHUB_ENV)
            per_source[SOURCE_FINNHUB] = _empty_news()

    if include_gdelt:
        gdelt_since = max(since.date(), now.date() - timedelta(days=90))
        gdelt_until = now.date()
        collected: list[pd.DataFrame] = []
        # One query per ticker keeps entity linking exact (plan §3) and stays rate-friendly.
        for position, ticker in enumerate(tickers):
            names = [ticker.upper(), *aliases_for(index, ticker)]
            query = build_gdelt_query(names)
            if not query:
                continue
            if position:
                time.sleep(GDELT_MIN_INTERVAL_S)
            try:
                frame = fetch_gdelt(
                    query,
                    gdelt_since,
                    gdelt_until,
                    index=index,
                    hint_ticker=ticker.upper(),
                    max_records=max_records,
                    stats=own_stats,
                )
            except Exception as exc:  # noqa: BLE001 - health row, not a crash
                errors[SOURCE_GDELT] = _source_errors(exc)
                LOG.warning("gdelt %s: %s", ticker, errors[SOURCE_GDELT])
                continue
            if not frame.empty:
                collected.append(frame)
        per_source[SOURCE_GDELT] = (
            pd.concat(collected, ignore_index=True) if collected else _empty_news()
        )

    if include_alpha_vantage:
        alpha_vantage_key = (
            alpha_vantage_key
            if alpha_vantage_key is not None
            else os.environ.get(ALPHA_VANTAGE_ENV)
        )
        try:
            per_source[SOURCE_AV] = fetch_alpha_vantage(
                list(tickers), index=index, api_key=alpha_vantage_key, stats=own_stats
            )
        except MissingCredentialError as exc:
            errors[SOURCE_AV] = _source_errors(exc)
            LOG.warning("alpha_vantage: %s not set; source skipped", ALPHA_VANTAGE_ENV)
            per_source[SOURCE_AV] = _empty_news()

    for frame in per_source.values():
        if not frame.empty and frame is not fetched:
            new_frames.append(frame)
    combined = pd.concat(new_frames, ignore_index=True) if new_frames else _empty_news()
    if not combined.empty:
        combined = _finalize_news(combined, policy, _calendar(since.date(), now.date()))
        combined, dedupe_counts = dedupe_live(combined)
        combined = _finalize_news(combined, policy, _calendar(since.date(), now.date()))
    else:
        dedupe_counts = {"by_url": 0, "by_title": 0, "near": 0}

    expected = [*(_source_of(f) for f in frames_to_watch)]
    if include_finnhub:
        expected.append(SOURCE_FINNHUB)
    if include_gdelt:
        expected.append(SOURCE_GDELT)
    if include_alpha_vantage:
        expected.append(SOURCE_AV)
    health = feed_health(per_source, combined, errors=errors, expected=expected)
    violations = validate_news(combined) if not combined.empty else []
    if not combined.empty:
        future = combined["published_at"] > now
        if bool(future.any()):
            violations.append(f"{int(future.sum())} rows with published_at in the future")
    return LiveCollectResult(
        frame=combined,
        per_source=per_source,
        health=health,
        dedupe=dedupe_counts,
        violations=violations,
    )


def aliases_for(index: AliasIndex, ticker: str) -> list[str]:
    """Aliases of ``ticker`` registered in the index (for the GDELT query)."""
    getter = getattr(index, "aliases_for", None)
    if callable(getter):
        return list(getter(ticker))
    return []


def _calendar(start: date, end: date) -> list[date]:
    from spp.data.normalize import nyse_trading_days

    return nyse_trading_days(start, end + timedelta(days=7))


def backfill(
    tickers: list[str],
    days: int,
    *,
    index: AliasIndex,
    finnhub_key: str | None = None,
    policy: SignalPolicy | None = None,
    until: datetime | None = None,
    include_gdelt: bool = True,
    gdelt_override: bool = False,
    stats: DropStats | None = None,
) -> LiveCollectResult:
    """Finnhub company-news over the last ``days`` calendar days (+ GDELT when in range).

    Plan §4: 365 days × 40 tickers ≈ 40 requests, ~1 min at 1.1 s apiece. GDELT only serves
    the last 3 months; by default the period older than that is covered by Finnhub alone.
    ``gdelt_override`` is a deliberate escape hatch and is logged.
    """
    if days <= 0:
        raise ValueError(f"days must be positive, got {days}")
    now = until or _now_utc()
    start = (now - timedelta(days=days)).date()
    policy = policy or SignalPolicy()
    own_stats = stats if stats is not None else DropStats()
    if include_gdelt and start < now.date() - timedelta(days=90) and not gdelt_override:
        LOG.info("backfill: window starts %s, GDELT serves only 90 days; GDELT disabled", start)
        include_gdelt = False
    since_dt = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
    return collect_live(
        tickers,
        index,
        since_dt,
        until=now,
        feeds=[],  # RSS only carries the last hours; backfill is API-only
        policy=policy,
        include_finnhub=True,
        include_gdelt=include_gdelt,
        finnhub_key=finnhub_key,
        stats=own_stats,
    )


def write_daily(
    frame: pd.DataFrame,
    live_dir: str | Path,
    day: date,
    *,
    policy: SignalPolicy,
) -> Path:
    """Merge ``frame`` with the existing day file and write it atomically (idempotent)."""
    target = live_news_path(live_dir, day)
    frames = [frame] if not frame.empty else []
    if target.is_file():
        existing = read_parquet(target)
        if not existing.empty:
            frames.append(existing)
    if not frames:
        merged = _empty_news()
    else:
        merged, _ = dedupe_live(pd.concat(frames, ignore_index=True))
    merged = _finalize_news(merged, policy, _calendar(day, day))
    write_parquet_atomic(merged, target, date32=("date", "signal_date"))
    return target
