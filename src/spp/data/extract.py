"""Article extraction on top of established open-source extractors.

Primary backend is :mod:`trafilatura` (Apache-2.0, used by Common Crawl);
:mod:`newspaper4k` (MIT) is an optional secondary opinion for the title and
the body when the primary returns nothing. We do not parse HTML ourselves:
the value here is the metadata contract and the date-precision flag, which the
signal-date policy depends on.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import trafilatura

LOG = logging.getLogger(__name__)

ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
ISO_DATETIME = re.compile(r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})")


@dataclass(frozen=True)
class ExtractedArticle:
    """One article after extraction.

    ``date_precision`` is part of the contract on purpose: ``"day"`` means the
    extractor found a calendar date without a time, so the news cannot be
    placed before the 16:00 ET cutoff and must be treated as next-day.
    """

    url: str | None
    title: str | None
    text: str
    published_at: datetime | None
    date_precision: str  # "minute" | "day" | "none"
    authors: tuple[str, ...] = ()
    site: str | None = None
    method: str = "trafilatura"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def _parse_date(value: object) -> tuple[datetime | None, str]:
    if not value:
        return None, "none"
    if isinstance(value, datetime):
        stamped = value if value.tzinfo else value.replace(tzinfo=UTC)
        return stamped.astimezone(UTC), "minute"
    text = str(value).strip()
    match = ISO_DATETIME.match(text)
    if match:
        try:
            parsed = datetime.strptime(text[:16], "%Y-%m-%dT%H:%M" if "T" in text[:17] else "%Y-%m-%d %H:%M")
        except ValueError:
            return None, "none"
        return parsed.replace(tzinfo=UTC), "minute"
    if ISO_DATE.match(text):
        parsed = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC)
        return parsed, "day"
    try:
        from dateutil import parser as date_parser

        parsed = date_parser.parse(text)
    except (ImportError, ValueError, OverflowError):
        return None, "none"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    precision = "minute" if (parsed.hour or parsed.minute) else "day"
    return parsed.astimezone(UTC), precision


def _as_tuple(value: object) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(";") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(part).strip() for part in value if str(part).strip())
    return (str(value),)


def _from_trafilatura(html: str, url: str | None, **kwargs: Any) -> ExtractedArticle | None:
    try:
        document = trafilatura.bare_extraction(
            html,
            url=url,
            with_metadata=True,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - extractor is third-party and raises broadly
        LOG.warning("trafilatura failed for %s: %s", url, exc)
        return None
    if not document:
        return None
    # trafilatura >= 2.0 returns a Document; older releases return a dict.
    payload = document.as_dict() if hasattr(document, "as_dict") else dict(document)
    text = (payload.get("text") or "").strip()
    title = (payload.get("title") or "").strip() or None
    if not text and not title:
        return None
    published, precision = _parse_date(payload.get("date"))
    return ExtractedArticle(
        url=url or payload.get("url"),
        title=title,
        text=text,
        published_at=published,
        date_precision=precision,
        authors=_as_tuple(payload.get("author")),
        site=payload.get("sitename"),
        method="trafilatura",
        extra={
            "description": payload.get("description"),
            "categories": _as_tuple(payload.get("categories")),
            "tags": _as_tuple(payload.get("tags")),
        },
    )


def _from_newspaper(html: str, url: str | None) -> ExtractedArticle | None:
    try:
        from newspaper import Article  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        article = Article(url or "https://example.invalid/", input_html=html)
        article.download_state = 2
        article.parse()
    except Exception as exc:  # noqa: BLE001 - optional backend
        LOG.debug("newspaper4k failed for %s: %s", url, exc)
        return None
    text = (article.text or "").strip()
    title = (article.title or "").strip() or None
    if not text and not title:
        return None
    published, precision = _parse_date(article.publish_date)
    return ExtractedArticle(
        url=url,
        title=title,
        text=text,
        published_at=published,
        date_precision=precision,
        authors=_as_tuple(article.authors),
        site=None,
        method="newspaper4k",
    )


def extract_from_html(html: str, url: str | None = None, *, allow_fallback: bool = True) -> ExtractedArticle | None:
    """Extract title, body and publication date from already-downloaded HTML."""
    if not html or not html.strip():
        return None
    article = _from_trafilatura(html, url)
    if article is not None and (article.text or article.title):
        if article.text or not allow_fallback:
            return article
    if not allow_fallback:
        return article
    fallback = _from_newspaper(html, url)
    if fallback is None:
        return article
    if article is None:
        return fallback
    return ExtractedArticle(
        url=url,
        title=article.title or fallback.title,
        text=article.text if len(article.text) >= len(fallback.text) else fallback.text,
        published_at=article.published_at or fallback.published_at,
        date_precision=(
            article.date_precision if article.published_at else fallback.date_precision
        ),
        authors=article.authors or fallback.authors,
        site=article.site or fallback.site,
        method=f"{article.method}+{fallback.method}",
    )


def extract_from_url(url: str, *, timeout: float = 15.0) -> ExtractedArticle | None:
    """Download and extract one article. Politeness is delegated to trafilatura."""
    if not url or not url.startswith(("http://", "https://")):
        return None
    html = None
    try:
        html = trafilatura.fetch_url(url)
    except Exception as exc:  # noqa: BLE001 - network layer is third-party
        LOG.warning("fetch failed for %s: %s", url, exc)
    if not html:
        return None
    return extract_from_html(html, url)
