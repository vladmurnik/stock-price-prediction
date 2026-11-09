"""T0 + T3 — keyless crypto RSS recorder and headline ticker attribution.

The recorder polls the public RSS/Atom feeds declared in ``configs/crypto_dataset.yaml:news``,
stamps ``first_seen_at`` the moment an item is observed and appends it to a parquet store.
``first_seen_at - published_at`` is the publication-to-detection latency that the GDELT archive
(T2) will later be measured against, so the store is append-only: prior rows are never dropped
or rewritten, and each poll only adds rows it has not seen before (key = ``guid``, else ``url``)
whose ``published_at`` is fresher than ``news.dedupe_minutes``.

Attribution (:func:`load_alias_index` / :func:`link_tickers`) is deliberately separate from the
equity matcher in :mod:`spp.data.normalize`: crypto aliases are ordinary English words
("link", "op", "ada", "ton"), so weak aliases only count when the headline also carries a
``crypto_context`` word, and everything matches on word boundaries ("sol" never matches inside
"solution").
"""

from __future__ import annotations

import html
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import pandas as pd
import requests
import yaml

from spp.data.store import read_parquet, write_parquet_atomic
from spp.logging import get_logger

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "RECORD_COLUMNS",
    "NewsConfig",
    "Recorder",
    "fetch_feed",
    "link_tickers",
    "load_alias_index",
    "load_news_config",
    "parse_feed",
]

LOG = get_logger("spp.crypto.news")

#: Repo root resolved from this file (``src/spp/crypto/news.py``), never from the cwd.
DEFAULT_CONFIG_PATH: Final[Path] = Path(__file__).resolve().parents[3] / "configs" / (
    "crypto_dataset.yaml"
)

#: Sleep between the two attempts of :func:`fetch_feed`; patched to 0 in tests.
_RETRY_SLEEP_SECONDS: Final[float] = 0.5

#: Keys of the dicts returned by :func:`parse_feed` (before the recorder adds its own).
_PARSE_COLUMNS: Final[tuple[str, ...]] = (
    "source",
    "title",
    "url",
    "guid",
    "published_at",
    "summary",
)

#: Exact column order of the parquet store; T1/T5 consume this schema unchanged.
RECORD_COLUMNS: Final[tuple[str, ...]] = (
    "source",
    "title",
    "url",
    "guid",
    "published_at",
    "first_seen_at",
    "latency_seconds",
    "negative_latency",
    "tickers",
    "event_kind",
)

#: Date fields tried in order: RSS ``pubDate``, Atom ``published``/``updated``, DC ``date``.
_DATE_KEYS: Final[tuple[str, ...]] = ("pubDate", "published", "updated", "date")
_SUMMARY_KEYS: Final[tuple[str, ...]] = ("description", "summary", "encoded", "content")

_HTML_TAG = re.compile(r"<[^>]{0,40}>")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class NewsConfig:
    """``news`` section of ``configs/crypto_dataset.yaml`` plus the attribution tables."""

    feeds: dict[str, str]
    poll_seconds: float
    request_timeout_seconds: float
    user_agent: str
    dedupe_minutes: float
    aliases: dict[str, list[str]]
    weak_aliases: dict[str, list[str]]
    crypto_context: list[str]


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def load_news_config(path: Path | str | None = None) -> NewsConfig:
    """Read the ``news`` section (feeds, polling, user agent) and the attribution tables.

    ``path=None`` resolves ``configs/crypto_dataset.yaml`` from the package root, so the
    loader works from any working directory. Frozen-config safety: every optional key uses
    ``.get(key, default)`` and never rewrites the YAML.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"news config not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {cfg_path}")

    news = raw.get("news")
    news = news if isinstance(news, dict) else {}
    feeds_raw = news.get("feeds")
    feeds = (
        {str(name): str(url) for name, url in feeds_raw.items()}
        if isinstance(feeds_raw, dict)
        else {}
    )

    aliases_raw = raw.get("aliases")
    weak_raw = raw.get("weak_aliases")
    return NewsConfig(
        feeds=feeds,
        poll_seconds=float(news.get("poll_seconds", 30.0)),
        request_timeout_seconds=float(news.get("request_timeout_seconds", 20.0)),
        user_agent=str(news.get("user_agent", "spp-crypto-recorder/0.1 (+research)")),
        dedupe_minutes=float(news.get("dedupe_minutes", 60.0)),
        aliases=(
            {str(t): _as_str_list(names) for t, names in aliases_raw.items()}
            if isinstance(aliases_raw, dict)
            else {}
        ),
        weak_aliases=(
            {str(t): _as_str_list(names) for t, names in weak_raw.items()}
            if isinstance(weak_raw, dict)
            else {}
        ),
        crypto_context=_as_str_list(raw.get("crypto_context")),
    )


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _local(tag: str) -> str:
    """Namespace-tolerant local name: ``{http://...}date`` -> ``date``."""
    return tag.rsplit("}", 1)[-1]


def _clean_text(raw: str | None) -> str:
    """Unescape HTML entities, drop tags, collapse whitespace."""
    if not raw:
        return ""
    text = html.unescape(raw)
    text = _HTML_TAG.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def _parse_date(raw: str | None) -> pd.Timestamp:
    """RFC 822 (``pubDate``) then ISO 8601 (Atom / ``dc:date``); naive input means UTC."""
    if not raw or not raw.strip():
        return pd.NaT
    text = raw.strip()
    stamp: datetime | None = None
    try:
        stamp = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        stamp = None
    if stamp is None:
        try:
            iso = f"{text[:-1]}+00:00" if text.endswith("Z") else text
            stamp = datetime.fromisoformat(iso)
        except ValueError:
            LOG.debug("unparseable feed date: %r", text)
            return pd.NaT
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return pd.Timestamp(stamp.astimezone(UTC))


def parse_feed(payload: str | bytes, source: str) -> list[dict[str, Any]]:
    """Parse an RSS 2.0 (``<item>``) or Atom (``<entry>``) payload into row dicts.

    Each dict has ``source, title, url, guid, published_at, summary``. ``published_at`` is a
    tz-aware UTC :class:`pd.Timestamp`, or ``pd.NaT`` when no usable date exists — the item is
    kept either way. An unparseable payload logs a warning and yields ``[]``.
    """
    try:
        root = ET.fromstring(payload.strip())
    except ET.ParseError as exc:
        LOG.warning("feed %s: unparseable XML (%s)", source, exc)
        return []

    rows: list[dict[str, Any]] = []
    for node in root.iter():
        if _local(node.tag) not in {"item", "entry"}:
            continue
        values: dict[str, str] = {}
        link = ""
        for child in node:
            name = _local(child.tag)
            text = (child.text or "").strip() or (child.attrib.get("href") or "").strip()
            if name == "link":
                link = link or text
            elif name not in values:
                values[name] = text

        published = pd.NaT
        for key in _DATE_KEYS:
            candidate = _parse_date(values.get(key))
            if not pd.isna(candidate):
                published = candidate
                break

        guid = values.get("guid") or values.get("id") or ""
        summary = next((values[k] for k in _SUMMARY_KEYS if values.get(k)), "")
        rows.append(
            {
                "source": source,
                "title": _clean_text(values.get("title")),
                "url": link,
                "guid": guid,
                "published_at": published,
                "summary": _clean_text(summary),
            }
        )
    return rows


def fetch_feed(url: str, config: NewsConfig) -> list[dict[str, Any]]:
    """One GET with the configured user agent/timeout and one short retry; never raises.

    A dead feed logs a warning and returns ``[]`` so one broken endpoint cannot take the
    recorder down.
    """
    headers = {"User-Agent": config.user_agent}
    last_error: Exception | None = None
    for attempt in range(2):
        if attempt:
            time.sleep(_RETRY_SLEEP_SECONDS)
        try:
            response = requests.get(
                url, headers=headers, timeout=config.request_timeout_seconds
            )
            response.raise_for_status()
        except Exception as exc:  # network/HTTP failure must never propagate
            last_error = exc
            continue
        return parse_feed(response.content, source=url)
    LOG.warning("feed unreachable after 2 attempts: %s (%s)", url, last_error)
    return []


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=512)
def _word_pattern(alias: str) -> re.Pattern[str]:
    """Case-insensitive whole-word matcher: ``sol`` never matches inside ``solution``."""
    return re.compile(rf"(?<![0-9A-Za-z]){re.escape(alias)}(?![0-9A-Za-z])", re.IGNORECASE)


def load_alias_index(config: NewsConfig) -> dict[str, list[str]]:
    """Lowercased alias -> tickers, strong aliases first, config order preserved."""
    index: dict[str, list[str]] = {}
    for mapping in (config.aliases, config.weak_aliases):
        for ticker, names in mapping.items():
            for name in names:
                key = name.strip().casefold()
                if not key:
                    continue
                tickers = index.setdefault(key, [])
                if ticker not in tickers:
                    tickers.append(ticker)
    return index


def link_tickers(text: str, index: dict[str, list[str]], config: NewsConfig) -> list[str]:
    """Tickers mentioned in ``text``; weak aliases need a ``crypto_context`` word.

    Matching is case-insensitive on word boundaries; weak aliases are skipped unless the same
    text contains a ``crypto_context`` word (also whole-word). Result is sorted and unique.
    """
    if not text or not index:
        return []
    weak = {
        alias.strip().casefold()
        for names in config.weak_aliases.values()
        for alias in names
        if alias.strip()
    }
    has_context = any(
        _word_pattern(word.strip().casefold()).search(text)
        for word in config.crypto_context
        if word.strip()
    )
    matched: set[str] = set()
    for alias, tickers in index.items():
        if alias in weak and not has_context:
            continue
        if _word_pattern(alias).search(text):
            matched.update(tickers)
    return sorted(matched)


# --------------------------------------------------------------------------- #
# Recorder
# --------------------------------------------------------------------------- #
def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _dedupe_key(guid: object, url: object) -> str:
    """Observed-item key: ``guid`` when present, else ``url``."""
    for value in (guid, url):
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _empty_frame() -> pd.DataFrame:
    """Store-shaped empty frame (exact :data:`RECORD_COLUMNS` order)."""
    return pd.DataFrame(
        {
            "source": pd.Series(dtype="str"),
            "title": pd.Series(dtype="str"),
            "url": pd.Series(dtype="str"),
            "guid": pd.Series(dtype="str"),
            "published_at": pd.Series(dtype="datetime64[us, UTC]"),
            "first_seen_at": pd.Series(dtype="datetime64[us, UTC]"),
            "latency_seconds": pd.Series(dtype="float64"),
            "negative_latency": pd.Series(dtype="bool"),
            "tickers": pd.Series(dtype="object"),
            "event_kind": pd.Series(dtype="str"),
        }
    )[list(RECORD_COLUMNS)]


class Recorder:
    """Append-only RSS recorder with publication-to-detection latency."""

    def __init__(
        self,
        config: NewsConfig,
        store: Path,
        clock: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        self.config = config
        self.store = Path(store)
        self._clock: Callable[[], pd.Timestamp] = clock if clock is not None else _utc_now
        self._index = load_alias_index(config)
        #: Items parsed from feeds across all polls (before dedupe) — drives the CLI exit code.
        self.fetched_total = 0

    def _read_store(self) -> pd.DataFrame:
        if not self.store.is_file():
            return _empty_frame()
        return read_parquet(self.store)

    def _ensure_store(self) -> None:
        """Create the store with the exact schema when it does not exist yet."""
        if not self.store.is_file():
            write_parquet_atomic(_empty_frame(), self.store)

    def _append_store(self, rows: pd.DataFrame) -> None:
        prior = self._read_store()
        combined = pd.concat([prior, rows], ignore_index=True) if len(prior) else rows
        write_parquet_atomic(combined[list(RECORD_COLUMNS)], self.store)

    def poll_once(self) -> pd.DataFrame:
        """Fetch every configured feed once; append unseen fresh rows; return only the new rows."""
        items: list[dict[str, Any]] = []
        for name, url in self.config.feeds.items():
            rows = fetch_feed(url, self.config)
            for row in rows:
                row["source"] = name
            items.extend(rows)
        self.fetched_total += len(items)
        first_seen = self._clock()

        if not items:
            self._ensure_store()
            return _empty_frame()

        frame = pd.DataFrame(items, columns=_PARSE_COLUMNS)
        frame["published_at"] = pd.to_datetime(
            frame["published_at"], utc=True, errors="coerce"
        )
        frame["_key"] = [
            _dedupe_key(guid, url)
            for guid, url in zip(frame["guid"], frame["url"], strict=True)
        ]
        # Same item observed twice in one poll (repeated guid across feeds) keeps the first row.
        frame = frame.drop_duplicates(subset="_key", keep="first")

        # Freshness: drop items older than dedupe_minutes; NaT dates survive (key dedupe
        # handles their re-observation on the next poll).
        age = (first_seen - frame["published_at"]).dt.total_seconds()
        frame = frame.loc[~(age > self.config.dedupe_minutes * 60.0)]

        prior = self._read_store()
        seen = {
            _dedupe_key(guid, url)
            for guid, url in zip(prior["guid"], prior["url"], strict=True)
        }
        fresh = frame.loc[[key not in seen for key in frame["_key"]]]

        if fresh.empty:
            self._ensure_store()
            return _empty_frame()

        fresh["first_seen_at"] = first_seen
        latency = (first_seen - fresh["published_at"]).dt.total_seconds()
        fresh["latency_seconds"] = latency.clip(lower=0.0)
        fresh["negative_latency"] = (latency < 0).astype(bool)
        fresh["tickers"] = [
            link_tickers(
                "" if pd.isna(title) else str(title), self._index, self.config
            )
            for title in fresh["title"]
        ]
        # Structural kinds (listing/unlock/...) arrive with the calendar; RSS rows carry none.
        fresh["event_kind"] = ""
        new_rows = fresh[list(RECORD_COLUMNS)].reset_index(drop=True)

        self._append_store(new_rows)
        LOG.info(
            "poll: fetched=%d new=%d store=%s", len(items), len(new_rows), self.store
        )
        return new_rows

    def run_recorder(
        self, duration_seconds: float, sleep: Callable[[float], None] | None = None
    ) -> int:
        """Poll until the clock passes the deadline; return the number of stored items.

        The deadline and the pauses both go through the injected ``clock``/``sleep``, so the
        loop is fully testable without waiting in real time.
        """
        sleep_fn = sleep if sleep is not None else time.sleep
        deadline = self._clock() + pd.Timedelta(seconds=duration_seconds)
        stored = 0
        while True:
            stored += len(self.poll_once())
            remaining = (deadline - self._clock()).total_seconds()
            if remaining <= 0:
                break
            sleep_fn(min(self.config.poll_seconds, remaining))
        return stored
