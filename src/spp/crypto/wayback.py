"""Wayback Machine RSS snapshot harvester — historical headlines with exact capture times.

Live RSS feeds expose only the last few hours and GDELT answers HTTP 429, so the
2023-01-01..2026-11-09 headline corpus is rebuilt from Internet Archive snapshots of the
same feeds:

* the CDX API (``collapse=timestamp:8``) lists one capture per day per feed;
* ``/web/<timestamp>id_/<url>`` returns the archived bytes without the Wayback toolbar;
* :func:`spp.crypto.news.parse_feed` turns each snapshot into rows and the capture
  timestamp becomes ``snapshot_ts`` — the exact UTC time the feed was archived.

Cost model: one CDX listing per feed-month (cheap) plus one capture fetch per archived
day. Every HTTP attempt passes through a counting wrapper, so ``requests``/``failed`` in
the summary stay honest across retries. Completed days are recorded as
``<feed>|<YYYYMMDD>`` in a JSON state file *after* their rows reach the parquet store: a
crash in between costs a refetch, which the url dedupe absorbs, never an article. Rows are
deduped on ``url`` (first snapshot wins) and rows without a parseable ``published_at``
are dropped.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

import pandas as pd
import requests

from spp.crypto.news import (
    NewsConfig,
    link_tickers,
    load_alias_index,
    load_news_config,
    parse_feed,
)
from spp.data.store import read_parquet, write_parquet_atomic
from spp.logging import get_logger

__all__ = [
    "CDX_ENDPOINT",
    "USER_AGENT",
    "WAYBACK_COLUMNS",
    "HarvestConfig",
    "feed_map",
    "fetch_capture",
    "fetch_captures",
    "harvest",
    "load_wayback",
    "parse_capture",
]

LOG = get_logger("spp.crypto.wayback")

CDX_ENDPOINT: Final[str] = "http://web.archive.org/cdx/search/cdx"
CAPTURE_PREFIX: Final[str] = "http://web.archive.org/web/"
USER_AGENT: Final[str] = "spp-crypto-research/0.1 (+research)"
HEADERS: Final[dict[str, str]] = {"User-Agent": USER_AGENT}

#: Exact column order of the parquet corpus; the news-to-bars join consumes this schema.
WAYBACK_COLUMNS: Final[tuple[str, ...]] = (
    "source",
    "title",
    "url",
    "guid",
    "published_at",
    "snapshot_ts",
    "tickers",
)

#: Wayback can be slow behind its own throttling; one attempt must not hang the budget.
_REQUEST_TIMEOUT_SECONDS: Final[float] = 60.0
#: One retry: enough for a transient 500/429, cheap enough under a request budget.
_MAX_ATTEMPTS: Final[int] = 2
_RETRY_SLEEP_SECONDS: Final[float] = 0.5
#: Cumulative counters stored beside the ``done`` day keys in the state file.
_STATE_KEYS: Final[tuple[str, ...]] = ("requests", "failed")


@dataclass(frozen=True)
class HarvestConfig:
    """Defaults for :func:`harvest` and the CLI; frozen so a run cannot mutate them."""

    max_requests: int = 1000
    max_minutes: float = 30.0
    pause_seconds: float = 0.6
    cdx_limit: int = 500
    user_agent: str = USER_AGENT
    request_timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS


# --------------------------------------------------------------------------- #
# Feed selection
# --------------------------------------------------------------------------- #
def _host_name(url: str) -> str:
    """First host label of ``url`` with ``www.`` stripped (``coindesk.com`` -> ``coindesk``)."""
    try:
        host = urlsplit(url).netloc
    except ValueError:
        return ""
    host = host.split("@")[-1].split(":")[0].lower()
    if host.startswith("www."):
        host = host[4:]
    return host.split(".")[0]


def feed_map(config: NewsConfig | None = None) -> dict[str, str]:
    """``{name: url}`` for either config shape: dict ``name -> url`` or a flat list of urls.

    ``load_news_config()`` collapses a list-shaped ``feeds`` to ``{}``, so the list branch
    also accepts a :class:`NewsConfig` constructed with a list; entries are named after
    their host (``coindesk`` for ``https://www.coindesk.com/...``) with a numeric suffix
    on collision.
    """
    cfg = config if config is not None else load_news_config()
    feeds = cfg.feeds
    if isinstance(feeds, Mapping):
        return {str(name): str(url) for name, url in feeds.items()}
    urls = [str(item) for item in feeds] if isinstance(feeds, (list, tuple)) else []
    named: dict[str, str] = {}
    for index, url in enumerate(urls):
        base = _host_name(url) or f"feed{index}"
        name = base
        suffix = 1
        while name in named:
            suffix += 1
            name = f"{base}{suffix}"
        named[name] = url
    return named


# --------------------------------------------------------------------------- #
# Range bounds
# --------------------------------------------------------------------------- #
def _parse_bound(value: str | date | datetime | pd.Timestamp, *, is_end: bool) -> pd.Timestamp:
    """Normalize a range bound to a tz-naive calendar day: ``YYYY-MM-DD`` or ``YYYYMM``.

    A ``YYYYMM`` bound expands to the first day (start) or the last day (end) of that
    month, matching the inclusive CDX ``from``/``to`` params. Naive input is kept naive;
    aware input is converted to UTC and stripped of the zone (bounds are calendar days).
    """
    if isinstance(value, str):
        text = value.strip()
        if len(text) == 6 and text.isdigit():
            month_start = pd.Timestamp(f"{text[:4]}-{text[4:]}-01")
            return month_start if not is_end else month_start + pd.offsets.MonthEnd(0)
        return pd.Timestamp(text).normalize()
    stamp = pd.Timestamp(value)
    if stamp.tz is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize()


def _month_groups(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Calendar-month slices of the inclusive ``[start, end]`` range, clipped to it."""
    groups: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor <= end:
        stop = min(cursor + pd.offsets.MonthEnd(0), end)
        groups.append((cursor, stop))
        cursor = stop + pd.Timedelta(days=1)
    return groups


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _request(
    url: str, *, params: Mapping[str, Any] | None, get: Callable[..., Any] | None
) -> Any:
    """One GET with a single retry on 429/5xx/timeouts; raise ``RuntimeError`` otherwise.

    A non-retryable status (404, 403, ...) raises immediately — the archive simply does
    not have that capture, and retrying a deterministic answer only burns the budget.
    """
    http_get = get if get is not None else requests.get
    last = "no attempt made"
    for attempt in range(_MAX_ATTEMPTS):
        if attempt:
            time.sleep(_RETRY_SLEEP_SECONDS)
        try:
            response = http_get(
                url,
                params=dict(params) if params else None,
                headers=HEADERS,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # network failure must never propagate unlabelled
            last = f"request error: {exc}"
            continue
        status = int(getattr(response, "status_code", 0) or 0)
        if status == 200:
            return response
        if status == 429 or status >= 500:
            last = f"HTTP {status}"
            continue
        raise RuntimeError(f"wayback GET {url}: HTTP {status}")
    raise RuntimeError(f"wayback GET {url} failed after {_MAX_ATTEMPTS} attempt(s): {last}")


def _json_payload(response: Any) -> Any:
    """Parsed JSON body, or ``None`` when a 200 body cannot be parsed (never raises)."""
    try:
        return response.json()
    except (ValueError, AttributeError, TypeError):
        pass
    text = str(getattr(response, "text", "") or "")
    try:
        return json.loads(text)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# CDX listing
# --------------------------------------------------------------------------- #
def _parse_cdx(response: Any) -> list[dict[str, str]]:
    """``output=json`` CDX body -> ``[{timestamp, original}, ...]``; ``[]`` on anything odd.

    The first row is the header (``urlkey, timestamp, original, ...``); an empty array, a
    header-only array or an HTML error page all mean "no captures" and yield ``[]``.
    """
    payload = _json_payload(response)
    if not isinstance(payload, list) or len(payload) < 2:
        return []
    header = payload[0]
    if not isinstance(header, list):
        return []
    try:
        ts_index = header.index("timestamp")
        original_index = header.index("original")
    except ValueError:
        return []
    last_index = max(ts_index, original_index)
    captures: list[dict[str, str]] = []
    for row in payload[1:]:
        if not isinstance(row, (list, tuple)) or len(row) <= last_index:
            continue
        timestamp = str(row[ts_index] or "")
        original = str(row[original_index] or "")
        if len(timestamp) >= 8 and original:
            captures.append({"timestamp": timestamp, "original": original})
    return captures


def fetch_captures(
    feed_url: str,
    start: str | date | datetime | pd.Timestamp,
    end: str | date | datetime | pd.Timestamp,
    *,
    limit: int = 500,
    get: Callable[..., Any] | None = None,
) -> list[dict[str, str]]:
    """One CDX listing for ``[start, end]`` (inclusive; ``YYYY-MM-DD`` or ``YYYYMM``).

    ``collapse=timestamp:8`` keeps one capture per calendar day. An empty array, an error
    page or a persistent HTTP failure all yield ``[]`` — a feed the archive never stored
    is a legitimate answer, not an exception. Each returned dict has at least ``timestamp``
    (14-digit UTC stamp) and ``original`` (the archived URL).
    """
    params = {
        "url": feed_url,
        "from": _parse_bound(start, is_end=False).strftime("%Y%m%d"),
        "to": _parse_bound(end, is_end=True).strftime("%Y%m%d"),
        "output": "json",
        "filter": "statuscode:200",
        "collapse": "timestamp:8",
        "limit": int(limit),
    }
    try:
        response = _request(CDX_ENDPOINT, params=params, get=get)
    except RuntimeError as exc:
        LOG.warning("fetch_captures %s: %s", feed_url, exc)
        return []
    return _parse_cdx(response)


# --------------------------------------------------------------------------- #
# Capture fetch + parse
# --------------------------------------------------------------------------- #
def fetch_capture(
    timestamp: str, url: str, *, get: Callable[..., Any] | None = None
) -> str:
    """The archived bytes of ``url`` at ``timestamp`` as text; raise on a non-200.

    The ``id_`` suffix requests the raw capture without the Wayback toolbar injected into
    the payload. 500/429 responses are retried once before the ``RuntimeError``.
    """
    capture_url = f"{CAPTURE_PREFIX}{timestamp}id_/{url}"
    response = _request(capture_url, params=None, get=get)
    content = getattr(response, "content", None)
    if isinstance(content, (bytes, bytearray)):
        return bytes(content).decode("utf-8", errors="replace")
    return str(getattr(response, "text", "") or "")


def parse_capture(payload: object, source: str) -> list[dict[str, Any]]:
    """Thin wrapper over :func:`spp.crypto.news.parse_feed` that never raises.

    Adds a ``snapshot_ts`` placeholder (:data:`pd.NaT`) to every row — :func:`harvest`
    overwrites it with the capture's UTC time — and returns ``[]`` for garbage payloads
    (None, HTML error pages, truncated XML) instead of propagating the parse error.
    """
    try:
        rows = parse_feed(payload, source)  # type: ignore[arg-type]
    except Exception as exc:  # a dead snapshot must not take the whole harvest down
        LOG.warning("parse_capture %s: %s", source, exc)
        return []
    for row in rows:
        row["snapshot_ts"] = pd.NaT
    return rows


def _snapshot_stamp(timestamp: str) -> pd.Timestamp:
    """Wayback ``YYYYMMDDhhmmss`` (UTC) -> tz-aware timestamp; date-only and garbage tolerated."""
    text = str(timestamp).strip()
    for fmt, size in (("%Y%m%d%H%M%S", 14), ("%Y%m%d", 8)):
        if len(text) >= size:
            try:
                parsed = datetime.strptime(text[:size], fmt)
            except ValueError:
                continue
            return pd.Timestamp(parsed.replace(tzinfo=UTC))
    return pd.NaT


# --------------------------------------------------------------------------- #
# Store helpers
# --------------------------------------------------------------------------- #
def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _nap(pause: float) -> None:
    if pause > 0:
        time.sleep(pause)


def _empty_frame() -> pd.DataFrame:
    """Corpus-shaped empty frame (exact :data:`WAYBACK_COLUMNS` order)."""
    return pd.DataFrame(
        {
            "source": pd.Series(dtype="str"),
            "title": pd.Series(dtype="str"),
            "url": pd.Series(dtype="str"),
            "guid": pd.Series(dtype="str"),
            "published_at": pd.Series(dtype="datetime64[us, UTC]"),
            "snapshot_ts": pd.Series(dtype="datetime64[us, UTC]"),
            "tickers": pd.Series(dtype="object"),
        }
    )[list(WAYBACK_COLUMNS)]


def load_wayback(path: Path | str) -> pd.DataFrame:
    """The stored corpus, sorted by ``published_at`` (stable: ties keep arrival order)."""
    target = Path(path)
    if not target.is_file():
        return _empty_frame()
    frame = read_parquet(target)
    return frame.sort_values("published_at", kind="stable").reset_index(drop=True)


def _append_rows(out_path: Path, rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Append ``rows`` to the corpus; return ``(kept, new)``.

    ``kept`` is the number of rows that survived the NaT and in-run url filters; ``new``
    is how many were actually added beyond what the store already held (``kept - new``
    are duplicates of rows from an earlier run). Dedupe is first-wins, so the earliest
    snapshot of a headline is the one that stays.
    """
    frame = pd.DataFrame(rows, columns=list(WAYBACK_COLUMNS))
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_ts"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["published_at"])
    frame = frame.drop_duplicates(subset="url", keep="first")
    kept = len(frame)
    if kept == 0:
        return 0, 0
    prior = read_parquet(out_path) if out_path.is_file() else _empty_frame()
    before = len(prior)
    combined = pd.concat([prior, frame], ignore_index=True) if before else frame
    combined = combined.drop_duplicates(subset="url", keep="first")
    new = len(combined) - before
    if new > 0:
        write_parquet_atomic(combined[list(WAYBACK_COLUMNS)], out_path)
    return kept, new


# --------------------------------------------------------------------------- #
# State file
# --------------------------------------------------------------------------- #
def _load_state(path: Path) -> tuple[set[str], dict[str, int]]:
    """``(done day keys, cumulative counters)``; a missing/corrupt file starts fresh —
    url dedupe makes a re-harvest correct, only wasteful, so state damage never blocks."""
    if not path.is_file():
        return set(), {key: 0 for key in _STATE_KEYS}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        LOG.warning("state %s unreadable (%s); starting fresh", path, exc)
        return set(), {key: 0 for key in _STATE_KEYS}
    if not isinstance(raw, dict):
        LOG.warning("state %s is not a JSON object; starting fresh", path)
        return set(), {key: 0 for key in _STATE_KEYS}
    done_raw = raw.get("done")
    done = {str(item) for item in done_raw} if isinstance(done_raw, list) else set()
    counters = {key: int(raw.get(key, 0) or 0) for key in _STATE_KEYS}
    return done, counters


def _write_state(path: Path, done: set[str], counters: dict[str, int]) -> None:
    """Atomic rewrite (``.tmp`` then ``os.replace``) so a kill cannot leave half a JSON file."""
    payload: dict[str, Any] = {"done": sorted(done)}
    payload.update(counters)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():  # pragma: no cover - only on failure paths
            try:
                tmp.unlink()
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# Harvest
# --------------------------------------------------------------------------- #
def _summary(
    requests: int = 0,
    failed: int = 0,
    captures: int = 0,
    kept: int = 0,
    new: int = 0,
    feeds: int = 0,
    seconds: float = 0.0,
    stopped: str = "done",
) -> dict[str, Any]:
    return {
        "requests": requests,
        "failed": failed,
        "captures": captures,
        "kept": kept,
        "new": new,
        "feeds": feeds,
        "seconds": seconds,
        "stopped": stopped,
    }


def harvest(
    feeds: Mapping[str, str],
    start: str | date | datetime | pd.Timestamp,
    end: str | date | datetime | pd.Timestamp,
    out_path: Path | str,
    state_path: Path | str,
    *,
    max_requests: int | None = 1000,
    max_minutes: float | None = 30,
    pause: float = 0.6,
    get: Callable[..., Any] | None = None,
    clock: Callable[[], pd.Timestamp] | None = None,
) -> dict[str, Any]:
    """Harvest the inclusive ``[start, end]`` range (``YYYY-MM-DD`` or ``YYYYMM``) per feed.

    One CDX listing per feed-month plus one capture fetch per archived day; every HTTP
    attempt counts toward ``max_requests`` and wall time toward ``max_minutes``. Completed
    days (``<feed>|<YYYYMMDD>``) are checkpointed to ``state_path`` *after* their rows
    reach ``out_path``, so a second run resumes where the first stopped and a crash in
    between only costs a refetch (url dedupe absorbs it). Rows with an unparseable
    ``published_at`` are dropped; remaining rows are deduped on ``url``, first snapshot
    wins, and get ``snapshot_ts`` plus ticker attribution from the news config.

    All counters are for *this run*: ``requests`` HTTP attempts (retries included),
    ``failed`` attempts that were not HTTP 200, ``captures`` snapshots fetched OK,
    ``kept`` rows surviving the NaT/url filters, ``new`` rows actually appended (``kept -
    new`` duplicates rows an earlier run already stored), ``feeds`` requested, ``seconds``
    of wall time, and ``stopped`` is ``"budget"`` or ``"done"``.
    """
    feed_items = {str(name): str(url) for name, url in dict(feeds).items()}
    out = Path(out_path)
    state = Path(state_path)

    start_day = _parse_bound(start, is_end=False)
    end_day = _parse_bound(end, is_end=True)

    now = clock if clock is not None else _utc_now
    started_at = now()
    done, base_counters = _load_state(state)
    counts = {"requests": 0, "failed": 0}
    base_get = get if get is not None else requests.get

    def counting_get(url: str, **kwargs: Any) -> Any:
        """Every attempt (including a retry) counts once; every non-200 counts as failed."""
        counts["requests"] += 1
        try:
            response = base_get(url, **kwargs)
        except Exception:
            counts["failed"] += 1
            raise
        if int(getattr(response, "status_code", 0) or 0) != 200:
            counts["failed"] += 1
        return response

    def over_budget() -> bool:
        if max_requests is not None and counts["requests"] >= max_requests:
            return True
        if max_minutes is None:
            return False
        elapsed = (now() - started_at).total_seconds()
        return elapsed >= max_minutes * 60.0

    def checkpoint() -> None:
        """Persist done keys plus cumulative counters (prior runs + this one)."""
        cumulative = {key: base_counters[key] + counts[key] for key in _STATE_KEYS}
        _write_state(state, done, cumulative)

    if not feed_items or start_day > end_day:
        return _summary(feeds=len(feed_items))

    news_config = load_news_config()
    alias_index = load_alias_index(news_config)
    kept = new = captures_done = 0
    stopped = "done"

    for name, url in feed_items.items():
        if stopped == "budget":
            break
        for group_start, group_end in _month_groups(start_day, end_day):
            if over_budget():
                stopped = "budget"
                break
            captures = fetch_captures(url, group_start, group_end, get=counting_get)
            _nap(pause)
            for capture in captures:
                timestamp = str(capture["timestamp"])
                day_key = f"{name}|{timestamp[:8]}"
                if day_key in done:
                    continue  # resume: a finished day costs no request
                if over_budget():
                    stopped = "budget"
                    break
                try:
                    payload = fetch_capture(timestamp, str(capture["original"]), get=counting_get)
                except RuntimeError as exc:
                    LOG.warning("harvest %s %s: %s", name, day_key, exc)
                    _nap(pause)
                    continue  # day stays unfinished; the next run retries exactly this
                _nap(pause)
                captures_done += 1
                rows = []
                for row in parse_capture(payload, name):
                    if pd.isna(row["published_at"]):
                        continue
                    row["snapshot_ts"] = _snapshot_stamp(timestamp)
                    row["tickers"] = link_tickers(row["title"], alias_index, news_config)
                    rows.append(row)
                if rows:
                    run_kept, run_new = _append_rows(out, rows)
                    kept += run_kept
                    new += run_new
                done.add(day_key)
                checkpoint()
            if stopped == "budget":
                break

    if counts["requests"]:
        checkpoint()
    seconds = round((now() - started_at).total_seconds(), 3)
    LOG.info(
        "harvest: requests=%d failed=%d captures=%d kept=%d new=%d stopped=%s in %.1fs",
        counts["requests"], counts["failed"], captures_done, kept, new, stopped, seconds,
    )
    return _summary(
        requests=counts["requests"],
        failed=counts["failed"],
        captures=captures_done,
        kept=kept,
        new=new,
        feeds=len(feed_items),
        seconds=seconds,
        stopped=stopped,
    )
