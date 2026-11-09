"""T2 — resumable, time-budgeted GDELT DOC 2.0 crawler for the historical crypto archive.

RSS gives no history and FNSPID ends in 2023, so the historical layer is built from the GDELT
DOC 2.0 artlist API with minute-precision ``seendate`` stamps. Rate limiting is the dominant
failure mode (HTTP 429 arrives often, sometimes on the very first request), so the crawler is
built around three ideas:

* :func:`fetch_window` retries 429/5xx/timeouts with the configured backoff and never raises
  on a bad body — a run that gives up loudly is worth more than one that dies quietly.
* :func:`crawl` records every completed ``asset|window`` task in a JSON state file and rewrites
  it after each window, so a kill (or a Ctrl-C during a backoff sleep) never loses progress.
* Saturation (``len(raw) >= max_records``) halves that asset's window exactly once, because a
  250-record ``datedesc`` page is a truncated sample of a busy two-hour slice.

Counter semantics: ``requests`` counts :func:`fetch_window` calls (internal retry attempts are
not counted), ``failed`` counts calls that returned zero rows — with the frozen
``(rows, saturated)`` return shape a rate-limited window and a genuinely empty window are
indistinguishable, and for these query terms an empty two-hour slice is rare. ``saturated``
counts calls that hit the record cap.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd
import requests
import yaml

from spp.data.store import read_parquet, write_parquet_atomic
from spp.logging import get_logger

__all__ = [
    "ARCHIVE_COLUMNS",
    "DEFAULT_CONFIG_PATH",
    "ArchiveConfig",
    "crawl",
    "fetch_window",
    "iter_windows",
    "load_archive",
    "load_archive_config",
    "parse_articles",
]

LOG = get_logger("spp.crypto.archive")

#: Repo root resolved from this file (``src/spp/crypto/archive.py``), never from the cwd.
DEFAULT_CONFIG_PATH: Final[Path] = Path(__file__).resolve().parents[3] / "configs" / (
    "crypto_dataset.yaml"
)

#: Column order of the parquet archive; T5 consumes this schema unchanged.
ARCHIVE_COLUMNS: Final[tuple[str, ...]] = (
    "asset",
    "query",
    "title",
    "url",
    "domain",
    "published_at",
    "source",
)

#: GDELT can be slow behind its own throttling; one attempt must not hang the budget.
_REQUEST_TIMEOUT_SECONDS: Final[float] = 60.0

#: Defaults used when the frozen YAML omits the ``archive`` keys (never rewrites the YAML).
_DEFAULT_ENDPOINT: Final[str] = "https://api.gdeltproject.org/api/v2/doc/doc"
_DEFAULT_WINDOW_HOURS: Final[int] = 2
_DEFAULT_MAX_RECORDS: Final[int] = 250
_DEFAULT_MAX_RETRIES: Final[int] = 5
_DEFAULT_BACKOFF: Final[tuple[int, ...]] = (15, 30, 60, 120, 300)

_SEENDATE_FORMAT: Final[str] = "%Y%m%dT%H%M%SZ"


@dataclass(frozen=True)
class ArchiveConfig:
    """``archive`` section of ``configs/crypto_dataset.yaml``: endpoint, windows, retry policy."""

    endpoint: str
    window_hours: int
    max_records: int
    max_retries: int
    backoff_seconds: tuple[int, ...]
    terms: dict[str, str]  # asset -> GDELT query string


def load_archive_config(path: Path | str | None = None) -> ArchiveConfig:
    """Read the ``archive`` section; ``path=None`` resolves the frozen YAML from the package root.

    Every optional key uses ``.get(key, default)`` (the file is frozen at version 1 and is never
    rewritten). ``query_terms`` is a flat list in the YAML, so it maps asset -> query with the
    term used verbatim on both sides.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"archive config not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {cfg_path}")

    section = raw.get("archive")
    section = section if isinstance(section, dict) else {}

    backoff_raw = section.get("backoff_seconds")
    if isinstance(backoff_raw, list) and backoff_raw:
        backoff = tuple(int(item) for item in backoff_raw)
    else:
        backoff = _DEFAULT_BACKOFF

    terms_raw = section.get("query_terms")
    terms = (
        {str(term): str(term) for term in terms_raw}
        if isinstance(terms_raw, list)
        else {}
    )

    return ArchiveConfig(
        endpoint=str(section.get("endpoint", _DEFAULT_ENDPOINT)),
        window_hours=int(section.get("window_hours", _DEFAULT_WINDOW_HOURS)),
        max_records=int(section.get("max_records", _DEFAULT_MAX_RECORDS)),
        max_retries=int(section.get("max_retries", _DEFAULT_MAX_RETRIES)),
        backoff_seconds=backoff,
        terms=terms,
    )


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
def _as_utc(value: pd.Timestamp) -> pd.Timestamp:
    """Naive input is interpreted as UTC; aware input is converted to UTC."""
    stamp = pd.Timestamp(value)
    if stamp.tz is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def iter_windows(
    start: pd.Timestamp, end: pd.Timestamp, hours: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Half-open ``[start, end)`` windows of ``hours`` hours: gapless, last one clipped to ``end``.

    Both bounds are normalized to tz-aware UTC first, so callers may pass naive timestamps
    (interpreted as UTC). ``start >= end`` yields no windows; a non-positive ``hours`` is a
    caller error.
    """
    if hours <= 0:
        raise ValueError(f"window hours must be positive, got {hours}")
    begin, stop = _as_utc(start), _as_utc(end)
    if begin >= stop:
        return []
    step = pd.Timedelta(hours=hours)
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = begin
    while cursor < stop:
        window_end = min(cursor + step, stop)
        windows.append((cursor, window_end))
        cursor = window_end
    return windows


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _parse_seendate(raw: object) -> pd.Timestamp | None:
    """GDELT ``seendate`` (``20240501T123000Z``) -> tz-aware UTC timestamp; ``None`` when bad."""
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.strptime(raw.strip(), _SEENDATE_FORMAT)
    except ValueError:
        return None
    return pd.Timestamp(parsed.replace(tzinfo=UTC))


def parse_articles(payload: dict | str, asset: str, query: str) -> list[dict[str, Any]]:
    """Normalize one artlist payload into archive rows.

    Each row carries ``asset, query, title, url, domain, published_at, source`` with
    ``published_at`` a tz-aware UTC :class:`pd.Timestamp` and ``source="gdelt"``. Tolerates a
    JSON string body, a missing ``articles`` key and articles whose ``seendate`` does not parse
    (those rows are dropped and counted in a warning, never raised).
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            LOG.debug("parse_articles %s: body is not JSON", asset)
            return []
    if not isinstance(payload, dict):
        return []
    articles = payload.get("articles")
    if not isinstance(articles, list):
        return []

    rows: list[dict[str, Any]] = []
    dropped = 0
    for article in articles:
        if not isinstance(article, dict):
            dropped += 1
            continue
        published = _parse_seendate(article.get("seendate"))
        if published is None:
            dropped += 1
            continue
        rows.append(
            {
                "asset": asset,
                "query": query,
                "title": str(article.get("title") or ""),
                "url": str(article.get("url") or ""),
                "domain": str(article.get("domain") or ""),
                "published_at": published,
                "source": "gdelt",
            }
        )
    if dropped:
        LOG.warning("parse_articles %s: dropped %d article(s) with a bad seendate", asset, dropped)
    return rows


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def _gdelt_stamp(value: pd.Timestamp) -> str:
    """``YYYYMMDDHHMMSS`` in UTC, the format the ``startdatetime``/``enddatetime`` params want."""
    return _as_utc(value).strftime("%Y%m%d%H%M%S")


def _response_payload(response: Any) -> Any:
    """Parsed JSON body, or ``None`` when a 200 body cannot be parsed as JSON (never raises)."""
    try:
        payload = response.json()
    except (ValueError, AttributeError, TypeError, requests.RequestException):
        text = str(getattr(response, "text", "") or "")
        try:
            payload = json.loads(text)
        except ValueError:
            return None
    return payload


def fetch_window(
    config: ArchiveConfig,
    query: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    get: Callable[..., Any] | None = None,
) -> tuple[list[dict], bool]:
    """One artlist request for ``[start, end)``; returns ``(rows, saturated)``.

    ``get`` is the injectable HTTP callable (default ``requests.get``); ``saturated`` means the
    raw article list reached ``config.max_records`` and the window should be halved. Retries
    429/5xx/timeouts up to ``config.max_retries`` attempts, sleeping ``backoff_seconds`` before
    each retry, then logs a warning and returns ``([], False)``. A 200 with a non-JSON body is
    reported the same way instead of raising.
    """
    http_get = get if get is not None else requests.get
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": int(config.max_records),
        "format": "json",
        "sort": "datedesc",
        "startdatetime": _gdelt_stamp(start),
        "enddatetime": _gdelt_stamp(end),
    }
    attempts = max(1, int(config.max_retries))
    backoff = config.backoff_seconds or (0,)
    last_error = "no attempt made"
    for attempt in range(attempts):
        if attempt:
            time.sleep(backoff[min(attempt - 1, len(backoff) - 1)])
        try:
            response = http_get(
                config.endpoint, params=params, timeout=_REQUEST_TIMEOUT_SECONDS
            )
        except Exception as exc:  # network failure must never propagate out of the crawler
            last_error = f"request error: {exc}"
            continue
        status = int(getattr(response, "status_code", 0) or 0)
        if status == 429 or status >= 500:
            last_error = f"HTTP {status}"
            continue
        if status != 200:
            LOG.warning(
                "fetch_window %s..%s: HTTP %d, giving up", params["startdatetime"],
                params["enddatetime"], status,
            )
            return [], False
        payload = _response_payload(response)
        if payload is None:
            LOG.warning(
                "fetch_window %s..%s: 200 with a non-JSON body, giving up",
                params["startdatetime"], params["enddatetime"],
            )
            return [], False
        raw = payload.get("articles") if isinstance(payload, dict) else None
        raw_list = raw if isinstance(raw, list) else []
        rows = parse_articles(payload, asset=query, query=query)
        return rows, len(raw_list) >= config.max_records
    LOG.warning(
        "fetch_window %s..%s failed after %d attempt(s): %s",
        params["startdatetime"], params["enddatetime"], attempts, last_error,
    )
    return [], False


# --------------------------------------------------------------------------- #
# State file
# --------------------------------------------------------------------------- #
_STATE_KEYS: Final[tuple[str, ...]] = ("requests", "failed", "saturated")


def _fresh_state() -> tuple[set[str], dict[str, int]]:
    return set(), {key: 0 for key in _STATE_KEYS}


def _load_state(path: Path) -> tuple[set[str], dict[str, int]]:
    """``(done keys, cumulative counters)``; a missing or unreadable file starts fresh (dedupe on
    ``url`` makes a re-crawl correct, only wasteful, so a corrupt state never blocks the run)."""
    if not path.is_file():
        return _fresh_state()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        LOG.warning("state %s unreadable (%s); starting fresh", path, exc)
        return _fresh_state()
    if not isinstance(raw, dict):
        LOG.warning("state %s is not a JSON object; starting fresh", path)
        return _fresh_state()
    done_raw = raw.get("done")
    done = {str(item) for item in done_raw} if isinstance(done_raw, list) else set()
    counters = {key: int(raw.get(key, 0) or 0) for key in _STATE_KEYS}
    return done, counters


def _write_state(path: Path, done: set[str], counters: dict[str, int]) -> None:
    """Atomic rewrite (``.tmp`` then ``os.replace``) so a kill can never leave a half JSON file."""
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
# Crawl
# --------------------------------------------------------------------------- #
def _state_key(asset: str, window_start: pd.Timestamp) -> str:
    return f"{asset}|{window_start.isoformat()}"


def _over_budget(
    max_requests: int | None,
    requests_used: int,
    max_minutes: float | None,
    started: float,
) -> bool:
    if max_requests is not None and requests_used >= max_requests:
        return True
    if max_minutes is not None and (time.monotonic() - started) >= max_minutes * 60.0:
        return True
    return False


def _append_rows(out_path: Path, rows: list[dict[str, Any]]) -> int:
    """Append ``rows`` to the parquet store, deduping on ``url`` (keep the first); returns
    the number of rows actually added beyond what the store already held."""
    frame = pd.DataFrame(rows, columns=list(ARCHIVE_COLUMNS))
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    prior = read_parquet(out_path) if out_path.is_file() else None
    if prior is not None and len(prior):
        combined = pd.concat([prior, frame], ignore_index=True)
    else:
        combined = frame
    before = len(prior) if prior is not None else 0
    combined = combined.drop_duplicates(subset="url", keep="first")
    kept = len(combined) - before
    if kept > 0:
        write_parquet_atomic(combined, out_path)
    return kept


def crawl(
    config: ArchiveConfig,
    assets: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    out_path: Path,
    state_path: Path,
    max_requests: int | None = None,
    max_minutes: float | None = None,
    get: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Walk windows x assets, append new rows to ``out_path``, checkpoint ``state_path``.

    Tasks recorded in the state file are skipped (resume), each completed task is marked
    ``asset|window_start_iso``, and the state is rewritten after every window — and again on a
    budget stop — so a kill never loses progress. Rows are appended to the parquet store
    *before* the state that marks their window done: a crash in between costs a refetch, which
    dedupe absorbs, never a lost article. A saturated window is halved for that asset exactly
    once. Stops as soon as either budget (``max_requests`` fetch calls, ``max_minutes`` wall
    time) is exhausted. Returns ``requests, failed, saturated, kept, windows_done, seconds``
    for *this run*; the state file holds the cumulative counters.
    """
    unknown = [asset for asset in assets if asset not in config.terms]
    if unknown:
        raise ValueError(f"unknown asset(s): {', '.join(unknown)}")
    windows = iter_windows(start, end, config.window_hours)
    if not assets or not windows:
        return {
            "requests": 0, "failed": 0, "saturated": 0,
            "kept": 0, "windows_done": 0, "seconds": 0.0,
        }

    done, base_counters = _load_state(state_path)
    run = {key: 0 for key in _STATE_KEYS}
    kept = 0
    windows_done = 0
    started = time.monotonic()

    def _checkpoint() -> None:
        """Persist done keys plus cumulative counters (prior runs + this one)."""
        cumulative = {key: base_counters[key] + run[key] for key in _STATE_KEYS}
        _write_state(state_path, done, cumulative)

    for window_start, window_end in windows:
        window_rows: list[dict[str, Any]] = []
        budget_hit = False
        for asset in assets:
            if _over_budget(max_requests, run["requests"], max_minutes, started):
                budget_hit = True
                break
            key = _state_key(asset, window_start)
            if key in done:
                continue
            query = config.terms[asset]
            rows, saturated = fetch_window(config, query, window_start, window_end, get=get)
            run["requests"] += 1
            if not rows:
                run["failed"] += 1
            if saturated:
                run["saturated"] += 1
            window_rows.extend({**row, "asset": asset} for row in rows)
            # The main fetch completed, so the task is recorded even if a budget stop skips
            # the halving pass below (dedupe makes the resumed run cheap either way).
            done.add(key)
            if saturated:
                midpoint = window_start + (window_end - window_start) / 2
                for half_start, half_end in (
                    (window_start, midpoint),
                    (midpoint, window_end),
                ):
                    if _over_budget(max_requests, run["requests"], max_minutes, started):
                        budget_hit = True
                        break
                    half_rows, half_saturated = fetch_window(
                        config, query, half_start, half_end, get=get
                    )
                    run["requests"] += 1
                    if not half_rows:
                        run["failed"] += 1
                    if half_saturated:
                        run["saturated"] += 1
                    window_rows.extend({**row, "asset": asset} for row in half_rows)
            if budget_hit:
                break

        # Rows go to parquet first, then the state that marks the window done: a crash in
        # between costs a refetch (dedupe absorbs it), never a lost article.
        if window_rows:
            kept += _append_rows(out_path, window_rows)
        _checkpoint()
        if not budget_hit:
            windows_done += 1
        if budget_hit:
            break

    seconds = round(time.monotonic() - started, 3)
    LOG.info(
        "crawl done: requests=%d failed=%d saturated=%d kept=%d windows=%d in %.1fs",
        run["requests"], run["failed"], run["saturated"], kept, windows_done, seconds,
    )
    return {
        "requests": run["requests"],
        "failed": run["failed"],
        "saturated": run["saturated"],
        "kept": kept,
        "windows_done": windows_done,
        "seconds": seconds,
    }


def load_archive(path: Path) -> pd.DataFrame:
    """The stored archive frame, sorted by ``published_at`` (stable: ties keep arrival order)."""
    frame = read_parquet(Path(path))
    return frame.sort_values("published_at", kind="stable").reset_index(drop=True)
