"""T3 — FNSPID historical news: streaming ingest, normalisation, coverage report.

Source: HF dataset ``Zihan1004/FNSPID``. The dataset is *not* loaded via
``datasets.load_dataset``: it ships two raw CSV blobs (5.7 GB ``All_external.csv`` and
23.2 GB ``nasdaq_exteral_data.csv``) whose ``Article`` / ``*_summary`` columns are
unused and dominate the bytes. Loading them through ``datasets`` buffered 5+ GB before
yielding the first row; instead this module streams the CSV over HTTP with range
requests, which keeps peak memory flat (~1 MB) and lets us parse only the five
columns the plan needs (``03-news-history.md`` §1).

Output: ``data/interim/news_raw.parquet`` with the plan schema::

    article_id    string          sha1(publisher|url|title)
    ticker        string          linked ticker
    published_at  timestamp [UTC] source timestamp when the source gives one
    date          date32          source date (UTC)
    signal_date   date32          first day the news is usable by the model
    title         string          cleaned headline
    title_norm    string          lowercase, punctuation-free
    url           string          canonical url
    publisher     string          domain/name (``unknown`` when blank)
    source        string          ``fnspid``
    text_hash     string          sha1(title_norm)

Idempotency: rows are deduped and sorted deterministically, dtypes pinned, and the
parquet is written atomically; a rerun over the same window yields a byte-identical
file. A sidecar manifest records the row counts of every stage.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd

from spp.data.normalize import (
    ENTITY_AMBIGUOUS,
    ENTITY_OK,
    NEWS_COLUMNS,
    AliasIndex,
    DropStats,
    SignalPolicy,
    article_id,
    assign_signal_date,
    clean_title,
    dedupe_news,
    link_ticker,
    near_dupe_pairs,
    normalize_title,
    nyse_trading_days,
    text_hash,
    validate_news,
)
from spp.data.store import (
    manifest_path,
    read_manifest,
    read_parquet,
    write_manifest,
    write_parquet_atomic,
)
from spp.logging import get_logger

__all__ = [
    "FNSPID_REPO",
    "FNSPID_SHARDS",
    "NewsResult",
    "fetch_news",
    "fixture_source",
    "load_news_raw",
    "news_coverage",
    "news_path",
    "stream_source",
    "write_bad_dates",
    "write_coverage_report",
]

LOG = get_logger("data.news_fnspid")

SOURCE: Final[str] = "fnspid"
FNSPID_REPO: Final[str] = "Zihan1004/FNSPID"
#: CSV shards scanned by default. ``All_external.csv`` carries Benzinga + other outlets
#: (264 MB gzip / 5.7 GB raw); ``nasdaq_exteral_data.csv`` (23 GB) is an opt-in shard.
FNSPID_SHARDS: Final[tuple[str, ...]] = ("Stock_news/All_external.csv",)
HF_RESOLVE: Final[str] = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
PROGRESS_EVERY: Final[int] = 500_000
READ_BATCH_ROWS: Final[int] = 50_000
NEWS_PARQUET: Final[str] = "interim/news_raw.parquet"

REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "Date",
    "Article_title",
    "Stock_symbol",
    "Url",
    "Publisher",
)

#: ``RowSource`` contract used by :func:`fetch_news`:
#: ``source(shard) -> iterator of (line_number, {field: value})`` with ``line_number`` counting CSV records.
RowSource = Callable[[str], Iterator[tuple[int, dict[str, str]]]]

# --------------------------------------------------------------------------- #
# Date parsing
# --------------------------------------------------------------------------- #
#: ``YYYY-MM-DD HH:MM:SS UTC`` (FNSPID majority format).
_WITH_CLOCK = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?"
)
_DATE_ONLY = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$")


def parse_published_at(raw: object) -> tuple[datetime | None, date | None]:
    """Parse an FNSPID ``Date`` cell into ``(published_at_utc, date)``.

    Handles ``YYYY-MM-DD``, ``YYYY-MM-DD HH:MM[:SS]`` (with or without a timezone token),
    ``YYYY/MM/DD`` and bare ``YYYY-MM-DD HH:MM:SS UTC``. Anything else returns
    ``(None, None)`` and is reported to ``reports/news_bad_dates.csv``.
    """
    if raw is None:
        return None, None
    text = str(raw).strip()
    if not text or text.lower() in {"none", "nan", "nat"}:
        return None, None
    match = _WITH_CLOCK.match(text)
    if match:
        year, month, day, hour, minute, second = (int(g) if g else 0 for g in match.groups())
        try:
            stamp = datetime(year, month, day, hour, minute, second)
        except ValueError:
            return None, None
        suffix = text[match.end() :].strip().upper()
        if suffix and suffix not in {"UTC", "GMT", "Z", "ET", "EST", "EDT", "US/Eastern"}:
            try:  # explicit numeric offset, e.g. "+00:00"

                offset = datetime.fromisoformat(f"2000-01-01T00:00:00{suffix}").tzinfo
                assert offset is not None
                stamp = stamp.replace(tzinfo=offset).astimezone(UTC).replace(tzinfo=None)
            except ValueError:
                return None, None
        elif suffix in {"ET", "EST", "EDT", "US/EASTERN"}:
            # FNSPID labels every record UTC, so an ET/EST/EDT tag here is not trustworthy;
            # the wall clock is taken as UTC (conservative for the intraday policy).
            pass
        return stamp, stamp.date()
    match = _DATE_ONLY.match(text)
    if match:
        try:
            day = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None, None
        return None, day
    return None, None


# --------------------------------------------------------------------------- #
# Streaming source
# --------------------------------------------------------------------------- #
def stream_source(
    shards: Sequence[str] = FNSPID_SHARDS,
    *,
    repo: str = FNSPID_REPO,
    timeout: float = 180.0,
    retries: int = 3,
    block_bytes: int = 1 << 20,
) -> RowSource:
    """Return a row source that streams the given CSV shards from the HF hub.

    ``HF_TOKEN`` is read from the environment when present (gated/rate-limited access);
    the token value never enters logs.
    """
    headers = {"User-Agent": "spp-data/0.1 (research; FNSPID ingest)"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def source(shard: str) -> Iterator[tuple[int, dict[str, str]]]:
        import requests

        url = HF_RESOLVE.format(repo=repo, path=shard)
        LOG.info("streaming %s", shard)
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                with requests.get(url, headers=headers, stream=True, timeout=timeout) as resp:
                    resp.raise_for_status()
                    total = resp.headers.get("Content-Length")
                    if total:
                        LOG.info("%s: %.2f GB", shard, int(total) / 1e9)
                    raw = resp.raw.read
                    buffer = b""
                    header: list[str] | None = None
                    line_no = 0
                    while True:
                        chunk = raw(block_bytes, decode_content=True)
                        if not chunk:
                            break
                        buffer += chunk
                        *complete, buffer = buffer.split(b"\n")
                        if not complete:
                            continue
                        text_block = b"\n".join(complete).decode("utf-8", errors="replace")
                        records = list(csv.reader(io.StringIO(text_block)))
                        if header is None:
                            if not records:
                                continue
                            header = _normalise_header(records[0])
                            _require_fields(header, shard)
                            records = records[1:]
                        for record in records:
                            line_no += 1
                            yield line_no, dict(zip(header, record, strict=False))
                if buffer:
                    text_tail = buffer.decode("utf-8", errors="replace")
                    rows = list(csv.reader(io.StringIO(text_tail)))
                    for record in rows:
                        line_no += 1
                        yield line_no, dict(zip(header or [], record, strict=False))
                return
            except Exception as exc:  # network drop / rate limit / HTTP error
                last_exc = exc
                LOG.warning(
                    "streaming %s failed on attempt %d/%d: %s", shard, attempt, retries, type(exc).__name__
                )
                if attempt < retries:
                    time.sleep(min(30.0, 2.0 * attempt))
        raise RuntimeError(f"could not stream {shard} after {retries} attempts: {last_exc}") from last_exc

    return source


def _normalise_header(row: Sequence[str]) -> list[str]:
    return [str(c).strip() for c in row]


def _require_fields(header: Sequence[str], shard: str) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in header]
    if missing:
        raise ValueError(f"{shard}: missing columns {missing}; header was {list(header)}")


def parquet_source(path: str | Path) -> RowSource:
    """Offline row source over a local parquet mirror of FNSPID rows.

    The parquet mirrors on the Hub are columnar, so the useful columns can be
    pulled with a row filter over HTTP instead of streaming a multi-gigabyte CSV.
    The resulting file is fed through the same normalisation as a live shard.
    """
    target = Path(path)

    def source(_shard: str) -> Iterator[tuple[int, dict[str, str]]]:
        LOG.info("reading parquet %s", target)
        frame = pd.read_parquet(target)
        missing = [f for f in REQUIRED_FIELDS if f not in frame.columns]
        if missing:
            raise ValueError(f"{target}: missing columns {missing}")
        for line_no, record in enumerate(frame.to_dict("records"), start=1):
            yield line_no, {k: "" if v is None else str(v) for k, v in record.items()}

    return source


def fixture_source(fixture: str | Path) -> RowSource:
    """Offline row source replaying a CSV fixture (same columns as FNSPID).

    Used by tests and by any run without network access; it never fabricates rows.
    """
    path = Path(fixture)

    def source(_shard: str) -> Iterator[tuple[int, dict[str, str]]]:
        LOG.info("replaying fixture %s", path)
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            _require_fields(reader.fieldnames or [], str(path))
            for line_no, record in enumerate(reader, start=1):
                yield line_no, {k: (v or "") for k, v in record.items()}

    return source


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NewsResult:
    """Outcome of :func:`fetch_news`."""

    frame: pd.DataFrame
    path: Path
    stats: DropStats
    bad_dates: pd.DataFrame
    violations: list[str] = field(default_factory=list)
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)
    cached: bool = False


def news_path(interim_dir: str | Path) -> Path:
    return Path(interim_dir) / "news_raw.parquet"


# --------------------------------------------------------------------------- #
# Row -> record normalisation
# --------------------------------------------------------------------------- #
def _canonical_url(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    try:
        parts = urlsplit(text)
    except ValueError:
        return text
    if not parts.scheme or not parts.netloc:
        return text
    query = parts.query
    # Drop tracking parameters; keep semantic ones (rare on news urls).
    if query:
        keep = [
            kv
            for kv in query.split("&")
            if kv and not re.match(r"^(utm_|fbclid=|gclid=|ref=|cmp=|mod=|rss=|soc_src=)", kv)
        ]
        query = "&".join(keep)
    netloc = parts.netloc.lower()
    return urlunsplit((parts.scheme.lower(), netloc, parts.path, query, ""))


def _publisher(raw: str) -> str:
    text = (raw or "").strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return "unknown"
    if " " in text or "." in text or "@" in text:  # author name, not a domain
        return text
    return text.lower()


def _normalise_row(
    row: dict[str, str],
    index: AliasIndex,
    stats: DropStats,
    start: date,
    end: date,
    bad_rows: list[dict[str, Any]],
    line_no: int,
    shard: str,
) -> dict[str, Any] | None:
    stats.scanned += 1
    ticker_raw = (row.get("Stock_symbol") or "").strip().upper()
    if ticker_raw not in index.tickers:
        stats.non_universe += 1
        return None

    published_at, source_date = parse_published_at(row.get("Date"))
    if source_date is None:
        stats.bad_date += 1
        if len(bad_rows) < 5_000:
            bad_rows.append(
                {
                    "shard": shard,
                    "row": line_no,
                    "stock_symbol": ticker_raw,
                    "date_raw": str(row.get("Date"))[:64],
                    "title": str(row.get("Article_title"))[:120],
                }
            )
        return None
    if not (start <= source_date <= end):
        stats.out_of_range += 1
        return None

    title = clean_title(row.get("Article_title"))
    if title is None:
        stats.short_or_boilerplate += 1
        return None

    ticker, status = link_ticker(title, index)
    if status == ENTITY_AMBIGUOUS:
        stats.ambiguous_alias += 1
        return None
    if status == ENTITY_OK and ticker != ticker_raw:
        # The headline names another company; the publisher's symbol is the
        # primary subject, so keep it and record the disagreement.
        stats.alias_mismatch += 1
    ticker = ticker_raw

    title_norm = normalize_title(title)
    if len(title_norm) < 5:
        stats.short_or_boilerplate += 1
        return None
    publisher = _publisher(row.get("Publisher"))
    url = _canonical_url(row.get("Url"))
    return {
        "article_id": article_id(publisher, url, title),
        "ticker": ticker,
        "published_at": published_at,
        "date": source_date,
        "title": title,
        "title_norm": title_norm,
        "url": url,
        "publisher": publisher,
        "source": SOURCE,
        "text_hash": text_hash(title_norm),
    }


def _normalise_batch(
    rows: Sequence[tuple[int, dict[str, str]]],
    index: AliasIndex,
    stats: DropStats,
    start: date,
    end: date,
    bad_rows: list[dict[str, Any]],
    shard: str,
) -> pd.DataFrame:
    records = [
        rec
        for line_no, row in rows
        if (rec := _normalise_row(row, index, stats, start, end, bad_rows, line_no, shard)) is not None
    ]
    if not records:
        return _empty_news()
    frame = pd.DataFrame.from_records(records)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    return frame


def _empty_news() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": pd.Series(dtype="string"),
            "ticker": pd.Series(dtype="string"),
            "published_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "date": pd.Series(dtype="datetime64[ns]"),
            "signal_date": pd.Series(dtype="datetime64[ns]"),
            "title": pd.Series(dtype="string"),
            "title_norm": pd.Series(dtype="string"),
            "url": pd.Series(dtype="string"),
            "publisher": pd.Series(dtype="string"),
            "source": pd.Series(dtype="string"),
            "text_hash": pd.Series(dtype="string"),
        }
    )


def _finalize_news(df: pd.DataFrame, policy: SignalPolicy, trading_days: Sequence[date]) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    published = pd.to_datetime(out["published_at"], utc=True)
    out["published_at"] = published
    out["signal_date"] = assign_signal_date(
        [ts.date() for ts in out["date"]],
        [None if pd.isna(ts) else ts.to_pydatetime() for ts in published],
        policy,
        trading_days,
    )
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    for col in ("article_id", "ticker", "title", "title_norm", "url", "publisher", "source", "text_hash"):
        out[col] = out[col].astype("string")
    out = out[list(NEWS_COLUMNS)]
    for col in ("date", "signal_date"):
        out[col] = out[col].dt.tz_localize(None).astype("datetime64[ns]") if out[col].dt.tz is not None else out[col].astype("datetime64[ns]")
    out = out.sort_values(["date", "ticker", "published_at", "article_id"], kind="stable", na_position="last")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Main entrypoint
# --------------------------------------------------------------------------- #
def _cache_key(start: date, end: date, shards: Sequence[str], strict: bool, cutoff: str) -> str:
    payload = json.dumps(
        {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "shards": list(shards),
            "strict": strict,
            "cutoff": cutoff,
            "repo": FNSPID_REPO,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()


def fetch_news(
    start: date,
    end: date,
    interim_dir: str | Path,
    reports_dir: str | Path,
    index: AliasIndex,
    policy: SignalPolicy,
    *,
    shards: Sequence[str] = FNSPID_SHARDS,
    source: RowSource | None = None,
    refresh: bool = False,
    calendar_pad_days: int = 7,
) -> NewsResult:
    """Build ``data/interim/news_raw.parquet`` for ``[start, end]``.

    Caching: the sidecar manifest stores the request parameters; a rerun without
    ``refresh`` and with the same parameters returns the existing artifact untouched.
    The trading calendar is built with a small forward pad so ``signal_date`` can roll
    past the last in-range day (``November 30`` -> next NYSE session in ``December``).
    """
    if start > end:
        raise ValueError(f"start {start} > end {end}")
    interim_dir = Path(interim_dir)
    reports_dir = Path(reports_dir)
    target = news_path(interim_dir)
    key = _cache_key(start, end, shards, policy.strict_dates, policy.cutoff_et.strftime("%H:%M"))

    if not refresh and target.is_file():
        manifest = read_manifest(manifest_path(target))
        if manifest.get("request_key") == key:
            LOG.info("news cache hit: %s (%d rows, no network)", target.name, int(manifest.get("rows", 0)))
            frame = load_news_raw(interim_dir)
            stats = DropStats(**{k: int(v) for k, v in (manifest.get("drops") or {}).items() if k != "by_ticker"})
            bad_dates = _read_csv_or_empty(reports_dir / "news_bad_dates.csv")
            violations = validate_news(frame, index.tickers)
            return NewsResult(
                frame=frame,
                path=target,
                stats=stats,
                bad_dates=bad_dates,
                violations=violations,
                coverage=news_coverage(frame, start, end, index.tickers),
                cached=True,
            )

    row_source = source or stream_source(shards)
    stats = DropStats()
    bad_rows: list[dict[str, Any]] = []
    parts: list[pd.DataFrame] = []
    shard_counts: dict[str, int] = {}

    for shard in shards:
        shard_stats = DropStats()
        batch: list[tuple[int, dict[str, str]]] = []
        seen_in_shard = 0
        last_progress = 0
        for line_no, row in row_source(shard):
            batch.append((line_no, row))
            seen_in_shard += 1
            if len(batch) >= READ_BATCH_ROWS:
                parts.append(_normalise_batch(batch, index, shard_stats, start, end, bad_rows, shard))
                batch = []
                if seen_in_shard - last_progress >= PROGRESS_EVERY:
                    last_progress = seen_in_shard
                    LOG.info(
                        "%s: %d rows scanned, %d kept so far",
                        shard,
                        seen_in_shard,
                        sum(len(p) for p in parts),
                    )
        if batch:
            parts.append(_normalise_batch(batch, index, shard_stats, start, end, bad_rows, shard))
        stats.merge(shard_stats)
        shard_counts[shard] = seen_in_shard
        LOG.info("%s: %d rows scanned and filtered", shard, seen_in_shard)

    raw = pd.concat(parts, ignore_index=True) if parts else _empty_news()
    LOG.info("rows after filters: %d (scanned %d)", len(raw), stats.scanned)

    deduped, dedupe_stats = dedupe_news(raw, threshold=0.9)
    stats.exact_dupe += dedupe_stats.exact_dupe
    stats.near_dupe += dedupe_stats.near_dupe
    LOG.info(
        "dedupe: %d exact, %d near (Jaccard > 0.9) removed; %d kept",
        dedupe_stats.exact_dupe,
        dedupe_stats.near_dupe,
        len(deduped),
    )

    trading_days = nyse_trading_days(start - timedelta(days=calendar_pad_days), end + timedelta(days=calendar_pad_days))
    frame = _finalize_news(deduped, policy, trading_days)

    violations = validate_news(frame, index.tickers) if not frame.empty else ["frame is empty"]
    pairs = near_dupe_pairs(frame) if not frame.empty else 0
    if pairs:
        violations.append(f"{pairs} near-duplicate pairs survived dedupe")

    write_parquet_atomic(frame, target, date32=("date", "signal_date"))
    bad = pd.DataFrame(bad_rows, columns=["shard", "row", "stock_symbol", "date_raw", "title"])
    write_bad_dates(bad, reports_dir / "news_bad_dates.csv")
    coverage = news_coverage(frame, start, end, index.tickers)
    write_coverage_report(coverage, reports_dir / "news_coverage.csv")

    counts = frame["ticker"].value_counts()
    for ticker, count in counts.items():
        stats.by_ticker[str(ticker)] = int(count)
    stats.kept = len(frame)
    write_manifest(
        manifest_path(target),
        {
            "artifact": target.name,
            "kind": "news_raw",
            "source": SOURCE,
            "repo": FNSPID_REPO,
            "shards": list(shards),
            "shard_rows_scanned": shard_counts,
            "request_start": start,
            "request_end": end,
            "request_key": key,
            "cutoff_et": policy.cutoff_et.strftime("%H:%M"),
            "strict_dates": policy.strict_dates,
            "rows": int(len(frame)),
            "tickers": int(counts.shape[0]),
            "min_date": frame["date"].min().date().isoformat() if len(frame) else None,
            "max_date": frame["date"].max().date().isoformat() if len(frame) else None,
            "drops": stats.as_dict(),
            "near_dupe_pairs_after": int(pairs),
        },
    )
    LOG.info("news_raw written: %s rows=%d tickers=%d", target, len(frame), counts.shape[0])
    return NewsResult(frame, target, stats, bad, violations, coverage, cached=False)


def load_news_raw(interim_dir: str | Path) -> pd.DataFrame:
    """Read ``news_raw.parquet`` with dtypes restored (dates as timestamp[ns])."""
    path = news_path(interim_dir)
    if not path.is_file():
        LOG.warning("news artifact missing: %s", path)
        return _empty_news()
    frame = read_parquet(path)
    for col in ("date", "signal_date"):
        if col in frame:
            frame[col] = pd.to_datetime(frame[col]).dt.tz_localize(None).astype("datetime64[ns]")
    if "published_at" in frame:
        frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    return frame[list(NEWS_COLUMNS)]


# --------------------------------------------------------------------------- #
# Coverage / reports
# --------------------------------------------------------------------------- #
def news_coverage(
    frame: pd.DataFrame,
    start: date,
    end: date,
    tickers: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-ticker article statistics on NYSE sessions (the features' own calendar)."""
    sessions = pd.DatetimeIndex(nyse_trading_days(start, end)).normalize()
    n_sessions = len(sessions)
    if tickers is None:
        tickers = sorted(frame["ticker"].astype(str).unique()) if not frame.empty else []
    rows: list[dict[str, Any]] = []
    counts: pd.Series
    if not frame.empty:
        work = frame.copy()
        signal = pd.to_datetime(work["signal_date"], utc=True)
        work["signal"] = signal.dt.tz_localize(None).dt.normalize()
        work = work.loc[work["signal"].isin(sessions)]
        counts = work.groupby(["ticker", "signal"], sort=True).size()
    else:
        counts = pd.Series(dtype="int64")
    by_ticker = counts.groupby(level=0) if not counts.empty else {}
    for ticker in tickers:
        if not counts.empty and ticker in by_ticker.groups:
            series = by_ticker.get_group(ticker)
            if isinstance(series.index, pd.MultiIndex):
                # groupby(level=0) keeps the ticker level in the index.
                series = series.droplevel(0)
        else:
            series = pd.Series(dtype="int64")
        per_day = (
            series.reindex(sessions, fill_value=0).astype("int64")
            if len(series)
            else pd.Series(0, index=sessions, dtype="int64")
        )
        rows.append(
            {
                "ticker": ticker,
                "articles": int(per_day.sum()),
                "sessions": n_sessions,
                "median_per_day": float(per_day.median()) if n_sessions else float("nan"),
                "p10_per_day": float(per_day.quantile(0.1)) if n_sessions else float("nan"),
                "p90_per_day": float(per_day.quantile(0.9)) if n_sessions else float("nan"),
                "empty_days": int((per_day == 0).sum()),
                "empty_share": float((per_day == 0).mean()) if n_sessions else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _atomic_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():  # pragma: no cover
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


def write_bad_dates(bad: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    frame = bad.copy().sort_values(["shard", "row"], kind="stable") if not bad.empty else bad
    return _atomic_text(target, frame.to_csv(index=False, lineterminator="\n"))


def write_coverage_report(coverage: pd.DataFrame, path: str | Path, publishers: pd.DataFrame | None = None) -> Path:
    target = Path(path)
    text = coverage.to_csv(index=False, lineterminator="\n") if not coverage.empty else ""
    if publishers is not None and not publishers.empty:
        text += "\n# publisher_top20\n" + publishers.head(20).to_csv(index=False, lineterminator="\n")
    return _atomic_text(target, text)


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["shard", "row", "stock_symbol", "date_raw", "title"])
    return pd.read_csv(path)


def summarize(news: pd.DataFrame, stats: DropStats, coverage: pd.DataFrame) -> dict[str, Any]:
    """Numbers for the CLI / report: totals, medians, empty share, top publishers."""
    result: dict[str, Any] = dict(stats.as_dict())
    result["kept"] = int(len(news))
    if not news.empty:
        result["publishers"] = news["publisher"].value_counts().head(20).to_dict()
        result["tickers"] = int(news["ticker"].nunique())
    if not coverage.empty:
        result["median_per_day_weighted"] = float(np.median(coverage["median_per_day"]))
        result["min_median_per_day"] = float(coverage["median_per_day"].min())
        result["max_empty_share"] = float(coverage["empty_share"].max())
    return result
