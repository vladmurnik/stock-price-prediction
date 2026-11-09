"""T1 — resumable 1-minute kline downloader from the Binance Vision public archive.

``https://data.binance.vision`` is the only keyless source with full Binance spot history:
monthly zips (``{SYMBOL}-1m-{YYYY}-{MM}.zip``, 2 169 570 bytes for BTCUSDT-1m-2024-01) plus
daily zips for the current month. Every zip holds one CSV with the twelve standard kline
fields; this module selects the eight the dataset needs and normalises them into the
canonical :data:`BAR_COLUMNS` frame.

Contract for downstream tasks (T5 dataset assembly):

* ``ts_open`` is the bar **open** time as ``datetime64[ns, UTC]`` — a row labelled
  ``2024-01-01 00:00`` describes the minute that started then, never the close time.
* ``open``..``quote_volume`` are float64, ``trades`` is int64; rows are sorted ascending by
  ``ts_open`` with unique timestamps (duplicates keep the last occurrence).
* Parquet files land in ``data/crypto/bars/1m/{SYMBOL}.parquet``; downloaded zips are
  cached under ``data/cache/crypto/bars_1m`` so an interrupted build resumes without
  re-downloading anything.

The downloader never invents data: a month the archive does not have raises
:class:`FileNotFoundError` with the exact URL, and :func:`build_all` reports such a symbol
with ``rows=0`` instead of padding it with partial or fabricated bars.
"""

from __future__ import annotations

import io
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Final

import pandas as pd
import requests
import yaml

from spp.data.store import write_parquet_atomic
from spp.logging import get_logger

__all__ = [
    "BAR_COLUMNS",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_OUT_DIR",
    "MANIFEST_COLUMNS",
    "build_all",
    "cache_path",
    "daily_url",
    "download_month",
    "empty_bars",
    "iter_months",
    "load_frozen_config",
    "load_symbol",
    "month_url",
    "quality_report",
    "read_month",
    "resolve_window",
]

LOG = get_logger("spp.crypto.bars")

#: Repo root resolved from this file (``src/spp/crypto/bars.py``), never from the cwd.
ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH: Final[Path] = ROOT / "configs" / "crypto_dataset.yaml"
DEFAULT_OUT_DIR: Final[Path] = ROOT / "data" / "crypto" / "bars" / "1m"
DEFAULT_CACHE_DIR: Final[Path] = ROOT / "data" / "cache" / "crypto" / "bars_1m"

_MONTHLY_BASE: Final[str] = "https://data.binance.vision/data/spot/monthly/klines"
_DAILY_BASE: Final[str] = "https://data.binance.vision/data/spot/daily/klines"
_HEADERS: Final[dict[str, str]] = {"User-Agent": "spp-crypto-bars/0.1"}
_MONTH_RE: Final[re.Pattern[str]] = re.compile(r"^(\d{4})-(\d{2})$")

#: Exact column order of every frame this module returns and every parquet it writes.
BAR_COLUMNS: Final[tuple[str, ...]] = (
    "ts_open",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
)

#: Exact column order of the manifest returned by :func:`build_all`.
MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "rows",
    "first",
    "last",
    "bytes",
    "gaps_over_2min",
    "ohlc_violations",
)


class _RetryableError(Exception):
    """HTTP 429/5xx, network failure or corrupt payload: safe to retry with backoff."""


class _InvalidArchive(ValueError):
    """A zip that will not open or does not contain exactly one CSV member."""


# --------------------------------------------------------------------------- #
# URLs and month arithmetic
# --------------------------------------------------------------------------- #
def month_url(symbol: str, year: int, month: int) -> str:
    """URL of the monthly 1m kline zip (e.g. BTCUSDT 2024-01)."""
    return f"{_MONTHLY_BASE}/{symbol}/1m/{symbol}-1m-{year:04d}-{month:02d}.zip"


def daily_url(symbol: str, date: pd.Timestamp) -> str:
    """URL of the daily 1m kline zip for ``date`` (its UTC calendar date)."""
    stamp = pd.Timestamp(date)
    name = f"{symbol}-1m-{stamp.year:04d}-{stamp.month:02d}-{stamp.day:02d}.zip"
    return f"{_DAILY_BASE}/{symbol}/1m/{name}"


def _parse_month(text: str, label: str) -> tuple[int, int]:
    match = _MONTH_RE.match(text.strip())
    if match is None:
        raise ValueError(f"{label} must look like 'YYYY-MM', got {text!r}")
    year, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        raise ValueError(f"{label} month out of range: {text!r}")
    return year, month


def iter_months(start: str, end: str) -> list[tuple[int, int]]:
    """Inclusive list of ``(year, month)`` pairs from ``start`` to ``end`` (both ``YYYY-MM``).

    ``"2023-11"`` to ``"2024-02"`` yields ``[(2023, 11), (2023, 12), (2024, 1), (2024, 2)]``.
    """
    first = _parse_month(start, "start")
    last = _parse_month(end, "end")
    if first > last:
        raise ValueError(f"start {start!r} is after end {end!r}")
    months: list[tuple[int, int]] = []
    year, month = first
    while (year, month) <= last:
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def cache_path(cache_dir: Path, symbol: str, year: int, month: int) -> Path:
    """Zip location in the cache: ``{cache_dir}/{SYMBOL}-1m-{YYYY}-{MM}.zip``."""
    return Path(cache_dir) / f"{symbol}-1m-{year:04d}-{month:02d}.zip"


# --------------------------------------------------------------------------- #
# Zip verification and HTTP
# --------------------------------------------------------------------------- #
def _single_csv_member(zf: zipfile.ZipFile, source: str) -> str:
    """Name of the one CSV member, or :class:`_InvalidArchive` when the count is wrong."""
    members = [info.filename for info in zf.infolist() if not info.is_dir()]
    csv_members = [name for name in members if name.lower().endswith(".csv")]
    if len(csv_members) != 1:
        raise _InvalidArchive(
            f"{source}: expected exactly one CSV member, found {len(csv_members)}"
        )
    return csv_members[0]


def _verify_zip_file(path: Path) -> bool:
    """True when ``path`` opens as a zip with exactly one CSV member (cache is usable)."""
    if not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            _single_csv_member(zf, str(path))
    except (zipfile.BadZipFile, _InvalidArchive, OSError):
        return False
    return True


def _verify_payload(payload: bytes, url: str) -> None:
    """Raise :class:`_InvalidArchive` unless ``payload`` is a zip with one CSV member."""
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            _single_csv_member(zf, url)
    except zipfile.BadZipFile as exc:
        raise _InvalidArchive(f"{url}: not a valid zip ({exc})") from exc


def _fetch_zip(url: str, timeout: float) -> bytes:
    """Fetch ``url`` once: 200 -> payload bytes.

    404 raises :class:`FileNotFoundError` carrying the URL (the archive simply does not
    have that month), 429 and 5xx raise :class:`_RetryableError`, any other non-200
    status raises :class:`RuntimeError`.
    """
    response = requests.get(url, headers=_HEADERS, timeout=timeout)
    status = response.status_code
    if status == 404:
        raise FileNotFoundError(f"HTTP 404: {url}")
    if status == 429 or 500 <= status < 600:
        raise _RetryableError(f"HTTP {status}: {url}")
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {url}")
    return response.content


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_month(
    symbol: str,
    year: int,
    month: int,
    cache_dir: Path,
    timeout: float = 180.0,
    retries: int = 3,
) -> Path:
    """Download the monthly zip for ``symbol`` into ``cache_dir`` and return its path.

    Resumable by construction: a cached file that opens as a zip with exactly one CSV
    member is returned without any network access; a corrupt or partial cache entry is
    replaced. ``retries`` counts the attempts *after* the first one: HTTP 429/5xx,
    network errors and corrupt payloads back off 2, 4, 8 ... seconds. HTTP 404 raises
    :class:`FileNotFoundError` with the URL — a month the archive does not have is
    reported, never faked. The payload is verified before it touches the cache and lands
    atomically (``.part`` then ``os.replace``), so an interrupted run never leaves a
    truncated zip behind.
    """
    cache_dir = Path(cache_dir)
    target = cache_path(cache_dir, symbol, year, month)
    url = month_url(symbol, year, month)
    if _verify_zip_file(target):
        LOG.debug("cache hit: %s", target)
        return target
    if target.is_file():
        LOG.warning("cache entry is not a valid zip, re-downloading: %s", target)

    cache_dir.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        try:
            payload = _fetch_zip(url, timeout)
            _verify_payload(payload, url)
        except FileNotFoundError:
            raise  # a missing month is final: report the URL, never retry or invent data
        except (_RetryableError, _InvalidArchive, OSError) as exc:
            # OSError also covers requests.RequestException (an IOError) and socket timeouts.
            attempt += 1
            if attempt > retries:
                raise
            delay = 2**attempt
            LOG.warning(
                "download %s attempt %d/%d failed (%s); retrying in %ds",
                url,
                attempt,
                retries + 1,
                exc,
                delay,
            )
            time.sleep(delay)
            continue
        break

    part = target.with_name(f"{target.name}.part")
    part.write_bytes(payload)
    os.replace(part, target)
    LOG.info("downloaded %s (%d bytes)", target, len(payload))
    return target


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def empty_bars() -> pd.DataFrame:
    """Zero-row frame with the exact :data:`BAR_COLUMNS` schema and dtypes."""
    return pd.DataFrame(
        {
            "ts_open": pd.Series(dtype="datetime64[ns, UTC]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "quote_volume": pd.Series(dtype="float64"),
            "trades": pd.Series(dtype="int64"),
        }
    )


def read_month(path: Path, symbol: str) -> pd.DataFrame:
    """Parse one monthly kline zip into the canonical :data:`BAR_COLUMNS` frame.

    The CSV normally has no header, but some archived files carry one; the header is
    detected by trying to parse the first field of the first line as an integer.
    ``open_time`` is milliseconds in modern files and microseconds in older ones — any
    value above ``10**14`` selects ``us``. ``ts_open`` becomes ``datetime64[ns, UTC]`` and
    is the bar **open** time; ``open``..``quote_volume`` are float64, ``trades`` int64,
    and the close-time / taker-buy / ``ignore`` columns are dropped.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        member = _single_csv_member(zf, str(path))
        raw = zf.read(member)
    text = raw.decode("utf-8-sig")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        LOG.warning("read_month: %s (%s) has no rows", path.name, symbol)
        return empty_bars()
    try:
        int(lines[0].split(",", 1)[0].strip())
        skiprows = 0
    except ValueError:
        skiprows = 1
    try:
        frame = pd.read_csv(io.StringIO(text), header=None, skiprows=skiprows)
    except pd.errors.EmptyDataError:
        LOG.warning("read_month: %s (%s) has a header but no rows", path.name, symbol)
        return empty_bars()
    if frame.empty:
        LOG.warning("read_month: %s (%s) has no data rows", path.name, symbol)
        return empty_bars()
    if frame.shape[1] < 9:
        raise _InvalidArchive(f"{path}: expected at least 9 columns, found {frame.shape[1]}")
    try:
        value = int(frame.iloc[0, 0])
    except (TypeError, ValueError) as exc:
        raise _InvalidArchive(f"{path}: cannot read open_time from the first row") from exc
    unit = "us" if value > 10**14 else "ms"
    ts_open = pd.to_datetime(frame[0], unit=unit, utc=True).astype("datetime64[ns, UTC]")
    out = pd.DataFrame(
        {
            "ts_open": ts_open,
            "open": frame[1].astype("float64"),
            "high": frame[2].astype("float64"),
            "low": frame[3].astype("float64"),
            "close": frame[4].astype("float64"),
            "volume": frame[5].astype("float64"),
            "quote_volume": frame[7].astype("float64"),
            "trades": frame[8].astype("int64"),
        }
    )
    LOG.debug("read_month: %s rows=%d unit=%s", symbol, len(out), unit)
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def load_symbol(
    symbol: str,
    start: str,
    end: str,
    cache_dir: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download/refresh every month in ``[start, end]`` and return one clean bar frame.

    Months are read in calendar order, concatenated, sorted by ``ts_open`` with a stable
    sort and de-duplicated on ``ts_open`` keeping the LAST row, so a value re-observed in
    a later month (or later in the same file) wins. The result is asserted monotonic and
    tz-aware UTC. A month the archive does not have is skipped with a warning, never filled
    with invented bars: the current, still incomplete month is absent by design, and a token
    listed mid-history has no earlier files. Only when **no** month of the window exists does
    the function raise :class:`FileNotFoundError`.
    """
    months = iter_months(start, end)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for year, month in months:
        if refresh:
            cached = cache_path(cache_dir, symbol, year, month)
            if cached.is_file():
                cached.unlink()
        try:
            path = download_month(symbol, year, month, cache_dir)
        except FileNotFoundError:
            missing.append(f"{year:04d}-{month:02d}")
            LOG.warning("load_symbol: %s has no %04d-%02d in the archive", symbol, year, month)
            continue
        frames.append(read_month(path, symbol))
    if missing:
        LOG.info("load_symbol: %s skipped %d month(s): %s", symbol, len(missing), missing)
    if not frames:
        raise FileNotFoundError(
            f"{symbol}: no month of {start}..{end} exists in the archive "
            f"({len(missing)} month(s) missing)"
        )
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        LOG.warning("load_symbol: %s %s..%s produced no rows", symbol, start, end)
        return empty_bars()
    frame = pd.concat(non_empty, ignore_index=True)
    frame = frame.sort_values("ts_open", kind="stable")
    frame = frame.drop_duplicates(subset="ts_open", keep="last").reset_index(drop=True)
    assert frame["ts_open"].is_monotonic_increasing, f"{symbol}: ts_open is not monotonic"
    assert str(frame["ts_open"].dtype) == "datetime64[ns, UTC]", f"{symbol}: UTC required"
    LOG.info(
        "load_symbol: %s months=%d rows=%d %s..%s",
        symbol,
        len(months),
        len(frame),
        frame["ts_open"].iloc[0],
        frame["ts_open"].iloc[-1],
    )
    return frame


def quality_report(frame: pd.DataFrame, symbol: str) -> dict[str, Any]:
    """JSON-ready diagnostics for ``symbol``; the keys are T1's report contract.

    ``gaps_over_2min`` counts consecutive ``ts_open`` differences greater than two
    minutes. Monthly files are contiguous in time, so a healthy frame scores 0; the count
    is a prompt to inspect the window, not an error by itself. ``ohlc_violations`` counts
    rows where ``high < max(open, close)``, ``low > min(open, close)`` or ``high < low``.
    """
    rows = int(len(frame))
    ts = frame["ts_open"]
    if rows:
        violations = int(
            (
                (frame["high"] < frame[["open", "close"]].max(axis=1))
                | (frame["low"] > frame[["open", "close"]].min(axis=1))
                | (frame["high"] < frame["low"])
            ).sum()
        )
        zero_volume_share = float((frame["volume"] == 0).mean())
        non_positive = int((frame[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
        first = ts.iloc[0].isoformat()
        last = ts.iloc[-1].isoformat()
    else:
        violations, zero_volume_share, non_positive = 0, 0.0, 0
        first = last = None
    return {
        "symbol": symbol,
        "rows": rows,
        "first": first,
        "last": last,
        "duplicate_ts": int(ts.duplicated().sum()),
        "gaps_over_2min": int((ts.diff() > pd.Timedelta(minutes=2)).sum()),
        "ohlc_violations": violations,
        "zero_volume_share": zero_volume_share,
        "non_positive_prices": non_positive,
    }


# --------------------------------------------------------------------------- #
# Config and build
# --------------------------------------------------------------------------- #
def load_frozen_config(path: Path | None = None) -> dict[str, Any]:
    """Read the frozen ``configs/crypto_dataset.yaml``; the file is never modified."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"crypto config not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {cfg_path}")
    return raw


def resolve_window(config: dict[str, Any], start: str | None, end: str | None) -> tuple[str, str]:
    """Effective ``[start, end]`` month window: explicit arguments win over the config.

    The frozen config declares ``period: {start, end}``; top-level ``start``/``end`` are
    tolerated as well, so both layouts resolve to the same window.
    """
    period = config.get("period")
    period = period if isinstance(period, dict) else {}
    resolved_start = start or period.get("start") or config.get("start")
    resolved_end = end or period.get("end") or config.get("end")
    if not resolved_start or not resolved_end:
        raise ValueError(
            f"cannot resolve the month window: start={resolved_start!r} end={resolved_end!r}"
        )
    return str(resolved_start), str(resolved_end)


def build_all(
    config: dict[str, Any] | None = None,
    out_dir: Path | None = None,
    symbols: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download, clean and write one parquet per symbol; return the build manifest.

    ``config=None`` loads the frozen config; ``symbols``/``start``/``end`` default to its
    ``symbols`` and ``period.start``/``period.end`` (top-level ``start``/``end`` are
    tolerated too). Parquet files go to ``out_dir`` (default :data:`DEFAULT_OUT_DIR`) as
    ``{SYMBOL}.parquet``; the zip cache is :data:`DEFAULT_CACHE_DIR`. A symbol whose
    archive month is missing is logged and reported with ``rows=0`` and no parquet —
    partial data is never published as if it were complete.
    """
    cfg = load_frozen_config() if config is None else config
    if symbols is not None:
        chosen = [str(item) for item in symbols]
    else:
        chosen = [str(item) for item in cfg.get("symbols") or []]
    if not chosen:
        raise ValueError("no symbols: pass symbols= or declare them in the config")
    window_start, window_end = resolve_window(cfg, start, end)
    target = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    LOG.info(
        "build_all: symbols=%d window=%s..%s refresh=%s out=%s",
        len(chosen),
        window_start,
        window_end,
        refresh,
        target,
    )

    rows: list[dict[str, Any]] = []
    for symbol in chosen:
        try:
            frame = load_symbol(
                symbol, window_start, window_end, DEFAULT_CACHE_DIR, refresh=refresh
            )
        except FileNotFoundError as exc:
            LOG.error("build_all: %s: %s (no parquet written)", symbol, exc)
            rows.append(
                {
                    "symbol": symbol,
                    "rows": 0,
                    "first": None,
                    "last": None,
                    "bytes": 0,
                    "gaps_over_2min": 0,
                    "ohlc_violations": 0,
                }
            )
            continue
        path = write_parquet_atomic(frame, target / f"{symbol}.parquet")
        report = quality_report(frame, symbol)
        rows.append(
            {
                "symbol": symbol,
                "rows": report["rows"],
                "first": report["first"],
                "last": report["last"],
                "bytes": int(path.stat().st_size),
                "gaps_over_2min": report["gaps_over_2min"],
                "ohlc_violations": report["ohlc_violations"],
            }
        )
        LOG.info("build_all: %s rows=%d -> %s", symbol, report["rows"], path)
    return pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
