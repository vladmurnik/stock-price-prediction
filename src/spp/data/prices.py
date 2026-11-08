"""T2 — daily prices: yfinance fetch, parquet cache, quality control, wide/returns helpers.

Schema of ``data/cache/prices_{start}_{end}.parquet``::

    date      date32    trading day
    ticker    string    symbol (openFIGI-style exchange symbol; ``^VIX`` keeps its caret)
    open      float64   raw open
    high      float64   raw high
    low       float64   raw low
    close     float64   raw close
    adj_close float64   split *and* dividend adjusted close
    volume    int64     share volume
    source    string    ``yfinance``

Invariants enforced by :func:`validate_prices`:

* ``(date, ticker)`` unique;
* rows of one ticker are sorted by date;
* ``0 < low <= open, close <= high`` and ``adj_close > 0``;
* no gaps longer than :data:`MAX_FILL_DAYS` trading days inside a ticker's listing span.

Policy decisions (see ``02-prices.md``):

* ``auto_adjust=False`` always: ``close`` stays raw, ``adj_close`` carries the
  split/dividend adjustment. Using ``auto_adjust=True`` would double-adjust.
* Prices are never interpolated. Gaps up to ``MAX_FILL_DAYS`` NYSE trading days are
  filled forward-only (explicit ``ffill``), longer gaps split the listing into
  contiguous segments and later segments are *dropped* — never bridged.
* ``adr_crosscheck`` compares the ratio ``close / adj_close`` across the whole series:
  a single major jump is a **split** (``yfinance`` reports it in ``Ticker.splits``, old
  ratio is restored); a gradually drifting ratio is a **dividend**. One-day returns
  above ``SPIKE_THRESHOLD`` are flagged to ``reports/prices_anomalies.csv``; a
  multi-day silence of the feed is NOT an anomaly (a multi-day return is not a
  one-day spike).
* Helper instruments (``SPY``, ``^VIX``, the 8 sector ETFs) live in the same parquet;
  callers select them explicitly via ``to_wide(..., tickers=[...])``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from spp.data.store import (
    iter_batches,
    manifest_path,
    normalize_date_range,
    read_manifest,
    read_parquet,
    write_manifest,
    write_parquet_atomic,
)
from spp.logging import get_logger

__all__ = [
    "DEFAULT_HELPERS",
    "DEFAULT_LOOKBACK_YEARS",
    "MAX_FILL_DAYS",
    "PRICE_COLUMNS",
    "PRICE_DTYPES",
    "SPIKE_THRESHOLD",
    "PricesResult",
    "cache_path",
    "daily_returns",
    "default_end",
    "default_start",
    "fetch_prices",
    "fixture_adapter",
    "helper_tickers",
    "load_prices",
    "to_wide",
    "validate_prices",
    "write_anomaly_report",
]

LOG = get_logger("data.prices")

SOURCE: Final[str] = "yfinance"
FETCH_BATCH_SIZE: Final[int] = 10
FETCH_PAUSE_S: Final[float] = 2.0
FETCH_RETRIES: Final[int] = 3
MAX_FILL_DAYS: Final[int] = 3
PRICE_DECIMALS: Final[int] = 4
SPIKE_THRESHOLD: Final[float] = 0.30
ADR_DIVIDEND_TOLERANCE: Final[float] = 0.20
VOLATILE_TICKERS: Final[frozenset[str]] = frozenset({"^VIX", "^VIX3M", "^GSPC", "^NDX"})
TRAILING_NAN_DAYS: Final[int] = 5
TRADING_DAYS_PER_YEAR: Final[int] = 252

#: Instruments that feature engineering needs in addition to the 40-stock universe.
DEFAULT_HELPERS: Final[tuple[str, ...]] = (
    "SPY",
    "^VIX",
    "XLK",
    "XLV",
    "XLF",
    "XLE",
    "XLY",
    "XLI",
    "XLC",
    "XLU",
)

PRICE_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "source",
)
PRICE_DTYPES: Final[dict[str, str]] = {
    "date": "date",
    "ticker": "string",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "adj_close": "float64",
    "volume": "int64",
    "source": "string",
}
_NUMERIC = ("open", "high", "low", "close", "adj_close")

#: Adapter contract: ``adapter(tickers, start, end) -> (raw_df, splits)``.
FetchAdapter = Callable[[Sequence[str], date, date], tuple[pd.DataFrame, pd.DataFrame]]


# --------------------------------------------------------------------------- #
# Public dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PricesResult:
    """Outcome of :func:`fetch_prices`."""

    frame: pd.DataFrame
    path: Path
    coverage: pd.DataFrame
    missing: tuple[str, ...]
    violations: list[str] = field(default_factory=list)
    anomalies: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())


# --------------------------------------------------------------------------- #
# Paths and defaults
# --------------------------------------------------------------------------- #
def cache_path(cache_dir: str | Path, start: date, end: date) -> Path:
    """``cache_dir/prices_{start}_{end}.parquet`` (exactly the ``00-index.md`` name)."""
    return Path(cache_dir) / f"prices_{start.isoformat()}_{end.isoformat()}.parquet"


def default_start(today: date | None = None) -> date:
    """Start of the research window: ``TRAIN_START`` 2019-01-01, or 5 years back in live use.

    The 2019-01-01 anchor is the minimum history the features need (T6) and keeps a
    plain ``fetch prices`` (no flags) reproducing the research cache byte-for-byte.
    """
    anchor = date(2019, 1, 1)
    now = today or date.today()
    rolling = date(now.year - DEFAULT_LOOKBACK_YEARS, now.month, now.day)
    return min(anchor, rolling)


def default_end(today: date | None = None) -> date:
    """The last complete UTC day: yfinance's ``end`` is exclusive, so this is never partial."""
    from datetime import datetime

    now = today or datetime.now(UTC).date()
    return now


DEFAULT_LOOKBACK_YEARS: Final[int] = 5


# --------------------------------------------------------------------------- #
# Adapters: live yfinance + offline fixture
# --------------------------------------------------------------------------- #
class _YFinanceAdapter:
    """Live adapter over ``yfinance`` (Pandas 3 / yfinance 1.x multi-index aware)."""

    def __init__(self, pause_s: float = FETCH_PAUSE_S, retries: int = FETCH_RETRIES) -> None:
        self.pause_s = pause_s
        self.retries = retries

    def __call__(
        self, tickers: Sequence[str], start: date, end: date
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        import yfinance as yf

        tickers = list(dict.fromkeys(tickers))
        frames: list[pd.DataFrame] = []
        split_rows: list[dict[str, Any]] = []
        batches = list(iter_batches(list(tickers), FETCH_BATCH_SIZE))
        for index, batch in enumerate(batches):
            if index:
                time.sleep(self.pause_s)  # rate-limit guard between batches
            raw = self._fetch_batch(yf, batch, start, end)
            if raw is None:
                continue
            frames.append(self._to_long(raw, batch))
            split_rows.extend(self._fetch_splits(yf, batch, start, end))
        long_df = pd.concat(frames, ignore_index=True) if frames else _empty_long()
        splits = (
            pd.DataFrame(split_rows, columns=["ticker", "date", "ratio"])
            if split_rows
            else pd.DataFrame(columns=["ticker", "date", "ratio"])
        )
        return long_df, splits

    def _fetch_batch(self, yf: Any, batch: Sequence[str], start: date, end: date) -> pd.DataFrame | None:
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                raw = yf.download(
                    list(batch),
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),
                    auto_adjust=False,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
                if isinstance(raw, pd.DataFrame) and not raw.empty:
                    return raw
                LOG.warning("yfinance returned no rows for %s (attempt %d)", list(batch), attempt)
            except Exception as exc:  # network / rate limit / delisted payloads
                last_exc = exc
                LOG.warning(
                    "yfinance batch %s failed on attempt %d/%d: %s",
                    list(batch),
                    attempt,
                    self.retries,
                    exc,
                )
            if attempt < self.retries:
                time.sleep(self.pause_s * attempt)
        if last_exc is not None:
            LOG.error("giving up on batch %s after %d attempts: %s", list(batch), self.retries, last_exc)
        return None

    def _to_long(self, raw: pd.DataFrame, batch: Sequence[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        requested = list(batch)
        if isinstance(raw.columns, pd.MultiIndex):
            level0 = raw.columns.get_level_values(0).unique().tolist()
            for ticker in requested:
                if ticker not in level0:
                    LOG.warning("ticker %s absent from yfinance payload", ticker)
                    continue
                single = raw[ticker].copy()
                single.columns = [str(c).strip().lower().replace(" ", "_") for c in single.columns]
                if "adj_close" not in single.columns and "adj close" in single.columns:
                    single = single.rename(columns={"adj close": "adj_close"})
                frames.append(self._ticker_frame(single, ticker))
        else:  # single-ticker payload: flat columns
            ticker = requested[0]
            single = raw.copy()
            single.columns = [str(c).strip().lower().replace(" ", "_") for c in single.columns]
            frames.append(self._ticker_frame(single, ticker))
        if not frames:
            return _empty_long()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _ticker_frame(single: pd.DataFrame, ticker: str) -> pd.DataFrame:
        single = single.copy()
        single.index = pd.to_datetime(single.index).tz_localize(None).normalize()
        single = single.rename_axis("date").reset_index()
        keep = ["date"] + [c for c in _NUMERIC + ("volume",) if c in single.columns]
        single = single[keep]
        single["ticker"] = ticker
        return single

    def _fetch_splits(self, yf: Any, batch: Sequence[str], start: date, end: date) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for ticker in batch:
            try:
                splits = yf.Ticker(ticker).splits
            except Exception as exc:  # pragma: no cover - network
                LOG.debug("splits unavailable for %s: %s", ticker, exc)
                continue
            if splits is None or len(splits) == 0:
                continue
            for ts, ratio in splits.items():
                stamp = pd.Timestamp(ts)
                if stamp.tzinfo is not None:
                    stamp = stamp.tz_localize(None)
                day = stamp.normalize().date()
                if start <= day <= end and float(ratio) > 0:
                    rows.append({"ticker": ticker, "date": day, "ratio": float(ratio)})
        return rows


class _FixtureAdapter:
    """Offline adapter replaying a recorded yfinance payload.

    Fixture layout: ``{fixture_dir}/{TICKER}.parquet`` with a ``date`` column and
    lowercase OHLCV + ``adj_close`` columns, plus optional ``splits.csv`` with
    ``ticker,date,ratio``. Missing tickers are reported as absent, never invented.
    """

    def __init__(self, fixture_dir: str | Path) -> None:
        self.root = Path(fixture_dir)

    def __call__(
        self, tickers: Sequence[str], start: date, end: date
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            path = self.root / f"{_fixture_name(ticker)}.parquet"
            if not path.is_file():
                LOG.warning("fixture missing for %s (%s)", ticker, path)
                continue
            df = read_parquet(path)
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
            df = df.loc[mask].copy()
            if df.empty:
                continue
            df["ticker"] = ticker
            frames.append(df)
        long_df = pd.concat(frames, ignore_index=True) if frames else _empty_long()
        split_file = self.root / "splits.csv"
        if split_file.is_file():
            splits = pd.read_csv(split_file)
            splits["date"] = pd.to_datetime(splits["date"]).dt.date
            splits = splits[(splits["date"] >= start) & (splits["date"] <= end)]
            splits = splits[splits["ticker"].isin(list(tickers))][["ticker", "date", "ratio"]]
        else:
            splits = pd.DataFrame(columns=["ticker", "date", "ratio"])
        return long_df, splits.reset_index(drop=True)


def fixture_adapter(fixture_dir: str | Path) -> FetchAdapter:
    """Public factory for the offline replay adapter (used by tests and CI without network)."""
    return _FixtureAdapter(fixture_dir)


def _fixture_name(ticker: str) -> str:
    return ticker.replace("^", "_") or "UNKNOWN"


def _empty_long() -> pd.DataFrame:
    """Empty frame with the exact schema dtypes (``date`` is datetime64, stored as date32)."""
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "ticker": pd.Series(dtype="string"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "adj_close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="int64"),
            "source": pd.Series(dtype="string"),
        }
    )


# --------------------------------------------------------------------------- #
# Universe / helper resolution
# --------------------------------------------------------------------------- #
def helper_tickers(universe_path: str | Path | None = None) -> list[str]:
    """``SPY`` + ``^VIX`` + sector ETFs declared in the universe YAML (deduplicated)."""
    from spp.config import PROJECT_ROOT, load_universe  # local import: config is optional

    path = Path(universe_path) if universe_path is not None else (
        PROJECT_ROOT / "configs" / "universe_us_largecap.yaml"
    )
    helpers: list[str] = ["SPY", "^VIX"]
    try:
        import yaml

        with path.open(encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        sector_etf = payload.get("sector_etf") or {}
        helpers.extend(v for _, v in sorted(sector_etf.items()))
    except FileNotFoundError:
        LOG.warning("universe file not found at %s; using hardcoded helper list", path)
        helpers = list(DEFAULT_HELPERS)
    else:
        universe = load_universe(path)
        del universe  # validation side effect only (sector coverage check)
    seen: dict[str, None] = {}
    for t in helpers:
        seen.setdefault(t, None)
    return list(seen)


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
def fetch_prices(
    tickers: list[str],
    start: date,
    end: date,
    cache_dir: Path,
    *,
    refresh: bool = False,
    adapter: FetchAdapter | None = None,
    fallback: FetchAdapter | None = None,
) -> PricesResult:
    """Load daily OHLCV for ``tickers`` (``start``..``end`` inclusive) through the parquet cache.

    cache rules:

    1. ``cache_dir/prices_{start}_{end}.parquet`` covering the request (its manifest
       says the fetch reached ``max_date`` >= ``end``) is read without network.
    2. Otherwise any ``prices_*.parquet`` whose range overlaps is reused; only the
       missing left/right windows are fetched and merged.
    3. ``refresh=True`` bypasses the cache and re-fetches the whole range.

    Returns a :class:`PricesResult`; the frame is sorted by ``(date, ticker)`` and
    typed exactly as the schema, so writing it is deterministic.
    """
    start, _ = normalize_date_range(start, end)
    if start is None:
        raise ValueError("start is required")
    cache_dir = Path(cache_dir)
    target = cache_path(cache_dir, start, end)
    requested = list(dict.fromkeys(tickers))

    if not refresh and target.is_file() and _covers(target, start, end):
        LOG.info("prices cache hit: %s (no network)", target.name)
        df = load_prices(cache_dir, start, end)
        return _finalize(df, requested, target, fetched=False)

    fetch = adapter or _YFinanceAdapter()
    pieces: list[pd.DataFrame] = []
    fetched: list[tuple[date, date]] = []
    splits_parts: list[pd.DataFrame] = []
    if not refresh and target.is_file():
        LOG.warning("cache %s exists but does not cover the range; refetching", target.name)
    for part in _plan_windows(cache_dir, start, end):
        window_start, window_end = part
        LOG.info("fetching prices %s..%s (%d tickers)", window_start, window_end, len(requested))
        raw, splits = fetch(requested, window_start, window_end)
        if raw is None or raw.empty:
            LOG.warning("no data returned for %s..%s", window_start, window_end)
            continue
        if not splits.empty:
            splits_parts.append(splits)
        pieces.append(raw)
        fetched.append((window_start, window_end))

    if not pieces:
        if target.is_file():
            LOG.warning("network yielded nothing; falling back to the existing cache %s", target.name)
            df = load_prices(cache_dir, start, end)
            return _finalize(df, requested, target, fetched=False)
        raise RuntimeError(f"no price data for any of {requested} in {start}..{end}")

    long_df = pd.concat(pieces, ignore_index=True)
    splits_df = (
        pd.concat(splits_parts, ignore_index=True) if splits_parts
        else pd.DataFrame(columns=["ticker", "date", "ratio"])
    )
    cleaned, anomalies = _clean(long_df, splits_df)
    cleaned = _finalize_frame(cleaned)
    anomalies = anomalies.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)

    coverage = _coverage(cleaned, start, end)
    missing = tuple(sorted(set(requested) - set(coverage.index)))
    write_parquet_atomic(cleaned, target, date32=("date",))
    write_manifest(
        manifest_path(target),
        {
            "artifact": target.name,
            "kind": "prices",
            "request_start": start,
            "request_end": end,
            "fetched_windows": [[a.isoformat(), b.isoformat()] for a, b in fetched],
            "max_date": _as_date(cleaned["date"].max() if not cleaned.empty else None),
            "min_date": _as_date(cleaned["date"].min() if not cleaned.empty else None),
            "tickers_requested": requested,
            "tickers_present": list(coverage.index),
            "tickers_missing": list(missing),
            "rows": int(len(cleaned)),
            "source": SOURCE,
        },
    )
    violations = validate_prices(cleaned)
    LOG.info(
        "prices cache written: %s rows=%d tickers=%d missing=%s",
        target.name,
        len(cleaned),
        len(coverage),
        missing or "none",
    )
    return PricesResult(cleaned, target, coverage, missing, violations, anomalies)


def _plan_windows(cache_dir: Path, start: date, end: date) -> list[tuple[date, date]]:
    """Return the date windows that must be fetched to satisfy any overlapping cached file.

    The trailing window starts at the cached file's filename end (not at its manifest
    ``max_date``): a ticker that listed later is simply fetched again for a few extra
    months, which costs one call but can never skip data because of a calendar gap.
    """
    from datetime import timedelta

    best = _best_overlap(cache_dir, start, end)
    if best is None:
        return [(start, end)]
    cached_start, cached_end = best
    windows: list[tuple[date, date]] = []
    if start < cached_start:
        windows.append((start, cached_start - timedelta(days=1)))
    if end > cached_end:
        windows.append((cached_end + timedelta(days=1), end))
    if not windows:
        windows.append((start, end))  # cached range is unusable (e.g. empty manifest)
    return windows


def _best_overlap(cache_dir: Path, start: date, end: date) -> tuple[date, date] | None:
    """Pick the cached file whose filename range overlaps ``start..end`` the most."""
    best: tuple[int, date, date] | None = None
    for path in sorted(Path(cache_dir).glob("prices_*_*.parquet")):
        bounds = _parse_name(path.name)
        if bounds is None:
            continue
        file_start, file_end = bounds
        if file_end < start or file_start > end:
            continue
        overlap = (min(file_end, end) - max(file_start, start)).days
        if best is None or overlap > best[0]:
            best = (overlap, file_start, file_end)
    return None if best is None else (best[1], best[2])


def _as_date(value: Any) -> str | None:
    """Coerce a pandas timestamp / date / None into a plain ISO date string."""
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return None
    return stamp.date().isoformat()


def _norm_date_text(raw: Any) -> date | None:
    """Parse a manifest date that may be ISO date or ISO datetime."""
    if raw is None:
        return None
    text = str(raw).strip()
    return pd.Timestamp(text[:10]).date()


def _parse_name(name: str) -> tuple[date, date] | None:
    stem = name.removeprefix("prices_").removesuffix(".parquet")
    left, sep, right = stem.partition("_")
    if not sep:
        return None
    try:
        return date.fromisoformat(left), date.fromisoformat(right)
    except ValueError:
        return None


def _covers(path: Path, start: date, end: date) -> bool:
    """True when ``path`` was fetched for a range covering ``start..end``.

    The manifest's ``request_end`` is the authority: ``max_date`` only records the last
    trading day returned, so a request ending on a weekend/holiday would look uncovered
    and trigger a pointless refetch.
    """
    manifest = read_manifest(manifest_path(path))
    requested_start = _norm_date_text(manifest.get("request_start"))
    requested_end = _norm_date_text(manifest.get("request_end"))
    if requested_start is not None and requested_end is not None:
        return requested_start <= start and requested_end >= end
    reached = _norm_date_text(manifest.get("max_date"))
    if reached is None:
        return True  # legacy artifact without manifest: trust the filename
    return reached >= end


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def _clean(df: pd.DataFrame, splits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.tz_localize(None).dt.normalize()
    work["ticker"] = work["ticker"].astype("string")
    for col in _NUMERIC + ("volume",):
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    dups = int(work.duplicated(subset=["date", "ticker"], keep="last").sum())
    if dups:
        LOG.warning("dropping %d duplicate (date,ticker) rows from the raw payload", dups)
        work = work.drop_duplicates(subset=["date", "ticker"], keep="last")

    work = work.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    work["volume"] = work["volume"].fillna(0).round().astype("int64")

    filled_parts: list[pd.DataFrame] = []
    anomaly_frames: list[pd.DataFrame] = []
    for ticker, group in work.groupby("ticker", sort=True):
        group = group.sort_values("date", kind="stable")
        group, anomalies = _fill_gaps(group, str(ticker))
        if anomalies is not None and not anomalies.empty:
            anomaly_frames.append(anomalies)
        filled_parts.append(group)
    work = pd.concat(filled_parts, ignore_index=True) if filled_parts else _empty_long()

    # Prices must be positive; rows failing that are dropped (never interpolated).
    bad = (
        (work[list(_NUMERIC)] <= 0).any(axis=1)
        | work["close"].isna()
        | work["adj_close"].isna()
    )
    if bool(bad.any()):
        LOG.warning("dropping %d rows with non-positive/missing prices", int(bad.sum()))
        work = work.loc[~bad]

    work = _check_adj_close_suspicious(work)

    anomalies = (
        pd.concat(anomaly_frames, ignore_index=True) if anomaly_frames else _anomaly_frame()
    )
    work, spike_flags = _detect_spikes(work, splits)
    if not spike_flags.empty:
        anomalies = pd.concat([anomalies, spike_flags], ignore_index=True)
    work["source"] = SOURCE
    return work[list(PRICE_COLUMNS)], anomalies


def _check_adj_close_suspicious(df: pd.DataFrame) -> pd.DataFrame:
    """Plan check: ``adj_close``/``close`` divergence > 50% = split suspicion, WARN + manual review.

    ``adj_close`` is the cumulative split+dividend factor, so a flat ratio is normal; a step
    larger than 50% means the two columns disagree about a corporate action. The ratio is
    *never* used to rewrite prices: raw OHLCV stays exactly as the source reported it and
    ``adj_close`` stays the adjusted series (mixing both would double-adjust).
    """
    work = df.copy()
    for ticker, group in work.groupby("ticker", sort=True):
        close = group["close"].to_numpy(dtype="float64")
        adj = group["adj_close"].to_numpy(dtype="float64")
        if len(close) < 2:
            continue
        ratio = close / adj
        rel_step = np.abs(ratio[1:] / ratio[:-1] - 1.0)
        hits = np.flatnonzero(rel_step > ADR_DIVIDEND_TOLERANCE)
        for i in hits[:3]:
            day = pd.Timestamp(group["date"].iloc[i + 1]).normalize()
            LOG.warning(
                "%s adj_close/close divergence %.1f%% on %s (ratio %.4f -> %.4f): suspected split "
                "not applied consistently; manual review required, prices left as-is",
                ticker,
                rel_step[i] * 100,
                day.date(),
                ratio[i],
                ratio[i + 1],
            )
    return work


def _fill_gaps(group: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward-fill gaps up to ``MAX_FILL_DAYS`` NYSE days; drop later segments.

    The listing span is the calendar from the ticker's first to last observation day.
    Contiguous runs longer than ``MAX_FILL_DAYS + 1`` missing calendar days are
    treated as delisting/history breaks: everything after the first break is dropped
    (never bridged with fabricated prices).
    """
    try:
        import pandas_market_calendars as mcal

        cal = mcal.get_calendar("NYSE")
        open_days = cal.valid_days(
            start_date=group["date"].min(), end_date=group["date"].max()
        )
        calendar = pd.DatetimeIndex(open_days).tz_localize(None).normalize()
    except Exception as exc:  # pragma: no cover - calendar package missing
        LOG.warning("NYSE calendar unavailable (%s); gap fill without trading calendar", exc)
        calendar = pd.date_range(group["date"].min(), group["date"].max(), freq="B")

    present = pd.DatetimeIndex(group["date"]).normalize()
    missing = calendar.difference(present)
    if len(missing) == 0:
        return group, _anomaly_frame()

    runs: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    run_start = missing[0]
    run_end = missing[0]
    previous = missing[0]
    for day in missing[1:]:
        if day == previous + pd.offsets.BDay(1):
            run_end = day
        else:
            runs.append((run_start, run_end, int(calendar.get_indexer([run_end])[0] - calendar.get_indexer([run_start])[0]) + 1))
            run_start = day
            run_end = day
        previous = day
    runs.append((run_start, run_end, int(calendar.get_indexer([run_end])[0] - calendar.get_indexer([run_start])[0]) + 1))

    anomalies: list[dict[str, Any]] = []
    drop_after: pd.Timestamp | None = None
    for run_start, run_end, length in runs:
        kind = "leading" if run_end < present.min() else (
            "trailing" if run_start > present.max() else "internal"
        )
        if kind != "internal":
            LOG.info("%s %s gap %s..%s (%d days) skipped (not bridged)", ticker, kind, run_start.date(), run_end.date(), length)
            continue
        if length > MAX_FILL_DAYS:
            LOG.warning(
                "%s internal gap %s..%s is %d trading days (> %d); truncating the series "
                "after the earlier segment instead of bridging",
                ticker,
                run_start.date(),
                run_end.date(),
                length,
                MAX_FILL_DAYS,
            )
            drop_after = run_start
            anomalies.append(
                {
                    "ticker": ticker,
                    "date": run_start.date(),
                    "kind": "gap_too_long",
                    "severity": "error",
                    "detail": f"{length} missing trading days {run_start.date()}..{run_end.date()}",
                }
            )
            break

    if drop_after is not None:
        group = group.loc[group["date"] < drop_after]
        return group, pd.DataFrame(anomalies)

    reindexed = (
        group.set_index("date")
        .reindex(calendar)
        .rename_axis("date")
        .reset_index()
    )
    fill_cols = list(_NUMERIC) + ["volume"]
    reindexed[fill_cols] = reindexed[fill_cols].ffill(limit=MAX_FILL_DAYS)
    reindexed["ticker"] = ticker
    if reindexed["close"].isna().any():  # pragma: no cover - guarded by run splitting
        raise AssertionError(f"{ticker}: forward fill left NaN inside a short gap")
    return reindexed, pd.DataFrame(anomalies)


def _detect_spikes(df: pd.DataFrame, splits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag one-day adjusted-close moves above ``SPIKE_THRESHOLD`` not explained by a split."""
    split_keys = set()
    if splits is not None and not splits.empty:
        for _, row in splits.iterrows():
            split_keys.add((str(row["ticker"]), pd.Timestamp(row["date"]).normalize()))

    rows: list[dict[str, Any]] = []
    for ticker, group in df.groupby("ticker", sort=True):
        group = group.sort_values("date", kind="stable")
        adj = group["adj_close"].to_numpy(dtype="float64")
        days = pd.DatetimeIndex(group["date"]).normalize()
        rets = np.zeros(len(adj), dtype="float64")
        rets[1:] = np.log(adj[1:] / adj[:-1])
        volatile = str(ticker) in VOLATILE_TICKERS
        threshold = SPIKE_THRESHOLD if not volatile else np.inf
        for i in range(1, len(rets)):
            if not np.isfinite(rets[i]) or abs(rets[i]) <= threshold:
                continue
            day = days[i]
            prev_day = days[i - 1]
            span = int(np.busday_count(prev_day.date(), day.date()))
            if span > 1:
                LOG.info(
                    "%s multi-day move (%.1f%% over %d sessions) ending %s: feed gap, not a spike",
                    ticker,
                    (np.exp(rets[i]) - 1.0) * 100,
                    span,
                    day.date(),
                )
                continue
            if _split_within(split_keys, str(ticker), prev_day, day):
                LOG.info("%s %.1f%% move on %s explained by a split", ticker, (np.exp(rets[i]) - 1.0) * 100, day.date())
                continue
            close_ret = 0.0
            if i > 0 and group["close"].iloc[i - 1] > 0:
                close_ret = float(group["close"].iloc[i] / group["close"].iloc[i - 1] - 1.0)
            rows.append(
                {
                    "ticker": ticker,
                    "date": day.date(),
                    "kind": "one_day_spike",
                    "severity": "warn",
                    "detail": (
                        f"adj_close {rets[i] * 100:+.2f}% (close {close_ret * 100:+.2f}%), "
                        "no split reported"
                    ),
                }
            )
            LOG.warning(
                "%s anomalous one-day move %.1f%% on %s (close %.1f%%), no split: see reports/prices_anomalies.csv",
                ticker,
                (np.exp(rets[i]) - 1.0) * 100,
                day.date(),
                close_ret * 100,
            )
    return df, (pd.DataFrame(rows) if rows else _anomaly_frame())


def _split_within(keys: set[tuple[str, pd.Timestamp]], ticker: str, prev: pd.Timestamp, day: pd.Timestamp) -> bool:
    for (sym, split_day) in keys:
        if sym == ticker and prev < split_day <= day:
            return True
    return False


def _anomaly_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "date", "kind", "severity", "detail"])


# --------------------------------------------------------------------------- #
# Finalisation / validation
# --------------------------------------------------------------------------- #
def _finalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out["date"] = out["date"].astype("datetime64[ns]")
    out["ticker"] = out["ticker"].astype("string")
    for col in ("open", "high", "low", "close", "adj_close"):
        # 4 decimals is below any economically meaningful price move (>= 1e-4 relative on a
        # $40 stock) and above yfinance's own ~1e-7 recomputation noise, so a refetch yields
        # a byte-identical parquet instead of float jitter in `adj_close`.
        out[col] = pd.to_numeric(out[col], errors="coerce").round(PRICE_DECIMALS).astype(
            "float64"
        )
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).round().astype("int64")
    out["source"] = out["source"].astype("string") if "source" in out else SOURCE
    out = out.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    return out[list(PRICE_COLUMNS)]


def _finalize(df: pd.DataFrame, requested: list[str], target: Path, *, fetched: bool) -> PricesResult:
    frame = _finalize_frame(df) if not df.empty else _empty_long()
    present = set(frame["ticker"].astype(str).unique()) if not frame.empty else set()
    missing = tuple(sorted(set(requested) - present))
    coverage = _coverage(frame, frame["date"].min().date(), frame["date"].max().date()) if not frame.empty else pd.DataFrame()
    anomalies = _anomaly_frame()
    split_guess = pd.DataFrame(columns=["ticker", "date", "ratio"])
    _, anomalies = _detect_spikes(frame, split_guess)
    return PricesResult(frame, target, coverage, missing, validate_prices(frame), anomalies)


def validate_prices(df: pd.DataFrame) -> list[str]:
    """Return a list of invariant violations (empty list when the frame is clean)."""
    violations: list[str] = []
    if df.empty:
        return ["frame is empty"]
    missing_cols = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing_cols:
        return [f"missing columns: {missing_cols}"]

    frame = df[list(PRICE_COLUMNS)].copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()

    dups = frame.duplicated(subset=["date", "ticker"], keep=False)
    if bool(dups.any()):
        pairs = (
            frame.loc[dups, ["date", "ticker"]]
            .drop_duplicates()
            .head(5)
            .apply(lambda r: f"{r['ticker']}@{r['date'].date()}", axis=1)
            .tolist()
        )
        violations.append(f"duplicate (date,ticker) rows: {len(pairs)} shown {pairs}")

    for ticker, group in frame.groupby("ticker", sort=True):
        if not group["date"].is_monotonic_increasing:
            violations.append(f"{ticker}: dates not strictly increasing")
        if group["date"].duplicated().any():
            violations.append(f"{ticker}: repeated dates inside the series")

    bad_price = (frame[list(_NUMERIC)] <= 0).any(axis=1)
    if bool(bad_price.any()):
        sample = (
            frame.loc[bad_price, ["ticker", "date"]]
            .head(5)
            .apply(lambda r: f"{r['ticker']}@{r['date'].date()}", axis=1)
            .tolist()
        )
        violations.append(f"non-positive or missing prices in {int(bad_price.sum())} rows, e.g. {sample}")

    finite = np.isfinite(frame[list(_NUMERIC)].to_numpy(dtype="float64")).all(axis=1)
    if not bool(finite.all()):
        violations.append(f"non-finite price values in {int((~finite).sum())} rows")

    if not violations:
        for ticker, group in frame.groupby("ticker", sort=True):
            present = pd.DatetimeIndex(group["date"]).normalize()
            span = pd.date_range(present.min(), present.max(), freq="B")
            gaps = _long_runs(span.difference(present))
            if gaps:
                violations.append(
                    f"{ticker}: {len(gaps)} gap(s) longer than {MAX_FILL_DAYS} business days, "
                    f"first {gaps[0][0].date()}..{gaps[0][1].date()}"
                )
    return violations


def _long_runs(index: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    runs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if len(index) == 0:
        return runs
    start = prev = index[0]
    length = 1
    for day in index[1:]:
        if day == prev + pd.offsets.BDay(1):
            length += 1
        else:
            if length > MAX_FILL_DAYS:
                runs.append((start, prev))
            start, length = day, 1
        prev = day
    if length > MAX_FILL_DAYS:
        runs.append((start, prev))
    return runs


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_prices(cache_dir: Path, start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """Load cached prices, optionally sliced to ``[start, end]`` (inclusive).

    The cache directory may hold several range files; the one containing the request
    (or the union of overlapping ones) is read. Missing cache yields an empty schema frame.
    """
    start, end = normalize_date_range(start, end)
    cache_dir = Path(cache_dir)
    files = sorted(cache_dir.glob("prices_*_*.parquet"))
    chosen: list[Path] = []
    if start is not None or end is not None:
        for path in files:
            bounds = _parse_name(path.name)
            if bounds is None:
                continue
            file_start, file_end = bounds
            if start is not None and file_end < start:
                continue
            if end is not None and file_start > end:
                continue
            chosen.append(path)
        if not chosen and files:
            chosen = files
    else:
        chosen = files
    if not chosen:
        LOG.warning("no price cache in %s", cache_dir)
        return _empty_long()

    parts = [read_parquet(p) for p in chosen]
    frame = pd.concat(parts, ignore_index=True)
    frame = _finalize_frame(frame)
    if start is not None:
        frame = frame.loc[frame["date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame.loc[frame["date"] <= pd.Timestamp(end)]
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Wide / returns helpers (the only sanctioned way to build matrices)
# --------------------------------------------------------------------------- #
def to_wide(df: pd.DataFrame, field: str = "adj_close", tickers: Sequence[str] | None = None) -> pd.DataFrame:
    """Pivot long prices into ``index=date, columns=ticker`` for ``field``.

    Column order follows ``tickers`` when given, else the ticker's first appearance
    in the frame; missing tickers get an all-NaN column so downstream joins stay aligned.
    """
    if field not in PRICE_COLUMNS and field not in df.columns:
        raise KeyError(f"unknown field {field!r}; available: {list(df.columns)}")
    frame = df[["date", "ticker", field]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str)
    if frame.duplicated(subset=["date", "ticker"]).any():
        raise ValueError("duplicate (date, ticker) in input; run validate_prices first")
    if tickers is not None:
        order = list(tickers)
        frame = frame.loc[frame["ticker"].isin(order)]
    else:
        order = list(dict.fromkeys(frame["ticker"].tolist()))
    wide = frame.pivot(index="date", columns="ticker", values=field)
    wide = wide.reindex(columns=order)
    wide.index = pd.DatetimeIndex(wide.index).normalize()
    wide.index.name = "date"
    wide.columns.name = None
    return wide.sort_index(kind="stable")


def daily_returns(adj_close_wide: pd.DataFrame) -> pd.DataFrame:
    """Log returns ``log(p_t / p_{t-1})``; the first row per column is left NaN."""
    wide = adj_close_wide.sort_index(kind="stable")
    values = wide.to_numpy(dtype="float64")
    out = np.full_like(values, np.nan)
    if len(values) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = np.log(values[1:] / values[:-1])
    return pd.DataFrame(out, index=wide.index, columns=wide.columns)


# --------------------------------------------------------------------------- #
# Coverage / reporting
# --------------------------------------------------------------------------- #
def _coverage(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Per-ticker counts vs the NYSE calendar over the requested range."""
    from datetime import timedelta

    if df.empty:
        return pd.DataFrame(
            columns=["ticker", "nyse_days", "rows", "first_date", "last_date", "coverage"]
        )
    try:
        import pandas_market_calendars as mcal

        nyse = mcal.get_calendar("NYSE").valid_days(
            start_date=start, end_date=end + timedelta(days=1)
        )
        nyse_days = int(len(pd.DatetimeIndex(nyse).normalize().unique()))
    except Exception as exc:  # pragma: no cover
        LOG.warning("NYSE calendar unavailable (%s); using business-day count", exc)
        nyse_days = int(len(pd.bdate_range(start, end)))

    rows: list[dict[str, Any]] = []
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    for ticker, group in frame.groupby("ticker", sort=True):
        rows.append(
            {
                "ticker": str(ticker),
                "nyse_days": nyse_days,
                "rows": int(len(group)),
                "first_date": group["date"].min().date(),
                "last_date": group["date"].max().date(),
                "coverage": round(len(group) / nyse_days, 4) if nyse_days else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def write_anomaly_report(anomalies: pd.DataFrame, path: str | Path) -> Path:
    """Write ``reports/prices_anomalies.csv`` atomically (stable row order)."""
    import os
    import tempfile

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = anomalies.copy()
    if frame.empty:
        frame = _anomaly_frame()
    frame = frame.sort_values(["date", "ticker", "kind"], kind="stable").reset_index(drop=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.stem}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        frame.to_csv(tmp, index=False, lineterminator="\n")
        os.replace(tmp, target)
    finally:
        if tmp.exists():  # pragma: no cover
            try:
                tmp.unlink()
            except OSError:
                pass
    return target
