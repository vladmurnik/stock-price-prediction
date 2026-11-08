"""Intraday bars for the event study, with the calendar mistakes made explicit.

The daily pipeline only ever needed adjusted closes, so it could afford to treat a session as an
opaque label. An event study cannot: it asks *what happened in the forty minutes after 14:07 ET*,
and that question is only answerable if four things are pinned down.

1. **Time zone.** Bars arrive tz-aware in UTC. Every decision here is made in ``America/New_York``
   because the trading session, the news timestamp and the DST rule are all New York facts. Mixing
   naive UTC with ET silently shifts half the year by an hour - the kind of bug that produces a
   plausible, wrong result.
2. **Session boundaries.** A bar is kept only if its *start* is inside 09:30–16:00 ET on a real
   session of that ticker. Bars that fall outside are dropped and counted, never trimmed silently.
3. **Bar labelling.** Providers label a bar by its start. The 60-minute bar starting at 15:30
   covers the last half hour of the session, so it is short. Treating it as a full hour would
   depress any volatility estimate built on it.
4. **Provenance.** ``interval`` and the fetch window go into the cache key, because a 60-minute
   cache is not a 5-minute cache and a stale one must not be reused for a different question.

``yfinance`` is the only loader because it is the only source reachable without a paid key. It
serves 60-minute bars for about two years and 5-minute bars for sixty days, which is why
``probe_depth`` exists: an experiment that silently receives six sessions instead of five hundred
must fail before it produces a number.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spp.logging import get_logger

__all__ = [
    "BAR_COLUMNS",
    "MinutesUnavailableError",
    "SESSION_CLOSE",
    "SESSION_OPEN",
    "bar_label_end",
    "fetch_minute_bars",
    "load_minute_bars",
    "probe_depth",
    "session_of",
    "to_new_york",
]

_LOG = get_logger("live.minutes")

ET = "America/New_York"
SESSION_OPEN = (9, 30)
SESSION_CLOSE = (16, 0)
BAR_COLUMNS = ("ticker", "ts", "open", "high", "low", "close", "volume", "session")


class MinutesUnavailableError(RuntimeError):
    """Raised when bars cannot be obtained; never caught to fabricate an empty frame."""


#: How far back each interval is actually served. Asking for more does not raise - the provider
#: returns an empty frame, which is indistinguishable from "no data" unless the window is clamped
#: first. 60-minute bars end around 730 days; every finer interval ends around 60 days.
MAX_DEPTH_DAYS = {
    "1m": 7,
    "2m": 59,
    "5m": 59,
    "15m": 59,
    "30m": 59,
    "60m": 729,
    "90m": 59,
}


def clamp_window(
    start: pd.Timestamp, end: pd.Timestamp, interval: str
) -> tuple[pd.Timestamp, bool]:
    """Pull ``start`` forward to the provider's limit; report whether anything was trimmed."""
    depth = MAX_DEPTH_DAYS.get(str(interval).lower())
    if depth is None:
        return start, False
    earliest = end - pd.Timedelta(days=depth)
    if start < earliest:
        _LOG.warning(
            "%s bars are served for %d days only; clamped the request from %s to %s",
            interval,
            depth,
            start.date(),
            earliest.date(),
        )
        return earliest, True
    return start, False


@dataclass(frozen=True)
class ProbeResult:
    ticker: str
    interval: str
    n_bars: int
    first: datetime | None
    last: datetime | None
    n_sessions: int
    bars_per_session: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "interval": self.interval,
            "n_bars": self.n_bars,
            "first": None if self.first is None else self.first.isoformat(),
            "last": None if self.last is None else self.last.isoformat(),
            "n_sessions": self.n_sessions,
            "bars_per_session": self.bars_per_session,
        }


# ------------------------------------------------------------------ time helpers
def to_new_york(ts: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert to ``America/New_York``; naive input is treated as UTC, never as local time."""
    index = pd.DatetimeIndex(ts)
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(ET)


def session_of(ts: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Session date of each timestamp: the ET calendar date, as ``date32``-friendly objects."""
    return pd.Series(to_new_york(ts).date, index=getattr(ts, "index", None))


def in_session(ts: pd.Series | pd.DatetimeIndex, *, minutes: int = 0) -> np.ndarray:
    """True where the timestamp lies inside 09:30–16:00 ET.

    ``minutes`` shifts the window, which the caller uses to keep an entry only if a full horizon
    still fits before the close.
    """
    ny = to_new_york(ts)
    seconds = ny.hour * 3600 + ny.minute * 60 + ny.second
    open_s = (SESSION_OPEN[0] * 3600 + SESSION_OPEN[1] * 60) + minutes * 60
    close_s = SESSION_CLOSE[0] * 3600 + SESSION_CLOSE[1] * 60
    return np.asarray((seconds >= open_s) & (seconds < close_s))


def bar_label_end(ts: pd.Series | pd.DatetimeIndex, interval: str) -> pd.DatetimeIndex:
    """End of each bar, given that providers label a bar by its start."""
    minutes = _interval_minutes(interval)
    return pd.DatetimeIndex(ts) + pd.Timedelta(minutes=minutes)


def _interval_minutes(interval: str) -> int:
    text = str(interval).strip().lower()
    if text.endswith("m"):
        return int(text[:-1])
    if text.endswith("h"):
        return int(text[:-1]) * 60
    raise ValueError(f"unsupported interval {interval!r}; use '5m', '15m', '30m' or '60m'")


# ------------------------------------------------------------------ cache layout
def _cache_path(cache_dir: Path, ticker: str, interval: str) -> Path:
    safe = ticker.replace("^", "_").replace("/", "_")
    return Path(cache_dir) / interval / f"{safe}.parquet"


def _normalise_frame(raw: pd.DataFrame, ticker: str, interval: str) -> pd.DataFrame:
    """Flatten whatever column layout the provider returned into ``BAR_COLUMNS``."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=list(BAR_COLUMNS))
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(col[0]) for col in frame.columns]
    frame = frame.rename(columns={c: str(c).lower() for c in frame.columns})
    frame = frame.reset_index()
    stamp_col = next(
        (c for c in frame.columns if str(c).lower() in {"datetime", "date", "index", "ts"}),
        None,
    )
    if stamp_col is None:
        raise MinutesUnavailableError(f"{ticker}: no timestamp column in {list(frame.columns)}")
    frame = frame.rename(columns={stamp_col: "ts"})
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    keep = ["ts", "open", "high", "low", "close", "volume"]
    for column in keep:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame.loc[:, keep]
    frame = frame.dropna(subset=["close"])
    mask = in_session(frame["ts"])
    frame["ticker"] = ticker
    frame["interval"] = interval
    frame["session"] = session_of(frame["ts"]).to_numpy()
    frame = frame.loc[mask, list(BAR_COLUMNS)]
    frame = frame.drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
    return frame.reset_index(drop=True)


def fetch_minute_bars(
    tickers: Sequence[str],
    start: date | datetime | str,
    end: date | datetime | str,
    *,
    interval: str = "60m",
    cache_dir: Path | str,
    refresh: bool = False,
    pause: float = 0.0,
) -> pd.DataFrame:
    """Bars for every ticker, cached per ``(ticker, interval)`` and merged on read.

    The cache stores the union of everything ever fetched for that ticker, so asking a second time
    for an overlapping window costs nothing. ``refresh`` forces a re-download - required when the
    question is whether today's bars are visible yet.
    """
    cache_root = Path(cache_dir)
    wanted_start = pd.Timestamp(start).tz_localize("UTC")
    wanted_end = pd.Timestamp(end).tz_localize("UTC")
    wanted_start, _clamped = clamp_window(wanted_start, wanted_end, interval)
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        path = _cache_path(cache_root, ticker, interval)
        cached = pd.read_parquet(path) if path.is_file() and not refresh else None
        if cached is not None and not cached.empty:
            cached["ts"] = pd.to_datetime(cached["ts"], utc=True)
            covers_end = cached["ts"].max() >= wanted_end - pd.Timedelta(days=3)
            first_ok = cached["ts"].min() <= wanted_start + pd.Timedelta(days=3)
            if covers_end and first_ok:
                frames.append(cached)
                continue
        downloaded = _download(ticker, wanted_start, wanted_end, interval)
        merged = downloaded
        if cached is not None and not cached.empty and not downloaded.empty:
            merged = (
                pd.concat([cached, downloaded], ignore_index=True)
                .drop_duplicates(subset=["ts"], keep="last")
                .sort_values("ts")
                .reset_index(drop=True)
            )
        if merged.empty:
            _LOG.warning("%s: no %s bars in %s..%s", ticker, interval, start, end)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path, index=False)
        frames.append(merged)
        if pause:
            time.sleep(pause)
    if not frames:
        return pd.DataFrame(columns=list(BAR_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    out = out.loc[(out["ts"] >= wanted_start) & (out["ts"] < wanted_end + pd.Timedelta(days=1))]
    return out.sort_values(["ticker", "ts"]).reset_index(drop=True)


def load_minute_bars(cache_dir: Path | str, interval: str = "60m") -> pd.DataFrame:
    """Read every cached ticker without touching the network."""
    root = Path(cache_dir) / interval
    if not root.is_dir():
        return pd.DataFrame(columns=list(BAR_COLUMNS))
    frames = []
    for path in sorted(root.glob("*.parquet")):
        frame = pd.read_parquet(path)
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=list(BAR_COLUMNS))
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "ts"]).reset_index(drop=True)


def _download(ticker: str, start: pd.Timestamp, end: pd.Timestamp, interval: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dependency is optional at runtime
        raise MinutesUnavailableError(
            "yfinance is required for intraday bars: pip install yfinance"
        ) from exc
    try:
        raw = yf.download(
            tickers=ticker,
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=False,
            actions=False,
            prepost=False,
            group_by="column",
            threads=False,
            progress=False,
        )
    except Exception as exc:  # noqa: BLE001 - provider errors must never be swallowed
        raise MinutesUnavailableError(f"{ticker}: yfinance failed for {interval}: {exc}") from exc
    return _normalise_frame(raw, ticker, interval)


def probe_depth(
    tickers: Sequence[str],
    *,
    interval: str = "60m",
    cache_dir: Path | str,
    lookback_days: int = 800,
) -> list[ProbeResult]:
    """How far back the provider actually serves this interval right now.

    This is the guard that turns "the backtest looks fine" into "the backtest has six sessions".
    It reports the real first and last bar rather than the window that was requested.
    """
    end = pd.Timestamp.now(tz="UTC").normalize()
    start = end - pd.Timedelta(days=int(lookback_days))
    frame = fetch_minute_bars(
        tickers, start.date(), end.date(), interval=interval, cache_dir=cache_dir
    )
    results: list[ProbeResult] = []
    for ticker in tickers:
        part = frame.loc[frame["ticker"] == ticker]
        if part.empty:
            results.append(ProbeResult(ticker, interval, 0, None, None, 0, float("nan")))
            continue
        sessions = int(part["session"].nunique())
        results.append(
            ProbeResult(
                ticker=ticker,
                interval=interval,
                n_bars=int(len(part)),
                first=part["ts"].min().to_pydatetime(),
                last=part["ts"].max().to_pydatetime(),
                n_sessions=sessions,
                bars_per_session=float(len(part) / sessions) if sessions else float("nan"),
            )
        )
    return results
