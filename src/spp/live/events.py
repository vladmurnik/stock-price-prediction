"""Turn articles into tradeable events: one headline, one position, one measurable outcome.

The daily experiment aggregated every article of a ticker-day into one number and then asked
whether tomorrow's return was predictable. That design cannot answer the question this module
exists for, because it destroys the only thing that matters for a reaction test: **the clock**.

Here an event is ``(article, ticker, published_at)``. The rule is mechanical and stated before
any result is looked at:

* entry at the **open of the first bar whose start is at or after** ``published_at + lag``;
* exit at the **close of the first bar at or after** ``entry + horizon``, or at the session close
  if the horizon runs past 16:00 ET (flagged ``truncated``);
* outcome is the ticker return **minus SPY over exactly the same two timestamps**.

Three properties are deliberately enforced rather than assumed:

1. **The lag invariant** - no event can be built whose entry precedes ``published_at + lag``.
   ``violates_lag`` counts violations and must be zero; the frame is rejected otherwise.
2. **No silent drops** - every article rejected for a structural reason increments a named
   counter. A run that keeps four hundred of fifty thousand articles must say so.
3. **Direction is explicit** - ``y`` is the return of the *intended* trade (long on positive
   sentiment, short on negative), so a mean above the cost means the rule made money. The raw
   excess return stays in ``excess`` for anyone wanting to undo the direction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from spp.live.minutes import ET, in_session
from spp.logging import get_logger

__all__ = [
    "DropStats",
    "EVENT_COLUMNS",
    "build_events",
    "signed_direction",
]

_LOG = get_logger("live.events")

EVENT_COLUMNS = (
    "article_id",
    "ticker",
    "published_at",
    "session",
    "horizon",
    "entry_ts",
    "entry_price",
    "exit_ts",
    "exit_price",
    "r_ticker",
    "r_spy",
    "excess",
    "direction",
    "y",
    "truncated",
    "entry_delay_min",
)


@dataclass
class DropStats:
    """Why articles did not become events. Every field is a reason, not a stage."""

    total: int = 0
    kept: int = 0
    duplicate: int = 0
    no_publication_time: int = 0
    no_bars_for_session: int = 0
    after_close: int = 0
    no_entry_bar: int = 0
    no_exit_bar: int = 0
    missing_spy: int = 0
    zero_direction: int = 0
    rejected_lag: int = 0
    extra: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, int]:
        payload = {
            "total": self.total,
            "kept": self.kept,
            "duplicate": self.duplicate,
            "no_publication_time": self.no_publication_time,
            "no_bars_for_session": self.no_bars_for_session,
            "after_close": self.after_close,
            "no_entry_bar": self.no_entry_bar,
            "no_exit_bar": self.no_exit_bar,
            "missing_spy": self.missing_spy,
            "zero_direction": self.zero_direction,
            "rejected_lag": self.rejected_lag,
        }
        payload.update(self.extra)
        return payload

    def check(self) -> None:
        """Fail loudly when the funnel does not add up, instead of reporting a subset as if it
        were the whole corpus."""
        dropped = (
            self.duplicate
            + self.no_publication_time
            + self.no_bars_for_session
            + self.after_close
            + self.no_entry_bar
            + self.no_exit_bar
            + self.missing_spy
            + self.zero_direction
            + self.rejected_lag
            + sum(self.extra.values())
        )
        if self.total and self.kept + dropped != self.total:
            raise ValueError(
                f"event funnel does not balance: total={self.total} kept={self.kept} "
                f"dropped={dropped} (difference {self.total - self.kept - dropped})"
            )


def signed_direction(score: pd.Series | np.ndarray, *, dead_zone: float = 0.0) -> np.ndarray:
    """-1 / +1 from a sentiment score, with an optional band that produces no trade."""
    values = np.asarray(score, dtype="float64")
    direction = np.where(values > dead_zone, 1.0, np.where(values < -dead_zone, -1.0, 0.0))
    direction[~np.isfinite(values)] = 0.0
    return direction


def _bar_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Sorted per ticker with a positional index usable by ``searchsorted``."""
    return frame.sort_values(["ticker", "ts"]).reset_index(drop=True)


def build_events(
    news: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    lag_minutes: int = 5,
    horizons: Sequence[int] = (60, 120, 180),
    spy: str = "SPY",
    score_col: str = "score",
    dead_zone: float = 0.0,
    dedupe_hours: float = 6.0,
) -> tuple[pd.DataFrame, DropStats]:
    """Build the event table. See the module docstring for the exact timing rule.

    ``news`` needs ``article_id``, ``ticker`` and ``published_at``; a sentiment ``score`` column is
    optional and, when absent, every event is directionless (``direction = 0``) and ``y`` equals
    the raw excess return - which is still useful for measuring the *size* of a reaction.
    """
    stats = DropStats(total=int(len(news)))
    if news.empty or bars.empty:
        stats.check()
        return pd.DataFrame(columns=list(EVENT_COLUMNS)), stats

    work = news.copy()
    work["published_at"] = pd.to_datetime(work["published_at"], utc=True, errors="coerce")
    missing_time = int(work["published_at"].isna().sum())
    if missing_time:
        stats.no_publication_time += missing_time
        work = work.loc[work["published_at"].notna()]
    if score_col not in work.columns:
        work[score_col] = np.nan

    # A wire story reprinted by ten outlets is one event, not ten. Keep the earliest copy inside
    # the window, keyed by ticker and the normalised headline where one is available.
    key = work["title_hash"] if "title_hash" in work.columns else work["article_id"]
    order = work.assign(_key=key, _ts=work["published_at"]).sort_values("_ts")
    seen: dict[tuple[str, Any], pd.Timestamp] = {}
    keep_mask = np.ones(len(order), dtype=bool)
    for position, (ticker, dedupe_key, stamp) in enumerate(
        zip(order["ticker"], order["_key"], order["_ts"], strict=True)
    ):
        pair = (str(ticker), dedupe_key)
        previous = seen.get(pair)
        if previous is not None and (stamp - previous) <= pd.Timedelta(hours=dedupe_hours):
            keep_mask[position] = False
            stats.duplicate += 1
        else:
            seen[pair] = stamp
    work = order.loc[keep_mask].drop(columns=["_key", "_ts"], errors="ignore")

    index = _bar_index(bars)
    spy_bars = index.loc[index["ticker"] == spy]
    if spy_bars.empty:
        raise ValueError(f"no benchmark bars for {spy!r}; the excess return is not computable")
    spy_lookup = spy_bars.set_index("ts")

    rows: list[dict[str, Any]] = []
    known = set(index["ticker"].unique())
    # Articles without usable bars must be counted before the loop: a ticker that never appears
    # in the bar frame never enters it, and an uncounted rejection breaks the funnel quietly.
    stats.no_bars_for_session += int(len(work.loc[~work["ticker"].isin(known)]))
    stats.no_bars_for_session += int((work["ticker"] == spy).sum())
    eligible = work.loc[work["ticker"].isin(known) & (work["ticker"] != spy)]
    for ticker, group in index.groupby("ticker", sort=False):
        if ticker == spy:
            continue
        ticker_bars = group.reset_index(drop=True)
        for record in eligible.loc[eligible["ticker"] == ticker].itertuples(index=False):
            published = pd.Timestamp(record.published_at)
            session = published.tz_convert(ET).date()
            session_bars = ticker_bars.loc[ticker_bars["session"] == session]
            if session_bars.empty:
                stats.no_bars_for_session += 1
                continue
            # Bars may arrive at microsecond resolution (pandas 3 defaults to us), so the ns
            # conversion has to be explicit - a bare astype('int64') would silently produce
            # microseconds and make every timestamp look like it is far in the future.
            session_ns = session_bars["ts"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
            target = published + pd.Timedelta(minutes=int(lag_minutes))
            position = int(np.searchsorted(session_ns, int(target.value), side="left"))
            if position >= len(session_ns):
                stats.after_close += 1
                continue
            entry_bar = session_bars.iloc[position]
            entry_ts = pd.Timestamp(entry_bar["ts"])
            if not in_session(pd.DatetimeIndex([entry_ts]))[0]:
                stats.no_entry_bar += 1
                continue
            entry_price = entry_bar["open"]
            if not np.isfinite(entry_price) or entry_price <= 0:
                entry_price = entry_bar["close"]
            if not np.isfinite(entry_price) or entry_price <= 0:
                stats.no_entry_bar += 1
                continue
            if entry_ts < target:
                stats.rejected_lag += 1
                continue
            if entry_ts not in spy_lookup.index:
                stats.missing_spy += 1
                continue
            direction = 0.0
            score_value = getattr(record, score_col, np.nan)
            if np.isfinite(score_value):
                direction = float(signed_direction([score_value], dead_zone=dead_zone)[0])
            for horizon in horizons:
                exit_row, truncated = _exit_bar(session_bars, position, int(horizon))
                if exit_row is None:
                    stats.no_exit_bar += 1
                    continue
                exit_ts = pd.Timestamp(exit_row["ts"])
                if exit_ts not in spy_lookup.index:
                    stats.missing_spy += 1
                    continue
                exit_price = float(exit_row["close"])
                r_ticker = exit_price / float(entry_price) - 1.0
                spy_entry = float(spy_lookup.loc[entry_ts, "open"])
                spy_exit = float(spy_lookup.loc[exit_ts, "close"])
                if not (np.isfinite(spy_entry) and np.isfinite(spy_exit)) or spy_entry <= 0:
                    stats.missing_spy += 1
                    continue
                r_spy = spy_exit / spy_entry - 1.0
                excess = r_ticker - r_spy
                rows.append(
                    {
                        "article_id": record.article_id,
                        "ticker": ticker,
                        "published_at": published,
                        "session": session,
                        "horizon": int(horizon),
                        "entry_ts": entry_ts,
                        "entry_price": float(entry_price),
                        "exit_ts": exit_ts,
                        "exit_price": exit_price,
                        "r_ticker": r_ticker,
                        "r_spy": r_spy,
                        "excess": excess,
                        "direction": direction,
                        "y": excess * direction if direction != 0.0 else excess,
                        "truncated": bool(truncated),
                        "entry_delay_min": float(
                            (entry_ts - published).total_seconds() / 60.0
                        ),
                    }
                )
            stats.kept += 1

    frame = pd.DataFrame(rows, columns=list(EVENT_COLUMNS)) if rows else pd.DataFrame(
        columns=list(EVENT_COLUMNS)
    )
    stats.check()
    _LOG.info(
        "events: kept=%d duplicate=%d after_close=%d",
        stats.kept,
        stats.duplicate,
        stats.after_close,
    )
    return frame, stats


def _exit_bar(
    session_bars: pd.DataFrame, entry_position: int, horizon_minutes: int, lookahead: int = 0
) -> tuple[pd.Series | None, bool]:
    """First bar at or after ``entry + horizon``; the session close when the horizon overruns."""
    entry_ts = pd.Timestamp(session_bars.iloc[entry_position]["ts"])
    target = entry_ts + pd.Timedelta(minutes=int(horizon_minutes) + int(lookahead))
    starts = session_bars["ts"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    position = int(np.searchsorted(starts, int(target.value), side="left"))
    if position < len(starts):
        return session_bars.iloc[position], False
    return session_bars.iloc[-1], True
