"""Turn articles into tradeable events: one headline, one position, one measurable outcome.

The daily experiment aggregated every article of a ticker-day into one number and then asked
whether tomorrow's return was predictable. That design cannot answer the question this module
exists for, because it destroys the only thing that matters for a reaction test: **the clock**.

Here an event is ``(article, ticker, published_at)``. The rule is mechanical and stated before any
result is looked at:

* entry at the **open of the first bar whose start is at or after** ``published_at + lag``;
* exit at the **close of the bar that closes at or after** ``entry + horizon`` - that is the last
  bar starting before the target time, so a 60-minute horizon on 60-minute bars is held for
  exactly one bar and not for two;
* outcome is the ticker return **minus SPY over exactly the same two timestamps**.

Four properties are enforced rather than assumed:

1. **The lag invariant** - no event can be built whose entry precedes ``published_at + lag``.
   ``rejected_lag`` counts violations and must be zero.
2. **No silent drops** - every article rejected for a structural reason increments a named
   counter, and :meth:`DropStats.check` refuses a funnel that does not balance.
3. **Direction is explicit** - ``y`` is the return of the *intended* trade (long on positive
   sentiment, short on negative). An article with no sentiment produces no trade; it is counted
   as ``zero_direction`` rather than quietly treated as a long.
4. **Diagnostics are separate from the funnel** - a missing SPY bar for one of three horizons does
   not change the number of articles kept, so it must not be added to the drop counters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from spp.live.minutes import ET
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
    #: Per-horizon losses that must not disturb the article-level funnel.
    diagnostics: dict[str, int] = field(default_factory=dict)

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
        payload.update({f"diag_{k}": v for k, v in self.diagnostics.items()})
        return payload

    def note(self, key: str, amount: int = 1) -> None:
        """Record a per-horizon loss without touching the balance."""
        self.diagnostics[key] = self.diagnostics.get(key, 0) + int(amount)

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


def _dedupe_key(frame: pd.DataFrame) -> pd.Series:
    """Headline identity: ``title_hash`` when a caller provides it, else ``text_hash``.

    ``news_raw.parquet`` has no ``title_hash``; keying the reprint check on ``article_id`` would
    make every copy a separate event and inflate both the count and the apparent significance.
    """
    for column in ("title_hash", "text_hash", "title_norm"):
        if column in frame.columns:
            key = frame[column]
            key = key.where(key.notna(), frame["article_id"])
            return key.astype(str)
    return frame["article_id"].astype(str)


def _starts_ns(frame: pd.DataFrame) -> np.ndarray:
    """Bar starts in nanoseconds since epoch.

    pandas 3 returns ``datetime64[us, UTC]``; ``astype('int64')`` would then be microseconds while
    ``Timestamp.value`` is nanoseconds, and every search would land past the end of the session.
    """
    return frame["ts"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()


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
    bar_minutes: int = 60,
) -> tuple[pd.DataFrame, DropStats]:
    """Build the event table. See the module docstring for the exact timing rule.

    ``news`` needs ``article_id``, ``ticker`` and ``published_at``; a sentiment ``score`` column is
    required for a directional ``y``, and an article without one is counted, not assumed long.
    """
    stats = DropStats(total=int(len(news)))
    if news.empty or bars.empty:
        # No bars at all is a data failure, not an empty result: every article is accounted for
        # under one reason so the funnel still balances and the caller can see why nothing came out.
        stats.no_bars_for_session += int(len(news))
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
    # the window, keyed by ticker and the normalised headline.
    order = work.assign(_key=_dedupe_key(work)).sort_values("published_at")
    seen: dict[tuple[str, str], pd.Timestamp] = {}
    keep_mask = np.ones(len(order), dtype=bool)
    for position, (ticker, dedupe_key, stamp) in enumerate(
        zip(order["ticker"], order["_key"], order["published_at"], strict=True)
    ):
        pair = (str(ticker), dedupe_key)
        previous = seen.get(pair)
        if previous is not None and (stamp - previous) <= pd.Timedelta(hours=dedupe_hours):
            keep_mask[position] = False
            stats.duplicate += 1
        else:
            seen[pair] = stamp
    work = order.loc[keep_mask].drop(columns=["_key"], errors="ignore")

    index = bars.sort_values(["ticker", "ts"]).reset_index(drop=True)
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
        articles = eligible.loc[eligible["ticker"] == ticker]
        if articles.empty:
            continue
        for record in articles.itertuples(index=False):
            published = pd.Timestamp(record.published_at)
            session = published.tz_convert(ET).date()
            session_bars = ticker_bars.loc[ticker_bars["session"] == session]
            if session_bars.empty:
                stats.no_bars_for_session += 1
                continue
            starts = _starts_ns(session_bars)
            target = published + pd.Timedelta(minutes=int(lag_minutes))
            position = int(np.searchsorted(starts, int(target.value), side="left"))
            if position >= len(starts):
                stats.after_close += 1
                continue
            entry_bar = session_bars.iloc[position]
            entry_ts = pd.Timestamp(entry_bar["ts"])
            if entry_ts < target:
                stats.rejected_lag += 1
                continue
            entry_price = entry_bar["open"]
            if not np.isfinite(entry_price) or entry_price <= 0:
                entry_price = entry_bar["close"]
            if not np.isfinite(entry_price) or entry_price <= 0:
                stats.no_entry_bar += 1
                continue
            score = float(getattr(record, score_col)) if np.isfinite(
                getattr(record, score_col, np.nan)
            ) else np.nan
            direction = (
                float(signed_direction([score], dead_zone=dead_zone)[0])
                if np.isfinite(score)
                else 0.0
            )
            if direction == 0.0:
                stats.zero_direction += 1
                continue
            if entry_ts not in spy_lookup.index:
                stats.missing_spy += 1
                continue
            spy_entry = float(spy_lookup.loc[entry_ts, "open"])
            if not np.isfinite(spy_entry) or spy_entry <= 0:
                stats.missing_spy += 1
                continue

            emitted = 0
            for horizon in horizons:
                exit_row, truncated = _exit_bar(
                    session_bars, position, int(horizon), int(bar_minutes)
                )
                if exit_row is None:
                    stats.no_exit_bar += 1
                    continue
                exit_ts = pd.Timestamp(exit_row["ts"])
                exit_price = float(exit_row["close"])
                if not np.isfinite(exit_price) or exit_price <= 0:
                    stats.note("nan_exit_price")
                    continue
                if exit_ts not in spy_lookup.index:
                    stats.note("missing_spy_exit")
                    continue
                spy_exit = float(spy_lookup.loc[exit_ts, "close"])
                if not np.isfinite(spy_exit) or spy_exit <= 0:
                    stats.note("missing_spy_exit")
                    continue
                r_ticker = exit_price / float(entry_price) - 1.0
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
                        "y": excess * direction,
                        "truncated": bool(truncated),
                        "entry_delay_min": float((entry_ts - published).total_seconds() / 60.0),
                    }
                )
                emitted += 1
            if emitted:
                stats.kept += 1
            else:
                stats.extra["no_usable_horizon"] = stats.extra.get("no_usable_horizon", 0) + 1

    frame = (
        pd.DataFrame(rows, columns=list(EVENT_COLUMNS))
        if rows
        else pd.DataFrame(columns=list(EVENT_COLUMNS))
    )
    stats.check()
    _LOG.info(
        "events: kept=%d duplicate=%d after_close=%d zero_direction=%d",
        stats.kept,
        stats.duplicate,
        stats.after_close,
        stats.zero_direction,
    )
    return frame, stats


def _exit_bar(
    session_bars: pd.DataFrame, entry_position: int, horizon_minutes: int, bar_minutes: int
) -> tuple[pd.Series | None, bool]:
    """The bar whose close is at or just past ``entry + horizon``.

    Bars are labelled by their start, so the last bar starting before the target is the one that
    straddles it. Taking the first bar starting *after* the target would hold the position for two
    bars on a 60-minute horizon - a silent doubling of every reported exposure.
    """
    starts = _starts_ns(session_bars)
    entry_ts = pd.Timestamp(session_bars.iloc[entry_position]["ts"])
    target_ns = int((entry_ts + pd.Timedelta(minutes=int(horizon_minutes))).value)
    position = int(np.searchsorted(starts, target_ns, side="left")) - 1
    if position < entry_position:
        position = entry_position
    if position < 0 or position >= len(starts):
        return None, False
    row = session_bars.iloc[position]
    close_ns = int(starts[position]) + int(bar_minutes) * 60 * 1_000_000_000
    truncated = bool(close_ns < target_ns)
    return row, truncated
