"""End-to-end checks on the event panel.

Each test here corresponds to a defect that a reviewer found by probing the real code rather than
by reading it, and each one was invisible to the unit tests that shipped with the module: they
tested the pieces separately, so nobody noticed the panel and the evaluator disagreed about the
name of the date column, or that a 60-minute horizon was actually held for two bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spp.live.events import build_events
from spp.live.intraday import attach_trend, evaluate_panel


def _bars() -> pd.DataFrame:
    starts = pd.date_range("2023-12-11 14:30", periods=7, freq="1h", tz="UTC")  # 09:30..15:30 ET
    session = pd.Series(starts).dt.tz_convert("America/New_York").dt.date.to_numpy()
    ticker = pd.DataFrame(
        {
            "ticker": "AAPL",
            "ts": starts,
            "open": np.linspace(100.0, 106.0, 7),
            "high": np.linspace(101.0, 107.0, 7),
            "low": np.linspace(100.0, 106.0, 7),
            "close": np.linspace(101.0, 107.0, 7),
            "volume": 1000,
            "session": session,
        }
    )
    spy = pd.DataFrame(
        {
            "ticker": "SPY",
            "ts": starts,
            "open": 400.0,
            "high": 402.0,
            "low": 400.0,
            "close": 402.0,
            "volume": 5000,
            "session": session,
        }
    )
    return pd.concat([ticker, spy], ignore_index=True)


def _news(rows: list[dict] | None = None) -> pd.DataFrame:
    base = {
        "article_id": "a1",
        "ticker": "AAPL",
        "text_hash": "h1",
        "published_at": pd.Timestamp("2023-12-11 15:20:00", tz="UTC"),  # 10:20 ET
        "score": 0.5,
    }
    return pd.DataFrame([{**base, **row} for row in (rows or [{}])])


def test_a_sixty_minute_horizon_is_held_for_exactly_one_bar() -> None:
    events, _ = build_events(_news(), _bars(), horizons=(60,), bar_minutes=60)
    row = events.iloc[0]
    entry = pd.Timestamp(row["entry_ts"])
    exit_ = pd.Timestamp(row["exit_ts"])
    # Entry at 10:30, exit at the close of that same hourly bar: one hour of exposure, not two.
    assert exit_ == entry
    assert entry.tz_convert("America/New_York").strftime("%H:%M") == "10:30"


def test_the_panel_can_be_evaluated_without_renaming_anything() -> None:
    events, stats = build_events(_news(), _bars(), horizons=(60, 120))
    assert stats.kept == 1
    report = evaluate_panel(events)
    # The regression this pins: build_events produced ``session`` while the evaluator grouped by
    # ``date``, so any real panel raised KeyError and no report was ever produced.
    assert report["n_events"] == 2
    assert set(report["horizons"]) == {"h60", "h120"}
    assert report["lag_violations"] == 0


def test_trend_flag_never_uses_the_session_it_labels() -> None:
    days = pd.bdate_range("2023-01-02", periods=60).date
    prices = []
    for ticker, level, slope in (("AAPL", 100.0, 1.0), ("SPY", 400.0, 0.0)):
        for index, day in enumerate(days):
            prices.append(
                {"ticker": ticker, "date": day, "adj_close": level + slope * index, "close": level}
            )
    frame = pd.DataFrame(prices)
    last_day = days[-1]
    events = pd.DataFrame(
        {"session": [last_day], "ticker": ["AAPL"], "y": [0.0], "entry_ts": [None]}
    )
    tagged = attach_trend(events, frame, window=50)
    # On the last day the previous close is far above its own mean, so the flag is True - and it
    # would still be True if today's close were collapsed, which is the property being tested.
    assert bool(tagged["trend_up"].iloc[0]) is True
    collapsed = frame.copy()
    collapsed.loc[
        (collapsed["ticker"] == "AAPL") & (collapsed["date"] == last_day), "adj_close"
    ] = 1.0
    tagged_collapsed = attach_trend(events, collapsed, window=50)
    assert bool(tagged_collapsed["trend_up"].iloc[0]) is True


def test_an_article_without_sentiment_is_counted_not_assumed_long() -> None:
    events, stats = build_events(_news([{"score": np.nan}]), _bars(), horizons=(60,))
    assert events.empty
    assert stats.zero_direction == 1
    assert stats.kept == 0


def test_reprints_are_matched_on_text_hash_when_title_hash_is_absent() -> None:
    news = _news(
        [
            {"article_id": "a1", "text_hash": "same"},
            {
                "article_id": "a2",
                "text_hash": "same",
                "published_at": pd.Timestamp("2023-12-11 15:25:00", tz="UTC"),
            },
        ]
    )
    _, stats = build_events(news, _bars(), horizons=(60,), dedupe_hours=6.0)
    assert stats.duplicate == 1
    assert stats.kept == 1


def test_a_missing_benchmark_bar_for_one_horizon_does_not_break_the_funnel() -> None:
    bars = _bars()
    spy_mask = (bars["ticker"] == "SPY") & (bars["ts"] == bars["ts"].iloc[2])  # 11:30 ET
    broken = bars.drop(index=bars.index[spy_mask]).reset_index(drop=True)
    # The entry still has its SPY bar; the 120-minute horizon loses its exit bar.
    events, stats = build_events(_news(), broken, horizons=(120, 60))
    assert stats.kept == 1
    assert stats.diagnostics.get("missing_spy_exit", 0) == 1
    assert len(events) == 1


def test_evaluation_refuses_to_invent_an_excess_return_without_spy() -> None:
    bars = _bars().loc[lambda f: f["ticker"] != "SPY"]
    with pytest.raises(ValueError, match="benchmark"):
        build_events(_news(), bars, horizons=(60,))
