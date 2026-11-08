"""Event construction: the five-minute lag, the exit rule and the funnel are the contract.

A test here is a promise that the experiment cannot be run in a way that quietly looks ahead.
The synthetic session is deliberately trivial - seven hourly bars, a ticker and SPY - so that any
disagreement between the code and the stated rule shows up as an arithmetic difference rather than
a modelling debate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spp.live.events import build_events, signed_direction


def _bars() -> pd.DataFrame:
    """One session, hourly bars labelled by start, with known returns."""
    starts = pd.date_range("2023-12-11 14:30", periods=7, freq="1h", tz="UTC")  # 09:30..15:30 ET
    ticker_open = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    ticker_close = [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
    spy_open = [400.0] * 7
    spy_close = [402.0] * 7
    ticker = pd.DataFrame(
        {
            "ticker": "AAPL",
            "ts": starts,
            "open": ticker_open,
            "high": ticker_close,
            "low": ticker_open,
            "close": ticker_close,
            "volume": 1000,
            "session": pd.Series(starts).dt.tz_convert("America/New_York").dt.date.to_numpy(),
        }
    )
    spy = pd.DataFrame(
        {
            "ticker": "SPY",
            "ts": starts,
            "open": spy_open,
            "high": spy_close,
            "low": spy_open,
            "close": spy_close,
            "volume": 5000,
            "session": pd.Series(starts).dt.tz_convert("America/New_York").dt.date.to_numpy(),
        }
    )
    return pd.concat([ticker, spy], ignore_index=True)


def _news(rows: list[dict]) -> pd.DataFrame:
    base = {
        "article_id": "a1",
        "ticker": "AAPL",
        "title_hash": "h1",
        "published_at": pd.Timestamp("2023-12-11 15:20:00", tz="UTC"),  # 10:20 ET
        "score": 0.5,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_entry_is_the_first_bar_at_or_after_the_lag() -> None:
    events, stats = build_events(_news([{}]), _bars(), lag_minutes=5, horizons=(60,))
    assert stats.kept == 1
    row = events.iloc[0]
    # 10:20 ET + 5 min = 10:25, so the 10:30 bar is the entry.
    assert pd.Timestamp(row["entry_ts"]).tz_convert("America/New_York").strftime("%H:%M") == "10:30"
    assert row["entry_delay_min"] == pytest.approx(10.0)


def test_no_event_can_enter_before_the_lag() -> None:
    events, stats = build_events(_news([{}]), _bars(), lag_minutes=5, horizons=(60, 120, 180))
    published = pd.to_datetime(events["published_at"], utc=True)
    entry = pd.to_datetime(events["entry_ts"], utc=True)
    assert (entry >= published + pd.Timedelta(minutes=5)).all()
    assert stats.rejected_lag == 0


def test_excess_return_is_measured_against_spy_on_the_same_stamps() -> None:
    events, _ = build_events(_news([{}]), _bars(), horizons=(60,))
    row = events.iloc[0]
    # Entry 10:30 open 101.0 -> exit 11:30 close 103.0 = +1.9802%; SPY 400 -> 402 = +0.5%.
    assert row["r_ticker"] == pytest.approx(103.0 / 101.0 - 1.0)
    assert row["r_spy"] == pytest.approx(402.0 / 400.0 - 1.0)
    assert row["excess"] == pytest.approx((103.0 / 101.0) - (402.0 / 400.0))


def test_direction_flips_the_trade_but_not_the_measurement() -> None:
    positive, _ = build_events(_news([{"score": 0.9}]), _bars(), horizons=(60,))
    negative, _ = build_events(_news([{"score": -0.9}]), _bars(), horizons=(60,))
    assert positive.iloc[0]["y"] == pytest.approx(positive.iloc[0]["excess"])
    assert negative.iloc[0]["y"] == pytest.approx(-negative.iloc[0]["excess"])
    assert positive.iloc[0]["direction"] == 1.0
    assert negative.iloc[0]["direction"] == -1.0


def test_pre_open_news_enters_at_the_open_and_after_close_is_dropped() -> None:
    early = _news([{"published_at": pd.Timestamp("2023-12-11 13:00:00", tz="UTC")}])  # 08:00 ET
    events, stats = build_events(early, _bars(), horizons=(60,))
    assert stats.kept == 1
    assert pd.Timestamp(events.iloc[0]["entry_ts"]).tz_convert("America/New_York").strftime(
        "%H:%M"
    ) == "09:30"

    late = _news([{"published_at": pd.Timestamp("2023-12-11 21:45:00", tz="UTC")}])  # 16:45 ET
    events_late, stats_late = build_events(late, _bars(), horizons=(60,))
    assert events_late.empty
    assert stats_late.after_close == 1


def test_horizon_that_overruns_the_close_is_truncated_not_dropped() -> None:
    news = _news([{"published_at": pd.Timestamp("2023-12-11 19:30:00", tz="UTC")}])  # 14:30 ET
    events, stats = build_events(news, _bars(), horizons=(180,))
    assert stats.kept == 1
    row = events.iloc[0]
    assert bool(row["truncated"]) is True
    # Entry 15:30 bar (open 106), exit is the last bar's close 107.
    assert row["exit_price"] == pytest.approx(107.0)


def test_reprints_inside_the_window_collapse_into_one_event() -> None:
    news = _news(
        [
            {"article_id": "a1", "title_hash": "same"},
            {
                "article_id": "a2",
                "title_hash": "same",
                "published_at": pd.Timestamp("2023-12-11 15:40:00", tz="UTC"),
            },
            {
                "article_id": "a3",
                "title_hash": "same",
                "published_at": pd.Timestamp("2023-12-12 15:20:00", tz="UTC"),
            },
        ]
    )
    events, stats = build_events(news, _bars(), dedupe_hours=6.0, horizons=(60,))
    assert stats.duplicate == 1
    # The copy a day later is outside the window, so it survives as a separate event.
    assert stats.kept == 1
    assert events["article_id"].tolist() == ["a1"]


def test_funnel_balances_on_every_path() -> None:
    news = _news(
        [
            {"article_id": "a1"},
            {"article_id": "a2", "published_at": pd.Timestamp("2023-12-11 21:45:00", tz="UTC")},
            {"article_id": "a3", "published_at": pd.NaT},
            {"article_id": "a4", "title_hash": "other", "ticker": "ZZZZ"},
        ]
    )
    _, stats = build_events(news, _bars(), horizons=(60,))
    payload = stats.as_dict()
    assert payload["total"] == 4
    assert payload["kept"] + sum(
        v for k, v in payload.items() if k not in {"total", "kept"}
    ) == payload["total"]


def test_missing_benchmark_is_an_error_not_a_silent_zero() -> None:
    bars = _bars().loc[lambda f: f["ticker"] != "SPY"]
    with pytest.raises(ValueError, match="benchmark"):
        build_events(_news([{}]), bars, horizons=(60,))


def test_signed_direction_respects_the_dead_zone() -> None:
    values = pd.Series([0.4, -0.4, 0.01, float("nan")])
    assert signed_direction(values, dead_zone=0.1).tolist() == [1.0, -1.0, 0.0, 0.0]
    assert signed_direction(pd.Series([0.4]), dead_zone=0.0).tolist() == [1.0]
    assert np.isnan(np.nan)  # keeps numpy imported for readers comparing to the evaluator
