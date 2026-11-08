"""Bar hygiene: the calendar is the part of an intraday study that fails silently.

The tests here are adversarial on purpose. Every one of them corresponds to a mistake that would
still produce a plausible backtest: naive timestamps read as local time, a 16:00 bar counted as
tradeable, a bar from an overnight session kept because nobody checked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spp.live.minutes import (
    _interval_minutes,
    _normalise_frame,
    bar_label_end,
    in_session,
    session_of,
    to_new_york,
)


def test_naive_timestamps_are_utc_not_local() -> None:
    naive = pd.Series(pd.to_datetime(["2023-12-11 14:30:00"]))
    converted = to_new_york(naive)
    assert str(converted.tz) == "America/New_York"
    assert converted[0].hour == 9 and converted[0].minute == 30


def test_summer_and_winter_shifts_differ() -> None:
    summer = to_new_york(pd.Series(pd.to_datetime(["2023-07-10 13:30:00"], utc=True)))
    winter = to_new_york(pd.Series(pd.to_datetime(["2023-12-11 14:30:00"], utc=True)))
    assert (summer[0].hour, summer[0].minute) == (9, 30)
    assert (winter[0].hour, winter[0].minute) == (9, 30)


def test_session_boundaries_are_half_open() -> None:
    stamps = pd.Series(
        pd.to_datetime(
            [
                "2023-12-11 09:29:00",
                "2023-12-11 09:30:00",
                "2023-12-11 15:59:00",
                "2023-12-11 16:00:00",
                "2023-12-11 20:00:00",
            ]
        ).tz_localize("America/New_York")
    )
    assert in_session(stamps).tolist() == [False, True, True, False, False]


def test_session_date_uses_new_york_not_utc() -> None:
    stamps = pd.Series(pd.to_datetime(["2023-12-12 01:00:00"], utc=True))
    # 01:00 UTC on the 12th is 20:00 ET on the 11th: the session, not the UTC date.
    assert session_of(stamps).iloc[0] == pd.Timestamp("2023-12-11").date()


def test_interval_parsing_and_bar_end() -> None:
    assert _interval_minutes("60m") == 60
    assert _interval_minutes("5m") == 5
    assert _interval_minutes("2h") == 120
    with pytest.raises(ValueError):
        _interval_minutes("1d")
    starts = pd.DatetimeIndex(pd.to_datetime(["2023-12-11 14:30:00"], utc=True))
    ends = bar_label_end(starts, "60m")
    assert ends[0] - starts[0] == pd.Timedelta(minutes=60)


def test_normalise_drops_out_of_session_bars_and_sorts() -> None:
    index = pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2023-12-11 15:30:00",  # 10:30 ET
                "2023-12-11 14:30:00",  # 09:30 ET
                "2023-12-11 21:00:00",  # 16:00 ET - outside the half-open session
                "2023-12-11 22:30:00",  # 17:30 ET - after hours
            ],
            utc=True,
        )
    )
    raw = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0, 4.0],
            "High": [1.0, 2.0, 3.0, 4.0],
            "Low": [1.0, 2.0, 3.0, 4.0],
            "Close": [1.1, 2.1, 3.1, 4.1],
            "Volume": [10, 20, 30, 40],
        },
        index=index,
    )
    frame = _normalise_frame(raw, "AAPL", "60m")
    assert len(frame) == 2
    assert frame["ts"].is_monotonic_increasing
    assert frame["close"].tolist() == [2.1, 1.1]
    assert frame["session"].iloc[0] == pd.Timestamp("2023-12-11").date()
    assert set(frame.columns) >= {"ticker", "ts", "open", "close", "session"}


def test_normalise_handles_a_multiindex_column_layout() -> None:
    index = pd.DatetimeIndex(pd.to_datetime(["2023-12-11 14:30:00"], utc=True))
    raw = pd.DataFrame(
        [[1.0, 1.0, 1.0, 1.2, 5]],
        index=index,
        columns=pd.MultiIndex.from_tuples(
            [
                ("Open", "AAPL"),
                ("High", "AAPL"),
                ("Low", "AAPL"),
                ("Close", "AAPL"),
                ("Volume", "AAPL"),
            ]
        ),
    )
    frame = _normalise_frame(raw, "AAPL", "60m")
    assert len(frame) == 1
    assert np.isclose(frame["close"].iloc[0], 1.2)
