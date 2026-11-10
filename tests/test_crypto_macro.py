"""Tests for the macro release calendar."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from spp.crypto import macro

BLS_PAGE = """
<html><body>
<a href="/news.release/archives/cpi_01112024.htm">CPI January 2024</a>
<a href="/news.release/archives/cpi_03122024.htm">CPI February 2024</a>
<a href="/news.release/archives/cpi_13022024.htm">not a date</a>
<a href="/news.release/archives/cpi_01112024.htm">duplicate</a>
</body></html>
"""

FOMC_PAGE = """
<td><a href="/monetarypolicy/fomcminutes20240131.htm">Minutes</a></td>
<td><a href="/monetarypolicy/fomcminutes20240320.htm">Minutes</a></td>
<td><a href="/monetarypolicy/fomcminutes20240131.htm">duplicate</a></td>
"""


def test_parse_bls_archive_keeps_only_valid_links() -> None:
    assert macro.parse_bls_archive(BLS_PAGE, "cpi") == [date(2024, 1, 11), date(2024, 3, 12)]


def test_parse_bls_archive_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        macro.parse_bls_archive(BLS_PAGE, "ppi")


def test_parse_fomc_archive_sorts_and_deduplicates() -> None:
    assert macro.parse_fomc_archive(FOMC_PAGE) == [date(2024, 1, 31), date(2024, 3, 20)]


@pytest.mark.parametrize(
    ("day", "kind", "expected"),
    [
        (date(2024, 1, 11), "cpi", "2024-01-11T13:30:00Z"),
        (date(2024, 3, 12), "cpi", "2024-03-12T12:30:00Z"),
        (date(2024, 7, 5), "nfp", "2024-07-05T12:30:00Z"),
        (date(2024, 1, 31), "fomc", "2024-01-31T19:00:00Z"),
        (date(2024, 7, 31), "fomc", "2024-07-31T18:00:00Z"),
    ],
)
def test_release_utc_follows_daylight_saving(day: date, kind: str, expected: str) -> None:
    assert macro.release_utc(day, kind) == pd.Timestamp(expected)


def test_release_utc_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        macro.release_utc(date(2024, 1, 11), "cpi_core")


def test_calendar_frame_sorts_filters_and_deduplicates() -> None:
    frame = macro.calendar_frame(
        {"cpi": BLS_PAGE, "nfp": BLS_PAGE.replace("cpi_", "empsit_")},
        [FOMC_PAGE],
        since=date(2024, 2, 1),
        until=date(2024, 12, 31),
    )
    assert list(frame.columns) == list(macro.CALENDAR_COLUMNS)
    assert frame["release_ts"].is_monotonic_increasing
    assert set(frame["event"]) == {"cpi", "nfp", "fomc"}
    assert len(frame[frame["event"] == "cpi"]) == 1  # January is filtered out by `since`
    assert len(frame) == 3  # one CPI, one payrolls, one FOMC


def test_calendar_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    frame = macro.calendar_frame({"cpi": BLS_PAGE}, [FOMC_PAGE])
    path = macro.write_calendar(frame, tmp_path / "calendar.csv")
    restored = macro.load_calendar(path)
    pd.testing.assert_frame_equal(restored.reset_index(drop=True), frame.reset_index(drop=True))
