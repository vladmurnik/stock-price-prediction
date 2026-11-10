"""Macro release calendar: when the print hits the tape, in UTC.

The crypto news branch kept losing to a 14-28 bps round trip because the headline is
read *after* the move. Macro releases have the opposite property: the timestamp is
published months in advance, so the only open question is whether the move is big
enough to pay the fee.

Two sources, both machine-readable:

* BLS publishes every CPI and Employment Situation release as
  ``https://www.bls.gov/news.release/archives/{cpi|empsit}_MMDDYYYY.htm`` and keeps an
  index of them, so the release date is literally part of the file name;
* the Federal Reserve keeps one page per year of FOMC meetings, and every scheduled
  meeting has a ``fomcminutesYYYYMMDD.htm`` document - again, the date is in the name.

Release times are fixed by policy: 08:30 America/New_York for CPI and payrolls, 14:00
America/New_York for the FOMC statement. The calendar therefore stores the *local* date
and converts it with the tz database, which is what keeps a January release at 13:30 UTC
and a July release at 12:30 UTC.

Nothing here reads prices; the calendar is independent of the sample it is tested on.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

__all__ = [
    "CALENDAR_COLUMNS",
    "EASTERN",
    "EVENT_KINDS",
    "RELEASE_TIME",
    "calendar_frame",
    "load_calendar",
    "parse_bls_archive",
    "parse_fomc_archive",
    "release_utc",
    "write_calendar",
]

EASTERN = ZoneInfo("America/New_York")

#: Local wall-clock time of each release, as published by the agency.
RELEASE_TIME: dict[str, time] = {
    "cpi": time(8, 30),
    "nfp": time(8, 30),
    "fomc": time(14, 0),
}

EVENT_KINDS: tuple[str, ...] = tuple(RELEASE_TIME)
CALENDAR_COLUMNS: tuple[str, ...] = ("event", "date", "release_ts")

_BLS_PATTERN = {
    "cpi": re.compile(r"cpi_(\d{2})(\d{2})(\d{4})\.htm", re.IGNORECASE),
    "nfp": re.compile(r"empsit_(\d{2})(\d{2})(\d{4})\.htm", re.IGNORECASE),
}
_FOMC_PATTERN = re.compile(r"fomcminutes(20\d{6})\.htm", re.IGNORECASE)


def _safe_date(month: int, day: int, year: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_bls_archive(text: str, kind: str) -> list[date]:
    """Release dates found in a BLS archive index page for ``kind``.

    The page lists one link per historical release and the link target carries the
    release date as ``MMDDYYYY``. Duplicates and impossible dates are dropped, and the
    result is sorted so the calendar is deterministic.
    """
    pattern = _BLS_PATTERN.get(kind)
    if pattern is None:
        raise ValueError(f"unknown BLS release kind: {kind!r}")
    found: set[date] = set()
    for month, day, year in pattern.findall(text):
        parsed = _safe_date(int(month), int(day), int(year))
        if parsed is not None and 1990 <= parsed.year <= 2100:
            found.add(parsed)
    return sorted(found)


def parse_fomc_archive(text: str) -> list[date]:
    """Scheduled FOMC meeting end dates from a Federal Reserve calendar page.

    Every scheduled meeting publishes minutes, and the minutes filename is the meeting
    date, so the calendar and the documents cannot drift apart. Emergency meetings are
    deliberately out of scope: their release time is not the 14:00 statement slot.
    """
    found: set[date] = set()
    for stamp in _FOMC_PATTERN.findall(text):
        parsed = datetime.strptime(stamp, "%Y%m%d").date()
        if 1990 <= parsed.year <= 2100:
            found.add(parsed)
    return sorted(found)


def release_utc(day: date, kind: str) -> pd.Timestamp:
    """The UTC instant at which ``kind`` is released on ``day``.

    ``day`` is the local (America/New_York) publication date, which is how both agencies
    label it. DST is handled by the tz database, not by a fixed offset.
    """
    local_time = RELEASE_TIME.get(kind)
    if local_time is None:
        raise ValueError(f"unknown release kind: {kind!r}")
    local = datetime.combine(day, local_time, tzinfo=EASTERN)
    return pd.Timestamp(local).tz_convert("UTC")


def calendar_frame(
    bls_pages: dict[str, str],
    fomc_pages: Iterable[str],
    *,
    since: date | None = None,
    until: date | None = None,
) -> pd.DataFrame:
    """Assemble the event calendar from raw index pages.

    ``bls_pages`` maps ``"cpi"``/``"nfp"`` to the text of its archive index; ``fomc_pages``
    is every yearly calendar page. The frame is sorted by release instant and carries no
    price information at all.
    """
    rows: list[dict[str, object]] = []
    for kind, text in bls_pages.items():
        for day in parse_bls_archive(text, kind):
            rows.append({"event": kind, "date": day, "release_ts": release_utc(day, kind)})
    for text in fomc_pages:
        for day in parse_fomc_archive(text):
            rows.append({"event": "fomc", "date": day, "release_ts": release_utc(day, "fomc")})
    frame = pd.DataFrame(rows, columns=list(CALENDAR_COLUMNS))
    stamps = pd.to_datetime(frame["release_ts"], utc=True)
    frame["release_ts"] = stamps.astype("datetime64[ns, UTC]")
    if since is not None:
        frame = frame[frame["date"] >= since]
    if until is not None:
        frame = frame[frame["date"] <= until]
    frame = frame.drop_duplicates(subset=["event", "date"]).sort_values("release_ts")
    return frame.reset_index(drop=True)


def write_calendar(frame: pd.DataFrame, path: Path | str) -> Path:
    """Persist the calendar as CSV (dates as ISO, instants as UTC)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out["release_ts"] = pd.to_datetime(out["release_ts"], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    out.to_csv(target, index=False)
    return target


def load_calendar(path: Path | str) -> pd.DataFrame:
    """Read a calendar written by :func:`write_calendar`."""
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["release_ts"] = pd.to_datetime(frame["release_ts"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    return frame[list(CALENDAR_COLUMNS)]
