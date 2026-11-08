"""Purged, embargoed walk-forward cross-validation (T8).

The scheme answers one question: *can a training row see the test period?* It cannot
if, and only if, three invariants hold on the NYSE spine:

* ``train_end`` is at least ``embargo_days`` **trading sessions** before ``test_start``
  (a calendar gap of "5 days" spans weekends and would leak three sessions);
* the internal ``valid`` slice lives entirely inside that gap (it is used for early
  stopping, so it must never touch test);
* the sets of ``(date, ticker)`` in train and test are disjoint — checked on the
  actual rows, not on the boundaries, so a ``date`` column built from the wrong
  frame is still caught.

Everything is expressed on a trading-session spine computed from the calendar, never
on calendar months: a fold boundary that lands on a weekend is snapped *backwards* to
the last open session so a fold can never contain a non-trading date.

No function here shuffles, samples or sorts by anything but ``(date, ticker)``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spp.config import CvCfg
from spp.logging import get_logger

__all__ = [
    "CV_WINDOW_END",
    "CV_WINDOW_START",
    "Fold",
    "assert_no_overlap",
    "fold_row_counts",
    "iter_splits",
    "make_folds",
    "summarize_folds",
    "write_folds_report",
]

LOG = get_logger("models.cv")

#: The CV window. 2023 is reserved for the single T10 evaluation and is hard-excluded.
CV_WINDOW_START = date(2019, 1, 1)
CV_WINDOW_END = date(2022, 12, 31)
CV_HELD_OUT_YEAR = 2023
MIN_FOLDS = 3


@dataclass(frozen=True)
class Fold:
    """One walk-forward split. All dates are inclusive and lie on the trading spine.

    ``frame`` is an optional, excluded reference to the matrix the fold belongs to.
    When present, :func:`assert_no_overlap` runs the row-level ``(date, ticker)`` check.

    ``embargo_days`` records the embargo the fold promises, so :func:`assert_no_overlap`
    can verify the gap independently of any config object.
    """

    fold_id: int
    train_start: date
    train_end: date
    valid_start: date
    valid_end: date
    test_start: date
    test_end: date
    embargo_days: int = 5
    frame: pd.DataFrame | None = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, str | int]:
        payload = asdict(self)
        del payload["frame"]
        return {k: (v.isoformat() if isinstance(v, date) else v) for k, v in payload.items()}


@dataclass
class _Spine:
    """Normalised trading calendar plus helpers (months are *sessions*, not calendar)."""

    sessions: pd.DatetimeIndex

    def month_end_session(self, year: int, month: int) -> pd.Timestamp:
        """Last session in ``(year, month)``; raises if the month is absent from the spine."""
        last_day = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        subset = self.sessions[self.sessions <= last_day]
        if len(subset) == 0 or subset[-1] < pd.Timestamp(year=year, month=month, day=1):
            raise ValueError(f"no trading session in {year}-{month:02d} on the spine")
        return subset[-1]

    def session_before(self, ts: pd.Timestamp, n: int = 1) -> pd.Timestamp:
        """Session ``n`` places before ``ts`` (``ts`` need not be a session)."""
        subset = self.sessions[self.sessions < ts]
        if len(subset) < n:
            raise ValueError(f"fewer than {n} session(s) before {ts.date()} on the spine")
        return subset[-n]

    def session_at_or_after(self, ts: pd.Timestamp) -> pd.Timestamp:
        subset = self.sessions[self.sessions >= ts]
        if len(subset) == 0:
            raise ValueError(f"no session at or after {ts.date()} on the spine")
        return subset[0]

    def session_at_or_before(self, ts: pd.Timestamp) -> pd.Timestamp:
        subset = self.sessions[self.sessions <= ts]
        if len(subset) == 0:
            raise ValueError(f"no session at or before {ts.date()} on the spine")
        return subset[-1]

    def sessions_between(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        return self.sessions[(self.sessions >= start) & (self.sessions <= end)]

    def count_between(self, start: pd.Timestamp, end: pd.Timestamp) -> int:
        return int(len(self.sessions_between(start, end)))

    def month_diff(self, start: pd.Timestamp, end: pd.Timestamp) -> float:
        """Distance in calendar months between two session timestamps (expanding guard)."""
        return (end - start).days / 30.4375


def _normalize_sessions(dates: Iterable[Any]) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return pd.DatetimeIndex(idx).normalize().unique().sort_values()


def _as_spine(calendar: Any) -> _Spine:
    """Accept a ``pd.DatetimeIndex`` spine, or an object exposing ``.sessions``."""
    if isinstance(calendar, pd.DatetimeIndex):
        sessions = calendar
    elif isinstance(calendar, _Spine):
        return calendar
    elif hasattr(calendar, "sessions"):
        sessions = calendar.sessions
    else:  # pragma: no cover - defensive
        raise TypeError(f"unsupported calendar type: {type(calendar)!r}")
    sessions = _normalize_sessions(sessions)
    if len(sessions) == 0:
        raise ValueError("empty trading calendar")
    return _Spine(sessions)


def _build_cv_spine(spine: _Spine) -> _Spine:
    """Clip the input spine to the CV window ``[2019-01-01, 2022-12-31]``.

    A spine spanning the whole window must be exactly the NYSE calendar there (a hole
    would move an embargo boundary); a shorter spine is used as given and its own
    coverage becomes the window, which is the "short period" edge case of the plan.
    Dates inside the window that fall on a non-NYSE day are always rejected.
    """
    sessions = spine.sessions
    in_window = sessions[
        (sessions >= pd.Timestamp(CV_WINDOW_START)) & (sessions <= pd.Timestamp(CV_WINDOW_END))
    ]
    if len(in_window) == 0:
        raise ValueError("trading spine has no sessions inside the CV window 2019-2022")
    _assert_cv_spine_against_nyse(sessions, strict_coverage=len(in_window) > 900)
    return _Spine(pd.DatetimeIndex(in_window))


def _assert_cv_spine_against_nyse(
    sessions: pd.DatetimeIndex, *, strict_coverage: bool = False
) -> None:
    """The CV spine must not invent a session and must not hide one.

    * Non-NYSE dates inside the window are always an error (a fold boundary on them
      would be a non-trading day).
    * When ``strict_coverage`` is set the spine claims to span 2019-2022 and a missing
      session would silently shift an embargo; that is an error too.
    """
    window = sessions[
        (sessions >= pd.Timestamp(CV_WINDOW_START)) & (sessions <= pd.Timestamp(CV_WINDOW_END))
    ]
    if len(window) == 0:
        raise ValueError("spine has no sessions inside the CV window")
    from spp.features.calendar import nyse_sessions

    nyse = _normalize_sessions(nyse_sessions(CV_WINDOW_START, CV_WINDOW_END))
    extra = sorted(set(window) - set(nyse))
    if extra:
        raise ValueError(
            f"CV spine contains non-NYSE date(s): {[str(d.date()) for d in extra[:3]]}"
        )
    if strict_coverage:
        missing = sorted(set(nyse) - set(window))
        if missing:
            raise ValueError(
                "CV spine is not the NYSE calendar inside 2019-2022: "
                f"{len(missing)} missing session(s) e.g. "
                f"{[str(d.date()) for d in missing[:3]]}"
            )


def _fold_backwards(spine: _Spine, fold_id: int, test_end: pd.Timestamp, cfg: CvCfg) -> Fold:
    """Fold ``fold_id`` whose test window ends at ``test_end`` (last session of a quarter)."""
    # Test window = the ``test_months`` calendar months that end at ``test_end``.
    base = pd.Timestamp(test_end.year, test_end.month, 1) - pd.DateOffset(
        months=cfg.test_months - 1
    )
    test_start = spine.session_at_or_after(base)
    if test_start > test_end:  # pragma: no cover - guarded by calendar checks
        raise ValueError(f"fold {fold_id}: test_start {test_start} > test_end {test_end}")

    # Embargo: ``embargo_days`` *trading sessions* strictly between train_end and
    # test_start. ``train_end`` is therefore the session ``embargo_days + 1`` places
    # before ``test_start``; the valid window is everything in between -> ``embargo_days``
    # sessions. Off-by-one here is the difference between a 4- and a 5-session gap.
    train_end = spine.session_before(test_start, cfg.embargo_days + 1)
    valid_start = spine.session_at_or_after(train_end + pd.Timedelta(days=1))
    valid_end = spine.session_at_or_before(test_start - pd.Timedelta(days=1))
    if valid_start > valid_end:
        raise ValueError(
            f"fold {fold_id}: empty valid window (embargo_days={cfg.embargo_days} leaves "
            "no session between train_end and test_start)"
        )

    train_start = spine.sessions[0]
    return Fold(
        fold_id=fold_id,
        train_start=train_start.date(),
        train_end=train_end.date(),
        valid_start=valid_start.date(),
        valid_end=valid_end.date(),
        test_start=test_start.date(),
        test_end=test_end.date(),
        embargo_days=cfg.embargo_days,
    )


def make_folds(dates: pd.DatetimeIndex, cfg: CvCfg, calendar: Any) -> list[Fold]:
    """Build ``n_folds`` expanding walk-forward folds ending on 2022-12-31.

    ``dates`` is accepted for API parity (the plan fixes the signature); the spine used
    for boundaries is ``calendar``. If ``dates`` is not a subset of the calendar, the
    call fails loudly: a fold built from a different calendar than the data is exactly
    the class of bug this module exists to prevent.

    Fewer than ``n_folds`` folds are produced only when the requested count does not
    fit the window; a ``WARN`` is logged and the count is clamped to ``>= MIN_FOLDS``.
    """
    spine = _build_cv_spine(_as_spine(calendar))
    sessions = spine.sessions

    provided = _normalize_sessions(dates) if dates is not None and len(dates) else sessions
    outside = sorted(set(provided) - set(sessions))
    if outside:
        raise ValueError(
            f"{len(outside)} supplied date(s) are not on the CV spine, e.g. "
            f"{[str(d.date()) for d in outside[:3]]}"
        )

    if cfg.test_months < 1:
        raise ValueError(f"cv.test_months must be >= 1, got {cfg.test_months}")
    if cfg.embargo_days < 1:
        raise ValueError(
            f"cv.embargo_days must be >= 1 trading session (got {cfg.embargo_days}); "
            "an embargo of zero means a training row can sit immediately before test"
        )

    window_end = sessions[-1]
    # Quarter-aligned test ends: walk back ``test_months`` at a time from the window end.
    test_ends: list[pd.Timestamp] = [window_end]
    cursor = window_end
    while len(test_ends) < cfg.n_folds:
        prev = cursor - pd.DateOffset(months=cfg.test_months)
        if prev <= sessions[0]:
            break
        # Snap to the last session of the month that ``prev`` belongs to; this keeps
        # test windows back-to-back at month granularity.
        snapped = spine.month_end_session(prev.year, prev.month)
        if snapped >= cursor:
            snapped = spine.session_before(cursor, 1)
        if snapped <= sessions[0]:
            break
        test_ends.append(snapped)
        cursor = snapped

    test_ends = sorted(test_ends)
    if len(test_ends) < cfg.n_folds:
        LOG.warning(
            "cv: only %d of %d folds fit the window %s..%s; using %d",
            len(test_ends),
            cfg.n_folds,
            sessions[0].date(),
            sessions[-1].date(),
            len(test_ends),
        )
    if len(test_ends) < MIN_FOLDS:
        raise ValueError(
            f"cv window too short: {len(test_ends)} fold(s) < minimum {MIN_FOLDS} "
            f"(test_months={cfg.test_months})"
        )

    # Drop folds without enough history to train; keeps the list contiguous from the end.
    drop = 0
    while len(test_ends) - drop >= MIN_FOLDS:
        candidate = test_ends[drop]
        try:
            probe = _fold_backwards(spine, 0, candidate, cfg)
        except ValueError:
            drop += 1
            continue
        months = spine.month_diff(pd.Timestamp(probe.train_start), pd.Timestamp(probe.train_end))
        if months < cfg.min_train_months:
            drop += 1
            continue
        break
    if drop:
        LOG.warning(
            "cv: dropped %d fold(s) whose train window is shorter than min_train_months=%d",
            drop,
            cfg.min_train_months,
        )
    test_ends = test_ends[drop:]
    if len(test_ends) < MIN_FOLDS:
        raise ValueError(
            f"cv window too short after min_train_months={cfg.min_train_months}: "
            f"{len(test_ends)} fold(s) < minimum {MIN_FOLDS}; widen the window"
        )

    folds = [_fold_backwards(spine, fid, test_end, cfg) for fid, test_end in enumerate(test_ends)]
    assert_no_overlap(folds, spine)
    return folds


def _mask(frame: pd.DataFrame, start: date, end: date) -> pd.Series:
    dates = pd.to_datetime(frame["date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    dates = dates.dt.normalize()
    return (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))


def assert_no_overlap(folds: list[Fold], calendar: Any) -> None:
    """Five hard invariants. Raises ``AssertionError`` with an actionable message.

    1. ``train_end < test_start`` and the gap is ``>= embargo`` trading sessions
       (derived from the fold itself, so a silently shrunk embargo is caught).
    2. ``valid`` lies strictly between train and test (no shared session).
    3. test windows are disjoint and strictly increasing in time.
    4. no fold date falls in 2023 (or outside the CV window).
    5. with a frame attached to a fold (``Fold.frame``): the sets of ``(date, ticker)``
       in train / valid / test are pairwise disjoint.
    """
    spine = _as_spine(calendar)
    sessions = spine.sessions
    if not folds:
        raise AssertionError("no folds supplied")

    ends_2023 = sessions[sessions.year == CV_HELD_OUT_YEAR]
    window_lo, window_hi = pd.Timestamp(CV_WINDOW_START), pd.Timestamp(CV_WINDOW_END)

    for fold in folds:
        spans = {
            "train": (fold.train_start, fold.train_end),
            "valid": (fold.valid_start, fold.valid_end),
            "test": (fold.test_start, fold.test_end),
        }
        # --- 0. every boundary is a session on the spine (new year's day is not) ---
        for name, (lo, hi) in spans.items():
            for bound in (lo, hi):
                if pd.Timestamp(bound) not in set(sessions):
                    raise AssertionError(
                        f"fold {fold.fold_id}: {name} boundary {bound} is not a trading session"
                    )
            if lo > hi:
                raise AssertionError(f"fold {fold.fold_id}: {name} start {lo} > end {hi}")

        # --- 1. train_end < test_start with an embargo of trading sessions ---
        train_end = pd.Timestamp(fold.train_end)
        test_start = pd.Timestamp(fold.test_start)
        if not train_end < test_start:
            raise AssertionError(
                f"fold {fold.fold_id}: train_end {fold.train_end} >= test_start {fold.test_start}"
            )
        gap = spine.sessions_between(train_end, test_start)
        # sessions strictly between train_end and test_start
        inner = [d for d in gap if d > train_end and d < test_start]
        if len(inner) < 1:
            raise AssertionError(
                f"fold {fold.fold_id}: zero-session embargo between {fold.train_end} "
                f"and {fold.test_start}"
            )
        # The embargo is the fold's own contract: valid must span exactly the gap.
        required = int(getattr(fold, "embargo_days", len(inner)) or len(inner))
        if len(inner) < required:
            raise AssertionError(
                f"fold {fold.fold_id}: embargo of {len(inner)} session(s) between "
                f"{fold.train_end} and {fold.test_start} < required {required}"
            )

        # --- 2. valid strictly inside the gap, disjoint from train and test ---
        valid_start, valid_end = pd.Timestamp(fold.valid_start), pd.Timestamp(fold.valid_end)
        if not (train_end < valid_start <= valid_end < test_start):
            raise AssertionError(
                f"fold {fold.fold_id}: valid {fold.valid_start}..{fold.valid_end} is not "
                f"strictly inside the gap {fold.train_end}..{fold.test_start}"
            )
        # The valid slice must be exactly the sessions in the gap (no hidden overlap).
        between = [d for d in inner]
        if between[0] != valid_start or between[-1] != valid_end:
            raise AssertionError(
                f"fold {fold.fold_id}: valid bounds {fold.valid_start}..{fold.valid_end} do "
                f"not span the embargo gap exactly ({between[0].date()}..{between[-1].date()})"
            )

        # --- 4. no fold date may be 2023 or outside the CV window ---
        for name, (lo, hi) in spans.items():
            lo_date = pd.Timestamp(lo).date()
            hi_date = pd.Timestamp(hi).date()
            if lo_date < window_lo.date() or hi_date > window_hi.date():
                raise AssertionError(
                    f"fold {fold.fold_id}: {name} {lo}..{hi} leaves the CV window "
                    f"{CV_WINDOW_START}..{CV_WINDOW_END}"
                )
            if any(pd.Timestamp(lo) <= d <= pd.Timestamp(hi) for d in ends_2023):
                raise AssertionError(
                    f"fold {fold.fold_id}: {name} window touches the held-out year 2023"
                )

        # --- 5. row-level disjointness on (date, ticker), when a frame is attached ---
        frame = fold.frame
        if frame is not None:
            train_rows = frame.loc[_mask(frame, fold.train_start, fold.train_end)]
            test_rows = frame.loc[_mask(frame, fold.test_start, fold.test_end)]
            valid_rows = frame.loc[_mask(frame, fold.valid_start, fold.valid_end)]
            train_keys = set(zip(train_rows["date"], train_rows["ticker"], strict=False))
            test_keys = set(zip(test_rows["date"], test_rows["ticker"], strict=False))
            valid_keys = set(zip(valid_rows["date"], valid_rows["ticker"], strict=False))
            shared = train_keys & test_keys
            if shared:
                raise AssertionError(
                    f"fold {fold.fold_id}: {len(shared)} (date, ticker) row(s) shared by "
                    f"train and test, e.g. {sorted(shared)[:3]}"
                )
            if train_keys & valid_keys:
                raise AssertionError(
                    f"fold {fold.fold_id}: train and valid share "
                    f"{len(train_keys & valid_keys)} (date, ticker) row(s)"
                )
            if valid_keys & test_keys:
                raise AssertionError(
                    f"fold {fold.fold_id}: valid and test share "
                    f"{len(valid_keys & test_keys)} (date, ticker) row(s)"
                )
            if test_rows.empty:
                raise AssertionError(f"fold {fold.fold_id}: test slice has no rows")

    # --- 3. test windows are disjoint and strictly increasing ---
    ordered = sorted(folds, key=lambda f: f.test_start)
    for left, right in zip(ordered, ordered[1:], strict=False):
        if not left.test_end < right.test_start:
            raise AssertionError(
                f"folds {left.fold_id} and {right.fold_id} have overlapping test windows "
                f"({left.test_start}..{left.test_end} / {right.test_start}..{right.test_end})"
            )
        left_tail = spine.sessions_between(
            pd.Timestamp(left.test_start), pd.Timestamp(left.test_end)
        )
        right_head = spine.sessions_between(
            pd.Timestamp(right.test_start), pd.Timestamp(right.test_end)
        )
        if set(left_tail) & set(right_head):
            raise AssertionError(f"folds {left.fold_id} and {right.fold_id} share a test session")
    for left, right in zip(ordered, ordered[1:], strict=False):
        # Expanding window: the next fold must never start training later.
        if right.train_start > left.train_start:
            raise AssertionError(
                f"fold {right.fold_id} trains later than fold {left.fold_id}: not expanding"
            )


def attach_frame(fold: Fold, frame: pd.DataFrame) -> Fold:
    """Return ``fold`` carrying ``frame`` so :func:`assert_no_overlap` runs check 5.

    The frame is a validation-only reference (excluded from equality and repr).
    """

    return dataclasses.replace(fold, frame=frame)


def count_rows(frame: pd.DataFrame, mask: pd.Series) -> int:
    return int(np.count_nonzero(mask.to_numpy()))


def fold_row_counts(frame: pd.DataFrame, folds: list[Fold]) -> list[dict[str, Any]]:
    """Rows per split, plus the label-horizon purge accounting for each fold."""

    labels_col = "y" if "y" in frame.columns else None
    rows: list[dict[str, Any]] = []
    for fold in folds:
        train_mask = _mask(frame, fold.train_start, fold.train_end)
        valid_mask = _mask(frame, fold.valid_start, fold.valid_end)
        test_mask = _mask(frame, fold.test_start, fold.test_end)
        block: dict[str, Any] = {
            "fold_id": fold.fold_id,
            "train_rows": count_rows(frame, train_mask),
            "valid_rows": count_rows(frame, valid_mask),
            "test_rows": count_rows(frame, test_mask),
        }
        if labels_col is not None:
            block["train_y_nonnull"] = int(
                pd.to_numeric(frame.loc[train_mask, labels_col], errors="coerce").notna().sum()
            )
            block["valid_y_nonnull"] = int(
                pd.to_numeric(frame.loc[valid_mask, labels_col], errors="coerce").notna().sum()
            )
            block["test_y_nonnull"] = int(
                pd.to_numeric(frame.loc[test_mask, labels_col], errors="coerce").notna().sum()
            )
        rows.append(block)
    return rows


def iter_splits(
    df: pd.DataFrame, folds: list[Fold]
) -> Iterator[tuple[Fold, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Yield ``(fold, train, valid, test)`` slices, sorted by ``(date, ticker)``.

    Rows are selected by *session date* from the frame; nothing is shuffled. The frame
    must carry ``date`` and ``ticker`` and is validated once up front.
    """
    missing = {"date", "ticker"} - set(df.columns)
    if missing:
        raise ValueError(f"frame is missing columns: {sorted(missing)}")
    dates = pd.to_datetime(df["date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    normalized = dates.dt.normalize()
    frame = df.assign(date=normalized).sort_values(["date", "ticker"], kind="stable")
    if frame.duplicated(["date", "ticker"]).any():
        dupes = int(frame.duplicated(["date", "ticker"]).sum())
        raise ValueError(f"duplicate (date, ticker) keys in frame: {dupes}")

    for fold in folds:
        train = frame.loc[_mask(frame, fold.train_start, fold.train_end)]
        valid = frame.loc[_mask(frame, fold.valid_start, fold.valid_end)]
        test = frame.loc[_mask(frame, fold.test_start, fold.test_end)]
        if test.empty:
            raise ValueError(f"fold {fold.fold_id}: test slice is empty")
        yield (
            fold,
            train.reset_index(drop=True),
            valid.reset_index(drop=True),
            test.reset_index(drop=True),
        )


def summarize_folds(folds: list[Fold], calendar: Any) -> dict[str, Any]:
    """Report skeleton: fold dates, embargo measured in sessions, test coverage share."""
    spine = _as_spine(calendar)
    rows = []
    total_test = 0
    for fold in folds:
        gap = [
            d
            for d in spine.sessions_between(
                pd.Timestamp(fold.train_end), pd.Timestamp(fold.test_start)
            )
            if pd.Timestamp(fold.train_end) < d < pd.Timestamp(fold.test_start)
        ]
        test_sessions = int(
            spine.count_between(pd.Timestamp(fold.test_start), pd.Timestamp(fold.test_end))
        )
        total_test += test_sessions
        rows.append(
            {
                **fold.as_dict(),
                "embargo_sessions": len(gap),
                "test_sessions": test_sessions,
            }
        )
    window_sessions = int(
        spine.count_between(pd.Timestamp(CV_WINDOW_START), pd.Timestamp(CV_WINDOW_END))
    )
    return {
        "window": {"start": CV_WINDOW_START.isoformat(), "end": CV_WINDOW_END.isoformat()},
        "held_out_year": CV_HELD_OUT_YEAR,
        "window_sessions": window_sessions,
        "total_test_sessions": total_test,
        "test_session_share": (total_test / window_sessions) if window_sessions else 0.0,
        "folds": rows,
    }


def write_folds_report(payload: dict[str, Any], path: str | Path) -> Path:
    """Atomic, sorted, stable JSON write (byte-identical between runs)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.stem}.", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    finally:
        if tmp.exists():  # pragma: no cover
            try:  # noqa: SIM105
                tmp.unlink()
            except OSError:
                pass
    return target
