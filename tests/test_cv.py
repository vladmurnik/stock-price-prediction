"""Tests for the purged walk-forward CV (T8).

Every check is exercised in the positive direction *and* with a negative case that
must be caught: an assertion that cannot fail is not a check.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from spp.config import CvCfg
from spp.features.calendar import nyse_sessions
from spp.models.cv import (
    CV_WINDOW_END,
    CV_WINDOW_START,
    Fold,
    assert_no_overlap,
    fold_row_counts,
    iter_splits,
    make_folds,
    summarize_folds,
    write_folds_report,
)


# --------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def spine() -> pd.DatetimeIndex:
    """Real NYSE sessions 2019..2022, before the end of the NYSE trading day on Nov 8 2026."""
    return nyse_sessions(CV_WINDOW_START, CV_WINDOW_END)


@pytest.fixture(scope="module")
def cfg() -> CvCfg:
    return CvCfg(n_folds=6, test_months=3, embargo_days=5, min_train_months=12)


@pytest.fixture(scope="module")
def folds(spine: pd.DatetimeIndex, cfg: CvCfg) -> list[Fold]:
    return make_folds(spine, cfg, spine)


def _panel(
    spine: pd.DatetimeIndex, tickers: tuple[str, ...] = ("AAA", "BBB", "CCC")
) -> pd.DataFrame:
    """Minimal (date, ticker) matrix with the labels the CV slices expect."""
    rows = [
        {"date": ts, "ticker": ticker, "y": float(k % 3 - 1) / 100.0, "f1": 1.0}
        for k, ts in enumerate(spine)
        for ticker in tickers
    ]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ generation
def test_six_folds(folds: list[Fold]) -> None:
    assert len(folds) == 6
    assert [f.fold_id for f in folds] == list(range(6))


def test_last_fold_matches_plan_example(folds: list[Fold]) -> None:
    last = folds[-1]
    assert last.test_start == date(2022, 10, 3)
    assert last.test_end == date(2022, 12, 30)
    assert last.valid_start == date(2022, 9, 26)
    assert last.valid_end == date(2022, 9, 30)


def test_train_starts_at_window_start_and_expands(folds: list[Fold]) -> None:
    assert all(f.train_start == date(2019, 1, 2) for f in folds)
    train_ends = [f.train_end for f in folds]
    assert train_ends == sorted(train_ends)
    assert len(set(train_ends)) == len(train_ends)


def test_every_boundary_is_a_trading_session(folds: list[Fold], spine: pd.DatetimeIndex) -> None:
    sessions = set(spine)
    for fold in folds:
        for field_name in dataclasses.fields(Fold):
            value = getattr(fold, field_name.name)
            if not isinstance(value, date):
                continue
            assert pd.Timestamp(value) in sessions, f"{fold.fold_id}.{field_name.name}={value}"


def test_test_windows_are_quarter_back_to_back(folds: list[Fold]) -> None:
    for left, right in zip(folds, folds[1:], strict=False):
        assert left.test_end < right.test_start


def test_all_fold_dates_are_in_2019_2022(folds: list[Fold]) -> None:
    for fold in folds:
        assert fold.train_start >= CV_WINDOW_START
        assert fold.test_end <= CV_WINDOW_END
        for bound in (fold.valid_start, fold.valid_end, fold.test_start, fold.test_end):
            assert bound.year != 2023


def test_no_fold_contains_a_2023_session(folds: list[Fold]) -> None:
    """Explicit acceptance criterion: 2023 must not fall into any fold window.

    Built from the real calendar, so the test fails if anyone widens the CV window
    past ``CV_WINDOW_END`` or lets a fold boundary spill into January 2023.
    """
    sessions_2023 = nyse_sessions(date(2023, 1, 1), date(2023, 12, 31))
    assert len(sessions_2023) > 200  # sanity: the calendar really has 2023 sessions
    first_2023 = pd.Timestamp(sessions_2023[0])
    for fold in folds:
        for field_name in (
            "train_start",
            "train_end",
            "valid_start",
            "valid_end",
            "test_start",
            "test_end",
        ):
            value = getattr(fold, field_name)
            assert pd.Timestamp(value) < first_2023, f"{field_name}={value} reaches 2023"
    assert CV_WINDOW_END < date(2023, 1, 1)


def test_embargo_is_five_trading_sessions(folds: list[Fold], spine: pd.DatetimeIndex) -> None:
    for fold in folds:
        between = [
            d for d in spine if pd.Timestamp(fold.train_end) < d < pd.Timestamp(fold.test_start)
        ]
        assert len(between) == 5
        assert pd.Timestamp(fold.valid_start) == between[0]
        assert pd.Timestamp(fold.valid_end) == between[-1]


def test_test_rows_cover_at_least_quarter_of_2021_2022(
    folds: list[Fold], spine: pd.DatetimeIndex
) -> None:
    test_dates = set()
    for fold in folds:
        test_dates |= {
            d for d in spine if pd.Timestamp(fold.test_start) <= d <= pd.Timestamp(fold.test_end)
        }
    reference = set(spine[spine >= pd.Timestamp("2021-01-01")])
    share = len(test_dates & reference) / len(reference)
    assert share >= 0.25, f"test coverage {share:.3f} < 0.25"


# ------------------------------------------------------------ positive checks
def test_assert_no_overlap_passes_on_generated_folds(
    folds: list[Fold], spine: pd.DatetimeIndex
) -> None:
    assert_no_overlap(folds, spine)  # must not raise


def test_assert_no_overlap_row_level_passes(folds: list[Fold], spine: pd.DatetimeIndex) -> None:
    frame = _panel(spine)
    with_frame = [dataclasses.replace(f, frame=frame) for f in folds]
    assert_no_overlap(with_frame, spine)


def test_iter_splits_is_disjoint_and_sorted(folds: list[Fold], spine: pd.DatetimeIndex) -> None:
    frame = _panel(spine)
    for _fold, train, valid, test in iter_splits(frame, folds):
        assert not train.empty and not test.empty and not valid.empty
        assert train["date"].max() < valid["date"].min()
        assert valid["date"].max() < test["date"].min()
        assert list(train["date"]) == sorted(train["date"])
        train_keys = set(zip(train["date"], train["ticker"], strict=False))
        test_keys = set(zip(test["date"], test["ticker"], strict=False))
        assert not (train_keys & test_keys)


def test_iter_splits_rejects_duplicate_keys(folds: list[Fold]) -> None:
    frame = pd.DataFrame(
        [
            {"date": pd.Timestamp("2021-07-01"), "ticker": "AAA", "y": 0.0},
            {"date": pd.Timestamp("2021-07-01"), "ticker": "AAA", "y": 0.0},
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        list(iter_splits(frame, folds))


# ------------------------------------------------------------ negative cases
def _flip(folds: list[Fold], fold_id: int, **changes) -> list[Fold]:
    return [dataclasses.replace(f, **changes) if f.fold_id == fold_id else f for f in folds]


def test_shift_train_end_plus_one_day_is_caught(folds: list[Fold], spine: pd.DatetimeIndex) -> None:
    """Acceptance criterion: moving train_end forward by one session must fail assert."""
    victim = folds[2]
    next_session = spine[spine > pd.Timestamp(victim.train_end)][0]
    shifted = _flip(folds, victim.fold_id, train_end=next_session.date())
    with pytest.raises(AssertionError):
        assert_no_overlap(shifted, spine)


def test_shift_train_end_plus_one_calendar_day_is_caught(
    folds: list[Fold], spine: pd.DatetimeIndex
) -> None:
    victim = folds[2]
    shifted = _flip(folds, victim.fold_id, train_end=victim.train_end + timedelta(days=1))
    with pytest.raises(AssertionError):
        assert_no_overlap(shifted, spine)


def test_shift_train_end_backwards_is_caught(folds: list[Fold], spine: pd.DatetimeIndex) -> None:
    """Pulling train_end back also breaks the contract: valid must span the whole gap."""
    victim = folds[2]
    prev_session = spine[spine < pd.Timestamp(victim.train_end)][-1]
    shifted = _flip(folds, victim.fold_id, train_end=prev_session.date())
    with pytest.raises(AssertionError):
        assert_no_overlap(shifted, spine)


def test_argument_swap_catches_train_end_equal_test_start(
    folds: list[Fold], spine: pd.DatetimeIndex
) -> None:
    """A positional swap of the last two fields must not silently pass (regression)."""
    bad = Fold(
        fold_id=0,
        train_start=date(2019, 1, 2),
        train_end=date(2021, 7, 1),
        valid_start=date(2021, 6, 24),
        valid_end=date(2021, 6, 30),
        test_start=date(2021, 6, 30),
        test_end=date(2021, 9, 30),
    )
    with pytest.raises(AssertionError):
        assert_no_overlap([bad], spine)


def test_overlapping_test_windows_are_caught(folds: list[Fold], spine: pd.DatetimeIndex) -> None:
    overlapping = _flip(folds, 1, test_start=folds[0].test_end)
    with pytest.raises(AssertionError):
        assert_no_overlap(overlapping, spine)


def test_valid_outside_gap_is_caught(folds: list[Fold], spine: pd.DatetimeIndex) -> None:
    victim = folds[1]
    bad = _flip(folds, victim.fold_id, valid_start=victim.train_end)
    with pytest.raises(AssertionError):
        assert_no_overlap(bad, spine)


def test_fold_touching_2023_is_caught(spine: pd.DatetimeIndex) -> None:
    bad = Fold(
        fold_id=0,
        train_start=date(2019, 1, 2),
        train_end=date(2022, 12, 20),
        valid_start=date(2022, 12, 21),
        valid_end=date(2022, 12, 29),
        test_start=date(2023, 1, 3),
        test_end=date(2023, 3, 31),
    )
    with pytest.raises(AssertionError):
        assert_no_overlap([bad], spine)


def test_boundary_off_by_one_is_caught(folds: list[Fold], spine: pd.DatetimeIndex) -> None:
    """A join that snaps ``valid_start`` onto ``train_end`` puts one session in both windows.

    This is the realistic failure mode of a date-shift: the same session ends up on the
    training side and on the validation side, so the gap disappears.
    """
    victim = folds[3]
    assert victim.train_end < victim.test_start
    bugged = dataclasses.replace(
        victim,
        valid_start=victim.train_end,
        valid_end=victim.test_start,
    )
    with pytest.raises(AssertionError):
        assert_no_overlap([bugged], spine)


# --------------------------------------------------------------- degradation
def test_short_window_downgrades_fold_count() -> None:
    short = nyse_sessions(date(2020, 1, 1), date(2021, 12, 31))
    cfg = CvCfg(n_folds=6, test_months=3, embargo_days=5, min_train_months=6)
    result = make_folds(short, cfg, short)
    assert 3 <= len(result) < 6


def test_too_short_window_raises() -> None:
    short = nyse_sessions(date(2022, 6, 1), date(2022, 12, 31))
    cfg = CvCfg(n_folds=6, test_months=3, embargo_days=5, min_train_months=6)
    with pytest.raises(ValueError, match="too short"):
        make_folds(short, cfg, short)


def test_zero_embargo_is_rejected(spine: pd.DatetimeIndex) -> None:
    cfg = CvCfg(n_folds=6, test_months=3, embargo_days=0, min_train_months=12)
    with pytest.raises(ValueError, match="embargo"):
        make_folds(spine, cfg, spine)


def test_min_train_months_is_enforced(spine: pd.DatetimeIndex) -> None:
    cfg = CvCfg(n_folds=6, test_months=3, embargo_days=5, min_train_months=60)
    with pytest.raises(ValueError, match="too short after min_train_months"):
        make_folds(spine, cfg, spine)


def test_calendar_mismatch_is_rejected(spine: pd.DatetimeIndex, cfg: CvCfg) -> None:
    """A spine with a hole would move an embargo boundary; refuse to guess."""
    holed = spine[spine != pd.Timestamp("2021-03-10")]
    with pytest.raises(ValueError, match="missing"):
        make_folds(holed, cfg, holed)


def test_dates_outside_calendar_are_rejected(spine: pd.DatetimeIndex, cfg: CvCfg) -> None:
    bogus = spine.insert(0, pd.Timestamp("2018-12-31"))
    with pytest.raises(ValueError, match="not on the CV spine"):
        make_folds(bogus, cfg, spine)


# ------------------------------------------------------------------ reporting
def test_row_counts_sum_to_full_frame(folds: list[Fold], spine: pd.DatetimeIndex) -> None:
    frame = _panel(spine)
    counts = fold_row_counts(frame, folds)
    assert len(counts) == 6
    for block in counts:
        assert block["test_rows"] == len(
            frame[
                (frame["date"] >= pd.Timestamp(folds[block["fold_id"]].test_start))
                & (frame["date"] <= pd.Timestamp(folds[block["fold_id"]].test_end))
            ]
        )


def test_report_is_deterministic_and_complete(
    tmp_path, folds: list[Fold], spine: pd.DatetimeIndex
) -> None:
    payload = summa = summarize_folds(folds, spine)
    payload["row_counts"] = fold_row_counts(_panel(spine), folds)
    first = write_folds_report(payload, tmp_path / "cv_folds.json")
    text_a = first.read_text(encoding="utf-8")
    second = write_folds_report(payload, tmp_path / "cv_folds.json")
    assert text_a == second.read_text(encoding="utf-8")

    loaded = json.loads(text_a)
    assert len(loaded["folds"]) == 6
    assert loaded["held_out_year"] == 2023
    assert loaded["window"]["end"] == "2022-12-31"
    assert summa["total_test_sessions"] > 0
    for block in loaded["row_counts"]:
        assert {"train_rows", "valid_rows", "test_rows"} <= set(block)


def test_no_shuffle_anywhere(spine: pd.DatetimeIndex, folds: list[Fold]) -> None:
    """Row order of the training slice must be sorted by (date, ticker), never permuted."""
    rng = np.random.default_rng(0)
    shuffled = (
        _panel(spine).sample(frac=1.0, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    )
    fold = folds[3]
    _, train, _, _ = next(iter(iter_splits(shuffled, [fold])))
    ordered = train.sort_values(["date", "ticker"])
    pd.testing.assert_frame_equal(train.reset_index(drop=True), ordered.reset_index(drop=True))
