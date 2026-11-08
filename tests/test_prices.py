"""Tests for T2 prices: cleaning, validation, cache semantics, wide/returns helpers."""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spp.data import prices as prices_mod
from spp.data.prices import (
    PRICE_COLUMNS,
    daily_returns,
    fetch_prices,
    load_prices,
    to_wide,
    validate_prices,
)
from spp.data.store import manifest_path, read_manifest, read_parquet


# --------------------------------------------------------------------------- #
# Synthetic helpers
# --------------------------------------------------------------------------- #
def _synthetic(
    tickers: tuple[str, ...] = ("AAA", "BBB"),
    start: str = "2020-01-02",
    periods: int = 40,
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    rows = []
    for k, ticker in enumerate(tickers):
        base = 100.0 + 10 * k
        for i, day in enumerate(dates):
            close = base + i
            rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "open": close - 0.5,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "adj_close": close * 0.98,
                    "volume": 1_000_000 + i,
                    "source": "yfinance",
                }
            )
    return pd.DataFrame(rows, columns=list(PRICE_COLUMNS))


def _drop_day(df: pd.DataFrame, ticker: str, day: str) -> pd.DataFrame:
    mask = (df["ticker"] == ticker) & (df["date"] == pd.Timestamp(day))
    return df.loc[~mask].reset_index(drop=True)


class Recorder:
    """Adapter wrapper counting calls and returning a fixed frame."""

    def __init__(self, frame: pd.DataFrame, split: pd.DataFrame | None = None) -> None:
        self.frame = frame
        self.splits = (
            split if split is not None else pd.DataFrame(columns=["ticker", "date", "ratio"])
        )
        self.calls: list[tuple[date, date, tuple[str, ...]]] = []

    def __call__(self, tickers, start, end):
        self.calls.append((start, end, tuple(tickers)))
        frame = self.frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))]
        return frame, self.splits


# --------------------------------------------------------------------------- #
# Plan-mandated tests
# --------------------------------------------------------------------------- #
def test_validate_flags_duplicate_negative_and_gap() -> None:
    df = _synthetic()
    dup = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    violations = validate_prices(dup)
    assert any("duplicate" in v for v in violations), violations

    negative = df.copy()
    negative.loc[negative.index[7], "close"] = -1.0
    violations = validate_prices(negative)
    assert any("non-positive" in v for v in violations), violations

    gapped = df.copy()
    gap = (gapped["ticker"] == "AAA") & gapped["date"].between("2020-02-03", "2020-02-14")
    gapped = gapped.loc[~gap]
    violations = validate_prices(gapped)
    assert any("gap" in v for v in violations), violations


def test_validate_clean_frame_is_empty() -> None:
    assert validate_prices(_synthetic()) == []


def test_daily_returns_log_ratio() -> None:
    wide = pd.DataFrame(
        {"AAA": [100.0, 110.0, 99.0]},
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
    )
    out = daily_returns(wide)
    assert np.isnan(out.iloc[0, 0])
    assert out.iloc[1, 0] == pytest.approx(np.log(1.1))
    assert out.iloc[2, 0] == pytest.approx(np.log(0.9))


def test_to_wide_preserves_requested_column_order() -> None:
    df = _synthetic(tickers=("BBB", "AAA"))
    wide = to_wide(df, "adj_close", tickers=["AAA", "BBB"])
    assert list(wide.columns) == ["AAA", "BBB"]
    wide_default = to_wide(df, "adj_close")
    assert list(wide_default.columns) == ["BBB", "AAA"]  # first-appearance order
    assert wide.index.is_monotonic_increasing
    assert wide.index.name == "date"


def test_to_wide_rejects_duplicates() -> None:
    df = _synthetic(("AAA",))
    with pytest.raises(ValueError, match="duplicate"):
        to_wide(pd.concat([df, df.iloc[[0]]], ignore_index=True))


# --------------------------------------------------------------------------- #
# Cleaning behaviour
# --------------------------------------------------------------------------- #
def _clean(df: pd.DataFrame, splits: pd.DataFrame | None = None):
    empty = pd.DataFrame(columns=["ticker", "date", "ratio"])
    return prices_mod._clean(df, splits if splits is not None else empty)


def test_short_gap_forward_filled_and_never_interpolated() -> None:
    df = _drop_day(_synthetic(("AAA",)), "AAA", "2020-01-07")
    cleaned, anomalies = _clean(df)
    row = cleaned.loc[cleaned["date"] == pd.Timestamp("2020-01-07")].iloc[0]
    previous = cleaned.loc[cleaned["date"] == pd.Timestamp("2020-01-06")].iloc[0]
    assert row["adj_close"] == previous["adj_close"]
    assert row["close"] == previous["close"]
    assert anomalies.empty


def test_long_gap_truncates_instead_of_bridging() -> None:
    df = _synthetic(("AAA",), periods=60)
    gap = pd.bdate_range("2020-01-20", periods=6)
    df = df.loc[~df["date"].isin(gap)]
    cleaned, anomalies = _clean(df)
    assert cleaned["date"].max() < gap[0]
    assert not anomalies.empty
    assert (anomalies["kind"] == "gap_too_long").any()
    assert validate_prices(cleaned) == []


def test_duplicate_rows_are_dropped() -> None:
    df = _synthetic(("AAA",))
    doubled = pd.concat([df, df.iloc[[3]].assign(close=999.0)], ignore_index=True)
    cleaned, _ = _clean(doubled)
    assert cleaned.duplicated(subset=["date", "ticker"]).sum() == 0
    assert validate_prices(cleaned) == []


def test_split_backward_adjusted_close_has_no_spike() -> None:
    """Raw close jumps 4x on the split day while adj_close is continuous: no anomaly."""
    df = _synthetic(("AAA",), periods=20)
    split_day = df["date"].iloc[10]
    raw = df.copy()
    post = raw["date"] >= split_day
    raw.loc[post, ["open", "high", "low", "close"]] *= 4.0
    raw.loc[~post, "adj_close"] = raw.loc[~post, "close"].to_numpy()  # pre-adjustment in raw units
    raw.loc[post, "adj_close"] = raw.loc[post, "close"].to_numpy()
    splits = pd.DataFrame({"ticker": ["AAA"], "date": [split_day.date()], "ratio": [4.0]})
    _, anomalies = _clean(raw, splits)
    assert anomalies.empty


def test_one_day_spike_flagged() -> None:
    df = _synthetic(("AAA",), periods=20)
    idx = df.index[10]
    df.loc[idx, "adj_close"] = df.loc[idx - 1, "adj_close"] * 1.5
    _, anomalies = _clean(df)
    assert not anomalies.empty
    assert anomalies["kind"].eq("one_day_spike").any()


def test_vix_is_not_flaggeable() -> None:
    df = _synthetic(("^VIX",), periods=20)
    idx = df.index[10]
    df.loc[idx, "adj_close"] = df.loc[idx - 1, "adj_close"] * 1.6
    _, anomalies = _clean(df)
    assert anomalies.empty


def test_multi_day_feed_gap_is_not_a_one_day_spike() -> None:
    df = _synthetic(("AAA",), periods=30)
    holidays = pd.bdate_range(df["date"].iloc[10], periods=2)
    df = df.loc[~df["date"].isin(holidays)].copy()
    df.loc[df["date"] >= holidays[0], "adj_close"] *= 1.6
    _, anomalies = _clean(df)
    assert anomalies.empty


def test_share_price_rescale_when_report_is_adjusted() -> None:
    """If close silently becomes post-split while adj_close stays raw, split day collapses."""
    df = _synthetic(("AAA",), periods=20)
    split_day = df["date"].iloc[10]
    # adj_close tracks close (no dividend), then on split day close drops 4x but adj_close does not.
    df["adj_close"] = df["close"].to_numpy()
    post = df["date"] >= split_day
    df.loc[post, ["open", "high", "low", "close"]] /= 4.0
    splits = pd.DataFrame({"ticker": ["AAA"], "date": [split_day.date()], "ratio": [4.0]})
    _, anomalies = _clean(df, splits)
    assert (anomalies["kind"] == "one_day_spike").sum() == 0


# --------------------------------------------------------------------------- #
# Cache semantics
# --------------------------------------------------------------------------- #
def test_cache_hit_makes_no_network_call(tmp_path: Path) -> None:
    frame = _synthetic(("AAA", "BBB"), periods=30)
    adapter = Recorder(frame)
    start, end = date(2020, 1, 2), date(2020, 2, 12)
    first = fetch_prices(["AAA", "BBB"], start, end, tmp_path, adapter=adapter)
    assert len(adapter.calls) == 1
    assert first.path.name == "prices_2020-01-02_2020-02-12.parquet"

    started = time.perf_counter()
    second = fetch_prices(["AAA", "BBB"], start, end, tmp_path, adapter=adapter)
    elapsed = time.perf_counter() - started
    assert len(adapter.calls) == 1, "cache hit must not hit the network"
    assert elapsed < 2.0
    pd.testing.assert_frame_equal(first.frame, second.frame)


def test_rerun_is_byte_identical(tmp_path: Path) -> None:
    frame = _synthetic(("AAA", "BBB"), periods=30)
    adapter = Recorder(frame)
    start, end = date(2020, 1, 2), date(2020, 2, 12)
    first = fetch_prices(["AAA", "BBB"], start, end, tmp_path, adapter=adapter)
    digest_a = first.path.read_bytes()
    refreshed = fetch_prices(["AAA", "BBB"], start, end, tmp_path, adapter=adapter, refresh=True)
    digest_b = refreshed.path.read_bytes()
    assert digest_a == digest_b
    pd.testing.assert_frame_equal(first.frame, refreshed.frame)


def test_manifest_records_reach_and_load_roundtrip(tmp_path: Path) -> None:
    adapter = Recorder(_synthetic(("AAA",), periods=20))
    result = fetch_prices(["AAA"], date(2020, 1, 2), date(2020, 1, 31), tmp_path, adapter=adapter)
    manifest = read_manifest(manifest_path(result.path))
    assert manifest["max_date"] == result.frame["date"].max().date().isoformat()
    assert "T" not in manifest["max_date"], "manifest dates must be plain ISO dates"
    assert manifest["tickers_missing"] == []
    assert manifest["fetched_windows"] == [["2020-01-02", "2020-01-31"]]
    loaded = load_prices(tmp_path, date(2020, 1, 6), date(2020, 1, 10))
    assert list(loaded.columns) == list(PRICE_COLUMNS)
    assert loaded["date"].min() == pd.Timestamp("2020-01-06")
    assert loaded["date"].max() == pd.Timestamp("2020-01-10")


def test_missing_ticker_is_reported_not_raised(tmp_path: Path) -> None:
    adapter = Recorder(_synthetic(("AAA",), periods=20))
    result = fetch_prices(
        ["AAA", "ZZZ"], date(2020, 1, 2), date(2020, 1, 31), tmp_path, adapter=adapter
    )
    assert result.missing == ("ZZZ",)
    assert "ZZZ" not in set(result.frame["ticker"])


def test_dtypes_match_schema(tmp_path: Path) -> None:
    adapter = Recorder(_synthetic(("AAA",), periods=10))
    result = fetch_prices(["AAA"], date(2020, 1, 2), date(2020, 1, 31), tmp_path, adapter=adapter)
    stored = read_parquet(result.path)
    assert list(stored.columns) == list(PRICE_COLUMNS)
    assert str(stored["ticker"].dtype) in {"string", "object"}
    for col in ("open", "high", "low", "close", "adj_close"):
        assert stored[col].dtype == np.float64
    assert pd.api.types.is_integer_dtype(stored["volume"])
    assert pd.api.types.is_datetime64_any_dtype(stored["date"])


def test_fixture_adapter_replays_offline(tmp_path: Path) -> None:
    fixture = tmp_path / "fixtures"
    fixture.mkdir()
    frame = _synthetic(("AAA",), periods=10)
    frame.assign(date=frame["date"].dt.strftime("%Y-%m-%d")).to_parquet(fixture / "AAA.parquet")
    adapter = prices_mod.fixture_adapter(fixture)
    result = fetch_prices(
        ["AAA", "NOPE"], date(2020, 1, 2), date(2020, 1, 31), tmp_path / "cache", adapter=adapter
    )
    assert result.missing == ("NOPE",)
    assert result.frame.shape[0] == 10


def test_covers_uses_request_end_not_last_bar(tmp_path: Path) -> None:
    """Coverage is decided by the requested range, not by the last returned bar."""
    adapter = Recorder(_synthetic(("AAA",), periods=30))
    start, end = date(2020, 1, 2), date(2020, 2, 12)
    result = fetch_prices(["AAA"], start, end, tmp_path, adapter=adapter)
    manifest = read_manifest(manifest_path(result.path))

    assert manifest["request_end"] == end.isoformat()
    assert prices_mod._covers(result.path, start, end) is True
    assert prices_mod._covers(result.path, start, date(2020, 2, 13)) is False
    assert len(adapter.calls) == 1
