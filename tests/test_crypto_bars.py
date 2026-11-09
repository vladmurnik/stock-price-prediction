"""T1 — Binance Vision 1m bars: URLs, month iteration, CSV parsing, dedupe, quality
diagnostics and the resumable downloader.

All tests are hermetic: every fixture is a zip built under ``tmp_path`` and every HTTP
response is a fake object — no network is ever touched.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from spp.crypto import bars
from spp.crypto.bars import (
    BAR_COLUMNS,
    daily_url,
    download_month,
    iter_months,
    load_symbol,
    month_url,
    quality_report,
    read_month,
)

#: The header Binance prepends to a few archived files.
CSV_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_asset_volume,"
    "number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,ignore"
)

#: 2024-01-01T00:00:00Z in milliseconds.
TS0_MS = 1_704_067_200_000


def kline(
    ts: int,
    *,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.5,
    volume: float = 10.0,
    trades: int = 42,
) -> str:
    """One standard 12-field Binance kline CSV row."""
    quote = volume * close
    fields = [ts, open_, high, low, close, volume, ts + 59_999, quote, trades, 1.0, 1.0, 0]
    return ",".join(str(field) for field in fields)


def zip_bytes(rows: list[str], *, header: bool = False) -> bytes:
    """Bytes of a monthly zip holding ``rows`` (optionally behind a header line)."""
    text = "\n".join(([CSV_HEADER] if header else []) + rows) + "\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.csv", text)
    return buffer.getvalue()


def write_month_zip(path: Path, rows: list[str], *, header: bool = False) -> Path:
    """Write a monthly zip fixture at ``path`` (parent dirs are created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zip_bytes(rows, header=header))
    return path


def seed_cache(cache_dir: Path, symbol: str, months: dict[tuple[int, int], list[str]]) -> None:
    """Pre-populate the download cache so ``load_symbol`` runs without any network."""
    for (year, month), rows in months.items():
        write_month_zip(bars.cache_path(cache_dir, symbol, year, month), rows)


def test_iter_months_inclusive_across_year_boundary() -> None:
    assert iter_months("2023-11", "2024-02") == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]
    assert iter_months("2024-06", "2024-06") == [(2024, 6)]
    # The frozen config window (2023-01 .. 2026-11) is 47 inclusive months.
    assert len(iter_months("2023-01", "2026-11")) == 47


def test_iter_months_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        iter_months("2024-02", "2024-01")  # reversed window
    with pytest.raises(ValueError):
        iter_months("2024", "2025-01")  # not YYYY-MM
    with pytest.raises(ValueError):
        iter_months("2024-13", "2024-14")  # month out of range


def test_url_builders() -> None:
    assert month_url("BTCUSDT", 2024, 1) == (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"
    )
    assert month_url("ETHUSDT", 2023, 11) == (
        "https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/1m/ETHUSDT-1m-2023-11.zip"
    )
    assert daily_url("BTCUSDT", pd.Timestamp("2024-03-05")) == (
        "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2024-03-05.zip"
    )


def test_read_month_without_header(tmp_path: Path) -> None:
    rows = [
        kline(TS0_MS, close=100.5, volume=12.0),
        kline(TS0_MS + 60_000, close=101.0, volume=7.5, trades=30),
    ]
    path = write_month_zip(tmp_path / "BTCUSDT-1m-2024-01.zip", rows)
    frame = read_month(path, "BTCUSDT")
    assert list(frame.columns) == list(BAR_COLUMNS)
    assert len(frame) == 2
    assert str(frame["ts_open"].dtype) == "datetime64[ns, UTC]"
    assert frame["ts_open"].iloc[0] == pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    assert frame["ts_open"].iloc[1] - frame["ts_open"].iloc[0] == pd.Timedelta(minutes=1)
    assert frame["open"].dtype == "float64"
    assert frame["quote_volume"].dtype == "float64"
    assert frame["trades"].dtype == "int64"
    assert frame["close"].iloc[0] == 100.5
    assert frame["quote_volume"].iloc[0] == pytest.approx(12.0 * 100.5)
    assert frame["trades"].iloc[1] == 30


def test_read_month_with_header(tmp_path: Path) -> None:
    rows = [kline(TS0_MS), kline(TS0_MS + 60_000)]
    plain = read_month(write_month_zip(tmp_path / "plain.zip", rows), "BTCUSDT")
    headed = read_month(write_month_zip(tmp_path / "headed.zip", rows, header=True), "BTCUSDT")
    pd.testing.assert_frame_equal(plain, headed)


def test_read_month_detects_microsecond_timestamps(tmp_path: Path) -> None:
    path = write_month_zip(tmp_path / "BTCUSDT-1m-2019-08.zip", [kline(TS0_MS * 1000)])
    frame = read_month(path, "BTCUSDT")
    assert str(frame["ts_open"].dtype) == "datetime64[ns, UTC]"
    assert frame["ts_open"].iloc[0] == pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    assert int(frame["ts_open"].iloc[0].value) == TS0_MS * 1_000_000


def test_load_symbol_keeps_last_duplicate_timestamp(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    seed_cache(
        cache,
        "BTCUSDT",
        {
            (2024, 1): [kline(TS0_MS, close=100.0), kline(TS0_MS + 60_000, close=101.0)],
            (2024, 2): [kline(TS0_MS, close=200.0)],  # same open time, later month wins
        },
    )
    frame = load_symbol("BTCUSDT", "2024-01", "2024-02", cache)
    assert len(frame) == 2
    first = frame.iloc[0]
    assert first["ts_open"] == pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    assert float(first["close"]) == 200.0
    assert float(frame.iloc[1]["close"]) == 101.0


def test_load_symbol_output_is_tz_aware_and_monotonic(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    seed_cache(
        cache,
        "BTCUSDT",
        {(2024, 1): [kline(TS0_MS + 120_000), kline(TS0_MS), kline(TS0_MS + 60_000)]},
    )
    frame = load_symbol("BTCUSDT", "2024-01", "2024-01", cache)
    assert list(frame.columns) == list(BAR_COLUMNS)
    assert str(frame["ts_open"].dtype) == "datetime64[ns, UTC]"
    assert frame["ts_open"].dt.tz is not None
    assert frame["ts_open"].is_monotonic_increasing
    assert not frame["ts_open"].duplicated().any()
    assert list(frame["ts_open"]) == [
        pd.Timestamp(TS0_MS, unit="ms", tz="UTC"),
        pd.Timestamp(TS0_MS + 60_000, unit="ms", tz="UTC"),
        pd.Timestamp(TS0_MS + 120_000, unit="ms", tz="UTC"),
    ]


def test_quality_report_flags_ohlc_violation_and_three_minute_gap() -> None:
    ts = pd.to_datetime(
        pd.Series([TS0_MS, TS0_MS + 60_000, TS0_MS + 240_000, TS0_MS + 300_000]),
        unit="ms",
        utc=True,
    ).astype("datetime64[ns, UTC]")
    frame = pd.DataFrame(
        {
            "ts_open": ts,
            "open": [10.0, 10.0, 10.0, 5.0],
            # Row 2: high 11 < max(open, close) 12 -> OHLC violation.
            "high": [12.0, 11.0, 12.0, 12.0],
            # Row 4: low and close are 0 -> non-positive prices, no extra OHLC violation.
            "low": [9.0, 9.0, 9.0, 0.0],
            "close": [11.0, 12.0, 11.0, 0.0],
            # Row 3 has zero volume.
            "volume": [5.0, 5.0, 0.0, 5.0],
            "quote_volume": [55.0, 60.0, 0.0, 0.0],
            "trades": [10, 11, 0, 12],
        }
    )
    report = quality_report(frame, "BTCUSDT")
    assert report["symbol"] == "BTCUSDT"
    assert report["rows"] == 4
    assert report["gaps_over_2min"] == 1  # 3 minutes between rows 2 and 3
    assert report["ohlc_violations"] == 1
    assert report["duplicate_ts"] == 0
    assert report["zero_volume_share"] == pytest.approx(0.25)
    assert report["non_positive_prices"] == 1
    assert report["first"] == pd.Timestamp(TS0_MS, unit="ms", tz="UTC").isoformat()
    assert report["last"] == pd.Timestamp(TS0_MS + 300_000, unit="ms", tz="UTC").isoformat()


def test_quality_report_counts_duplicates_and_handles_empty_frame() -> None:
    ts = pd.to_datetime(
        pd.Series([TS0_MS, TS0_MS, TS0_MS + 60_000]), unit="ms", utc=True
    ).astype("datetime64[ns, UTC]")
    frame = pd.DataFrame(
        {
            "ts_open": ts,
            "open": [10.0, 10.0, 10.0],
            "high": [11.0, 11.0, 11.0],
            "low": [9.0, 9.0, 9.0],
            "close": [10.5, 10.5, 10.5],
            "volume": [1.0, 1.0, 1.0],
            "quote_volume": [10.5, 10.5, 10.5],
            "trades": [5, 5, 5],
        }
    )
    assert quality_report(frame, "BTCUSDT")["duplicate_ts"] == 1

    blank = quality_report(bars.empty_bars(), "ETHUSDT")
    assert blank["rows"] == 0
    assert blank["first"] is None and blank["last"] is None
    assert blank["duplicate_ts"] == 0
    assert blank["gaps_over_2min"] == 0
    assert blank["ohlc_violations"] == 0
    assert blank["zero_volume_share"] == 0.0
    assert blank["non_positive_prices"] == 0


def test_download_month_returns_cached_path_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    expected = bars.cache_path(cache, "BTCUSDT", 2024, 1)
    write_month_zip(expected, [kline(TS0_MS)])

    def fail_on_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network touched although the zip is already cached")

    monkeypatch.setattr(bars.requests, "get", fail_on_network)
    assert download_month("BTCUSDT", 2024, 1, cache) == expected


def test_download_month_404_raises_filenotfound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def not_found(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(status_code=404, content=b"")

    monkeypatch.setattr(bars.requests, "get", not_found)
    url = bars.month_url("BTCUSDT", 2024, 1)
    with pytest.raises(FileNotFoundError) as excinfo:
        download_month("BTCUSDT", 2024, 1, tmp_path, retries=0)
    assert url in str(excinfo.value)
    assert not bars.cache_path(tmp_path, "BTCUSDT", 2024, 1).exists()


def test_download_month_retries_5xx_with_exponential_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = zip_bytes([kline(TS0_MS)])
    calls = {"count": 0}

    def flaky_get(*_args: object, **_kwargs: object) -> SimpleNamespace:
        calls["count"] += 1
        if calls["count"] < 3:
            return SimpleNamespace(status_code=500, content=b"")
        return SimpleNamespace(status_code=200, content=payload)

    sleeps: list[int] = []
    monkeypatch.setattr(bars.requests, "get", flaky_get)
    monkeypatch.setattr(bars.time, "sleep", sleeps.append)
    path = download_month("BTCUSDT", 2024, 1, tmp_path, retries=3)
    assert path.is_file()
    assert calls["count"] == 3
    assert sleeps == [2, 4]

def test_load_symbol_skips_a_month_the_archive_lacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The current month and a pre-listing month are absent from the archive by design."""
    cache = tmp_path / "cache"
    seed_cache(
        cache,
        "TONUSDT",
        {
            (2024, 1): [kline(TS0_MS)],
            (2024, 3): [kline(TS0_MS + 60_000, open_=101.0, close=101.5)],
        },
    )
    requested: list[str] = []

    def not_found(url: str, *_args: object, **_kwargs: object) -> SimpleNamespace:
        requested.append(url)
        return SimpleNamespace(status_code=404, content=b"")

    monkeypatch.setattr(bars.requests, "get", not_found)
    frame = load_symbol("TONUSDT", "2024-01", "2024-03", cache)
    assert len(frame) == 2
    assert requested == [bars.month_url("TONUSDT", 2024, 2)]


def test_load_symbol_raises_when_no_month_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def not_found(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(status_code=404, content=b"")

    monkeypatch.setattr(bars.requests, "get", not_found)
    with pytest.raises(FileNotFoundError):
        load_symbol("ARBUSDT", "2023-01", "2023-02", tmp_path / "cache")