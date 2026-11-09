"""T5 — event dataset: entry semantics, exit windows, excess, splits, funnel, checks.

Every test is hermetic: bars and news are synthesised inside the test (no network, no
2M-row parquet read). The only real file touched is ``configs/crypto_dataset.yaml`` via
:func:`load_dataset_config`, because the leakage checks resolve the frozen lag/splits
from it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from spp.crypto.dataset import (
    CANONICAL_ASSETS,
    DATASET_COLUMNS,
    FUNNEL_DROP_KEYS,
    FUNNEL_KEYS,
    SYMBOL_MAP,
    asset_symbol_map,
    build_dataset,
    empty_dataset,
    empty_funnel,
    load_bars,
    load_dataset_config,
    resolve_symbol,
    run_checks,
)

CONFIG = load_dataset_config()

#: Bars rise by ``step`` per minute: open[i] = base + step*i, close[i] = open[i] + step/2.
def make_bars(
    start: object,
    count: int,
    base: float = 100.0,
    step: float = 1.0,
    quote0: float = 1_000_000.0,
    trades0: int = 100,
) -> pd.DataFrame:
    ts = pd.date_range(start, periods=count, freq="min", tz="UTC")
    i = np.arange(count, dtype="float64")
    open_ = base + step * i
    close = open_ + step * 0.5
    return pd.DataFrame(
        {
            "ts_open": ts,
            "open": open_,
            "high": close + 0.1,
            "low": open_ - 0.1,
            "close": close,
            "volume": np.ones(count),
            "quote_volume": quote0 + i,
            "trades": (trades0 + i).astype("int64"),
        }
    )


def news_row(
    published: object,
    url: str,
    tickers: list[str],
    title: str = "story",
    **extra: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "source": "wire",
        "title": title,
        "url": url,
        "guid": url,
        "published_at": published,
        "snapshot_ts": pd.NaT,
        "tickers": tickers,
    }
    row.update(extra)
    return row


def build(rows: list[dict[str, object]], bars: dict[str, pd.DataFrame]):
    """build_dataset over a news frame built from ``rows`` with the frozen config."""
    return build_dataset(pd.DataFrame(rows), bars, CONFIG, allow_benchmark=True)


def full_build() -> tuple[pd.DataFrame, dict[str, int], dict[str, pd.DataFrame]]:
    """One ethereum headline over 70 minutes of ETHUSDT + BTCUSDT bars (11:55..13:04)."""
    eth = make_bars("2026-03-01 11:55:00", 70, base=100.0)
    btc = make_bars("2026-03-01 11:55:00", 70, base=50_000.0, step=10.0)
    rows = [
        news_row(
            "2026-03-01 12:00:00",
            "https://x/full",
            ["ethereum"],
            title="Ethereum story",
        )
    ]
    bars = {"ETHUSDT": eth, "BTCUSDT": btc}
    frame, funnel = build_dataset(pd.DataFrame(rows), bars, CONFIG)
    return frame, funnel, bars


# --------------------------------------------------------------------------- #
# 1-2. Entry: the lag gate and the next minute open
# --------------------------------------------------------------------------- #
def test_entry_waits_for_bar_exactly_at_target():
    # A bar exists at 12:00 == published_at, but the target is published + 1 min.
    frame, funnel, _ = full_build()
    assert funnel["kept"] == 1
    assert frame["entry_ts"].iat[0] == pd.Timestamp("2026-03-01 12:01:00", tz="UTC")
    assert frame["entry_ts"].iat[0] != pd.Timestamp("2026-03-01 12:00:00", tz="UTC")
    assert frame["lag_minutes"].iat[0] == 1.0


def test_mid_minute_news_enters_at_next_minute_open():
    # 12:00:30 + 1 min = 12:01:30 -> the first bar at or after that opens 12:02.
    eth = make_bars("2026-03-01 11:55:00", 70, base=100.0)
    btc = make_bars("2026-03-01 11:55:00", 70, base=50_000.0, step=10.0)
    rows = [news_row("2026-03-01 12:00:30", "https://x/mid", ["ethereum"])]
    frame, funnel = build(rows, {"ETHUSDT": eth, "BTCUSDT": btc})
    assert funnel["kept"] == 1
    assert frame["entry_ts"].iat[0] == pd.Timestamp("2026-03-01 12:02:00", tz="UTC")
    assert frame["entry_price"].iat[0] == 107.0  # open of the 12:02 bar
    assert frame["lag_minutes"].iat[0] == 1.5


# --------------------------------------------------------------------------- #
# 3-4. Exit window: close of the bar starting at entry + (h - 1)
# --------------------------------------------------------------------------- #
def test_exit_bar_is_the_one_containing_entry_plus_h():
    frame, _, _ = full_build()
    # entry 12:01 (open 106); 5m exit bar opens 12:05 (close 110.5);
    # 15m exit bar opens 12:15 (close 120.5); 60m exit bar opens 13:00 (close 165.5).
    assert frame["ret_5m"].iat[0] == 110.5 / 106.0 - 1.0
    assert frame["ret_15m"].iat[0] == 120.5 / 106.0 - 1.0
    assert frame["ret_60m"].iat[0] == 165.5 / 106.0 - 1.0
    # BTC measured on the identical stamps: open 12:01 (50060) -> close 12:05 (50105).
    assert frame["bench_5m"].iat[0] == 50105.0 / 50060.0 - 1.0
    assert frame["entry_price"].iat[0] == 106.0


def test_extra_bar_after_window_is_not_used():
    frame, _, _ = full_build()  # bars continue to 13:04, well past every exit
    used = 110.5 / 106.0 - 1.0
    extra = 111.5 / 106.0 - 1.0  # close of the 12:06 bar: one minute too far
    assert frame["ret_5m"].iat[0] == used
    assert frame["ret_5m"].iat[0] != extra
    assert frame["ret_5m"].iat[0] < extra  # rising prices: the extra minute inflates it


# --------------------------------------------------------------------------- #
# 5. Excess against BTC over the same stamps
# --------------------------------------------------------------------------- #
def test_excess_uses_btc_over_identical_stamps():
    frame, _, bars = full_build()
    ret = frame["ret_5m"].iat[0]
    bench = frame["bench_5m"].iat[0]
    assert frame["excess_5m"].iat[0] == ret - bench
    assert bench == 50105.0 / 50060.0 - 1.0
    checks = run_checks(frame, bars)
    assert checks["bench_same_window"] is True
    assert checks["bench_same_window_count"] == 0


# --------------------------------------------------------------------------- #
# 6-7. Dedupe and unknown tickers
# --------------------------------------------------------------------------- #
def test_duplicate_url_symbol_keeps_first_published():
    btc = make_bars("2026-03-01 11:55:00", 70, base=50_000.0, step=10.0)
    rows = [
        news_row("2026-03-01 12:30:00", "https://x/dup", ["bitcoin"], title="later"),
        news_row("2026-03-01 12:00:00", "https://x/dup", ["bitcoin"], title="first"),
    ]
    frame, funnel = build(rows, {"BTCUSDT": btc})
    assert funnel["total_pairs"] == 2
    assert funnel["duplicate"] == 1
    assert funnel["kept"] == 1
    assert frame["title"].iat[0] == "first"
    assert frame["published_at"].iat[0] == pd.Timestamp("2026-03-01 12:00:00", tz="UTC")
    # The same url on a *different* symbol is a different event.
    eth = make_bars("2026-03-01 11:55:00", 70, base=100.0)
    rows.append(news_row("2026-03-01 12:05:00", "https://x/dup", ["ethereum"]))
    frame2, funnel2 = build(rows, {"BTCUSDT": btc, "ETHUSDT": eth})
    assert funnel2["duplicate"] == 1
    assert funnel2["kept"] == 2
    assert sorted(frame2["symbol"].tolist()) == ["BTCUSDT", "ETHUSDT"]


def test_unknown_ticker_dropped_as_no_symbol():
    btc = make_bars("2026-03-01 11:55:00", 70, base=50_000.0, step=10.0)
    rows = [news_row("2026-03-01 12:00:00", "https://x/unk", ["monero", "bitcoin"])]
    frame, funnel = build(rows, {"BTCUSDT": btc})
    assert funnel["total_pairs"] == 2
    assert funnel["no_symbol"] == 1
    assert funnel["kept"] == 1
    assert frame["symbol"].tolist() == ["BTCUSDT"]
    assert "monero" not in frame["asset"].tolist()


# --------------------------------------------------------------------------- #
# 8-10. Coverage drops (no bar at or after the target)
# --------------------------------------------------------------------------- #
def test_news_before_first_bar_dropped():
    btc = make_bars("2026-03-01 12:00:00", 30, base=100.0)
    rows = [news_row("2026-03-01 11:00:00", "https://x/early", ["bitcoin"])]
    frame, funnel = build(rows, {"BTCUSDT": btc})
    assert funnel["no_bar_at_or_after"] == 1
    assert funnel["kept"] == 0
    assert len(frame) == 0


def test_news_after_last_bar_dropped():
    btc = make_bars("2026-03-01 11:55:00", 5, base=100.0)  # 11:55 .. 11:59
    rows = [news_row("2026-03-01 12:00:00", "https://x/late", ["bitcoin"])]
    frame, funnel = build(rows, {"BTCUSDT": btc})
    assert funnel["no_bar_at_or_after"] == 1
    assert funnel["kept"] == 0
    assert len(frame) == 0


def test_later_listing_tonusdt_and_missing_symbol_do_not_crash():
    ton = make_bars("2024-08-08 10:00:00", 120, base=5.0)  # TONUSDT lists in 2024-08
    btc = make_bars("2024-08-08 09:00:00", 300, base=50_000.0, step=10.0)
    rows = [
        news_row("2023-05-01 10:00:00", "https://x/ton-old", ["toncoin"]),
        news_row("2024-08-08 10:30:00", "https://x/ton-new", ["toncoin"]),
        news_row("2024-08-08 10:30:00", "https://x/sol", ["solana"]),  # no SOLUSDT bars
    ]
    frame, funnel = build(rows, {"TONUSDT": ton, "BTCUSDT": btc})
    assert funnel["total_pairs"] == 3
    assert funnel["no_bar_at_or_after"] == 2
    assert funnel["kept"] == 1
    assert len(frame) == 1
    assert frame["symbol"].iat[0] == "TONUSDT"
    assert frame["entry_ts"].iat[0] == pd.Timestamp("2024-08-08 10:31:00", tz="UTC")


# --------------------------------------------------------------------------- #
# 11. NaT publication times
# --------------------------------------------------------------------------- #
def test_na_published_at_dropped():
    btc = make_bars("2026-03-01 11:55:00", 70, base=50_000.0, step=10.0)
    rows = [news_row(pd.NaT, "https://x/nat", ["bitcoin"])]
    frame, funnel = build(rows, {"BTCUSDT": btc})
    assert funnel["total_pairs"] == 1
    assert funnel["na_published_at"] == 1
    assert funnel["kept"] == 0
    assert len(frame) == 0
    assert funnel["kept"] + sum(funnel[k] for k in FUNNEL_DROP_KEYS) == funnel["total_pairs"]


# --------------------------------------------------------------------------- #
# 12-13. Splits and the embargo
# --------------------------------------------------------------------------- #
def bars_around(stamps: list[pd.Timestamp], base: float = 100.0) -> pd.DataFrame:
    """120 minutes of bars around each stamp (entry + full 60m exit), merged."""
    frames = [
        make_bars(stamp - pd.Timedelta(minutes=3), 120, base=base)
        for stamp in stamps
    ]
    merged = pd.concat(frames, ignore_index=True).sort_values("ts_open", kind="stable")
    return merged.drop_duplicates("ts_open", keep="first").reset_index(drop=True)


def test_split_assignment_four_boundaries():
    cases = [
        ("2025-06-15 12:00:00", "train"),
        ("2026-05-15 12:00:00", "val"),
        ("2026-10-01 12:00:00", "holdout"),
        ("2026-11-15 12:00:00", "outside"),
    ]
    stamps = [pd.Timestamp(ts, tz="UTC") for ts, _ in cases]
    rows = [
        news_row(ts, f"https://x/split-{i}", ["bitcoin"])
        for i, (ts, _) in enumerate(cases)
    ]
    frame, funnel = build(rows, {"BTCUSDT": bars_around(stamps)})
    assert funnel["kept"] == 4
    assert frame["split"].tolist() == [expected for _, expected in cases]


def test_split_embargo_near_every_boundary():
    # Boundaries are midnight UTC: 2026-01-01 (train/val), 2026-09-15 (val/holdout),
    # 2026-11-09 (holdout end). Within +/-60 min -> "embargo".
    cases = [
        ("2025-12-31 21:00:00", "train"),  # 3 h before boundary 1
        ("2025-12-31 23:30:00", "embargo"),  # 30 min before boundary 1
        ("2026-01-01 00:30:00", "embargo"),  # 30 min after boundary 1
        ("2026-09-14 21:00:00", "val"),  # 3 h before boundary 2
        ("2026-09-14 23:00:00", "embargo"),  # exactly 60 min before boundary 2
        ("2026-09-15 01:00:00", "embargo"),  # exactly 60 min after boundary 2
        ("2026-09-15 02:00:00", "holdout"),
        ("2026-11-08 23:30:00", "embargo"),  # 30 min before boundary 3
        ("2026-11-09 00:30:00", "embargo"),  # 30 min after boundary 3
        ("2026-11-09 02:00:00", "outside"),
    ]
    stamps = [pd.Timestamp(ts, tz="UTC") for ts, _ in cases]
    rows = [
        news_row(ts, f"https://x/emb-{i}", ["bitcoin"])
        for i, (ts, _) in enumerate(cases)
    ]
    frame, funnel = build(rows, {"BTCUSDT": bars_around(stamps)})
    assert funnel["kept"] == 10
    assert frame["split"].tolist() == [expected for _, expected in cases]


# --------------------------------------------------------------------------- #
# 14. Funnel balance on a mixed batch
# --------------------------------------------------------------------------- #
def test_funnel_balance_on_mixed_batch():
    # BTCUSDT bars 08:00 .. 09:39.
    btc = make_bars("2026-03-01 08:00:00", 100, base=50_000.0, step=10.0)
    rows = [
        news_row("2026-03-01 08:30:00", "https://x/good", ["bitcoin"]),  # kept
        news_row("2026-03-01 08:40:00", "https://x/good", ["bitcoin"]),  # duplicate
        news_row(pd.NaT, "https://x/nat", ["bitcoin"]),  # na_published_at
        news_row("2026-03-01 08:30:00", "https://x/unk", ["monero"]),  # no_symbol
        news_row("2026-03-01 07:00:00", "https://x/before", ["bitcoin"]),  # no bar
        news_row("2026-03-01 09:36:00", "https://x/noexit", ["bitcoin"]),  # kept, NaN
    ]
    frame, funnel = build(rows, {"BTCUSDT": btc})
    assert funnel["total_pairs"] == 6
    assert funnel["kept"] == 2
    assert funnel["duplicate"] == 1
    assert funnel["na_published_at"] == 1
    assert funnel["no_symbol"] == 1
    assert funnel["no_bar_at_or_after"] == 1
    dropped = sum(funnel[key] for key in FUNNEL_DROP_KEYS)
    assert funnel["kept"] + dropped == funnel["total_pairs"] == 6
    # The 09:36 row enters 09:37; every exit lies beyond the last bar (09:39).
    assert funnel["missing_exit_5m"] == 1
    assert funnel["missing_exit_15m"] == 1
    assert funnel["missing_exit_60m"] == 1
    late = frame[frame["url"] == "https://x/noexit"].iloc[0]
    assert np.isnan(late["ret_5m"])
    assert np.isnan(late["excess_60m"])
    # Every kept row is present and the balance holds for any subset of drop reasons.
    assert len(frame) == funnel["kept"]


# --------------------------------------------------------------------------- #
# 15-16. Schema
# --------------------------------------------------------------------------- #
def test_empty_input_returns_schema_and_zeroed_funnel():
    empty_news = pd.DataFrame(
        columns=["source", "title", "url", "guid", "published_at", "snapshot_ts", "tickers"]
    )
    frame, funnel = build_dataset(empty_news, {}, CONFIG)
    assert len(frame) == 0
    assert list(frame.columns) == list(DATASET_COLUMNS)
    assert frame.dtypes.equals(empty_dataset().dtypes)
    assert funnel == empty_funnel()
    assert set(funnel) == set(FUNNEL_KEYS)
    assert str(frame["published_at"].dtype) == "datetime64[ns, UTC]"
    assert frame["trades_entry"].dtype == "int64"
    assert frame["cost"].dtype == "float64"


def test_dataset_columns_order_matches_contract():
    frame, _, _ = full_build()
    assert list(frame.columns) == list(DATASET_COLUMNS)
    assert list(frame.columns) == [
        "asset",
        "symbol",
        "source",
        "title",
        "url",
        "published_at",
        "entry_ts",
        "entry_price",
        "lag_minutes",
        "ret_5m",
        "bench_5m",
        "excess_5m",
        "ret_15m",
        "bench_15m",
        "excess_15m",
        "ret_60m",
        "bench_60m",
        "excess_60m",
        "cost",
        "quote_volume_entry",
        "trades_entry",
        "snapshot_ts",
        "split",
        "event_kind",
    ]
    assert frame["event_kind"].iat[0] == ""


# --------------------------------------------------------------------------- #
# 17-18. A real run: 2 news rows x 3 bars, exact numbers
# --------------------------------------------------------------------------- #
def test_build_two_news_rows_over_three_bars():
    three = make_bars("2026-03-01 12:00:00", 3, base=100.0)  # 12:00, 12:01, 12:02
    rows = [
        {  # recorder-style row: first_seen_at instead of snapshot_ts
            "source": "a",
            "title": "A",
            "url": "https://x/a",
            "guid": "g-a",
            "published_at": pd.Timestamp("2026-03-01 11:59:00", tz="UTC"),
            "first_seen_at": pd.Timestamp("2026-03-01 11:59:20", tz="UTC"),
            "tickers": ["bitcoin"],
        },
        {
            "source": "b",
            "title": "B",
            "url": "https://x/b",
            "guid": "g-b",
            "published_at": pd.Timestamp("2026-03-01 12:00:30", tz="UTC"),
            "first_seen_at": pd.Timestamp("2026-03-01 12:01:05", tz="UTC"),
            "tickers": ["ethereum"],
        },
    ]
    bars = {"BTCUSDT": three, "ETHUSDT": three.copy()}
    frame, funnel = build(rows, bars)

    assert funnel["total_pairs"] == 2
    assert funnel["kept"] == 2
    expected = empty_funnel()
    expected.update(
        {"total_pairs": 2, "kept": 2, "missing_exit_5m": 2, "missing_exit_15m": 2,
         "missing_exit_60m": 2}
    )
    assert funnel == expected

    first, second = frame.iloc[0], frame.iloc[1]
    # Row A: 11:59 + 1 min = 12:00 exactly -> the 12:00 bar is the entry bar.
    assert first["asset"] == "bitcoin"
    assert first["symbol"] == "BTCUSDT"
    assert first["entry_ts"] == pd.Timestamp("2026-03-01 12:00:00", tz="UTC")
    assert first["entry_price"] == 100.0
    assert first["lag_minutes"] == 1.0
    assert first["quote_volume_entry"] == 1_000_000.0
    assert first["trades_entry"] == 100
    assert first["snapshot_ts"] == pd.Timestamp("2026-03-01 11:59:20", tz="UTC")
    # Row B: 12:00:30 + 1 min = 12:01:30 -> entry opens 12:02 (last of the 3 bars).
    assert second["asset"] == "ethereum"
    assert second["symbol"] == "ETHUSDT"
    assert second["entry_ts"] == pd.Timestamp("2026-03-01 12:02:00", tz="UTC")
    assert second["entry_price"] == 102.0
    assert second["lag_minutes"] == 1.5
    assert second["quote_volume_entry"] == 1_000_002.0
    assert second["trades_entry"] == 102
    # All exits lie beyond the last bar: returns NaN, benchmarks NaN, excess NaN.
    for col in ("ret_5m", "ret_15m", "ret_60m", "bench_5m", "excess_60m"):
        assert np.isnan(first[col]), col
        assert np.isnan(second[col]), col
    # 2026-03-01 sits between train_end and val_end -> "val".
    assert frame["split"].tolist() == ["val", "val"]
    assert frame["cost"].iat[0] == 14.0 / 10000.0
    checks = run_checks(frame, bars)
    assert all(v is True for k, v in checks.items() if isinstance(v, bool))


def test_cost_and_entry_bar_fields():
    frame, _, _ = full_build()
    assert frame["cost"].iat[0] == 14.0 / 10000.0
    # Entry bar 12:01 is index 6 of the series starting 11:55.
    assert frame["quote_volume_entry"].iat[0] == 1_000_006.0
    assert frame["trades_entry"].iat[0] == 106
    assert frame["trades_entry"].dtype == "int64"


# --------------------------------------------------------------------------- #
# 19-21. Leakage checks
# --------------------------------------------------------------------------- #
def test_run_checks_all_pass_on_clean_build():
    frame, _, bars = full_build()
    checks = run_checks(frame, bars)
    assert all(v is True for k, v in checks.items() if isinstance(v, bool)), checks
    counts = {k: v for k, v in checks.items() if isinstance(v, int) and not isinstance(v, bool)}
    assert counts and all(v == 0 for v in counts.values()), counts


def test_run_checks_flags_entry_before_news_lag():
    frame, _, bars = full_build()
    # Tamper: pretend the entry happened at 12:00, one minute before the allowed target.
    frame.loc[0, "entry_ts"] = pd.Timestamp("2026-03-01 12:00:00", tz="UTC")
    frame.loc[0, "entry_price"] = 105.0  # matches the 12:00 bar's open
    checks = run_checks(frame, bars)
    assert checks["entry_not_before_news_lag"] is False
    assert checks["entry_not_before_news_lag_count"] == 1
    assert checks["entry_price_matches_bar"] is True


def test_run_checks_flags_duplicate_and_bench_tamper():
    frame, _, bars = full_build()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.loc[0, "bench_5m"] = frame.loc[0, "bench_5m"] + 1e-4
    checks = run_checks(frame, bars)
    assert checks["no_duplicate_url_symbol"] is False
    assert checks["no_duplicate_url_symbol_count"] == 1
    assert checks["bench_same_window"] is False
    assert checks["bench_same_window_count"] == 1


# --------------------------------------------------------------------------- #
# 22. Config resolution
# --------------------------------------------------------------------------- #
def test_config_missing_keys_fall_back_with_warning(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("labels:\n  news_lag_minutes: 2\n", encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="missing"):
        cfg = load_dataset_config(path)
    assert cfg["labels"]["horizons"] == [5, 15, 60]
    assert cfg["labels"]["news_lag_minutes"] == 2.0  # the present key is kept
    assert cfg["labels"]["excess_vs"] == "BTCUSDT"
    assert cfg["labels"]["dead_zone"] == 0.05
    assert cfg["costs"]["round_trip_perp_bps"] == 14.0
    assert cfg["splits"]["train_end"] == "2025-12-31"
    assert cfg["splits"]["holdout_end"] == "2026-11-08"
    assert cfg["splits"]["embargo_minutes"] == 60


def test_config_accepts_horizons_minutes_spelling_silently(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "labels:\n"
        "  horizons_minutes: [5, 15, 60]\n"
        "  news_lag_minutes: 1\n"
        "  excess_vs: BTCUSDT\n"
        "  dead_zone: 0.05\n"
        "costs:\n"
        "  round_trip_perp_bps: 14.0\n"
        "splits:\n"
        '  train_end: "2025-12-31"\n'
        '  val_end: "2026-09-14"\n'
        '  holdout_start: "2026-09-15"\n'
        '  holdout_end: "2026-11-08"\n'
        "  embargo_minutes: 60\n",
        encoding="utf-8",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_dataset_config(path)
    messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert messages == []
    assert cfg["labels"]["horizons"] == [5, 15, 60]


def test_frozen_config_resolves_every_key():
    cfg = load_dataset_config()
    assert cfg["labels"]["horizons"] == [5, 15, 60]
    assert cfg["labels"]["news_lag_minutes"] == 1.0
    assert cfg["labels"]["excess_vs"] == "BTCUSDT"
    assert cfg["costs"]["round_trip_perp_bps"] == 14.0
    assert cfg["splits"]["holdout_start"] == "2026-09-15"
    assert cfg["splits"]["embargo_minutes"] == 60


# --------------------------------------------------------------------------- #
# 23. Asset mapping
# --------------------------------------------------------------------------- #
def test_asset_symbol_map_resolves_all_twelve_canonical_names():
    mapping = asset_symbol_map(CONFIG)
    for name, symbol in CANONICAL_ASSETS.items():
        assert mapping.get(name) == symbol, name
        assert resolve_symbol(name, mapping) == symbol
        assert resolve_symbol(symbol) == symbol
    assert SYMBOL_MAP["bitcoin"] == "BTCUSDT"
    assert SYMBOL_MAP["btcusdt"] == "BTCUSDT"
    assert SYMBOL_MAP.get("btc") == "BTCUSDT"  # suffix rule on the symbol base
    assert SYMBOL_MAP.get("link") == "LINKUSDT"  # weak alias resolves via the suffix rule
    assert resolve_symbol("ETH") == "ETHUSDT"
    assert resolve_symbol("monero") is None
    assert resolve_symbol("shib") is None  # SHIBUSDT is not one of the twelve
    assert resolve_symbol(None) is None
    assert resolve_symbol("  ") is None


# --------------------------------------------------------------------------- #
# 24. Lazy bar loading
# --------------------------------------------------------------------------- #
def test_load_bars_reads_lazily_and_windows(tmp_path):
    # Three days of bars: 2026-02-27, then 12:00..12:04 on 2026-03-01, then 2026-03-03.
    frame = pd.concat(
        [
            make_bars("2026-02-27 12:00:00", 1, base=100.0),
            make_bars("2026-03-01 12:00:00", 5, base=100.0),
            make_bars("2026-03-03 12:00:00", 1, base=100.0),
        ],
        ignore_index=True,
    )
    frame.to_parquet(tmp_path / "BTCUSDT.parquet", index=False)
    frame.to_parquet(tmp_path / "ETHUSDT.parquet", index=False)
    bars = load_bars(
        tmp_path,
        ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        since="2026-03-01 12:01",
        until="2026-03-01 12:02",
    )
    assert bars.loaded == set()  # nothing read yet
    assert "BTCUSDT" in bars
    assert "SOLUSDT" not in bars  # no parquet on disk
    assert len(bars) == 2
    with pytest.raises(KeyError):
        bars["SOLUSDT"]
    btc = bars["BTCUSDT"]
    # since/until are padded by a day so the 12:00 entry bar just before `since` and the
    # exits just after `until` survive; the 02-27 and 03-03 rows fall outside the padding.
    assert len(btc) == 5
    assert btc["ts_open"].iloc[0] == pd.Timestamp("2026-03-01 12:00:00", tz="UTC")
    assert btc["ts_open"].iloc[-1] == pd.Timestamp("2026-03-01 12:04:00", tz="UTC")
    assert set(btc.columns) >= {"ts_open", "open", "close", "quote_volume", "trades"}
    assert bars.loaded == {"BTCUSDT"}
