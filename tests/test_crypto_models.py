"""T7 — causal features, purged walk-forward folds, quantile models and the hurdle
policy of the crypto event study.

All tests are hermetic: events and 1-minute bar frames are synthetic and built in
memory; no parquet under ``data/`` and no network is ever touched.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from spp.crypto import models

BAR_START = "2024-01-01 00:00:00"


def make_bars(open0: float = 100.0, step: float = 1.0, n: int = 121) -> pd.DataFrame:
    """Contiguous 1-minute bars: open = open0 + i*step, close = open + 0.5."""
    ts = pd.date_range(BAR_START, periods=n, freq="min", tz="UTC")
    opens = open0 + step * np.arange(n, dtype="float64")
    return pd.DataFrame({"ts_open": ts, "open": opens, "close": opens + 0.5})


def events_frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """rows: ``(symbol, entry_ts, lag_minutes)``."""
    return pd.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "entry_ts": pd.to_datetime([r[1] for r in rows], utc=True),
            "lag_minutes": [r[2] for r in rows],
            "split": ["train"] * len(rows),
        }
    )


def feature_fixture() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Three events over two symbols; every entry bar's close is poisoned to 999 so a
    feature that touches the entry bar cannot go unnoticed."""
    btc, eth = make_bars(100.0, 1.0), make_bars(50.0, 0.5)
    events = events_frame(
        [
            ("ETHUSDT", "2024-01-01 01:00:00", 2.0),
            ("BTCUSDT", "2024-01-01 01:10:00", 1.5),
            ("ETHUSDT", "2024-01-01 01:40:00", 3.0),
        ]
    )
    for sym, frame in (("BTCUSDT", btc), ("ETHUSDT", eth)):
        for stamp in events.loc[events["symbol"] == sym, "entry_ts"]:
            frame.loc[frame["ts_open"] == stamp, "close"] = 999.0
    return events, {"BTCUSDT": btc, "ETHUSDT": eth}


# --------------------------------------------------------------------------- features
def test_feature_names_shape_and_alignment() -> None:
    events, bars = feature_fixture()
    X, names = models.build_features(events, bars)
    assert names == list(models.FEATURE_NAMES)
    assert list(X.columns) == names
    assert X.index.equals(events.index)
    assert len(X) == len(events)


def test_feature_names_exclude_forbidden_columns() -> None:
    events, bars = feature_fixture()
    _, names = models.build_features(events, bars)
    assert not set(names) & set(models.FORBIDDEN_FEATURES)
    for forbidden in ("quote_volume_entry", "trades_entry", "excess_5m"):
        assert forbidden not in names
    assert "quote_volume_entry" in models.FORBIDDEN_FEATURES
    assert "trades_entry" in models.FORBIDDEN_FEATURES


def test_hour_sin_cos_and_day_of_week() -> None:
    events, bars = feature_fixture()
    X, _ = models.build_features(events, bars)
    # 2024-01-01 01:00 UTC is a Monday (dow = 0), hour = 1.
    assert X.loc[0, "dow"] == 0
    assert X.loc[0, "hour_sin"] == pytest.approx(math.sin(2 * math.pi / 24))
    assert X.loc[0, "hour_cos"] == pytest.approx(math.cos(2 * math.pi / 24))
    assert float(X["hour_sin"].abs().max()) <= 1.0
    assert float(X["hour_cos"].abs().max()) <= 1.0


def test_asset_code_is_sorted_categorical() -> None:
    events, bars = feature_fixture()
    X, _ = models.build_features(events, bars)
    # Sorted categories: BTCUSDT = 0, ETHUSDT = 1.
    assert X.loc[0, "asset_code"] == 1  # ETHUSDT row
    assert X.loc[1, "asset_code"] == 0  # BTCUSDT row
    assert X["asset_code"].between(0, 1).all()


def test_log_lag_transform() -> None:
    events, bars = feature_fixture()
    X, _ = models.build_features(events, bars)
    assert X.loc[0, "log_lag"] == pytest.approx(math.log(3.0))  # lag_minutes = 2.0
    assert X.loc[1, "log_lag"] == pytest.approx(math.log(2.5))  # lag_minutes = 1.5


def test_trailing_returns_are_hand_computed_and_strictly_pre_entry() -> None:
    events, bars = feature_fixture()
    X, _ = models.build_features(events, bars)
    # First event enters ETHUSDT at 01:00. The 5-minute return is close(00:59) over
    # open(00:55) = (50.5 + 0.5*59) / (50 + 0.5*55) - 1; the poisoned close of the
    # entry bar (999) must not appear anywhere.
    expected_5m = (50.5 + 0.5 * 59) / (50 + 0.5 * 55) - 1.0
    expected_60m = (50.5 + 0.5 * 59) / 50.0 - 1.0
    assert X.loc[0, "tr_ret_5m"] == pytest.approx(expected_5m)
    assert X.loc[0, "tr_ret_60m"] == pytest.approx(expected_60m)
    assert 999.0 not in (X.loc[0, "tr_ret_5m"], X.loc[0, "tr_ret_60m"])
    # BTC row: its own series (open0 = 100, step = 1).
    expected_btc_5m = (100.5 + 1.0 * 69) / (100.0 + 1.0 * 65) - 1.0
    assert X.loc[1, "tr_ret_5m"] == pytest.approx(expected_btc_5m)
    # The ETH 60-minute window of the third event starts at 00:40.
    expected_third_60m = (50.5 + 0.5 * 99) / (50.0 + 0.5 * 40) - 1.0
    assert X.loc[2, "tr_ret_60m"] == pytest.approx(expected_third_60m)


def test_trailing_vol_window_excludes_entry_bar() -> None:
    events, bars = feature_fixture()
    X, _ = models.build_features(events, bars)
    closes = bars["ETHUSDT"]["close"].to_numpy(dtype="float64")
    window = closes[30:60]  # [01:00 - 30min, 01:00) — bars 00:30 .. 00:59
    expected = float(np.std(np.diff(np.log(window)), ddof=1))
    poisoned = float(np.std(np.diff(np.log(np.append(window[1:], 999.0))), ddof=1))
    assert X.loc[0, "tr_vol_30m"] == pytest.approx(expected)
    assert X.loc[0, "tr_vol_30m"] != pytest.approx(poisoned)


def test_trailing_vol_zero_for_constant_prices() -> None:
    ts = pd.date_range(BAR_START, periods=121, freq="min", tz="UTC")
    flat = pd.DataFrame({"ts_open": ts, "open": np.full(121, 100.0), "close": np.full(121, 100.0)})
    events = events_frame([("BTCUSDT", "2024-01-01 01:00:00", 1.0)])
    X, _ = models.build_features(events, {"BTCUSDT": flat})
    assert X.loc[0, "tr_vol_30m"] == 0.0
    assert X.loc[0, "tr_vol_60m"] == 0.0
    assert X.loc[0, "tr_ret_15m"] == 0.0


def test_btc_context_features_use_btc_series() -> None:
    events, bars = feature_fixture()
    X, _ = models.build_features(events, bars)
    # The ETH row must carry BTC's context, not its own: the series differ.
    assert X.loc[0, "btc_ret_5m"] != X.loc[0, "tr_ret_5m"]
    # The BTC row's own trailing return and its BTC context coincide.
    assert X.loc[1, "tr_ret_5m"] == pytest.approx(X.loc[1, "btc_ret_5m"])
    # BTC vol window for the BTC entry (bars 00:40 .. 01:09 for entry 01:10).
    closes = bars["BTCUSDT"]["close"].to_numpy(dtype="float64")
    window = closes[40:70]
    assert X.loc[1, "btc_vol_30m"] == pytest.approx(
        float(np.std(np.diff(np.log(window)), ddof=1))
    )


def test_news_burst_counts_only_same_asset_strictly_earlier_events() -> None:
    rows = [
        ("BTCUSDT", "2024-01-01 10:00:00", 1.0),
        ("BTCUSDT", "2024-01-01 10:10:00", 1.0),
        ("ETHUSDT", "2024-01-01 10:05:00", 1.0),
        ("BTCUSDT", "2024-01-01 11:00:00", 1.0),  # window [10:00, 11:00) -> 2
        ("BTCUSDT", "2024-01-01 11:10:00", 1.0),  # window [10:10, 11:10) -> 2
    ]
    events = events_frame(rows)
    ts = pd.date_range(BAR_START, periods=4000, freq="min", tz="UTC")
    bars = {
        sym: pd.DataFrame(
            {"ts_open": ts, "open": np.full(len(ts), 100.0), "close": np.full(len(ts), 100.0)}
        )
        for sym in ("BTCUSDT", "ETHUSDT")
    }
    X, _ = models.build_features(events, bars)
    assert X["news_burst_60m"].tolist() == [0, 1, 0, 2, 2]


# --------------------------------------------------------------------------- guard
def test_guard_passes_on_causal_features() -> None:
    events, bars = feature_fixture()
    _, names = models.build_features(events, bars)
    models.assert_no_future_features(events, bars, names)  # must not raise


def test_guard_fires_on_injected_future_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject the close of the entry bar (one minute of look-ahead) and prove the guard
    recomputation catches it."""
    events, bars = feature_fixture()
    original = models.build_features

    def patched(events_arg: pd.DataFrame, bars_arg: dict[str, pd.DataFrame]):
        X, names = original(events_arg, bars_arg)
        entry = pd.to_datetime(events_arg["entry_ts"], utc=True)
        values = []
        for sym, stamp in zip(events_arg["symbol"], entry, strict=True):
            frame = bars_arg[str(sym)]
            stamps = pd.to_datetime(frame["ts_open"], utc=True)
            hit = frame.loc[stamps == stamp, "close"]
            values.append(float(hit.iloc[0]) if len(hit) else np.nan)
        X["entry_bar_close"] = np.asarray(values, dtype="float64")
        return X, [*names, "entry_bar_close"]

    monkeypatch.setattr(models, "build_features", patched)
    _, injected_names = patched(events, bars)
    with pytest.raises(AssertionError, match="entry_ts"):
        models.assert_no_future_features(events, bars, injected_names)


def test_guard_rejects_forbidden_feature_name() -> None:
    events, bars = feature_fixture()
    with pytest.raises(AssertionError, match="quote_volume_entry"):
        models.assert_no_future_features(events, bars, ["quote_volume_entry"])
    with pytest.raises(AssertionError, match="excess_5m"):
        models.assert_no_future_features(events, bars, ["excess_5m"])


def test_guard_rejects_unknown_feature_name() -> None:
    events, bars = feature_fixture()
    _, names = models.build_features(events, bars)
    with pytest.raises(AssertionError, match="cannot be certified"):
        models.assert_no_future_features(events, bars, [*names, "made_up_feature"])


# --------------------------------------------------------------------------- folds
def train_fixture(n: int = 600) -> pd.DataFrame:
    """One event every 10 minutes, all in the train split."""
    stamps = pd.date_range("2024-01-01 00:00:00", periods=n, freq="10min", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * n,
            "entry_ts": stamps,
            "lag_minutes": np.full(n, 1.0),
            "split": ["train"] * n,
        }
    )


def test_walk_forward_folds_are_expanding_and_disjoint() -> None:
    events = train_fixture()
    folds = models.make_walk_forward_folds(events, n_folds=5)
    assert len(folds) == 5
    all_test: list[pd.Index] = []
    for k, fold in enumerate(folds):
        assert fold.fold_id == k
        assert len(fold.train_idx) > 0 and len(fold.test_idx) > 0
        assert fold.train_idx.intersection(fold.test_idx).empty
        assert events.loc[fold.train_idx, "entry_ts"].max() < events.loc[
            fold.test_idx, "entry_ts"
        ].min()
        all_test.append(fold.test_idx)
    # Expanding: each fold's training block contains the previous one.
    for left, right in zip(folds, folds[1:], strict=False):
        assert set(left.train_idx) < set(right.train_idx)
    # Test blocks are disjoint and cover every row except the initial block.
    seen = pd.Index([])
    for idx in all_test:
        assert seen.intersection(idx).empty
        seen = seen.union(idx)
    assert len(seen) == 500  # 600 rows - the first 100-row block


def test_embargo_drops_boundary_rows() -> None:
    events = train_fixture()
    folds = models.make_walk_forward_folds(
        events, n_folds=5, embargo_minutes=60, label_minutes=60
    )
    # Rows are 10 minutes apart: the 6 rows within 60 minutes before each test block
    # are purged, from both the label-overlap and the embargo condition.
    assert all(fold.purged_rows == 6 for fold in folds)
    fold0 = folds[0]
    train_entries = events.loc[fold0.train_idx, "entry_ts"]
    test_start = fold0.test_start
    assert len(fold0.train_idx) == 94
    assert train_entries.max() + pd.Timedelta(minutes=60) < test_start
    # Without the embargo the boundary row (test_start - 60min) would be in train.
    assert (test_start - pd.Timedelta(minutes=60)) not in set(train_entries)


def test_fold_never_trains_on_its_own_test_rows() -> None:
    events = train_fixture()
    folds = models.make_walk_forward_folds(events, n_folds=5)
    for fold in folds:
        assert fold.train_idx.intersection(fold.test_idx).empty
        # every purged row really was inside the 60-minute label window of the boundary
        purged = set(events.index) - set(fold.train_idx) - set(fold.test_idx)
        for label in sorted(purged)[:3]:
            entry = events.loc[label, "entry_ts"]
            assert entry + pd.Timedelta(minutes=60) >= fold.test_start


# --------------------------------------------------------------------------- policy
def test_policy_mutes_inside_hurdle() -> None:
    pred = np.array([0.0010, 0.0014, 0.0015, -0.0020, np.nan])
    pos = models.policy_positions(pred, hurdle=0.0014, leverage=2)
    assert pos.tolist() == [0.0, 2.0, 2.0, -2.0, 0.0]
    # A prediction exactly at the hurdle trades (>=), one below it does not.
    at = models.policy_positions(np.array([0.0014, 0.001399]), hurdle=0.0014, leverage=1)
    assert at.tolist() == [1.0, 0.0]


def test_leverage_scales_edge_and_cost() -> None:
    excess = np.array([0.0030, -0.0020, -0.0005])
    cost = np.full(3, 0.0014)
    pred = np.array([0.002, 0.002, 0.002])
    net1 = models.policy_net_pnl(models.policy_positions(pred, 0.0014, 1), excess, cost)
    net5 = models.policy_net_pnl(models.policy_positions(pred, 0.0014, 5), excess, cost)
    np.testing.assert_allclose(net1, [0.0016, -0.0034, -0.0019])
    np.testing.assert_allclose(net5, 5.0 * net1)
    # A negative-expectancy trade stays negative under leverage: no free alpha.
    assert net1[2] < 0.0 and net5[2] < 0.0
    assert net5[2] < net1[2]


def test_sharpe_annual_on_known_series() -> None:
    daily = pd.Series([0.01, -0.005, 0.0025])
    # mean = 0.0025, sample std = 0.0075 -> mean/std = 1/3, sqrt(365) annualisation.
    assert models.sharpe_annual(daily) == pytest.approx(math.sqrt(365.0) / 3.0)
    assert models.sharpe_annual(pd.Series([0.0, 0.0])) == 0.0
    assert math.isnan(models.sharpe_annual(pd.Series([0.01])))


def test_max_drawdown_on_known_series() -> None:
    daily = pd.Series([0.01, -0.02, 0.015])
    # Cumulative: 0.01, -0.01, 0.005; peak 0.01 -> max drawdown 0.02 = 200 bps.
    assert models.max_drawdown_bps(daily) == pytest.approx(200.0)
    assert models.max_drawdown_bps(pd.Series([], dtype="float64")) == 0.0


def test_cluster_t_on_known_series() -> None:
    daily = pd.Series([0.01, 0.03])
    # mean 0.02, sample std 0.01*sqrt(2), se = std/sqrt(2) = 0.01 -> t = 2.
    assert models.cluster_t_days(daily) == pytest.approx(2.0)
    assert math.isnan(models.cluster_t_days(pd.Series([0.01])))


def test_daily_pnl_fills_missing_days() -> None:
    entry = pd.to_datetime(
        ["2024-01-01 10:00", "2024-01-01 11:00", "2024-01-03 10:00"], utc=True
    )
    daily = models.daily_pnl(entry, [0.01, 0.02, 0.04])
    assert len(daily) == 3  # Jan 1, Jan 2 (zero), Jan 3
    assert daily.iloc[1] == 0.0
    assert float(daily.sum()) == pytest.approx(0.07)


def test_evaluate_policy_reports_required_metrics() -> None:
    entry = pd.to_datetime(
        ["2024-01-01 10:00", "2024-01-01 11:00", "2024-01-02 10:00"], utc=True
    )
    pos = np.array([1.0, -1.0, 0.0])
    exc = np.array([0.0030, -0.0010, 0.0020])
    m = models.evaluate_policy(entry, pos, exc, 0.0014)
    assert m["n_eval"] == 3
    assert m["n_trades"] == 2
    assert m["muted_share"] == pytest.approx(1 / 3)
    assert m["gross_bps"] == pytest.approx(20.0)  # mean(0.003, +0.001) * 1e4
    # net: long +0.003-0.0014 = +0.0016, short -1*-0.001-0.0014 = -0.0004, muted 0.
    assert m["net_bps"] == pytest.approx(1e4 * (0.0016 - 0.0004) / 3)
    assert m["net_bps_per_trade"] == pytest.approx(1e4 * (0.0016 - 0.0004) / 2)
    for key in ("t_cluster_days", "sharpe_annual", "max_drawdown_bps", "n_days"):
        assert key in m
    assert m["n_days"] == 2


def test_baselines_positions() -> None:
    t15 = np.array([0.01, -0.02, np.nan, 0.0])
    asset = np.array(["AAAUSDT", "AAAUSDT", "BBBUSDT", "BBBUSDT"])
    means = {"AAAUSDT": 0.001, "BBBUSDT": -0.002}
    assert models.baseline_positions("flat", t15, asset).tolist() == [0.0, 0.0, 0.0, 0.0]
    assert models.baseline_positions("long", t15, asset).tolist() == [1.0] * 4
    assert models.baseline_positions("short", t15, asset).tolist() == [-1.0] * 4
    assert models.baseline_positions("momentum", t15, asset).tolist() == [1.0, -1.0, 0.0, 0.0]
    assert models.baseline_positions("per_asset_constant", t15, asset, means).tolist() == [
        1.0,
        1.0,
        -1.0,
        -1.0,
    ]
    with pytest.raises(ValueError):
        models.baseline_positions("nope", t15, asset)
    with pytest.raises(ValueError):
        models.baseline_positions("per_asset_constant", t15, asset)


# --------------------------------------------------------------------------- models
def test_model_params_match_spec() -> None:
    assert models.MODEL_PARAMS == {
        "num_leaves": 15,
        "min_data_in_leaf": 100,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "random_state": 42,
    }
    if models.model_backend() == "lightgbm":
        model = models.make_quantile_model(0.1)
        params = model.get_params()
        assert params["objective"] == "quantile"
        assert params["alpha"] == 0.1
        assert params["num_leaves"] == 15


def test_fit_predict_roundtrip() -> None:
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = 0.01 * X["a"].to_numpy() + 0.001 * rng.normal(size=400)
    model = models.fit_quantile_model(0.5, X, y)
    pred = np.asarray(model.predict(X), dtype="float64")
    assert pred.shape == (400,)
    assert np.all(np.isfinite(pred))


def test_sklearn_fallback_when_lightgbm_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "LGBMRegressor", None)
    assert models.model_backend() == "sklearn"
    model = models.make_quantile_model(0.5)
    assert isinstance(model, models.SklearnQuantileModel)
    X = pd.DataFrame({"a": np.r_[np.nan, np.arange(99, dtype="float64")]})
    y = np.linspace(0.0, 1.0, 100)
    model.fit(X, y)
    pred = np.asarray(model.predict(X), dtype="float64")
    assert pred.shape == (100,)
    assert np.all(np.isfinite(pred))  # NaN was median-imputed, not propagated


# --------------------------------------------------------------------------- empty
def test_empty_input_is_handled() -> None:
    empty = pd.DataFrame(
        {
            "symbol": pd.Series(dtype="str"),
            "entry_ts": pd.Series(dtype="datetime64[ns, UTC]"),
            "lag_minutes": pd.Series(dtype="float64"),
            "split": pd.Series(dtype="str"),
        }
    )
    X, names = models.build_features(empty, {})
    assert len(X) == 0
    assert list(X.columns) == names == list(models.FEATURE_NAMES)
    models.assert_no_future_features(empty, {}, names)  # nothing to violate
    m = models.evaluate_policy([], [], [], [])
    assert m["n_eval"] == 0
    assert m["n_trades"] == 0
    assert math.isnan(m["gross_bps"])
    assert m["max_drawdown_bps"] == 0.0
    with pytest.raises(ValueError):
        models.make_walk_forward_folds(empty)
