from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from spp.config import load_config
from spp.features import fixtures
from spp.features.assemble import SplitCfg, assemble_features, feature_columns, split_label
from spp.features.calendar import calendar_spine, ghost_mask, horizon_returns, shift_by_sessions
from spp.features.labels import build_labels, demean_by_date


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def prices():
    return fixtures.make_prices()


@pytest.fixture(scope="module")
def articles():
    return fixtures.make_articles()


def _wide(prices: pd.DataFrame, ticker: str = "AAA") -> pd.DataFrame:
    return prices.pivot(index="date", columns="ticker", values="adj_close")[[ticker]]


def test_target_is_next_session_log_return():
    prices = fixtures.make_prices(tickers=("AAA",), helpers=False, periods=30)
    wide = _wide(prices)
    labels = build_labels(wide, demean_target=False)
    expected = np.log(wide["AAA"].shift(-1) / wide["AAA"]).dropna()

    got = labels.set_index("date")["y_1d"].dropna()
    shared = got.index.intersection(expected.index)
    assert len(shared) >= len(expected) - 6
    np.testing.assert_allclose(
        got.loc[shared].to_numpy(), expected.loc[shared].to_numpy(), atol=1e-12
    )


def test_target_never_equals_the_realised_same_day_return():
    prices = fixtures.make_prices(tickers=("AAA",), helpers=False, periods=30)
    wide = _wide(prices)
    labels = build_labels(wide, demean_target=False).set_index("date")
    same_day = np.log(wide["AAA"] / wide["AAA"].shift(1))
    overlap = labels["y_1d"].dropna().index.intersection(same_day.dropna().index)
    deviation = (labels.loc[overlap, "y_1d"] - same_day.loc[overlap]).abs()
    assert deviation.max() > 1e-6


def test_label_tail_is_masked():
    prices = fixtures.make_prices(tickers=("AAA",), helpers=False, periods=40)
    labels = build_labels(_wide(prices), demean_target=False)
    tail = labels.sort_values("date").tail(5)
    assert tail["y"].isna().all()
    assert not tail["label_ok"].any()


def test_demean_by_date_gives_zero_median_per_date():
    labels = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 3 + ["2020-01-03"] * 3),
            "y": [0.01, 0.02, 0.03, -0.01, 0.0, 0.02],
        }
    )
    out = demean_by_date(labels["y"], labels["date"])
    medians = out.groupby(labels["date"]).median()
    np.testing.assert_allclose(medians.to_numpy(), 0.0, atol=1e-12)


def test_ghost_mask_flags_flat_series():
    dates = pd.bdate_range("2020-01-02", periods=40)
    flat = pd.Series(100.0, index=dates)
    moving = pd.Series(np.linspace(100, 120, 40), index=dates)
    mask = ghost_mask(pd.DataFrame({"AAA": flat, "BBB": moving}), lookback=5)
    assert mask["AAA"].iloc[-1]
    assert not mask["BBB"].iloc[-1]


def test_horizon_returns_use_future_prices():
    dates = pd.bdate_range("2020-01-02", periods=10)
    close = pd.DataFrame({"AAA": np.arange(100.0, 110.0)}, index=dates)
    horizons = horizon_returns(close)
    assert {1, 5} <= set(horizons)
    first = horizons[1]["AAA"].iloc[0]
    assert first == pytest.approx(np.log(close["AAA"].iloc[1] / close["AAA"].iloc[0]))
    assert np.isnan(horizons[1]["AAA"].iloc[-1])


def test_calendar_spine_contains_only_sessions():
    prices = fixtures.make_prices(periods=20)
    dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    spine = calendar_spine(dates, dates[0].date(), dates[-1].date())
    assert not (spine.day_of_week >= 5).any()
    assert spine.is_monotonic_increasing
    assert spine.is_unique


def test_shift_by_sessions_maps_onto_the_spine():
    spine = pd.DatetimeIndex(pd.bdate_range("2020-01-02", periods=10))
    later = spine + pd.Timedelta(days=1)
    mapped = shift_by_sessions(later, spine)
    assert len(mapped) == len(spine)
    assert all(day in spine for day in mapped.dropna())


def test_split_label_boundaries():
    split = SplitCfg(
        train_start=date(2019, 1, 1),
        train_end=date(2021, 12, 31),
        valid_end=date(2022, 12, 31),
        test_end=date(2023, 12, 31),
    )
    dates = pd.Series(pd.to_datetime(["2019-06-03", "2022-06-01", "2023-06-01", "2024-06-03"]))
    assert list(split_label(dates, split)) == ["train", "valid", "test", "unused"]


def test_feature_columns_have_no_target():
    cols = feature_columns(include_sentiment=True)
    assert not any(c == "y" or c.startswith("y_") for c in cols)
    assert len(cols) == len(set(cols))


def test_assemble_without_sentiment_keeps_sentiment_features_empty(config, prices):
    result = assemble_features(prices, sentiment=None, news=None, config=config)
    frame = result.features
    sentiment_cols = [c for c in feature_columns(True) if c not in feature_columns(False)]
    assert sentiment_cols
    assert frame[sentiment_cols].isna().all().all(), "no news must give NaN, never zero"
    assert {"y", "y_1d", "date", "ticker", "split"} <= set(frame.columns)
    assert set(frame["split"].unique()) <= {"train", "valid", "test", "excluded"}


def test_assemble_with_sentiment_produces_finite_scores(config, prices, articles):
    sentiment = fixtures.make_sentiment(articles)
    news = fixtures.make_news(articles)
    start, end = fixtures.fixture_window(120)
    result = assemble_features(
        prices, sentiment=sentiment, news=news, config=config, start=start, end=end
    )
    frame = result.features
    sentiment_cols = [c for c in feature_columns(True) if c not in feature_columns(False)]
    assert frame[sentiment_cols].notna().any().any()
    assert frame["sent_mean"].dropna().between(-1, 1).all()


def test_assemble_is_deterministic(config, prices, articles):
    sentiment = fixtures.make_sentiment(articles)
    news = fixtures.make_news(articles)
    kwargs = dict(sentiment=sentiment, news=news, config=config)
    first = assemble_features(prices, **kwargs).features
    second = assemble_features(prices, **kwargs).features
    pd.testing.assert_frame_equal(first, second, check_like=True)


def test_ghost_rows_are_masked(config, prices):
    frame = assemble_features(prices, sentiment=None, news=None, config=config).features
    if "ghost" in frame.columns:
        ghost = frame.loc[frame["ghost"].astype(bool)]
        if len(ghost):
            assert ghost[feature_columns(False)].isna().all().all()
