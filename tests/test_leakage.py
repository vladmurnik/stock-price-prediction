from __future__ import annotations

import pandas as pd
import pytest

from spp.config import load_config
from spp.features import fixtures
from spp.features.assemble import assemble_features, feature_columns
from spp.features.leakage import (
    check_article_signal_dates,
    check_disguised_target,
    check_feature_provenance,
    check_future_shift,
    check_shuffled_label,
    check_target_alignment,
    feature_matrix_hash,
    run_all_checks,
)


@pytest.fixture(scope="module")
def built():
    config = load_config()
    prices = fixtures.make_prices()
    articles = fixtures.make_articles()
    sentiment = fixtures.make_sentiment(articles)
    news = fixtures.make_news(articles)
    return assemble_features(prices, sentiment=sentiment, news=news, config=config)


@pytest.fixture(scope="module")
def train_mask(built):
    return (built.features["split"] == "train").to_numpy()


def test_target_alignment_passes_on_clean_labels(built):
    result = check_target_alignment(built.labels, built.adj_close)
    assert result.passed


def test_target_alignment_fails_when_labels_are_shifted(built):
    broken = built.labels.copy()
    broken["y_1d"] = broken.groupby("ticker")["y_1d"].shift(1)
    assert not check_target_alignment(broken, built.adj_close).passed


def test_disguised_target_passes_on_real_features(built, train_mask):
    cols = feature_columns(True)
    assert check_disguised_target(built.features, cols, train_mask).passed


def test_disguised_target_catches_the_target_as_a_feature(built, train_mask):
    leaky = built.features.copy()
    leaky["leak"] = leaky["y"]
    assert not check_disguised_target(leaky, [*feature_columns(True), "leak"], train_mask).passed


def test_shuffled_label_passes_without_leakage(built, train_mask):
    assert check_shuffled_label(built.features, feature_columns(True), train_mask, seed=42).passed


def test_future_shift_passes_on_real_features(built, train_mask):
    assert check_future_shift(built.features, feature_columns(True), train_mask).passed


def test_article_signal_dates_reject_the_future():
    spine = pd.DatetimeIndex(pd.bdate_range("2020-01-02", periods=20))
    good = pd.DataFrame(
        {"date": [spine[5]], "signal_date": [spine[5]], "ticker": ["AAA"], "article_id": ["a1"]}
    )
    assert check_article_signal_dates(good, spine).passed

    late = good.copy()
    late["signal_date"] = spine[7]
    assert not check_article_signal_dates(late, spine).passed

    off_spine = good.copy()
    off_spine["signal_date"] = pd.Timestamp("2020-01-04")  # a Saturday
    assert not check_article_signal_dates(off_spine, spine).passed


def test_article_signal_dates_skip_when_no_articles():
    empty = pd.DataFrame(columns=["date", "signal_date"])
    assert check_article_signal_dates(empty, pd.DatetimeIndex([])).passed


def test_feature_provenance_skips_without_sentiment():
    result = check_feature_provenance(pd.DataFrame(), pd.DataFrame(), feature_columns(True))
    assert result.passed
    assert "skipped" in str(result.details)


def test_feature_matrix_hash_is_stable_and_sensitive(built):
    cols = feature_columns(True)
    first = feature_matrix_hash(built.features, cols)
    assert first == feature_matrix_hash(built.features, cols)
    assert len(first) == 64

    changed = built.features.copy()
    changed.loc[changed.index[0], cols[0]] = 12345.0
    assert feature_matrix_hash(changed, cols) != first


def test_run_all_checks_reports_six_checks(built, train_mask):
    spine = pd.DatetimeIndex(built.adj_close.index)
    articles = built.articles[pd.to_datetime(built.articles["signal_date"]).isin(spine)]
    report = run_all_checks(
        features=built.features,
        feature_columns=feature_columns(True),
        panel=built.panel,
        inventory=built.inventory,
        articles=articles,
        labels=built.labels,
        adj_close=built.adj_close,
        train_mask=train_mask,
    )
    assert len(report.results) == 6
    assert report.passed, report.failed
