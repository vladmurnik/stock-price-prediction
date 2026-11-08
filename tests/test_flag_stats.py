"""Regression tests for the open-jev flag aggregates.

The bug these pin: ``_flag_stats`` used to emit ``is_company_specific_n/_d`` and
``is_novel_n/_d`` while ``_NUMERIC_DAY`` consumed ``cs_n/cs_d`` and ``nov_n/nov_d``, so
``company_specific_share`` and ``novelty_mean`` were silently all-NaN in every matrix.
A second, hidden defect sat behind the first: ``is_company_specific`` and ``is_novel`` are
*probabilities* in ``[0, 1]``, so the ``astype("boolean")`` cast raised
``TypeError: Need to pass bool-like values`` as soon as the column stopped being all-NaN.
Both are locked down here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spp.features.sentiment_daily import _NUMERIC_DAY, _flag_stats, prepare_articles


def _flags(values: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAPL"] * len(values),
            "date": pd.to_datetime(["2023-03-06"] * len(values)),
            "is_company_specific": pd.Series(values, dtype="object"),
        }
    )


def test_flag_stats_emits_the_keys_that_derive_consumes() -> None:
    stats = _flag_stats(_flags([0.9, 0.1, None]), "is_company_specific", "cs")
    assert list(stats.columns) == ["cs_n", "cs_d"]
    for column in stats.columns:
        assert column in _NUMERIC_DAY


def test_flag_stats_averages_probabilities() -> None:
    """Sum of probabilities over count of non-null values, i.e. the mean probability."""
    stats = _flag_stats(_flags([0.9, 0.1, None]), "is_company_specific", "cs")
    assert stats.loc[("AAPL", pd.Timestamp("2023-03-06")), "cs_n"] == pytest.approx(1.0)
    assert stats.loc[("AAPL", pd.Timestamp("2023-03-06")), "cs_d"] == 2


def test_flag_stats_reports_no_coverage_when_the_backend_has_no_flag() -> None:
    """FinBERT leaves the flags NaN.

    ``groupby.sum`` of an all-NaN column is 0, not NaN, so the aggregate cannot itself signal
    "no coverage".  The signal lives in the denominator: ``_derive`` guards on ``cs_d > 0`` and
    emits NaN, which keeps ``company_specific_share`` NaN for FinBERT exactly as it was.
    """
    stats = _flag_stats(_flags([None, None]), "is_company_specific", "cs")
    assert stats["cs_d"].tolist() == [0]
    assert stats["cs_n"].tolist() == [0]


def test_prepare_articles_keeps_flags_numeric() -> None:
    sentiment = pd.DataFrame(
        {
            "article_id": ["a1", "a2"],
            "ticker": ["AAPL", "AAPL"],
            "signal_date": pd.to_datetime(["2023-03-06", "2023-03-06"]),
            "score_5": [0.4, -0.4],
            "score_3": [0.4, -0.4],
            "p_neg": [0.1, 0.8],
            "p_neu": [0.3, 0.1],
            "p_pos": [0.6, 0.1],
            "confidence": [0.8, 0.8],
            "is_company_specific": [0.55, 0.45],
            "is_novel": [0.7, 0.2],
            "label_5": ["positive", "negative"],
            "skipped": [False, False],
        }
    )
    prepared = prepare_articles(sentiment, None)
    assert prepared["is_company_specific"].dtype == "float64"
    assert prepared["is_novel"].dtype == "float64"
    assert prepared["is_novel"].tolist() == [0.7, 0.2]


def test_prepare_articles_fills_missing_flags_with_nan() -> None:
    sentiment = pd.DataFrame(
        {
            "article_id": ["a1"],
            "ticker": ["AAPL"],
            "signal_date": pd.to_datetime(["2023-03-06"]),
            "score_5": [0.4],
            "label_5": ["positive"],
            "skipped": [False],
        }
    )
    prepared = prepare_articles(sentiment, None)
    assert np.isnan(prepared["is_company_specific"]).all()
    assert np.isnan(prepared["is_novel"]).all()
