"""Tests for the event-level evaluator (``spp.backtest.intraday_eval``).

The module is the arbiter of the intraday experiment, so the tests pin the properties that
decide a verdict, not the arithmetic of individual means:

* clustering on dates must widen the standard error when events inside a day move together;
* the cost model must hit the mean exactly, with a break-even hit rate that matches it;
* a score that genuinely ranks outcomes must produce a positive quintile spread and event IC;
* power must be monotone and must fail loudly (NaN) rather than return a confident zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spp.backtest.intraday_eval import (
    cluster_tstat,
    evaluate_by_horizon,
    evaluate_events,
    minimum_detectable_effect,
    quintile_spread,
)


def _frame(n_dates: int = 40, per_date: int = 8, seed: int = 7) -> pd.DataFrame:
    """Events with a market-wide component: every date shares one draw of ``market``."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(n_dates):
        market = rng.normal(0.0, 0.01)  # the common move every event on this date inherits
        for k in range(per_date):
            score = rng.normal(0.0, 1.0)
            noise = rng.normal(0.0, 0.004)
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=day),
                    "ticker": f"T{k:02d}",
                    "horizon": 120,
                    "score": score,
                    "y": market + 0.002 * score + noise,
                }
            )
    return pd.DataFrame(rows)


def test_clustering_widens_the_error_when_dates_share_a_shock() -> None:
    frame = _frame()
    naive_se = frame["y"].std(ddof=1) / np.sqrt(len(frame))
    result = cluster_tstat(frame["y"].to_numpy(), frame["date"].to_numpy())
    assert result["n_clusters"] == frame["date"].nunique()
    assert result["se"] > naive_se  # the shared market move cannot average away
    assert result["inflation"] > 1.0


def test_clustering_is_undefined_with_one_date() -> None:
    frame = _frame(n_dates=1)
    result = cluster_tstat(frame["y"].to_numpy(), frame["date"].to_numpy())
    assert result["n_clusters"] == 1
    assert np.isnan(result["se"])


def test_cost_is_subtracted_exactly_once() -> None:
    frame = _frame()
    report = evaluate_events(frame, cost_bps=5.0, spread_bps=5.0)
    assert report["cost_per_round_trip"] == pytest.approx(15.0 / 1e4)
    assert report["net_mean"] == pytest.approx(report["mean_y"] - 15.0 / 1e4)


def test_breakeven_hit_rate_matches_the_break_even_condition() -> None:
    frame = _frame()
    report = evaluate_events(frame, cost_bps=5.0, spread_bps=5.0)
    hit = report["breakeven_hit_rate"]
    mean_abs = report["mean_abs_y"]
    cost = report["cost_per_round_trip"]
    # At the break-even hit rate a symmetric rule earns exactly the cost.
    assert (2.0 * hit - 1.0) * mean_abs == pytest.approx(cost)


def test_a_predictive_score_gives_positive_spread_and_ic() -> None:
    frame = _frame()
    report = evaluate_events(frame)
    assert report["score_quintiles"]["spread"] > 0.0
    assert report["event_ic"]["mean"] > 0.0
    assert report["hit_rate_directional"] > 0.5


def test_a_useless_score_does_not_fake_a_spread() -> None:
    frame = _frame(seed=11)
    rng = np.random.default_rng(0)
    frame["score"] = rng.normal(size=len(frame))  # rebuild the score from scratch
    spread = quintile_spread(frame)["spread"]
    assert abs(spread) < 0.004  # nothing systematic, within one event's dispersion


def test_horizon_filter_and_multi_horizon_report() -> None:
    frame = _frame()
    other = frame.assign(horizon=180, y=frame["y"] * 1.5)
    both = pd.concat([frame, other], ignore_index=True)
    single = evaluate_events(both, horizon=120)
    assert single["n_events"] == len(frame)
    assert single["horizon_minutes"] == 120
    report = evaluate_by_horizon(both)
    assert set(report) == {"h120", "h180"}
    assert report["h180"]["n_events"] == len(other)


def test_empty_frame_reports_zero_events_instead_of_raising() -> None:
    empty = pd.DataFrame(columns=["date", "ticker", "horizon", "score", "y"])
    report = evaluate_events(empty)
    assert report["n_events"] == 0


def test_power_improves_with_more_events_and_degrades_with_clustering() -> None:
    tight = minimum_detectable_effect(0.009, 10_000)
    loose = minimum_detectable_effect(0.009, 1_000)
    assert tight < loose
    clustered = minimum_detectable_effect(0.009, 10_000, inflation=2.2)
    assert clustered == pytest.approx(2.2 * tight)
    assert np.isnan(minimum_detectable_effect(0.009, 0))
    assert np.isnan(minimum_detectable_effect(float("nan"), 100))


def test_a_t_statistic_from_two_clusters_is_not_called_reliable() -> None:
    """Two day-means produce a large |t|; the report must refuse to present it as a result."""
    frame = pd.DataFrame(
        {
            "date": ["2026-10-01"] * 16 + ["2026-10-02"],
            "y": [0.01] * 16 + [0.0105],
        }
    )
    report = evaluate_events(frame)
    assert report["cluster"]["n_clusters"] == 2
    assert report["cluster_reliable"] is False
    assert report["cluster"]["t"] > 5.0  # arithmetic, not evidence