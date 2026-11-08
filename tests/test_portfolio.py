"""Tests for the long/short portfolio engine (T11).

Synthetic panels only: dates, prices and scores are generated inside the test, so
every number here is checkable by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spp.backtest.portfolio import (
    build_day_weights,
    build_execution_frame,
    compute_turnover,
    leg_holdings,
    run_backtest,
)
from spp.backtest.report import (
    block_bootstrap_sharpe,
    ic_diagnostics,
    quintile_returns,
    reproducible_stats,
)


# --------------------------------------------------------------------------- fixtures
def synthetic_panel(n_dates: int = 60, n_tickers: int = 20, seed: int = 0) -> tuple:
    """Prices with an exact, tradable link between score and open→close return."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    tickers = [f"T{index:02d}" for index in range(n_tickers)]
    rows = []
    for ticker in tickers:
        price = 100.0
        for day in dates:
            open_px = price
            close_px = open_px * (1.0 + rng.normal(0, 0.01))
            rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "open": open_px,
                    "close": close_px,
                    "adj_close": close_px,
                }
            )
            price = close_px
    prices = pd.DataFrame(rows)
    scores = [
        {"date": day, "ticker": ticker, "y_pred_seed_mean": rng.normal()}
        for day in dates[:-1]  # no prediction on the last day: no forward return
        for ticker in tickers
    ]
    return pd.DataFrame(scores), prices


def oracle_frame(n_dates: int = 40, exact: int = 10) -> pd.DataFrame:
    """Frame where the score is exactly the realised return: perfect foresight."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    rows = []
    for day in dates:
        rets = rng.normal(0, 0.01, size=20)
        for ticker, value in zip((f"T{k:02d}" for k in range(20)), rets, strict=True):
            rows.append({"date": day, "ticker": ticker, "ret": value, "adj_ret": value})
    frame = pd.DataFrame(rows)
    # Perfect foresight: the rank of the score equals the rank of the realised return.
    frame["score"] = frame["ret"] * 100.0
    frame["sector"] = "tech"
    frame["halted"] = False
    return frame


# --------------------------------------------------------------------------- invariants
def test_net_identity_holds_to_1e12() -> None:
    scores, prices = synthetic_panel()
    frame = build_execution_frame(scores, prices, "next_open_hold_close").frame
    result = run_backtest(frame)
    residual = (
        result.daily["trade_ret"] - result.daily["cost"] - result.daily["borrow"]
        - result.daily["net_ret"]
    ).abs().max()
    assert residual <= 1e-12


def test_book_is_dollar_neutral_every_day() -> None:
    scores, prices = synthetic_panel()
    frame = build_execution_frame(scores, prices, "next_open_hold_close").frame
    result = run_backtest(frame)
    assert result.daily["net_weight"].abs().max() < 1e-9
    assert (result.daily["gross_long"] - result.daily["gross_short"]).abs().max() < 1e-9
    assert abs(result.daily["gross_long"].mean() - 0.5) < 1e-9


def test_zero_costs_make_net_equal_gross() -> None:
    scores, prices = synthetic_panel()
    frame = build_execution_frame(scores, prices, "next_open_hold_close").frame
    result = run_backtest(
        frame, commission_bps=0.0, slippage_bps=0.0, borrow_annual=0.0
    )
    assert result.summary["sharpe_net"] == pytest.approx(result.summary["sharpe_gross"], abs=1e-12)
    assert (result.daily["net_ret"] - result.daily["trade_ret"]).abs().max() <= 1e-15


def test_turnover_formula_matches_the_plan() -> None:
    day_one = pd.Series({"A": 0.5, "B": -0.5})
    day_two = pd.Series({"A": -0.5, "B": 0.5})
    out = compute_turnover([day_one, day_two], cost_per_side=0.001)
    # Day 1 opens from cash: 0.5 * (0.5 + 0.5) = 0.5; day 2 flips both: 1.0.
    assert out["turnover"].tolist() == pytest.approx([0.5, 1.0])
    assert out["cost"].tolist() == pytest.approx([0.001, 0.002])


def test_borrow_is_zero_without_shorts() -> None:
    n_days, n_names = 20, 20
    rows = pd.DataFrame(
        {
            "date": pd.bdate_range("2022-01-03", periods=n_days).repeat(n_names),
            "ticker": [f"T{k:02d}" for k in range(n_names)] * n_days,
            "ret": np.tile(np.linspace(-0.01, 0.01, n_names), n_days),
            "adj_ret": np.tile(np.linspace(-0.01, 0.01, n_names), n_days),
            "sector": "tech",
            "halted": False,
        }
    )
    rows["score"] = rows["ret"]  # top/bottom exist, so this is a normal book
    result = run_backtest(rows)
    daily = result.daily
    assert np.isfinite(daily["borrow"]).all()
    # Day 0 opens the book (no prior position), every later day carries short exposure.
    assert (daily["borrow"].iloc[1:] > 0).all()
    expected = 0.05 / 252.0 * daily["gross_short"]
    assert np.allclose(daily["borrow"], expected, atol=1e-15)


def test_thin_legs_skip_the_day() -> None:
    day = pd.Timestamp("2022-01-03")
    frame = pd.DataFrame(
        {
            "date": day,
            "ticker": list("ABCD"),
            "ret": [0.01, 0.02, -0.01, -0.02],
            "score": [0.9, 0.8, 0.1, 0.2],
            "sector": "tech",
        }
    )
    legs = leg_holdings(frame, n_quantiles=2)
    book = build_day_weights(legs, n_quantiles=2, min_names=3, min_leg_names=3)
    assert not book.traded
    assert book.reason in {"few_names", "thin_leg"}


def test_few_names_skip_the_day() -> None:
    day = pd.Timestamp("2022-01-03")
    frame = pd.DataFrame(
        {
            "date": day,
            "ticker": list("ABC"),
            "ret": [0.01, 0.02, -0.01],
            "score": [0.9, 0.8, 0.1],
            "sector": "tech",
        }
    )
    legs = leg_holdings(frame, n_quantiles=2)
    book = build_day_weights(legs, n_quantiles=2, min_names=5)
    assert not book.traded and book.reason == "few_names"


def test_halt_rows_are_flagged_not_filled() -> None:
    _, prices = synthetic_panel(n_dates=10, n_tickers=6)
    # Freeze one session into a zero range: open == close.
    mask = (prices["ticker"] == "T00") & (prices["date"] == pd.Timestamp("2022-01-05"))
    prices.loc[mask, "close"] = prices.loc[mask, "open"]
    scores = pd.DataFrame(
        {
            "date": pd.Timestamp("2022-01-04"),
            "ticker": ["T00", "T01", "T02", "T03", "T04", "T05"],
            "y_pred_seed_mean": [0.9, 0.8, 0.7, 0.1, 0.2, 0.3],
        }
    )
    frame = build_execution_frame(scores, prices, "next_open_hold_close").frame
    assert bool(frame.loc[frame["ticker"] == "T00", "halted"].iloc[0]) is True
    assert frame.loc[frame["ticker"] == "T00", "ret"].iloc[0] == pytest.approx(0.0)


def test_out_of_sample_replay_has_no_lookahead() -> None:
    """Changing a future price must not change today's weights or today's return."""
    scores, prices = synthetic_panel(n_dates=30, n_tickers=12)
    base = build_execution_frame(scores, prices, "next_open_hold_close").frame
    result_a = run_backtest(base)

    mutated = prices.copy()
    late = mutated["date"] > pd.Timestamp("2022-02-01")
    mutated.loc[late, "close"] *= 1.5
    mutated.loc[late, "open"] *= 1.5
    changed = build_execution_frame(scores, mutated, "next_open_hold_close").frame
    result_b = run_backtest(changed)

    early = result_a.daily["date"] <= pd.Timestamp("2022-02-01")
    pd.testing.assert_series_equal(
        result_a.daily.loc[early, "net_ret"], result_b.daily.loc[early, "net_ret"]
    )
    for book_a, book_b in zip(result_a.books, result_b.books, strict=True):
        if book_a.date > pd.Timestamp("2022-02-01"):
            break
        pd.testing.assert_series_equal(book_a.weights, book_b.weights)


# --------------------------------------------------------------------------- skill
def test_oracle_scores_earn_positive_sharpe() -> None:
    frame = oracle_frame()
    result = run_backtest(frame, commission_bps=0.0, slippage_bps=0.0, borrow_annual=0.0)
    assert result.summary["sharpe_gross"] > 3.0
    assert result.daily["gross_ret"].mean() > 0


def test_permuted_scores_stay_near_zero() -> None:
    frame = oracle_frame(n_dates=120)
    rng = np.random.default_rng(3)
    shuffled = frame.copy()
    shuffled["score"] = (
        shuffled.groupby("date")["score"].transform(lambda s: rng.permutation(s.to_numpy()))
    )
    result = run_backtest(shuffled, commission_bps=0.0, slippage_bps=0.0, borrow_annual=0.0)
    assert abs(result.summary["sharpe_net"]) < 3.0


def test_quintiles_are_monotone_for_oracle() -> None:
    frame = oracle_frame()
    table = quintile_returns(frame, n_quantiles=5)
    assert table.attrs["monotonicity_spearman"] > 0.9
    assert table["mean_ret"].is_monotonic_increasing


def test_ic_blocks_are_finite() -> None:
    frame = oracle_frame()
    block = ic_diagnostics(frame)
    assert block["pooled"]["ic"] > 0.5
    assert set(block["by_year"]) == {"2022"}
    assert block["trimmed_1pct"]["ic"] > 0.5


def test_bootstrap_ci_brackets_the_point() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.01, size=500)
    boot = block_bootstrap_sharpe(returns, n_draws=300)
    assert boot.lo < boot.point < boot.hi
    assert 0.0 <= boot.p_value <= 1.0


def test_legs_use_score_order() -> None:
    day = pd.Timestamp("2022-01-03")
    frame = pd.DataFrame(
        {
            "date": day,
            "ticker": list("ABCDEFGHIJ"),
            "ret": np.arange(10) / 100.0,
            "score": np.arange(10, dtype=float),
            "sector": "tech",
        }
    )
    ranked = leg_holdings(frame, n_quantiles=5)
    assert set(ranked.loc[ranked["leg"] == "long", "ticker"]) == {"I", "J"}
    assert set(ranked.loc[ranked["leg"] == "short", "ticker"]) == set("AB")


def test_stats_reproduce_from_csv(tmp_path) -> None:
    scores, prices = synthetic_panel(n_dates=80)
    frame = build_execution_frame(scores, prices, "next_open_hold_close").frame
    result = run_backtest(frame)
    path = tmp_path / "daily.csv"
    result.daily.to_csv(path, index=False)
    stats = reproducible_stats(path)
    assert stats["identity_residual_max"] <= 1e-12
    assert stats["sharpe_net"] == pytest.approx(result.summary["sharpe_net"], abs=1e-10)
    assert stats["max_dd"] == pytest.approx(result.summary["max_dd"], abs=1e-12)
    assert stats["max_abs_net_weight"] < 1e-9


def test_sector_cap_limits_exposure() -> None:
    day = pd.Timestamp("2022-01-03")
    frame = pd.DataFrame(
        {
            "date": np.repeat(day, 10),
            "ticker": [f"T{k}" for k in range(10)],
            "ret": np.linspace(-0.01, 0.01, 10),
            "score": np.linspace(0.0, 1.0, 10),
            "sector": ["tech"] * 5 + ["health"] * 5,
        }
    )
    legs = leg_holdings(frame, n_quantiles=5)
    book = build_day_weights(legs, n_quantiles=5, sector_cap=0.3)
    exposure = book.weights.abs().groupby(
        frame.set_index("ticker")["sector"].reindex(book.weights.index)
    ).sum()
    assert (exposure <= 0.3 * 0.5 + 1e-12).all()
    assert abs(book.weights.sum()) < 1e-12
