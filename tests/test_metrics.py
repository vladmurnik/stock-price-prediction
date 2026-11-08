from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spp.models.baselines import BASELINE_CODES, fit_ridge, predict_baselines, ridge_predict
from spp.models.metrics import (
    bootstrap_ic,
    decile_returns,
    diebold_mariano,
    direction_metrics,
    ic,
    newey_west_tstat,
    regression_metrics,
)


# --------------------------------------------------------------------------- IC
def _panel(n_dates: int = 40, n_tickers: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    rows = [
        {"date": day, "ticker": f"T{k:02d}", "y": rng.normal(0, 0.01)}
        for day in dates
        for k in range(n_tickers)
    ]
    return pd.DataFrame(rows)


def test_ic_perfect_prediction_is_one():
    panel = _panel()
    res = ic(panel["y"], panel["y"], panel["date"])
    assert res["ic"] == pytest.approx(1.0, abs=1e-12)
    assert res["n_dates"] == 40
    assert res["n_rows"] == 40 * 30


def test_ic_inverted_prediction_is_minus_one():
    panel = _panel()
    res = ic(panel["y"], -panel["y"], panel["date"])
    assert res["ic"] == pytest.approx(-1.0, abs=1e-12)


def test_ic_constant_prediction_is_zero_and_logged():
    panel = _panel()
    res = ic(panel["y"], np.zeros(len(panel)), panel["date"])
    assert res["ic"] == pytest.approx(0.0)
    assert res["ic_std"] == pytest.approx(0.0)
    assert res["n_constant_dates"] == res["n_dates"]


def test_ic_skips_dates_with_fewer_than_ten_tickers():
    panel = _panel(n_dates=10, n_tickers=30)
    thin_day = pd.Timestamp("2021-12-31")
    extra = pd.DataFrame(
        {"date": thin_day, "ticker": ["A", "B", "C"], "y": [0.01, 0.02, 0.03]}
    )
    frame = pd.concat([panel, extra], ignore_index=True)
    res = ic(frame["y"], frame["y"], frame["date"])
    assert res["n_dates"] == 10
    assert res["n_dates_skipped"] == 1


def test_ic_contract_keys_present():
    panel = _panel()
    res = ic(panel["y"], panel["y"], panel["date"])
    for key in (
        "ic",
        "ic_std",
        "ic_se",
        "ic_t",
        "ic_t_nw",
        "ic_ci95_lo",
        "ic_ci95_hi",
        "ic_hit",
        "n_dates",
        "n_rows",
        "ic_series",
    ):
        assert key in res, key


def test_newey_west_minus1():
    z = np.array([0.01, -0.01, 0.02, 0.0, 0.03, -0.02, 0.01, 0.015])
    t_nw, se = newey_west_tstat(z, 2)
    assert np.isfinite(t_nw)
    assert se > 0


# --------------------------------------------------------------------------- regression / direction
def test_mae_constant_zero_equals_mean_abs_y():
    rng = np.random.default_rng(3)
    y = rng.normal(0, 0.01, size=500)
    out = regression_metrics(y, np.zeros_like(y))
    assert out["mae"] == pytest.approx(float(np.mean(np.abs(y))))
    assert out["rmse"] == pytest.approx(float(np.sqrt(np.mean(y**2))))
    assert out["r2_vs_zero"] == pytest.approx(0.0, abs=1e-12)


def test_direction_metrics_explicit_threshold():
    y = np.array([0.01, -0.01, 0.0, 0.02, -0.03])
    pred = np.array([0.01, -0.02, 0.0, -0.01, 0.02])
    out = direction_metrics(y, pred)
    # predictions: +, -, 0(skip), -, +  -> 4 directional; correct: row1 (+) and row5 (+)
    assert out["n_directional"] == 4
    assert out["accuracy"] == pytest.approx(0.5)
    assert out["hit_rate_long"] == pytest.approx(0.5)
    assert out["hit_rate_short"] == pytest.approx(0.5)
    assert out["coverage"] == pytest.approx(0.8)


def test_decile_returns_increases_with_signal():
    rng = np.random.default_rng(5)
    y = rng.normal(0, 0.01, size=200)
    pred = y + rng.normal(0, 0.001, size=200)
    dec = decile_returns(y, pred, n=5)
    assert len(dec) == 5
    assert dec.iloc[0] < dec.iloc[-1]


# --------------------------------------------------------------------------- DM / bootstrap
def test_diebold_mariano_identical_series_p_is_one():
    loss = np.abs(np.random.default_rng(4).normal(0, 0.01, size=100))
    dm, p = diebold_mariano(loss, loss)
    assert dm == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_diebold_mariano_detects_clearly_worse_forecast():
    rng = np.random.default_rng(6)
    good = np.abs(rng.normal(0, 0.005, size=200))
    bad = np.abs(rng.normal(0, 0.02, size=200))
    dm, p = diebold_mariano(bad, good)
    assert dm > 0  # loss_a (bad) larger -> positive
    assert p < 1e-6


def test_bootstrap_ci_contains_point():
    panel = _panel()
    noisy = panel["y"] + np.random.default_rng(9).normal(0, 0.01, size=len(panel))
    point, lo, hi = bootstrap_ic(panel["y"], noisy, panel["date"], n=200)
    assert lo <= point <= hi


def test_bootstrap_of_perfect_prediction_is_one():
    panel = _panel()
    point, lo, hi = bootstrap_ic(panel["y"], panel["y"], panel["date"], n=100)
    assert point == pytest.approx(1.0)
    assert lo <= 1.0 + 1e-12 and hi >= 1.0 - 1e-12


# --------------------------------------------------------------------------- ridge / baselines
def test_ridge_recovers_linear_signal_on_standardised_columns():
    rng = np.random.default_rng(11)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = 0.02 * X["a"] - 0.01 * X["b"] + 0.0005
    model = fit_ridge(X, y, alpha=1e-6)
    pred = ridge_predict(model, X)
    assert regression_metrics(y, pred)["r2"] > 0.99


def _matrix(seed: int = 0) -> pd.DataFrame:
    """Small synthetic feature matrix with the real split labels."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-06-01", "2022-12-30")
    tickers = [f"T{k:02d}" for k in range(12)]
    rows = []
    for day in dates:
        split = "train" if day.year == 2021 else "valid"
        for k, ticker in enumerate(tickers):
            y = float(rng.normal(0, 0.01))
            rows.append(
                {
                    "ticker": ticker,
                    "date": day,
                    "y": y,
                    "y_raw": y,
                    "label_ok": True,
                    "split": split,
                    "ret_1": float(rng.normal(0, 0.01)),
                    "ret_5": float(rng.normal(0, 0.02)),
                    "sent_mean": float(rng.normal(0, 0.3)) if k % 2 else np.nan,
                    "n_news": float(k % 3),
                }
            )
    return pd.DataFrame(rows)


def test_baselines_never_read_2023():
    frame = _matrix()
    frames = predict_baselines(frame)
    for code in BASELINE_CODES:
        valid = frames[code]
        assert pd.to_datetime(valid["date"]).max() <= pd.Timestamp("2022-12-31")
        assert pd.to_datetime(valid["date"]).min() >= pd.Timestamp("2022-01-01")


def test_b0_mae_equals_mean_abs_y():
    frames = predict_baselines(_matrix())
    b0 = frames["B0"]
    assert regression_metrics(b0["y"], b0["y_pred"])["mae"] == pytest.approx(
        float(np.mean(np.abs(b0["y"])))
    )
    assert ic(b0["y"], b0["y_pred"], b0["date"])["ic"] == pytest.approx(0.0)


def test_b3_directions_are_opposites():
    frames = predict_baselines(_matrix())
    np.testing.assert_allclose(
        frames["B3_mom"]["y_pred"].to_numpy(), -frames["B3_rev"]["y_pred"].to_numpy()
    )
