"""Evaluation metrics for cross-sectional return prediction (T7).

Everything here answers one question: *is the ranking of tickers inside a date
informative about next-session returns?* The unit of observation is the **date**,
because the portfolio lives or dies by the cross-section, not by a single row.

Functions
---------

``ic``
    Spearman rank correlation per date, then the mean over dates with a standard
    error from the time series of daily ICs, a t-statistic and a Newey-West
    (lag 5) t-stat for overlapping horizons. A constant prediction on a date gives
    an undefined (NaN) Spearman; it is treated as IC = 0 and logged — that is what
    a "no opinion" prediction means relative to the cross-section.

``regression_metrics`` / ``direction_metrics`` / ``decile_returns``
    Row-level diagnostics. ``r2_vs_zero`` measures the prediction against the
    constant zero predictor, the honest benchmark for a demeaned target.

``diebold_mariano``
    Test on *loss differentials* ``loss_a - loss_b`` (positive mean means B is
    better). Direct forecast comparison (``h = 1``) uses the Bartlett/HAC
    variance with ``h - 1`` lags; ``h > 1`` falls back to the Harvey-Leybourne-
    Newbold small-sample correction. Identical series give ``dm = 0``, ``p = 1``.

``bootstrap_ic``
    Circular block bootstrap of *daily* ICs, because daily IC is
    autocorrelated; resampling individual dates independently would understate
    the CI. Returns ``(mean, lo, hi)`` at 95 % by default.

Never a price level anywhere: ``y_true`` and ``y_pred`` are returns.

Rules pinned by the plan (T7):
* a date contributes to IC only when it has ``>= 10`` tickers;
* Newey-West lag = 5;
* every public function returns a plain ``dict`` (or ``pd.Series`` for deciles)
  with explicit key names so the JSON report needs no hand assembly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "IC_MIN_TICKERS_PER_DATE",
    "NEWEY_WEST_LAG",
    "bootstrap_ic",
    "decile_returns",
    "diebold_mariano",
    "direction_metrics",
    "ic",
    "newey_west_tstat",
    "regression_metrics",
    "valuation",
]

IC_MIN_TICKERS_PER_DATE = 10
NEWEY_WEST_LAG = 5


# --------------------------------------------------------------------------- helpers
def _as_float(values: Iterable[float]) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values.astype("float64", copy=False)
    return np.asarray(list(values), dtype="float64")


def _date_key(dates: pd.Series | pd.Index | Sequence) -> np.ndarray:
    if isinstance(dates, pd.Series):
        return pd.to_datetime(dates).to_numpy()
    return pd.to_datetime(pd.Index(dates)).to_numpy()


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via ranks; NaN when either side is constant."""
    if len(x) < 2:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy(dtype="float64")
    ry = pd.Series(y).rank(method="average").to_numpy(dtype="float64")
    sx, sy = rx.std(ddof=0), ry.std(ddof=0)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


# --------------------------------------------------------------------------- IC
def newey_west_tstat(
    values: Sequence[float] | np.ndarray, lag: int = NEWEY_WEST_LAG
) -> tuple[float, float]:
    """Mean, standard error and Newey-West t-stat of a series of per-date statistics.

    Returns ``(t_nw, se_nw)``. With fewer observations than ``lag + 2`` the lag is
    truncated to ``len - 2`` (zero variance information left otherwise).
    """
    x = _as_float(values)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    mean = float(x.mean())
    centred = x - mean
    lags = int(min(max(lag, 0), n - 2))
    gamma0 = float(np.dot(centred, centred) / n)
    s = gamma0
    for k in range(1, lags + 1):
        weight = 1.0 - k / (lags + 1.0)
        gamma_k = float(np.dot(centred[k:], centred[:-k]) / n)
        s += 2.0 * weight * gamma_k
    variance = max(s, 0.0) / n
    if variance <= 0.0:
        return float("nan"), float("nan")
    se = float(np.sqrt(variance))
    return mean / se, se


def ic(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    dates: pd.Series | pd.Index | Sequence,
    *,
    min_tickers: int = IC_MIN_TICKERS_PER_DATE,
    nw_lag: int = NEWEY_WEST_LAG,
) -> dict[str, object]:
    """Cross-sectional Spearman IC per date, then the summary time series.

    Parameters are aligned by *position* — callers must pass the same rows.
    Rows with a NaN ``y_true`` or ``y_pred`` are dropped before grouping; a date
    left with fewer than ``min_tickers`` usable rows is skipped.

    Returns a dict with::

        ic              mean daily Spearman
        ic_std          cross-date standard deviation of the daily ICs
        ic_se           standard error of the mean (sample std / sqrt(n))
        ic_t            t-statistic of the mean (time-series t, df = n - 1)
        ic_t_nw         Newey-West t-statistic (lag ``nw_lag``)
        ic_ci95_lo/hi   95 % normal CI on the mean (``ic +- 1.96 * ic_se``)
        ic_hit          share of dates with IC > 0 (directional consistency)
        n_dates         dates used
        n_dates_skipped dates with < ``min_tickers`` rows
        n_rows          rows used
        n_constant_dates dates whose prediction was constant (IC treated as 0)
        ic_series       DataFrame ``date, ic, n`` (for the bootstrap / plots)
    """
    true = _as_float(y_true)
    pred = _as_float(y_pred)
    if not (len(true) == len(pred)):
        raise ValueError(f"length mismatch: y_true {len(true)} vs y_pred {len(pred)}")
    keys = _date_key(dates)
    if len(keys) != len(true):
        raise ValueError(f"length mismatch: dates {len(keys)} vs y_true {len(true)}")

    frame = pd.DataFrame({"date": keys, "y": true, "p": pred})
    frame = frame.loc[np.isfinite(frame["y"].to_numpy()) & np.isfinite(frame["p"].to_numpy())]
    n_rows = int(len(frame))

    rows: list[dict[str, object]] = []
    skipped = 0
    constant = 0
    for day, group in frame.groupby("date", sort=True):
        if len(group) < min_tickers:
            skipped += 1
            continue
        rho = _spearman(group["y"].to_numpy(), group["p"].to_numpy())
        if not np.isfinite(rho):
            constant += 1
            rho = 0.0
        rows.append({"date": day, "ic": rho, "n": len(group)})

    series = pd.DataFrame(rows, columns=["date", "ic", "n"])
    n = len(series)
    if n == 0:
        return {
            "ic": float("nan"),
            "ic_std": float("nan"),
            "ic_se": float("nan"),
            "ic_t": float("nan"),
            "ic_t_nw": float("nan"),
            "ic_ci95_lo": float("nan"),
            "ic_ci95_hi": float("nan"),
            "ic_hit": float("nan"),
            "n_dates": 0,
            "n_dates_skipped": skipped,
            "n_rows": n_rows,
            "n_constant_dates": constant,
            "ic_series": series,
        }

    values = series["ic"].to_numpy(dtype="float64")
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else float("nan")
    se = float(std / np.sqrt(n)) if n > 1 else float("nan")
    t_stat = float(mean / se) if n > 1 and se > 0 else float("nan")
    t_nw, _ = newey_west_tstat(values, nw_lag)
    ci = 1.96 * se if np.isfinite(se) else float("nan")
    return {
        "ic": mean,
        "ic_std": std,
        "ic_se": se,
        "ic_t": t_stat,
        "ic_t_nw": t_nw,
        "ic_ci95_lo": mean - ci,
        "ic_ci95_hi": mean + ci,
        "ic_hit": float(np.mean(values > 0.0)),
        "n_dates": int(n),
        "n_dates_skipped": int(skipped),
        "n_rows": n_rows,
        "n_constant_dates": int(constant),
        "ic_series": series,
    }


def valuation(result: dict[str, object]) -> dict[str, object]:
    """Strip the non-serialisable ``ic_series`` so the dict lands in JSON as-is."""
    return {k: v for k, v in result.items() if k != "ic_series"}


# --------------------------------------------------------------------------- regression
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE, R2 and R2 against the constant zero prediction ``R2_vs_zero``."""
    true = _as_float(y_true)
    pred = _as_float(y_pred)
    keep = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[keep], pred[keep]
    n = len(true)
    if n == 0:
        return dict.fromkeys(
            ["mae", "rmse", "r2", "r2_vs_zero", "bias", "pearson", "n"], float("nan")
        )
    err = pred - true
    sst = float(np.sum((true - true.mean()) ** 2))
    sse = float(np.sum(err**2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    r2_zero = 1.0 - sse / float(np.sum(true**2)) if np.any(true != 0.0) else float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        pearson = float(np.corrcoef(true, pred)[0, 1]) if n > 1 else float("nan")
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": r2,
        "r2_vs_zero": r2_zero,
        "bias": float(np.mean(err)),
        "pearson": pearson,
        "n": float(n),
    }


# --------------------------------------------------------------------------- direction
def direction_metrics(y_true, y_pred, *, threshold: float = 0.0) -> dict[str, float]:
    """Directional accuracy around an explicit threshold (default: the zero return).

    ``pred_class = +1`` when ``y_pred > threshold``, ``-1`` when ``< -threshold``,
    ``0`` otherwise. Rows with a zero prediction are excluded from accuracy (the
    model expressed no direction); ``coverage`` is the share it did express.
    ``hit_rate_long/short`` are conditional accuracies of the up/down calls.
    """
    true = _as_float(y_true)
    pred = _as_float(y_pred)
    keep = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[keep], pred[keep]
    if len(true) == 0:
        return dict.fromkeys(
            ["accuracy", "balanced_accuracy", "auc", "hit_rate_long", "hit_rate_short",
             "coverage", "n", "n_directional"],
            float("nan"),
        )
    sign_true = np.sign(true)
    long_mask = pred > threshold
    short_mask = pred < -threshold
    directional = long_mask | short_mask
    out: dict[str, float] = {
        "n": float(len(true)),
        "coverage": float(directional.mean()),
        "n_directional": float(directional.sum()),
    }
    if not np.any(directional):
        out.update(
            accuracy=float("nan"), balanced_accuracy=float("nan"), auc=float("nan"),
            hit_rate_long=float("nan"), hit_rate_short=float("nan"),
        )
        return out
    correct = np.where(long_mask, sign_true > 0, sign_true < 0)
    acc = float(correct[directional].mean())
    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())
    hit_long = float(correct[long_mask].mean()) if n_long else float("nan")
    hit_short = float(correct[short_mask].mean()) if n_short else float("nan")
    tpr = hit_long if n_long else float("nan")
    tnr = hit_short if n_short else float("nan")
    if np.isfinite(tpr) or np.isfinite(tnr):
        balanced = float(np.nanmean([tpr, tnr]))
    else:
        balanced = float("nan")
    auc = _auc_from_scores(sign_true > 0, pred)
    out.update(
        accuracy=acc,
        balanced_accuracy=balanced,
        auc=auc,
        hit_rate_long=hit_long,
        hit_rate_short=hit_short,
    )
    return out


def _auc_from_scores(label: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC; NaN when only one class is present."""
    pos = label.astype(bool)
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype="float64")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# --------------------------------------------------------------------------- deciles
def decile_returns(
    y_true, y_pred, *, n: int = 10, dates: pd.Series | None = None
) -> pd.Series:
    """Mean realised return by prediction bucket, averaged over dates.

    When ``dates`` is given the bucketing happens **inside each date** (cross-
    sectional deciles — the only meaningful definition for a trading signal) and
    the result is the equal-weight mean across dates. Without dates the ranks are
    pooled over the whole sample, which is still monotone but not tradeable.
    """
    true = _as_float(y_true)
    pred = _as_float(y_pred)
    keep = np.isfinite(true) & np.isfinite(pred)
    frame = pd.DataFrame({"y": true[keep], "p": pred[keep]})
    if dates is not None:
        frame["date"] = _date_key(dates)[keep]
        buckets: dict[int, list[float]] = {}
        for _, group in frame.groupby("date", sort=True):
            if len(group) < n:
                continue
            # qcut on ranks: unique labels per bucket, deterministic ordering.
            rank = group["p"].rank(method="first")
            codes = pd.qcut(rank, n, labels=False, duplicates="drop")
            for code, mean in group.groupby(codes)["y"].mean().items():
                buckets.setdefault(int(code), []).append(float(mean))
        out = {k + 1: float(np.mean(v)) for k, v in sorted(buckets.items())}
    else:
        codes = pd.qcut(frame["p"].rank(method="first"), n, labels=False, duplicates="drop")
        out = {
            int(k) + 1: float(v) for k, v in frame.groupby(codes)["y"].mean().items()
        }
    return pd.Series(out, name="mean_return").sort_index()


# --------------------------------------------------------------------------- DM
def diebold_mariano(loss_a, loss_b, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy on per-period losses.

    Inputs are **losses** (``loss_a = |err_a|`` for MAE). The statistic is the
    mean loss differential ``d = loss_a - loss_b`` divided by its HAC standard
    error, so ``DM > 0`` means A has the larger loss (B is better). The p-value is
    two-sided normal; ``h > 1`` applies the Harvey-Leybourne-Newbold correction.
    """
    a = _as_float(loss_a)
    b = _as_float(loss_b)
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    n = len(d)
    if n < 3:
        return float("nan"), float("nan")
    if np.allclose(d, d[0], atol=0.0, rtol=0.0):
        return 0.0, 1.0
    mean = float(d.mean())
    lags = int(min(max(h - 1, 0), n - 2))
    s = float(np.dot(d, d) / n)
    for k in range(1, lags + 1):
        s += 2.0 * float(np.dot(d[k:], d[:-k]) / n)
    variance = max(s, 0.0) / n
    if variance <= 0.0:
        return float("nan"), float("nan")
    dm = mean / float(np.sqrt(variance))
    if h > 1:  # HLN small-sample correction
        dm *= float(np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n))
    from scipy.stats import norm

    p = float(2.0 * (1.0 - norm.cdf(abs(dm))))
    return float(dm), p


# --------------------------------------------------------------------------- bootstrap
def bootstrap_ic(
    y_true,
    y_pred,
    dates: pd.Series | pd.Index | Sequence,
    n: int = 1000,
    *,
    min_tickers: int = IC_MIN_TICKERS_PER_DATE,
    seed: int = 42,
    alpha: float = 0.05,
    block_length: int | None = None,
) -> tuple[float, float, float]:
    """Circular block bootstrap CI of the mean daily IC.

    Resampling *dates* independently ignores the serial dependence induced by
    overlapping 1-day returns (autocorrelated residuals); a circular moving-block
    bootstrap with ``block_length = ceil(n_dates**(1/3))`` keeps it. Returns
    ``(point, lo, hi)`` where ``point`` is the observed mean IC.
    """
    result = ic(y_true, y_pred, dates, min_tickers=min_tickers)
    series = result["ic_series"]
    values = series["ic"].to_numpy(dtype="float64")
    nd = len(values)
    if nd == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(values.mean())
    if nd == 1 or n <= 0:
        return point, point, point
    block = block_length if block_length is not None else int(np.ceil(nd ** (1.0 / 3.0)))
    block = int(max(1, min(block, nd)))
    rng = np.random.default_rng(seed)
    starts_per_draw = int(np.ceil(nd / block))
    means = np.empty(n, dtype="float64")
    extended = np.concatenate([values, values])  # circular wrapping
    for i in range(n):
        starts = rng.integers(0, nd, size=starts_per_draw)
        sample = np.concatenate([extended[s : s + block] for s in starts])[:nd]
        means[i] = float(sample.mean())
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return point, lo, hi
