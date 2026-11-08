"""Evaluation of event-level trades produced by the intraday experiment.

The daily pipeline asks *is the ranking of tickers inside a date informative?* The intraday
experiment asks a different question: **given one article, does the trade opened five minutes
later earn more than the market over the next one to three hours?** That changes three things,
and every function here is shaped by them.

1. The unit of observation is the **event**, not the ticker-day. One date can carry forty events,
   so a naive standard error over events is far too small: events on the same day share the market
   move they are measured against. ``cluster_tstat`` therefore clusters by date, which is the
   minimum honest correction.
2. The target is an **excess return over SPY**, so the benchmark is zero by construction. There is
   no "R2 versus zero" to argue about - the mean must simply beat the cost of the round trip.
3. The result is a **trading rule**, so the report has to state the cost explicitly and the
   break-even hit rate next to the realised one. A mean excess return of 6 bps is a finding; the
   same number net of 15 bps is a loss.

Nothing here knows about bars, tickers or databases: the caller passes a frame with the columns
``date`` and ``y`` (excess return, decimal), plus optional ``score``, ``horizon``, ``ticker``.
That keeps the module testable without any market data.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "BREAKEVEN_TOLERANCE",
    "MDE_Z_SUM",
    "cluster_tstat",
    "evaluate_by_horizon",
    "evaluate_events",
    "minimum_detectable_effect",
    "quintile_spread",
]

#: Two-sided alpha 0.05 and power 0.80 give ``1.96 + 0.84``; the sum is the multiplier on the
#: standard error that makes a true effect detectable four times out of five.
MDE_Z_SUM = 2.802

#: A cost estimate is never exact, so the break-even hit rate is reported with this tolerance
#: before it is called a failure.
BREAKEVEN_TOLERANCE = 1e-12


def _as_float(values: Iterable[float]) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values.astype("float64", copy=False)
    return np.asarray(list(values), dtype="float64")


def _finite(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for column in columns:
        if column in frame.columns:
            mask &= np.isfinite(_as_float(frame[column]))
    return frame.loc[mask]


def cluster_tstat(values: Sequence[float], clusters: Sequence[Any]) -> dict[str, float]:
    """Mean of ``values`` with a standard error clustered on ``clusters``.

    The correction is the textbook one: the variance of the total is the sum of squared
    *cluster sums* of deviations, not of individual squared deviations. With one cluster the
    standard error is undefined (no independent replication), so NaN is returned rather than a
    number that would look like precision.

    Returns ``mean, se, t, p, n, n_clusters, inflation`` where ``inflation = se / se_naive``
    tells the reader how much the clustering cost them.
    """
    x = _as_float(values)
    keys = np.asarray(list(clusters), dtype=object)
    if len(x) != len(keys):
        raise ValueError(f"length mismatch: values {len(x)} vs clusters {len(keys)}")
    keep = np.isfinite(x)
    x, keys = x[keep], keys[keep]
    n = len(x)
    empty = {
        "mean": float("nan"),
        "se": float("nan"),
        "t": float("nan"),
        "p": float("nan"),
        "n": float(n),
        "n_clusters": 0,
        "inflation": float("nan"),
    }
    if n < 3:
        return empty
    mean = float(x.mean())
    frame = pd.DataFrame({"x": x, "k": keys})
    summed = frame.groupby("k", sort=True)["x"].agg(["sum", "count"])
    n_clusters = int(len(summed))
    if n_clusters < 2:
        return {**empty, "mean": mean, "n": float(n), "n_clusters": n_clusters}
    resid = summed["sum"].to_numpy(dtype="float64") - mean * summed["count"].to_numpy("float64")
    variance = float(np.sum(resid**2)) / (n * n)
    variance *= n_clusters / (n_clusters - 1.0)
    se = float(np.sqrt(max(variance, 0.0)))
    se_naive = float(x.std(ddof=1) / np.sqrt(n))
    if se <= 0.0:
        return {
            **empty,
            "mean": mean,
            "n": float(n),
            "n_clusters": n_clusters,
            "inflation": float("nan"),
        }
    t_stat = mean / se
    from scipy.stats import norm

    return {
        "mean": mean,
        "se": se,
        "t": float(t_stat),
        "p": float(2.0 * (1.0 - norm.cdf(abs(t_stat)))),
        "n": float(n),
        "n_clusters": n_clusters,
        "inflation": float(se / se_naive) if se_naive > 0 else float("nan"),
    }


def quintile_spread(
    frame: pd.DataFrame, *, score_col: str = "score", n: int = 5
) -> dict[str, float]:
    """Mean excess return of the top minus the bottom ``1/n`` of events by ``score``.

    Buckets are formed **inside each date**: a rule that sorts all events of five years at once is
    not tradeable, because at decision time the future scores do not exist.
    """
    if score_col not in frame.columns or frame.empty:
        return {"top": float("nan"), "bottom": float("nan"), "spread": float("nan"), "n_dates": 0.0}
    work = _finite(frame, "y", score_col)
    work = work.assign(date=pd.to_datetime(work["date"]))
    tops: list[float] = []
    bottoms: list[float] = []
    for _, group in work.groupby("date", sort=True):
        if len(group) < n or group[score_col].nunique() < 2:
            continue
        ordered = group.sort_values(score_col)
        size = max(1, len(ordered) // n)
        bottoms.append(float(ordered["y"].head(size).mean()))
        tops.append(float(ordered["y"].tail(size).mean()))
    if not tops:
        return {"top": float("nan"), "bottom": float("nan"), "spread": float("nan"), "n_dates": 0.0}
    top = float(np.mean(tops))
    bottom = float(np.mean(bottoms))
    return {
        "top": top,
        "bottom": bottom,
        "spread": top - bottom,
        "n_dates": float(len(tops)),
    }


def _breakeven_hit_rate(mean_abs_y: float, cost: float) -> float:
    """Hit rate at which ``hit * mean_abs_y - (1 - hit) * mean_abs_y`` equals ``cost``.

    With symmetric win/loss magnitudes the mean return of a directional rule is
    ``(2 * hit - 1) * mean_abs_y``, so the cost is recovered at
    ``hit = 0.5 + cost / (2 * mean_abs_y)``.
    """
    if not np.isfinite(mean_abs_y) or mean_abs_y <= 0.0:
        return float("nan")
    return 0.5 + cost / (2.0 * mean_abs_y)


def evaluate_events(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    cost_bps: float = 5.0,
    spread_bps: float = 5.0,
    min_events_per_date: int = 5,
    horizon: int | None = None,
) -> dict[str, Any]:
    """Full report for one event set: level, significance, ranking power and costs.

    ``cost_bps`` is charged **per side** and ``spread_bps`` is the half-spread assumption; the
    round trip therefore costs ``2 * cost_bps + spread_bps``, which is subtracted from the mean
    excess return to give ``net_mean``. Every number is JSON-serialisable.

    ``horizon`` only filters; it is not used in any formula, so callers can loop over horizons
    without worrying that the metric changes shape.
    """
    if frame.empty:
        return {"n_events": 0, "note": "empty event frame"}
    work = frame.copy()
    if horizon is not None:
        if "horizon" not in work.columns:
            raise ValueError("horizon filter requested but the frame has no 'horizon' column")
        work = work.loc[work["horizon"] == horizon]
    work = _finite(work, "y")
    if work.empty:
        return {"n_events": 0, "note": "no finite y after filtering"}
    work["date"] = pd.to_datetime(work["date"])

    y = work["y"].to_numpy(dtype="float64")
    clustered = cluster_tstat(y, work["date"].to_numpy())
    cost = (2.0 * float(cost_bps) + float(spread_bps)) / 1e4
    mean_abs = float(np.mean(np.abs(y)))
    hit_up = float(np.mean(y > 0.0))
    report: dict[str, Any] = {
        "n_events": int(len(work)),
        "n_dates": int(work["date"].nunique()),
        "events_per_date": float(len(work) / max(1, work["date"].nunique())),
        "horizon_minutes": None if horizon is None else int(horizon),
        "mean_y": float(np.mean(y)),
        "median_y": float(np.median(y)),
        "std_y": float(np.std(y, ddof=1)) if len(y) > 1 else float("nan"),
        "mean_abs_y": mean_abs,
        "hit_rate_up": hit_up,
        "cluster": clustered,
        "cost_per_round_trip": cost,
        "net_mean": float(np.mean(y)) - cost,
        "breakeven_hit_rate": _breakeven_hit_rate(mean_abs, cost),
    }

    if score_col in work.columns and work[score_col].notna().any():
        scored = _finite(work, "y", score_col)
        by_date: list[float] = []
        for _, group in scored.groupby("date", sort=True):
            if len(group) < min_events_per_date or group[score_col].nunique() < 2:
                continue
            rho = _spearman(group[score_col].to_numpy(), group["y"].to_numpy())
            if np.isfinite(rho):
                by_date.append(rho)
        if by_date:
            values = np.asarray(by_date, dtype="float64")
            se = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else float("nan")
            )
            report["event_ic"] = {
                "mean": float(values.mean()),
                "se": se,
                "t": float(values.mean() / se) if np.isfinite(se) and se > 0 else float("nan"),
                "n_dates": int(len(values)),
            }
        else:
            report["event_ic"] = {"mean": float("nan"), "n_dates": 0}
        report["score_quintiles"] = quintile_spread(scored, score_col=score_col)
    else:
        report["event_ic"] = {"mean": float("nan"), "n_dates": 0}

    # Directional error: a rule that goes long on score > 0 is only as good as its sign agreement.
    if score_col in work.columns:
        scored = _finite(work, "y", score_col)
        directional = scored.loc[scored[score_col] != 0.0]
        if len(directional):
            agree = np.sign(directional[score_col].to_numpy()) == np.sign(
                directional["y"].to_numpy()
            )
            report["hit_rate_directional"] = float(np.mean(agree))
            report["n_directional"] = int(len(directional))
    return report


def evaluate_by_horizon(
    frame: pd.DataFrame, *, horizons: Sequence[int] | None = None, **kwargs: Any
) -> dict[str, dict[str, Any]]:
    """``evaluate_events`` for every horizon present, keyed by the horizon name."""
    if "horizon" not in frame.columns:
        raise ValueError("frame has no 'horizon' column")
    present = sorted(int(h) for h in pd.unique(frame["horizon"].dropna()))
    wanted = present if horizons is None else [int(h) for h in horizons if int(h) in present]
    return {f"h{minutes}": evaluate_events(frame, horizon=minutes, **kwargs) for minutes in wanted}


def minimum_detectable_effect(
    sigma: float,
    n_events: float,
    *,
    inflation: float = 1.0,
    alpha_z: float = 1.96,
    power_z: float = 0.84,
) -> float:
    """Excess return per event that this many events can detect at 5 % / 80 %.

    ``sigma`` is the event-level dispersion of the excess return and ``inflation`` the factor by
    which clustering on dates widens the standard error (measure it with ``cluster_tstat`` on
    realised data; 2.0 is a typical value for forty events a day sharing a market move).

    This is the number that decides whether an experiment is worth running: if the detectable
    effect exceeds the effect the literature reports, a null result is uninformative.
    """
    if not np.isfinite(sigma) or sigma <= 0.0 or n_events <= 0:
        return float("nan")
    return float((alpha_z + power_z) * inflation * sigma / np.sqrt(n_events))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy(dtype="float64")
    ry = pd.Series(y).rank(method="average").to_numpy(dtype="float64")
    sx, sy = rx.std(ddof=0), ry.std(ddof=0)
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))
