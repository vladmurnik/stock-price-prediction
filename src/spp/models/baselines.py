"""Baselines B0–B3 (T7): the bar the LightGBM model has to clear.

| code    | prediction                                    |
|---------|-----------------------------------------------|
| B0      | ``0`` for every row (no information)          |
| B1      | Ridge on market features                      |
| B2      | Ridge on sentiment features (NaN -> train med)|
| B3_mom  | ``+ret_5`` (trend continuation)                |
| B3_rev  | ``-ret_5`` (contrarian)                        |

Leakage rules baked into this module
------------------------------------
* Every fitted quantity (imputation medians, ridge coefficients, ridge alpha)
  is estimated on ``split == "train"`` only. The validation year is never read
  before the prediction is formed.
* Alpha is selected on an internal split: the last ``INNER_VALID_MONTHS`` months
  of **train** act as the inner validation set. 2022 is never used for tuning.
* Target is ``y`` (forward log return), never a price level.
* 2023 is *not* touched: the loader asserts ``test`` rows are not read.

The ridge is deliberately plain: ``np.linalg.solve`` on standardised columns,
``lambda = alpha * n`` so the penalty strength does not drift with row count.

``predict_baselines(features, cfg) -> dict[str, pd.DataFrame]`` returns, per
code, ``ticker, date, split, y, y_raw, y_pred`` rows so the metrics module and
the CLI report agree on the exact sample.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spp.features.assemble import feature_columns
from spp.features.sentiment_daily import SENTIMENT_FEATURES
from spp.logging import get_logger
from spp.models.metrics import (
    bootstrap_ic,
    decile_returns,
    diebold_mariano,
    direction_metrics,
    ic,
    regression_metrics,
)

__all__ = [
    "BASELINE_CODES",
    "INNER_VALID_MONTHS",
    "RIDGE_ALPHAS",
    "BaselinesError",
    "compute_baselines",
    "fit_ridge",
    "predict_baselines",
    "prepare_matrix",
    "render_report",
    "write_baselines_report",
]

LOG = get_logger("models.baselines")

BASELINE_CODES: tuple[str, ...] = ("B0", "B1", "B2", "B3_mom", "B3_rev")
RIDGE_ALPHAS: tuple[float, ...] = (0.1, 1.0, 10.0)
INNER_VALID_MONTHS = 3
#: Columns that are *not* model inputs: identifiers, target block, split tag.
NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "ticker",
    "date",
    "split",
    "y_1d",
    "y_2d",
    "y_3d",
    "y_5d",
    "y",
    "y_dir",
    "y_raw",
    "y_used_demeaned",
    "ghost",
    "label_ok",
)


class BaselinesError(RuntimeError):
    """The feature matrix cannot be used for the baseline evaluation."""


# --------------------------------------------------------------------------- matrix prep
def prepare_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Select train + valid rows with a usable label; hard-fail if 2023 leaks in."""
    if "split" not in frame.columns or "y" not in frame.columns:
        raise BaselinesError("feature matrix lacks 'split' or 'y'")
    splits = set(frame["split"].astype(str).unique())
    if "test" in splits:
        test_rows = int((frame["split"].astype(str) == "test").sum())
        LOG.warning(
            "input frame also contains %d test rows; they are dropped before fitting "
            "(T7 never touches 2023)",
            test_rows,
        )
    keep = frame["split"].astype(str).isin({"train", "valid"})
    work = frame.loc[keep].copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    if (work["date"] >= pd.Timestamp("2023-01-01")).any():
        raise BaselinesError(
            "2023 rows present in the train/valid slice: refusing to fit (T7 boundary)"
        )
    work = work.loc[work["y"].notna() & work["label_ok"].astype(bool)].reset_index(drop=True)
    if work.empty:
        raise BaselinesError("no labelled train/valid rows")
    return work


def _numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Float view of the feature block; ``dow`` (category) is cast, others coerced."""
    out = pd.DataFrame(index=frame.index)
    for name in columns:
        series = frame[name]
        if str(series.dtype) == "category" or str(series.dtype) == "Int8":
            out[name] = pd.to_numeric(series, errors="coerce").astype("float64")
        else:
            out[name] = pd.to_numeric(series, errors="coerce").astype("float64")
    return out


def _impute(X_train: pd.DataFrame, X_other: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Median imputation with statistics from ``X_train`` only.

    Constants cannot influence a demeaned target (they are removed by the
    centring inside :func:`fit_ridge`), so they are dropped as unusable rather
    than silently kept as zero-variance regressors.
    """
    medians = X_train.median(skipna=True)
    usable = [c for c in X_train.columns if np.isfinite(medians.get(c, np.nan))]
    dropped = [c for c in X_train.columns if c not in usable]
    if dropped:
        LOG.info("dropping all-NaN feature columns: %s", dropped)
    return (
        X_train.loc[:, usable].fillna(medians.loc[usable]),
        X_other.loc[:, usable].fillna(medians.loc[usable]),
    )


# --------------------------------------------------------------------------- ridge
def fit_ridge(
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Closed-form ridge on standardised columns with an explicit intercept.

    The penalty is ``alpha`` on the *standardised* coefficient vector:
    ``w = (Z'Z + alpha*I)^{-1} Z'y_centred``. Returns a dict (picklable, JSON-able
    apart from the arrays) so a fitted baseline can be inspected without sklearn.
    """
    train = pd.DataFrame(X_train, dtype="float64")
    yv = np.asarray(y_train, dtype="float64").reshape(-1)
    if len(train) != len(yv):
        raise ValueError(f"X/y length mismatch: {len(train)} vs {len(yv)}")
    if not np.all(np.isfinite(yv)):
        raise ValueError("y_train contains NaN/inf")
    means = train.mean()
    stds = train.std(ddof=0).replace(0.0, np.nan)
    columns = [c for c in train.columns if np.isfinite(stds.get(c, np.nan))]
    z = ((train.loc[:, columns] - means.loc[columns]) / stds.loc[columns]).to_numpy()
    y_mean = float(yv.mean())
    target = yv - y_mean
    gram = z.T @ z
    penalty = float(alpha) * np.eye(gram.shape[0])
    coef = np.linalg.solve(gram + penalty, z.T @ target)
    return {
        "columns": list(columns),
        "means": means.loc[columns].to_numpy(dtype="float64"),
        "stds": stds.loc[columns].to_numpy(dtype="float64"),
        "coef": coef.astype("float64"),
        "intercept": y_mean,
        "alpha": float(alpha),
        "n_train": int(len(train)),
    }


def ridge_predict(model: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Apply a :func:`fit_ridge` model. Missing columns become the fitted mean (= z 0)."""
    cols = model["columns"]
    if len(cols) == 0:
        return np.full(len(X), model["intercept"], dtype="float64")
    block = X.reindex(columns=cols).to_numpy(dtype="float64")
    means = model["means"]
    stds = model["stds"]
    with np.errstate(invalid="ignore"):
        z = (block - means) / stds
    z = np.where(np.isfinite(z), z, 0.0)  # median imputation == z 0 after centring
    return (z @ model["coef"] + model["intercept"]).astype("float64")


def _inner_split(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Last ``INNER_VALID_MONTHS`` months of train become the inner validation set."""
    dates = pd.to_datetime(train["date"])
    cutoff = dates.max() - pd.DateOffset(months=INNER_VALID_MONTHS)
    inner_valid = (dates > cutoff).to_numpy()
    if inner_valid.all() or not inner_valid.any():
        raise BaselinesError(
            f"inner validation slice empty (train ends {dates.max().date()}, "
            f"cutoff {cutoff.date()})"
        )
    return ~inner_valid, inner_valid


def _select_alpha(
    X_train: pd.DataFrame, y_train: np.ndarray, inner_valid: np.ndarray
) -> tuple[float, dict[str, float]]:
    """Pick ``alpha`` by MAE on the inner validation slice of train (never 2022)."""
    inner_fit = ~inner_valid
    X_fit, X_val = _impute(X_train.loc[inner_fit], X_train.loc[inner_valid])
    # Recover the same columns on the fit slice: _impute drops only all-NaN columns,
    # which are all-NaN on every slice, so the column set is shared.
    X_fit = X_fit.loc[:, X_val.columns]
    scores: dict[str, float] = {}
    best_alpha, best_mae = float(RIDGE_ALPHAS[0]), float("inf")
    for alpha in RIDGE_ALPHAS:
        model = fit_ridge(X_fit, y_train[inner_fit], alpha)
        pred = ridge_predict(model, X_val)
        mae = float(np.mean(np.abs(pred - y_train[inner_valid])))
        scores[f"{alpha:g}"] = mae
        if mae < best_mae:
            best_alpha, best_mae = float(alpha), mae
    LOG.info("inner alpha selection: %s -> alpha=%g", scores, best_alpha)
    return best_alpha, scores


# --------------------------------------------------------------------------- baselines
def _fit_linear_baseline(
    train: pd.DataFrame, valid: pd.DataFrame, columns: Sequence[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Shared B1/B2 path: impute on train, tune alpha on inner train, predict valid."""
    cols = [c for c in columns if c in train.columns]
    if not cols:
        raise BaselinesError("no usable feature columns for the linear baseline")
    X_all = _numeric(train, cols)
    X_valid = _numeric(valid, cols)
    y_train = train["y"].to_numpy(dtype="float64")
    fit_idx, val_idx = _inner_split(train)
    alpha, alpha_scores = _select_alpha(X_all, y_train, val_idx)
    X_fit, X_val = _impute(X_all.loc[fit_idx], X_all.loc[val_idx])
    X_fit = X_fit.loc[:, X_val.columns]
    model = fit_ridge(X_fit, y_train[fit_idx], alpha)
    X_full_train, X_valid_imp = _impute(X_all, X_valid)
    X_full_train = X_full_train.loc[:, X_valid_imp.columns]
    model = fit_ridge(X_full_train, y_train, alpha)
    pred = ridge_predict(model, X_valid_imp)
    meta = {
        "alpha": float(model["alpha"]),
        "alpha_scores": alpha_scores,
        "n_features": len(model["columns"]),
        "inner_valid_rows": int(val_idx.sum()),
        "train_rows": int(len(train)),
    }
    return pred, meta


def predict_baselines(
    features: pd.DataFrame, *, train_end: date = date(2021, 12, 31)
) -> dict[str, pd.DataFrame]:
    """Fit every baseline on train, predict valid; returns one frame per code.

    Each frame carries ``ticker, date, y, y_raw, y_pred`` for the **valid** slice.
    ``train_end`` is asserted to precede the validation window so a mis-dated
    matrix cannot smuggle the holdout into fitting.
    """
    work = prepare_matrix(features)
    train = work.loc[work["split"].astype(str) == "train"].reset_index(drop=True)
    valid = work.loc[work["split"].astype(str) == "valid"].reset_index(drop=True)
    if train.empty or valid.empty:
        raise BaselinesError(f"empty slice: train={len(train)} valid={len(valid)}")
    if pd.to_datetime(valid["date"]).min() <= pd.Timestamp(train_end):
        raise BaselinesError(
            f"valid starts {valid['date'].min().date()} but train_end is {train_end}"
        )
    if pd.to_datetime(valid["date"]).min() < pd.Timestamp("2022-01-01"):
        raise BaselinesError("valid slice reaches before 2022")
    if pd.to_datetime(valid["date"]).max() > pd.Timestamp("2022-12-31"):
        raise BaselinesError("valid slice reaches into 2023")

    base = pd.DataFrame(
        {
            "ticker": valid["ticker"].to_numpy(),
            "date": valid["date"].to_numpy(),
            "y": valid["y"].to_numpy(dtype="float64"),
            "y_raw": valid["y_raw"].to_numpy(dtype="float64"),
        }
    )
    out: dict[str, pd.DataFrame] = {}

    # B0 — the zero forecast.
    b0 = base.copy()
    b0["y_pred"] = 0.0
    out["B0"] = b0

    # B1 — market-only ridge.
    pred_b1, meta_b1 = _fit_linear_baseline(train, valid, feature_columns(False))
    b1 = base.copy()
    b1["y_pred"] = pred_b1
    out["B1"] = b1
    LOG.info("B1 fit: %s", meta_b1)

    # B2 — sentiment-only ridge (NaN -> train median; no-news is *not* zero).
    pred_b2, meta_b2 = _fit_linear_baseline(train, valid, SENTIMENT_FEATURES)
    b2 = base.copy()
    b2["y_pred"] = pred_b2
    out["B2"] = b2
    LOG.info("B2 fit: %s", meta_b2)

    # B3 — 5-session momentum, both directions.
    ret5 = pd.to_numeric(valid["ret_5"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(ret5).any():
        raise BaselinesError("ret_5 is all-NaN on the valid slice")
    for code, sign in (("B3_mom", 1.0), ("B3_rev", -1.0)):
        frame = base.copy()
        frame["y_pred"] = sign * np.where(np.isfinite(ret5), ret5, 0.0)
        out[code] = frame

    out["_meta"] = pd.DataFrame(
        [
            {"code": "B1", **meta_b1},
            {"code": "B2", **meta_b2},
        ]
    )
    return out


# --------------------------------------------------------------------------- evaluation
def _evaluate(frame: pd.DataFrame, *, n_deciles: int = 10, n_boot: int = 1000) -> dict:
    y = frame["y"].to_numpy(dtype="float64")
    p = frame["y_pred"].to_numpy(dtype="float64")
    ic_res = ic(y, p, frame["date"])
    point, lo, hi = bootstrap_ic(y, p, frame["date"], n=n_boot)
    reg = regression_metrics(y, p)
    direc = direction_metrics(y, p)
    return {
        "ic": ic_res["ic"],
        "ic_std": ic_res["ic_std"],
        "ic_se": ic_res["ic_se"],
        "ic_t": ic_res["ic_t"],
        "ic_t_nw": ic_res["ic_t_nw"],
        "ic_ci95_lo": ic_res["ic_ci95_lo"],
        "ic_ci95_hi": ic_res["ic_ci95_hi"],
        "ic_boot_point": point,
        "ic_boot_lo": lo,
        "ic_boot_hi": hi,
        "ic_hit": ic_res["ic_hit"],
        "n_dates": ic_res["n_dates"],
        "n_dates_skipped": ic_res["n_dates_skipped"],
        "n_constant_dates": ic_res["n_constant_dates"],
        "n_rows": ic_res["n_rows"],
        "mae": reg["mae"],
        "rmse": reg["rmse"],
        "r2": reg["r2"],
        "r2_vs_zero": reg["r2_vs_zero"],
        "bias": reg["bias"],
        "pearson": reg["pearson"],
        "dir_acc": direc["accuracy"],
        "dir_balanced_acc": direc["balanced_accuracy"],
        "auc": direc["auc"],
        "hit_rate_long": direc["hit_rate_long"],
        "hit_rate_short": direc["hit_rate_short"],
        "dir_coverage": direc["coverage"],
        "deciles": decile_returns(y, p, n=n_deciles, dates=frame["date"]).to_dict(),
        "_ic_series": ic_res["ic_series"],
    }


def compute_baselines(
    frames: dict[str, pd.DataFrame],
    *,
    n_deciles: int = 10,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Evaluate B0–B3 on the valid slice; B0 is the reference for the DM test."""
    meta = frames.get("_meta")
    results: dict[str, Any] = {}
    series: dict[str, np.ndarray] = {}
    losses: dict[str, np.ndarray] = {}
    row_dates: dict[str, np.ndarray] = {}
    dates: np.ndarray | None = None
    ic_index: pd.DatetimeIndex | None = None
    for code in BASELINE_CODES:
        frame = frames[code]
        aligned = frame.loc[np.isfinite(frame["y"]) & np.isfinite(frame["y_pred"])].reset_index(
            drop=True
        )
        evald = _evaluate(aligned, n_deciles=n_deciles, n_boot=n_boot)
        ic_series = evald.pop("_ic_series")
        series[code] = ic_series["ic"].to_numpy(dtype="float64")
        if dates is None:
            ic_index = pd.DatetimeIndex(pd.to_datetime(ic_series["date"]))
            dates = ic_index.to_numpy()
        row_dates[code] = pd.to_datetime(aligned["date"]).to_numpy()
        err = aligned["y_pred"].to_numpy() - aligned["y"].to_numpy()
        losses[code] = np.abs(err)
        results[code] = evald

    # Diebold-Mariano vs the zero forecast: per-date MAE loss differentials.
    date_index = pd.DatetimeIndex(dates)
    dm_block: dict[str, dict[str, float]] = {}
    for code in BASELINE_CODES[1:]:
        per_date = pd.DataFrame(
            {
                "date": row_dates[code],
                "a": losses[code],
                "b": losses["B0"],
            }
        ).groupby("date", sort=True)[["a", "b"]].mean()
        dm, p = diebold_mariano(per_date["a"].to_numpy(), per_date["b"].to_numpy())
        dm_block[code] = {"dm_vs_b0": float(dm), "p_vs_b0": float(p)}
    results["dm"] = dm_block

    # Pairwise IC matrix: which baseline's ranking dominates which.
    codes = list(BASELINE_CODES)
    ic_matrix: dict[str, dict[str, float]] = {}
    # B0 is a constant on every date: its daily IC is exactly 0, so correlation with
    # it is undefined by construction. Report it as such rather than inventing a 0.
    per_date_ic = pd.DataFrame({c: series[c] for c in codes}, index=ic_index)
    for a in codes:
        row: dict[str, float] = {}
        for b in codes:
            if a == b:
                row[b] = 1.0
            elif "B0" in (a, b):
                row[b] = float("nan")
            else:
                row[b] = float(per_date_ic[a].corr(per_date_ic[b]))
        ic_matrix[a] = row
    results["ic_correlation"] = ic_matrix

    results["_meta"] = meta
    results["_series"] = {c: series[c] for c in codes}
    results["_dates"] = date_index
    return results


# --------------------------------------------------------------------------- report
def _clean(value: Any) -> Any:
    """Make a nested dict JSON-safe: numpy scalars -> python, NaN -> None."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return str(value)


def render_report(results: dict[str, Any], *, split: str = "valid_2022") -> dict[str, Any]:
    """Flatten into the plan's JSON schema, keeping the full metric block per code."""
    payload: dict[str, Any] = {"split": split}
    for code in BASELINE_CODES:
        payload[code] = _clean(results[code])
    payload["n_dates"] = int(results["B0"]["n_dates"])
    payload["n_rows"] = int(results["B0"]["n_rows"])
    payload["dm"] = _clean(results.get("dm", {}))
    payload["ic_correlation"] = _clean(results.get("ic_correlation", {}))
    return payload


def write_baselines_report(
    results: dict[str, Any], path: str | Path, *, split: str = "valid_2022"
) -> Path:
    """Write ``reports/baselines.json`` atomically and return its path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = render_report(results, split=split)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(target)
    return target


def series_to_frame(results: dict[str, Any]) -> pd.DataFrame:
    """Daily IC series of every baseline, long format (persisted next to the JSON)."""
    dates_index = results.get("_dates", pd.DatetimeIndex([]))
    frames = []
    for code, series in results.get("_series", {}).items():
        frame = pd.DataFrame({"date": pd.DatetimeIndex(dates_index), "ic": series})
        frame.insert(0, "code", code)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
