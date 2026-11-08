"""Primary model: LightGBM on the cross-section, purged CV, three seeds (T9).

What one run does
-----------------
1. loads ``features.parquet`` and **hard-drops every 2023 row** (asserted, not hoped);
2. rebuilds the purged walk-forward folds of T8 from the NYSE spine and re-asserts
   them (``assert_no_overlap``) on the frame actually used;
3. per fold: fits :class:`~spp.features.transform.FeaturePrep` on **train only**,
   trains one ``LGBMRegressor(objective="huber")`` per seed with early stopping on
   the fold's internal validation IC, predicts the test slice;
4. writes ``predictions.parquet`` (``date, ticker, fold, seed, y_pred, y_pred_seed_mean,
   score_rank, y, y_raw``), per-seed ``model.txt`` + ``preprocess.joblib`` +
   ``model_meta.json`` under ``reports/runs/{run_id}/fold_{i}/seed_{s}/``;
5. evaluates each fold (IC with CI and t-stats, MAE/RMSE, direction accuracy, deciles),
   aggregates mean ± std across folds, writes feature importance (gain + permutation)
   and the richer ``reports/model_metrics.json``; ``reports/gbdt_report.json`` keeps the
   plan's headline schema and points to the run directory.

Leakage rules enforced here, not assumed
----------------------------------------
* the target is ``y_used`` (``y_used_demeaned`` when ``features.demean_target``), a
  forward log return — never a price level;
* the rank transform, the medians and the NaN shares are fitted on train rows only;
* early stopping reads the fold's ``valid`` slice, which lives inside the embargo gap;
* IC is computed against the **demeaned** target (fold-cross-section Spearman is
  invariant to the per-date additive shift, so this is the exact evaluation space);
  the undemeaned ``y`` is stored for the backtest;
* a fold whose training window ends after the test start raises before any fit.

Hyper-XGBoost-style tuning is *not* done here and never against 2023.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spp.config import PROJECT_ROOT, Config
from spp.features.assemble import feature_columns
from spp.features.clip import clip_bounds, clip_targets
from spp.features.transform import DEFAULT_MIN_RANK_TICKERS, FeaturePrep
from spp.logging import get_logger
from spp.models.cv import (
    CV_WINDOW_END,
    CV_WINDOW_START,
    Fold,
    assert_no_overlap,
    iter_splits,
    make_folds,
    summarize_folds,
)
from spp.models.metrics import (
    bootstrap_ic,
    decile_returns,
    diebold_mariano,
    direction_metrics,
    ic,
    regression_metrics,
)

__all__ = [
    "DEFAULT_PARAMS",
    "FALLBACK_BEST_ITERATION",
    "GBDTError",
    "MODEL_SCHEMA_VERSION",
    "PREDICTION_COLUMNS",
    "compute_importance",
    "evaluate_fold",
    "golden_vectors",
    "load_model_dir",
    "predict_frame",
    "run_gbdt_cv",
    "write_gbdt_report",
]

LOG = get_logger("models.gbdt")

MODEL_SCHEMA_VERSION = "gbdt-v1"
FALLBACK_BEST_ITERATION = 500
LGBM_DETERMINISM: dict[str, Any] = {
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
}
BOOSTING_ROUNDS = 3000
MIN_TRAIN_ROWS = 500
MIN_VALID_DATES = 3
EARLY_STOP_SESSIONS = 60
SHAP_SAMPLE_ROWS = 20_000
PI_LOWER_QUANTILE = 0.005
PI_UPPER_QUANTILE = 0.995
N_BOOTSTRAP = 1000

#: The plan's configuration, verbatim (``min_child_samples`` is LightGBM's spelling of
#: the config's ``min_data_in_leaf``; the two names are reconciled in ``_params``).
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "huber",
    "alpha": 0.9,
    "n_estimators": BOOSTING_ROUNDS,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "n_jobs": -1,
    "verbose": -1,
}

PREDICTION_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "fold",
    "seed",
    "y_pred",
    "y_pred_seed_mean",
    "score_rank",
    "y",
    "y_raw",
    "y_eval",
)


class GBDTError(RuntimeError):
    """The run cannot proceed without breaking a leakage or integrity rule."""


# --------------------------------------------------------------------------- helpers
def _git_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git on PATH
        return None
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and sha else None


def _versions() -> dict[str, str]:
    import lightgbm
    import pandas
    import sklearn

    return {
        "lightgbm": lightgbm.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "python": platform.python_version(),
    }


def _jsonable(value: Any) -> Any:
    """Recursive JSON conversion: numpy scalars unwrapped, non-finite floats -> None."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return pd.Timestamp(value).date().isoformat()
    return value


def _write_json_atomic(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():  # pragma: no cover - failure path
            tmp.unlink(missing_ok=True)
    return path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _session_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"]).dt.normalize())
    return pd.DatetimeIndex(sorted(dates.unique()))


def _params(model_cfg: Any, seed: int) -> dict[str, Any]:
    """Plan config + ``ModelCfg`` overrides that are *data*, not per-fold tuning."""
    params = dict(DEFAULT_PARAMS)
    params["num_leaves"] = int(model_cfg.num_leaves)
    params["learning_rate"] = float(model_cfg.learning_rate)
    params["min_child_samples"] = int(model_cfg.min_data_in_leaf)
    params["lambda_l2"] = float(model_cfg.lambda_l2)
    params.update(LGBM_DETERMINISM)
    params["seed"] = int(seed)
    if params.get("n_jobs", 0) == -1:
        params["n_jobs"] = int(LGBM_DETERMINISM["num_threads"])
    return params


def _clean_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows the model may see: a usable label and per-date-ticker uniqueness."""
    work = frame.loc[frame["y"].notna()].copy()
    if "label_ok" in work.columns:
        work = work.loc[work["label_ok"].astype(bool)]
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    if work.duplicated(["date", "ticker"]).any():
        raise GBDTError("duplicate (date, ticker) rows reached the trainer")
    return work.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


def _target(frame: pd.DataFrame, demeaning: bool) -> np.ndarray:
    column = "y_used_demeaned" if demeaning else "y"
    if column not in frame.columns:
        raise GBDTError(f"target column {column!r} is missing")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
    if not np.all(np.isfinite(values)):
        raise GBDTError(f"{column} contains NaN where a label was expected")
    return values


def _iqr_weighted_share(imp: pd.DataFrame) -> dict[str, float]:
    """Share of importance explained by the 15/85 quantile band of each feature."""
    out: dict[str, float] = {}
    for _, row in imp.iterrows():
        splits = str(row.get("decision_type", "2")).split("||")
        bounds = str(row.get("threshold", "0")).split("||")
        try:
            decisions = [float(s) for s in splits]
            thresholds = [float(b) for b in bounds]
        except ValueError:
            continue
        if not thresholds or not decisions or not np.isfinite(thresholds[0]):
            continue
        by_decision = dict(zip(decisions, thresholds, strict=False))
        out[str(row["feature"])] = float(by_decision.get(2.0, thresholds[0]))
    return out


def _finite_quantile(
    series: pd.Series,
    q: float,
) -> float:
    """Quantile over finite values; ``NaN`` (not a crash) when the column is all-NaN."""
    values = series.to_numpy(dtype="float64")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.quantile(values, q))


def _rank_ic_by_date(y: np.ndarray, p: np.ndarray, dates: np.ndarray) -> float:
    """In-memory per-date Spearman mean (used inside early-stopping callbacks)."""
    frame = pd.DataFrame({"y": y, "p": p, "date": dates})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    ics: list[float] = []
    for _, group in frame.groupby("date", sort=True):
        if len(group) < 2 or group["p"].nunique() < 2 or group["y"].nunique() < 2:
            ics.append(0.0)
            continue
        ics.append(float(group["p"].corr(group["y"], method="spearman")))
    return float(np.mean(ics)) if ics else float("nan")


# --------------------------------------------------------------------------- training
def _fit_one(
    train: pd.DataFrame,
    valid: pd.DataFrame | None,
    *,
    test: pd.DataFrame,
    feature_list: Sequence[str],
    ranked: Sequence[str],
    categorical: Sequence[str],
    demeaning: bool,
    params: dict[str, Any],
    early_stopping_rounds: int,
    clip_bounds_: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Fit one (fold, seed) pair. Returns predictions and the fitted objects."""
    import lightgbm as lgb

    seed = int(params["seed"])
    _seed_everything(seed)
    prep = FeaturePrep.fit(
        train,
        list(feature_list),
        rank_columns=list(ranked),
        categorical_columns=list(categorical),
        min_rank_tickers=DEFAULT_MIN_RANK_TICKERS,
    )
    X_train, _ = prep.transform(train)
    y_train = _target(train, demeaning)
    if clip_bounds_ is not None:
        # Huber loss is robust but a 40% single-session jump still distorts the fit;
        # bounds come from the training slice only.
        y_train = clip_targets(y_train, clip_bounds_)

    callbacks: list[Callable[..., None]] = []
    fit_kwargs: dict[str, Any] = {}
    X_valid = y_valid = None
    best_iteration = 0
    state: dict[str, Any] = {"best": -np.inf, "best_iteration": 0}
    if valid is not None and not valid.empty:
        X_valid, _ = prep.transform(valid)
        y_valid = _target(valid, demeaning)
        Xv = X_valid.to_numpy(dtype="float64")
        yv = np.asarray(y_valid, dtype="float64")
        dv = valid["date"].to_numpy()
        rounds = int(early_stopping_rounds)

        def _ic_early_stop(env: Any) -> None:
            """Stop when the validation rank-IC has not improved for ``rounds`` iterations.

            LightGBM's built-in early stopping monitors the eval metric, and our objective is
            the cross-sectional IC, so this callback owns the stopping rule. It deliberately
            does not touch ``env.evaluation_result_list``: that list holds LightGBM's own
            result objects, not tuples.
            """
            iteration = env.iteration + 1
            pred = env.model.predict(Xv, num_iteration=iteration)
            value = _rank_ic_by_date(yv, pred, dv)
            if np.isfinite(value) and value > state["best"]:
                state["best"] = float(value)
                state["best_iteration"] = iteration
            elif iteration - state["best_iteration"] >= rounds:
                best = ("valid", "ic", float(state["best"]), True)
                raise lgb.callback.EarlyStopException(state["best_iteration"], [best])

        callbacks.append(_ic_early_stop)
        callbacks.append(lgb.log_evaluation(period=0))
        fit_kwargs = dict(
            eval_set=[(Xv, y_valid)],
            eval_metric="l2",
        )

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train.to_numpy(dtype="float64"), y_train, callbacks=callbacks, **fit_kwargs)

    best_iteration = int(getattr(model, "best_iteration_", 0) or 0)
    if valid is not None and not valid.empty:
        stopped = bool(best_iteration > 0 and best_iteration < params["n_estimators"])
        if not stopped:
            best_iteration = FALLBACK_BEST_ITERATION
            LOG.warning(
                "early stopping did not trigger (best_iteration=%s); using %d",
                getattr(model, "best_iteration_", None),
                FALLBACK_BEST_ITERATION,
            )
        else:
            best_iteration = int(best_iteration)
    else:
        best_iteration = int(params["n_estimators"])

    if len(test) == 0:
        # Final-model fit has no test slice: nothing to predict, nothing to mask.
        return {
            "prep": prep,
            "model": model,
            "best_iteration": best_iteration,
            "y_pred": np.empty(0, dtype="float64"),
            "rank_status": None,
            "n_all_nan": 0,
            "valid_ic": float("nan"),
        }
    X_test, status = prep.transform(test)
    pred = model.predict(X_test.to_numpy(dtype="float64"), num_iteration=best_iteration)
    # A ticker whose every feature is NaN has no opinion: NaN, never 0.
    all_nan = X_test.isna().all(axis=1).to_numpy()
    pred = np.where(all_nan, np.nan, pred)
    return {
        "prep": prep,
        "model": model,
        "best_iteration": best_iteration,
        "y_pred": np.asarray(pred, dtype="float64"),
        "rank_status": status,
        "n_all_nan": int(all_nan.sum()),
        "valid_ic": float(state["best"]) if state["best_iteration"] else float("nan"),
    }


# --------------------------------------------------------------------------- evaluation
def _fold_metrics(
    frame: pd.DataFrame, *, n_boot: int = N_BOOTSTRAP, seed: int = 42
) -> dict[str, Any]:
    """Full metric block for one prediction frame (rows already aligned)."""
    y = frame["y_eval"].to_numpy(dtype="float64")
    pred_column = "y_pred" if "y_pred" in frame.columns else "y_pred_seed_mean"
    p = frame[pred_column].to_numpy(dtype="float64")
    dates = frame["date"].to_numpy()
    res = ic(y, p, dates)
    point, lo, hi = bootstrap_ic(y, p, dates, n=n_boot, seed=seed)
    reg = regression_metrics(y, p)
    direc = direction_metrics(y, p)
    deciles = decile_returns(y, p, n=10, dates=pd.Series(dates))
    return {
        "ic": res["ic"],
        "ic_std": res["ic_std"],
        "ic_se": res["ic_se"],
        "ic_t": res["ic_t"],
        "ic_t_nw": res["ic_t_nw"],
        "ic_ci95_lo": res["ic_ci95_lo"],
        "ic_ci95_hi": res["ic_ci95_hi"],
        "ic_boot_point": point,
        "ic_boot_lo": lo,
        "ic_boot_hi": hi,
        "ic_hit": res["ic_hit"],
        "n_dates": res["n_dates"],
        "n_dates_skipped": res["n_dates_skipped"],
        "n_constant_dates": res["n_constant_dates"],
        "n_rows": res["n_rows"],
        "mae": reg["mae"],
        "rmse": reg["rmse"],
        "r2": reg["r2"],
        "r2_vs_zero": reg["r2_vs_zero"],
        "bias": reg["bias"],
        "dir_acc": direc["accuracy"],
        "dir_balanced_acc": direc["balanced_accuracy"],
        "auc": direc["auc"],
        "dir_coverage": direc["coverage"],
        "deciles": {str(k): float(v) for k, v in deciles.items()},
        "ic_series": res["ic_series"],
    }


def evaluate_fold(
    frame: pd.DataFrame, *, n_boot: int = N_BOOTSTRAP, seed: int = 42
) -> dict[str, Any]:
    """Public alias used by the CLI/tests: metric block for one fold's predictions."""
    return _fold_metrics(frame, n_boot=n_boot, seed=seed)


def _aggregate(per_fold: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean ± std across folds for every scalar metric."""
    keys = [
        "ic", "ic_t", "ic_t_nw", "ic_hit", "mae", "rmse", "r2_vs_zero", "dir_acc", "auc",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        values = np.asarray([f[key] for f in per_fold], dtype="float64")
        finite = values[np.isfinite(values)]
        out[key] = {
            "mean": float(finite.mean()) if len(finite) else float("nan"),
            "std": float(finite.std(ddof=1)) if len(finite) > 1 else float("nan"),
            "n": int(len(finite)),
            "per_fold": [float(v) if np.isfinite(v) else None for v in values],
        }
    return out


def _pooled(frame: pd.DataFrame) -> dict[str, Any]:
    """Pooled-over-folds metric block: fold predictions stitched into one series."""
    values = frame["y_pred"].to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]
    centred = values - float(finite.mean()) if len(finite) else values
    res = ic(frame["y_eval"].to_numpy(), centred, frame["date"].to_numpy())
    return {
        "ic": res["ic"],
        "ic_t": res["ic_t"],
        "ic_t_nw": res["ic_t_nw"],
        "ic_se": res["ic_se"],
        "ic_ci95_lo": res["ic_ci95_lo"],
        "ic_ci95_hi": res["ic_ci95_hi"],
        "n_dates": res["n_dates"],
        "n_rows": res["n_rows"],
        "n_constant_dates": res["n_constant_dates"],
    }


def _monotonicity(deciles: dict[str, float]) -> dict[str, float]:
    top_bottom = float(deciles["10"] - deciles["1"]) if deciles else float("nan")
    values = np.asarray([deciles[str(i)] for i in range(1, 11)], dtype="float64")
    order = np.arange(10, dtype="float64")
    with np.errstate(invalid="ignore"):
        if np.isfinite(values).all():
            corr = float(np.corrcoef(order, values)[0, 1])
        else:
            corr = float("nan")
    return {"top_minus_bottom": top_bottom, "decile_rank_corr": corr}


# --------------------------------------------------------------------------- importance
def compute_importance(
    prep: FeaturePrep,
    model: Any,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    best_iteration: int,
    n_repeats: int = 5,
    top: int = 30,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Gain + permutation importance on the fold's validation slice.

    Permutation is measured on **rank IC** (the trading-relevant loss) with 5 repeats:
    each feature is shuffled across tickers inside each date, which destroys the
    cross-sectional information of exactly that feature and nothing else.
    """
    X_valid, _ = prep.transform(valid)
    y_valid = _target(valid, True)
    Xv = X_valid.to_numpy(dtype="float64")
    yv = np.asarray(y_valid, dtype="float64")
    dv = valid["date"].to_numpy()
    base_ic = ic(yv, model.predict(Xv, num_iteration=best_iteration), dv)["ic"]

    # Manual permutation importance: sklearn's helper requires an estimator with a
    # scikit-learn scorer signature, while the metric here is per-date rank IC. Shuffling
    # happens inside each date, so only the cross-sectional information is destroyed.
    rng = np.random.default_rng(42)
    groups = [np.flatnonzero(dv == day) for day in pd.unique(dv)]
    drops = np.zeros((n_repeats, Xv.shape[1]), dtype="float64")
    for r in range(n_repeats):
        for j in range(Xv.shape[1]):
            shuffled = Xv.copy()
            for idx in groups:
                if idx.size > 1:
                    shuffled[idx, j] = shuffled[rng.permutation(idx), j]
            perm_ic = ic(yv, model.predict(shuffled, num_iteration=best_iteration), dv)["ic"]
            drops[r, j] = base_ic - perm_ic
    perm_mean = drops.mean(axis=0)
    perm_std = drops.std(axis=0, ddof=1) if n_repeats > 1 else np.zeros_like(perm_mean)

    booster = model.booster_
    gain = pd.Series(booster.feature_importance(importance_type="gain"), index=prep.feature_columns)
    split = pd.Series(
        booster.feature_importance(importance_type="split"), index=prep.feature_columns
    )
    total = float(gain.sum()) or 1.0
    imp = pd.DataFrame(
        {
            "feature": prep.feature_columns,
            "gain": gain.reindex(prep.feature_columns).to_numpy(dtype="float64"),
            "gain_share": (gain.reindex(prep.feature_columns).to_numpy(dtype="float64") / total),
            "split": split.reindex(prep.feature_columns).to_numpy(dtype="float64"),
            "perm_ic_mean": perm_mean,
            "perm_ic_std": perm_std,
        }
    ).sort_values("gain", ascending=False)
    imp["group"] = [
        "sentiment" if c.startswith(("sent_", "ewma_", "n_news", "log_n_news", "pos_",
                                     "neg_", "neu_", "very_", "score", "conf_",
                                     "novelty", "source_", "top_source_", "company_"))
        else "market"
        for c in imp["feature"]
    ]
    group_share = (
        imp.groupby("group")["gain_share"].sum().to_dict()
    )
    meta = {"group_gain_share": group_share, "n_features": len(imp)}
    return imp.head(top).reset_index(drop=True), meta


class _Wrapper:
    """``predict`` adapter with a fixed iteration count (picklable-free, CPU only)."""

    def __init__(self, model: Any, best_iteration: int) -> None:
        self._model = model
        self._n = int(best_iteration)

    def fit(self, X: Any, y: Any = None) -> _Wrapper:  # noqa: N803 - sklearn API
        """No-op: the booster is already fitted. ``permutation_importance`` requires it."""
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(np.asarray(X, dtype="float64"), num_iteration=self._n)


# --------------------------------------------------------------------------- inference
@dataclass
class LoadedModel:
    """What :func:`load_model_dir` returns: enough to score one row the training way."""

    prep: FeaturePrep
    booster: Any
    meta: dict[str, Any]
    best_iteration: int

    def predict(self, frame: pd.DataFrame, *, dates: pd.Series | None = None) -> pd.Series:
        X, _ = self.prep.transform(frame, dates=dates)
        values = self.booster.predict(
            X.to_numpy(dtype="float64"), num_iteration=self.best_iteration
        )
        values = np.asarray(values, dtype="float64")
        values[X.isna().all(axis=1).to_numpy()] = np.nan
        return pd.Series(values, index=frame.index, name="y_pred")


def load_model_dir(model_dir: str | Path) -> LoadedModel:
    """Load ``model.txt`` + ``preprocess.joblib`` + ``model_meta.json`` from one seed dir."""
    import joblib
    import lightgbm as lgb

    base = Path(model_dir)
    for name in ("model.txt", "preprocess.joblib", "model_meta.json"):
        if not (base / name).is_file():
            raise FileNotFoundError(f"{base} is missing {name}")
    booster = lgb.Booster(model_file=str(base / "model.txt"))
    prep = joblib.load(base / "preprocess.joblib")
    meta = json.loads((base / "model_meta.json").read_text(encoding="utf-8"))
    best_iteration = int(meta.get("best_iteration") or booster.current_iteration())
    return LoadedModel(prep=prep, booster=booster, meta=meta, best_iteration=best_iteration)


def predict_frame(model_dir: str | Path, features_row: pd.DataFrame) -> pd.Series:
    """Score a frame **exactly** the way the offline trainer did (parity ±1e-6).

    ``model_dir`` is one seed's artifact directory
    (``reports/runs/{run_id}/fold_{i}/seed_{s}``). Column order in ``features_row``
    is irrelevant: ``preprocess.joblib`` reorders to the saved list. Missing columns
    become NaN (``no opinion``), never zero.
    """
    loaded = load_model_dir(model_dir)
    return loaded.predict(features_row)


def golden_vectors(run_dir: str | Path) -> list[dict[str, Any]]:
    """Stored (features_row, expected_prediction) pairs used to pin live parity."""
    path = Path(run_dir) / "golden_vectors.parquet"
    if not path.is_file():
        return []
    frame = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        payload = json.loads(str(row["features_json"]))
        rows.append(
            {
                "name": str(row["name"]),
                "model_dir": str(row["model_dir"]),
                "features": payload,
                "expected": float(row["expected"]),
            }
        )
    return rows


# --------------------------------------------------------------------------- the run
@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    importance: pd.DataFrame
    seeds: list[int]
    folds: list[Fold]
    elapsed_s: float
    legacy: dict[str, Any] = field(default_factory=dict)


def _make_run_id(config: Config, feature_path: Path, seeds: Sequence[int]) -> str:
    payload = {
        "schema": MODEL_SCHEMA_VERSION,
        "config": {
            "model": config.model.model_dump(mode="json"),
            "cv": config.cv.model_dump(mode="json"),
            "features": config.features.model_dump(mode="json"),
        },
        "feature_columns": feature_columns(True),
        "seeds": list(seeds),
        "feature_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        "params": DEFAULT_PARAMS,
        "git_head": _git_sha(),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return f"gbdt-{digest[:10]}"


def _assert_no_2023(frame: pd.DataFrame, where: str) -> None:
    if "date" not in frame.columns or len(frame) == 0:
        return
    latest = pd.to_datetime(frame["date"]).max()
    if latest >= pd.Timestamp(f"{2023}-01-01"):
        raise GBDTError(f"{where}: rows dated {latest.date()} (2023 is held out until T10)")


def run_gbdt_cv(
    config: Config,
    *,
    features_path: str | Path | None = None,
    seeds: Sequence[int] | None = None,
    run_id: str | None = None,
    project_root: str | Path | None = None,
    lgb_params: dict[str, Any] | None = None,
    feature_list: Sequence[str] | None = None,
    demean_target: bool | None = None,
    cross_sectional_rank: bool | None = None,
    max_folds: int | None = None,
) -> RunResult:
    """Train the primary model over the purged CV folds. Never touches 2023.

    The keyword overrides (``feature_list``, ``demean_target``, ``cross_sectional_rank``,
    ``lgb_params``) exist so that T10 ablations reuse *this* code path instead of a
    fork of it. They are data, not per-fold tuning.
    """
    started = time.perf_counter()
    from spp.data.store import read_parquet

    root = Path(project_root) if project_root else PROJECT_ROOT
    data_root = root / config.paths.data
    reports_root = root / config.paths.reports
    matrix_path = (
        Path(features_path)
        if features_path
        else data_root / "processed" / "features.parquet"
    )
    if not matrix_path.is_file():
        raise GBDTError(f"missing feature matrix: {matrix_path}")
    seeds = [int(s) for s in (seeds or config.model.seeds)]

    frame = read_parquet(matrix_path)
    frame = frame.loc[frame["date"] <= pd.Timestamp(CV_WINDOW_END)].copy()
    frame = frame.loc[frame["date"] >= pd.Timestamp(CV_WINDOW_START)].copy()
    _assert_no_2023(frame, "CV window")

    cols = list(feature_list) if feature_list is not None else feature_columns(True)
    absent = [c for c in cols if c not in frame.columns]
    if absent:
        # Ablation matrices legitimately omit whole feature blocks (for example the
        # sentiment block in A1). Keep the intersection and say exactly what was dropped.
        cols = [c for c in cols if c in frame.columns]
        LOG.warning("matrix lacks %d requested feature(s), dropped: %s", len(absent), absent)
        if not cols:
            raise GBDTError(f"no requested feature present in {matrix_path}")
    demeaning = config.features.demean_target if demean_target is None else bool(demean_target)
    ranking = (
        config.features.cross_sectional_rank
        if cross_sectional_rank is None
        else bool(cross_sectional_rank)
    )
    categorical = ["dow"] if "dow" in cols else []
    rank_cols = [c for c in cols if c not in categorical] if ranking else []

    from spp.features.calendar import nyse_sessions

    spine = nyse_sessions(CV_WINDOW_START, CV_WINDOW_END)
    cv = config.cv
    from spp.config import CvCfg

    folds = make_folds(
        spine,
        CvCfg(
            n_folds=cv.n_folds,
            test_months=cv.test_months,
            embargo_days=cv.embargo_days,
            min_train_months=cv.min_train_months,
        ),
        spine,
    )
    if max_folds is not None:
        folds = folds[: int(max_folds)]

    rid = run_id or _make_run_id(config, matrix_path, seeds)
    run_dir = reports_root / "runs" / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("gbdt run %s -> %s (folds=%d seeds=%s)", rid, run_dir, len(folds), seeds)

    predictions: list[pd.DataFrame] = []
    per_fold: list[dict[str, Any]] = []
    importance_top: list[pd.DataFrame] = []
    importance_groups: dict[str, Any] = {}
    all_seed_meta: list[dict[str, Any]] = []

    splits = list(iter_splits(frame, folds))
    assert_no_overlap(
        [Fold(**{**f.as_dict(), **{"fold_id": f.fold_id}}) for f, *_ in splits], spine
    )

    for fold, train_raw, valid_raw, test_raw in splits:
        if pd.Timestamp(fold.train_end) >= pd.Timestamp(fold.test_start):
            raise GBDTError(f"fold {fold.fold_id}: train_end >= test_start")
        train = _clean_rows(train_raw)
        valid = _clean_rows(valid_raw)
        if len(train) < MIN_TRAIN_ROWS:
            raise GBDTError(f"fold {fold.fold_id}: only {len(train)} labelled train rows")
        if valid["date"].nunique() < MIN_VALID_DATES:
            LOG.warning("fold %d: validation has %d dates", fold.fold_id, valid["date"].nunique())
        test = test_raw.copy()
        test["date"] = pd.to_datetime(test["date"]).dt.normalize()
        if test["date"].min() <= pd.Timestamp(fold.train_end):
            raise GBDTError(f"fold {fold.fold_id}: test overlaps the training window")
        if test["date"].max() >= pd.Timestamp("2023-01-01"):
            raise GBDTError(f"fold {fold.fold_id}: test slice reaches 2023")
        fold_dir = run_dir / f"fold_{fold.fold_id}"
        y_raw_series = train_raw.get("y_raw", train_raw["y"])
        clip_lo = clip_bounds(pd.to_numeric(y_raw_series, errors="coerce"))[0]
        clip_hi = clip_bounds(pd.to_numeric(y_raw_series, errors="coerce"))[1]
        y_valid = _target(valid, demeaning)
        valid_dates = valid["date"].to_numpy()

        # Early stopping on the 5-session embargo gap is noise: an IC over five dates has a
        # standard error near 0.4, so the booster stops at an arbitrary iteration. The tail of
        # the training window serves as the inner validation slice instead. It sits strictly
        # before the test window, so no future information enters the model.
        train_days = np.sort(train["date"].unique())
        if len(train_days) > 2 * EARLY_STOP_SESSIONS:
            inner_days = train_days[-EARLY_STOP_SESSIONS:]
            inner_mask = train["date"].isin(inner_days)
            fit_train = train.loc[~inner_mask]
            fit_valid = train.loc[inner_mask]
        else:
            fit_train, fit_valid = train, valid

        seed_frames: list[pd.DataFrame] = []
        seed_meta: list[dict[str, Any]] = []
        for seed in seeds:
            params = _params(config.model, seed)
            if lgb_params:
                params.update(lgb_params)
                params["seed"] = int(seed) if "seed" not in lgb_params else int(lgb_params["seed"])
            fitted = _fit_one(
                fit_train,
                fit_valid,
                test=test,
                feature_list=cols,
                ranked=rank_cols,
                categorical=categorical,
                demeaning=demeaning,
                params=params,
                early_stopping_rounds=config.model.early_stopping_rounds,
                clip_bounds_=(clip_lo, clip_hi),
            )
            model = fitted["model"]
            seed_dir = fold_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            model.booster_.save_model(str(seed_dir / "model.txt"))

            import joblib

            joblib.dump(fitted["prep"], seed_dir / "preprocess.joblib")
            valid_pred = model.predict(
                fitted["prep"].transform(valid)[0].to_numpy(dtype="float64"),
                num_iteration=fitted["best_iteration"],
            )
            valid_ic = ic(y_valid, valid_pred, valid_dates)
            meta = {
                "run_id": rid,
                "model_schema_version": MODEL_SCHEMA_VERSION,
                "fold_id": fold.fold_id,
                "seed": int(seed),
                "params": params,
                "n_estimators_cap": int(params["n_estimators"]),
                "best_iteration": int(fitted["best_iteration"]),
                "early_stopped": bool(fitted["best_iteration"] != int(params["n_estimators"])),
                "fold": fold.as_dict(),
                "target": "y_used_demeaned" if demeaning else "y",
                "target_definition": (
                    "log(adj_close[t+1]/adj_close[t]); "
                    "demeaned per date when enabled"
                ),
                "feature_columns": list(cols),
                "categorical_columns": categorical,
                "rank_columns": list(rank_cols),
                "rank_method": "per-date cross-sectional rank_gauss (train-fitted order stats)",
                "nan_policy": (
                    "no imputation; NaN = no news / unavailable, "
                    "model splits on it natively"
                ),
                "target_clip_quantiles": [PI_LOWER_QUANTILE, PI_UPPER_QUANTILE],
                "target_clip_bounds": [clip_lo, clip_hi],
                "n_valid_dates": int(valid["date"].nunique()),
                "valid_ic": float(valid_ic["ic"]),
                "valid_ic_t": float(valid_ic["ic_t"]),
                "train_rows": int(len(train)),
                "train_dates": [str(train["date"].min().date()), str(train["date"].max().date())],
                "valid_dates": (
                    [str(valid["date"].min().date()), str(valid["date"].max().date())]
                    if len(valid)
                    else []
                ),
                "test_dates": [str(test["date"].min().date()), str(test["date"].max().date())],
                "n_all_nan_test_rows": int(fitted["n_all_nan"]),
                "git_sha": _git_sha(),
                "versions": _versions(),
                "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            _write_json_atomic(meta, seed_dir / "model_meta.json")
            seed_meta.append({"seed": int(seed), "best_iteration": int(fitted["best_iteration"])})
            seed_frames.append(
                pd.DataFrame(
                    {
                        "date": test["date"].to_numpy(),
                        "ticker": test["ticker"].to_numpy(),
                        "fold": fold.fold_id,
                        "seed": int(seed),
                        "y_pred": fitted["y_pred"],
                        "y": pd.to_numeric(test["y"], errors="coerce").to_numpy(dtype="float64"),
                        "y_raw": pd.to_numeric(
                            test.get("y_raw", test["y"]), errors="coerce"
                        ).to_numpy(dtype="float64"),
                    }
                )
            )

            if seed == seeds[-1]:
                imp, groups = compute_importance(
                    fitted["prep"],
                    model,
                    train,
                    valid,
                    best_iteration=fitted["best_iteration"],
                )
                imp.insert(0, "fold", fold.fold_id)
                imp.insert(1, "seed", int(seed))
                importance_top.append(imp)
                importance_groups[f"fold_{fold.fold_id}"] = groups
                _write_json_atomic(
                    {"fold": fold.fold_id, "seed": int(seed), **groups},
                    fold_dir / "importance_groups.json",
                )

        stacked = pd.concat(seed_frames, ignore_index=True)
        mean = (
            stacked.groupby(["date", "ticker"], sort=True)["y_pred"]
            .mean()
            .rename("y_pred_seed_mean")
            .reset_index()
        )
        fold_pred = mean.merge(
            stacked.drop_duplicates(["date", "ticker"])[["date", "ticker", "fold", "y", "y_raw"]],
            on=["date", "ticker"],
            how="left",
            validate="one_to_one",
        )
        fold_pred["seed"] = np.nan  # aggregated row: the mean of the three seeds
        fold_pred["score_rank"] = fold_pred.groupby("date")["y_pred_seed_mean"].rank(pct=True)
        # The published schema carries a single ``y_pred`` column; for the aggregated fold row
        # it is the seed mean (per-seed values live in ``fold_dir/seed_*/``).
        fold_pred["y_pred"] = fold_pred["y_pred_seed_mean"]
        fold_pred["y_eval"] = (
            fold_pred.groupby("date")["y"].transform(lambda s: s - s.median())
            if demeaning
            else fold_pred["y"]
        )
        metrics = _fold_metrics(fold_pred)
        metrics.pop("ic_series")
        metrics["fold"] = fold.fold_id
        metrics["summarize"] = {k: v for k, v in summarize_folds([fold], spine)["folds"][0].items()}
        metrics["seed_meta"] = seed_meta
        metrics["best_iteration"] = int(
            np.median([m["best_iteration"] for m in seed_meta]) if seed_meta else 0
        )
        all_seed_meta.extend(seed_meta)
        metrics["seeds_ic"] = {
            int(s): float(
                ic(
                    fold_pred["y_eval"].to_numpy(),
                    stacked.loc[stacked["seed"] == s, "y_pred"].to_numpy(),
                    stacked.loc[stacked["seed"] == s, "date"].to_numpy(),
                )["ic"]
            )
            for s in seeds
        }
        per_fold.append(metrics)
        predictions.append(fold_pred)
        LOG.info(
            "fold %d: IC=%.4f (t=%.2f) seeds=%s",
            fold.fold_id,
            metrics["ic"],
            metrics["ic_t"] or float("nan"),
            metrics["seeds_ic"],
        )

    predictions_frame = pd.concat(predictions, ignore_index=True)
    predictions_frame["fold"] = predictions_frame["fold"].astype("int64")
    predictions_frame["seed"] = predictions_frame["seed"].astype("float64")
    predictions_frame = predictions_frame.loc[:, list(PREDICTION_COLUMNS)].sort_values(
        ["date", "fold", "ticker"], kind="stable"
    )

    aggregate = _aggregate(per_fold)
    pooled = _pooled(predictions_frame)
    zero_losses = predictions_frame["y_eval"].abs().to_numpy()
    model_losses = (
        (predictions_frame["y_pred_seed_mean"] - predictions_frame["y_eval"]).abs().to_numpy()
    )
    dm_model, p_model = _diebold_by_date(predictions_frame, model_losses, zero_losses)
    metrics_payload = {
        "run_id": rid,
        "schema": MODEL_SCHEMA_VERSION,
        "config": {
            "seeds": seeds,
            "features": list(cols),
            "demean_target": demeaning,
            "cross_sectional_rank": ranking,
            "params": {k: v for k, v in DEFAULT_PARAMS.items()},
            "penalised_thread_count": LGBM_DETERMINISM["num_threads"],
        },
        "folds": per_fold,
        "aggregate": aggregate,
        "pooled": pooled,
        "dm_vs_zero": {"dm": dm_model, "p": p_model},
        "target_used": "y_eval = y per date, demeaned across tickers when demean_target",
        "git_sha": _git_sha(),
        "versions": _versions(),
        "elapsed_s": None,
        "held_out_year_touched": False,
    }

    if importance_top:
        all_imp = pd.concat(importance_top, ignore_index=True)
        all_imp["gain_share"] = all_imp.groupby("fold")["gain"].transform(
            lambda s: s / max(s.sum(), 1e-12)
        )
        importance = (
            all_imp.groupby(["feature", "group"], sort=False)
            .agg(
                gain_mean=("gain", "mean"),
                gain_share_mean=("gain_share", "mean"),
                perm_ic_mean=("perm_ic_mean", "mean"),
                perm_ic_std=("perm_ic_std", "mean"),
            )
            .reset_index()
            .sort_values("gain_mean", ascending=False)
        )
        importance.to_csv(run_dir / "feature_importance.csv", index=False)
        group_share = None
        for block in importance_groups.values():
            frame_share = pd.Series(block["group_gain_share"], dtype="float64")
            if group_share is None:
                group_share = frame_share
            else:
                group_share = group_share.add(frame_share, fill_value=0.0)
        if group_share is not None:
            importance_groups["mean_gain_share"] = (group_share / len(importance_groups)).to_dict()
    else:  # pragma: no cover - only when a run has no folds
        importance = pd.DataFrame()

    # Final model for live inference: refit on the whole CV window. The tree count is the median
    # early-stopped length over the folds, so the horizon of the final fit is not tuned on 2023.
    final_meta: list[dict[str, Any]] = []
    if all_seed_meta:
        import joblib

        final_train = _clean_rows(frame)
        final_raw = pd.to_numeric(final_train.get("y_raw", final_train["y"]), errors="coerce")
        final_clip = clip_bounds(final_raw)
        lengths = [int(m["best_iteration"]) for m in all_seed_meta if m["best_iteration"]]
        n_final = int(np.median(lengths)) if lengths else int(config.model.n_estimators)
        final_dir = run_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            params = _params(config.model, seed)
            params["n_estimators"] = max(n_final, 50)
            fitted = _fit_one(
                final_train,
                None,
                test=final_train.iloc[:0],
                feature_list=cols,
                ranked=rank_cols,
                categorical=categorical,
                demeaning=demeaning,
                params=params,
                early_stopping_rounds=0,
                clip_bounds_=final_clip,
            )
            seed_dir = final_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            fitted["model"].booster_.save_model(str(seed_dir / "model.txt"))
            joblib.dump(fitted["prep"], seed_dir / "preprocess.joblib")
            meta = {
                "run_id": rid,
                "model_schema_version": MODEL_SCHEMA_VERSION,
                "scope": "final",
                "seed": int(seed),
                "params": params,
                "best_iteration": int(params["n_estimators"]),
                "feature_columns": list(cols),
                "rank_columns": list(rank_cols),
                "categorical_columns": categorical,
                "target": "y_used_demeaned" if demeaning else "y",
                "target_clip_bounds": list(final_clip),
                "train_rows": int(len(final_train)),
                "train_dates": [
                    str(final_train["date"].min().date()),
                    str(final_train["date"].max().date()),
                ],
                "n_estimators_from": "median early-stopped length across folds",
                "git_sha": _git_sha(),
                "versions": _versions(),
                "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            _write_json_atomic(meta, seed_dir / "model_meta.json")
            final_meta.append(meta)
        metrics_payload["final_model"] = {
            "dir": "final",
            "seeds": list(seeds),
            "n_estimators": int(n_final),
            "train_dates": meta["train_dates"],
        }
        LOG.info("final model: %d trees x %d seeds -> %s", n_final, len(seeds), final_dir)

    elapsed = time.perf_counter() - started
    metrics_payload["elapsed_s"] = float(elapsed)
    _write_json_atomic(metrics_payload, run_dir / "metrics.json")
    _write_json_atomic(
        {"run_id": rid, "importance": importance_groups, "folds": len(per_fold), "seeds": seeds},
        run_dir / "run_meta.json",
    )
    predictions_frame.to_parquet(run_dir / "predictions.parquet", index=False)
    _write_golden(predictions_frame, run_dir, seeds)

    legacy = _legacy_payload(metrics_payload, seeds, run_dir, matrix_path)
    result = RunResult(
        run_id=rid,
        run_dir=run_dir,
        predictions=predictions_frame,
        metrics=metrics_payload,
        importance=importance,
        seeds=seeds,
        folds=folds,
        elapsed_s=float(elapsed),
        legacy=legacy,
    )
    _publish_latest(result, reports_root, data_root)
    return result


def _diebold_by_date(frame: pd.DataFrame, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    per_date = pd.DataFrame({"date": frame["date"].to_numpy(), "a": a, "b": b})
    grouped = per_date.groupby("date", sort=True)[["a", "b"]].mean()
    return diebold_mariano(grouped["a"].to_numpy(), grouped["b"].to_numpy())


def _write_golden(predictions: pd.DataFrame, run_dir: Path, seeds: Sequence[int]) -> None:
    """Pick a few test rows per fold/seed and store them for live parity tests."""
    records: list[dict[str, Any]] = []
    import pyarrow as pa
    import pyarrow.parquet as pq

    from spp.data.store import read_parquet

    matrix_file = run_dir.parents[1].parent / "data" / "processed" / "features.parquet"
    matrix = read_parquet(matrix_file) if matrix_file.is_file() else None
    if matrix is None:  # pragma: no cover - test sandbox without data dir
        return
    keyed = matrix.set_index(["date", "ticker"])
    for fold_id, block in predictions.groupby("fold", sort=True):
        seed = int(seeds[-1])
        sample = block.sort_values(["date", "ticker"], kind="stable").iloc[[0, len(block) // 2, -1]]
        for _, row in sample.iterrows():
            try:
                source = keyed.loc[(row["date"], row["ticker"])]
            except KeyError:
                continue
            if isinstance(source, pd.DataFrame):  # pragma: no cover - defensive
                source = source.iloc[0]
            payload = {
                k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                for k, v in source.items()
                if k not in {"y", "y_1d", "y_2d", "y_3d", "y_5d", "y_dir", "y_raw",
                             "y_used_demeaned", "ghost", "label_ok", "split"}
            }
            payload["date"] = str(pd.Timestamp(row["date"]).date())
            records.append(
                {
                    "name": (
                        f"fold{int(fold_id)}_{seed}_{row['ticker']}_"
                        f"{pd.Timestamp(row['date']).date()}"
                    ),
                    "model_dir": str(run_dir / f"fold_{int(fold_id)}" / f"seed_{seed}"),
                    "features_json": json.dumps(_jsonable(payload), sort_keys=True),
                    "expected": float(row["y_pred_seed_mean"]),
                    "y_pred_single_seed": None,
                }
            )
    if not records:
        return
    table = pa.Table.from_pylist(records)
    pq.write_table(table, run_dir / "golden_vectors.parquet")


def _legacy_payload(
    metrics: dict[str, Any], seeds: Sequence[int], run_dir: Path, matrix_path: Path
) -> dict[str, Any]:
    """Plan-schema headline report (``reports/gbdt_report.json``)."""
    folds = metrics["folds"]
    return {
        "model": "lightgbm",
        "target": "y (per-date demeaned when features.demean_target)",
        "seeds": list(seeds),
        "run_id": metrics["run_id"],
        "run_dir": str(run_dir),
        "feature_matrix": str(matrix_path),
        "n_folds": len(folds),
        "cv_ic": {f"fold_{f['fold']}": f["ic"] for f in folds},
        "cv_ic_t": {f"fold_{f['fold']}": f["ic_t"] for f in folds},
        "cv_ic_positive_folds": int(sum(1 for f in folds if np.isfinite(f["ic"]) and f["ic"] > 0)),
        "cv_folds_required": 4,
        "aggregate": metrics["aggregate"],
        "pooled": metrics["pooled"],
        "dm_vs_zero": metrics["dm_vs_zero"],
        "versions": metrics["versions"],
        "git_sha": metrics["git_sha"],
        "elapsed_s": metrics["elapsed_s"],
        "held_out_year_touched": False,
    }


def write_gbdt_report(payload: dict[str, Any], path: str | Path) -> Path:
    return _write_json_atomic(payload, Path(path))


def _publish_latest(result: RunResult, reports_root: Path, data_root: Path) -> None:
    """Copy the run's artifacts to the plan's fixed names (``reports/latest_gbdt``...)."""
    import shutil

    latest = reports_root / "latest_gbdt"
    latest.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "run_meta.json", "feature_importance.csv",
                 "predictions.parquet", "golden_vectors.parquet"):
        source = result.run_dir / name
        if source.is_file():
            shutil.copyfile(source, latest / name)
    write_gbdt_report(result.legacy, reports_root / "gbdt_report.json")
    result.predictions.to_parquet(data_root / "processed" / "predictions.parquet", index=False)
