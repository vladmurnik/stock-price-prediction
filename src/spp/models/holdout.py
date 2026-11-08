"""Single-shot 2023 holdout.

The test year is used exactly once. The runner refuses to start when a lock file already
exists, and the lock records the config hash, the matrix hash and the wall-clock time so a
second run can be detected even if the reports are deleted.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spp.config import Config
from spp.models.gbdt import (
    EARLY_STOP_SESSIONS,
    MODEL_SCHEMA_VERSION,
    GBDTError,
    _clean_rows,
    _diebold_by_date,
    _fit_one,
    _fold_metrics,
    _git_sha,
    _jsonable,
    _params,
    _write_json_atomic,
    feature_columns,
)

HOLDOUT_START = pd.Timestamp("2023-01-01")
LOCK_NAME = "test_2023.lock"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_frame(frame: pd.DataFrame) -> str:
    cols = ["date", "ticker", "y"] + [c for c in ("y_raw", "split") if c in frame.columns]
    payload = frame[cols].sort_values(["date", "ticker"]).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(payload, index=False).values).hexdigest()


def _decile_rank_corr(deciles: dict[str, float]) -> float:
    ordered = [deciles[str(i)] for i in range(1, 11) if str(i) in deciles]
    if len(ordered) < 3 or not np.isfinite(ordered).all():
        return float("nan")
    ranks = pd.Series(ordered).rank().to_numpy()
    order = np.arange(1, len(ordered) + 1, dtype="float64")
    return float(np.corrcoef(order, ranks)[0, 1])


def run_holdout(
    config: Config,
    *,
    features_path: str | Path | None = None,
    seeds: Sequence[int] | None = None,
    project_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Train on 2019-2022 and score 2023 once. Writes the lock before touching the model."""
    started = time.perf_counter()
    from spp.data.store import read_parquet

    root = Path(project_root) if project_root else Path.cwd()
    data_root = root / config.paths.data
    reports_root = root / config.paths.reports
    matrix_path = (
        Path(features_path) if features_path else data_root / "processed" / "features.parquet"
    )
    if not matrix_path.is_file():
        raise GBDTError(f"missing feature matrix: {matrix_path}")

    lock_path = reports_root / LOCK_NAME
    if lock_path.exists() and not force:
        raise GBDTError(
            f"{lock_path} exists: the 2023 holdout has already been used. "
            "Repeat runs require an explicit --force and belong in a separate experiment."
        )

    frame = read_parquet(matrix_path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if "split" not in frame.columns:
        raise GBDTError("feature matrix has no 'split' column")

    fit_frame = frame.loc[frame["split"].isin(["train", "valid"])].copy()
    test_frame = frame.loc[frame["split"] == "test"].copy()
    if test_frame.empty:
        raise GBDTError("no rows with split == 'test'")
    if test_frame["date"].min() < HOLDOUT_START:
        raise GBDTError("test rows start before 2023-01-01")
    if fit_frame["date"].max() >= HOLDOUT_START:
        raise GBDTError("training rows reach into 2023")
    if test_frame["date"].max() > pd.Timestamp(config.dates.test_end):
        raise GBDTError("test rows extend past dates.test_end")

    seeds = [int(s) for s in (seeds or config.model.seeds)]
    cols = [c for c in feature_columns(True) if c in frame.columns]
    absent = [c for c in feature_columns(True) if c not in frame.columns]
    if absent:
        raise GBDTError(f"matrix lacks {len(absent)} documented feature(s): {absent}")
    categorical = ["dow"] if "dow" in cols else []
    rank_cols = (
        [c for c in cols if c not in categorical] if config.features.cross_sectional_rank else []
    )
    demeaning = bool(config.features.demean_target)

    train = _clean_rows(fit_frame)
    test = test_frame.copy()
    if len(train) < 500:
        raise GBDTError(f"only {len(train)} labelled training rows")

    train_days = np.sort(train["date"].unique())
    if len(train_days) > 2 * EARLY_STOP_SESSIONS:
        inner_days = train_days[-EARLY_STOP_SESSIONS:]
        inner_mask = train["date"].isin(inner_days)
        fit_train = train.loc[~inner_mask]
        inner_valid = train.loc[inner_mask]
    else:
        fit_train, inner_valid = train, train

    y_raw = pd.to_numeric(fit_frame.get("y_raw", fit_frame["y"]), errors="coerce")
    from spp.features.clip import clip_bounds

    clip_lo, clip_hi = clip_bounds(y_raw)

    lock_payload = {
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "config_hash": hashlib.sha256(
            json.dumps(config.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest(),
        "matrix_sha256": _sha256_frame(frame),
        "matrix_path": str(matrix_path),
        "file_sha256": _sha256_file(matrix_path),
        "git_sha": _git_sha(),
        "seeds": seeds,
        "train_rows": int(len(fit_train)),
        "inner_valid_rows": int(len(inner_valid)),
        "test_rows": int(len(test)),
        "test_window": [str(test["date"].min().date()), str(test["date"].max().date())],
    }
    reports_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(lock_payload, lock_path)

    preds: list[np.ndarray] = []
    seed_meta: list[dict[str, Any]] = []
    for seed in seeds:
        params = _params(config.model, seed)
        fitted = _fit_one(
            fit_train,
            inner_valid,
            test=test,
            feature_list=cols,
            ranked=rank_cols,
            categorical=categorical,
            demeaning=demeaning,
            params=params,
            early_stopping_rounds=config.model.early_stopping_rounds,
            clip_bounds_=(clip_lo, clip_hi),
        )
        preds.append(np.asarray(fitted["y_pred"], dtype="float64"))
        seed_meta.append(
            {
                "seed": int(seed),
                "best_iteration": int(fitted["best_iteration"]),
                "valid_ic": float(fitted["valid_ic"]),
            }
        )

    stacked = np.vstack(preds)
    y_pred = np.nanmean(stacked, axis=0)

    scored = test.copy()
    scored["y_pred"] = y_pred
    scored["y_eval"] = pd.to_numeric(scored["y"], errors="coerce")
    scored = scored.loc[scored["y_eval"].notna()].copy()

    metrics = _fold_metrics(scored)
    dm, p_value = _diebold_by_date(scored, scored["y_eval"].to_numpy(), y_pred[: len(scored)])
    deciles = metrics["deciles"]
    monotonicity = _decile_rank_corr(deciles)
    criteria = {
        "ic_gt_0.02": bool(metrics["ic"] > 0.02),
        "ic_ci_lo_gt_0": bool(metrics["ic_ci95_lo"] > 0.0),
        "dm_p_lt_0.05": bool(p_value < 0.05),
        "decile_monotonicity_gt_0.5": bool(
            np.isfinite(monotonicity) and monotonicity > 0.5
        ),
    }
    verdict = "confirmed" if all(criteria.values()) else "not_confirmed"

    payload = {
        "run": "test_2023",
        "created_at": lock_payload["created_at"],
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "config_hash": lock_payload["config_hash"],
        "matrix_sha256": lock_payload["matrix_sha256"],
        "git_sha": lock_payload["git_sha"],
        "seeds": seed_meta,
        "rows": {
            "train": int(len(fit_train)),
            "inner_valid": int(len(inner_valid)),
            "test": int(len(scored)),
            "test_dates": int(scored["date"].nunique()),
        },
        "test_window": lock_payload["test_window"],
        "metrics": {k: _jsonable(v) for k, v in metrics.items() if k != "ic_series"},
        "dm_vs_zero": {"dm": float(dm), "p": float(p_value)},
        "decile_rank_corr": monotonicity,
        "criteria": criteria,
        "verdict": verdict,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }

    out_json = reports_root / "test_2023.json"
    _write_json_atomic(payload, out_json)
    scored.to_parquet(reports_root / "test_2023_predictions.parquet", index=False)
    (reports_root / "test_2023.md").write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def _render_markdown(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines = [
        "# 2023 holdout — single run",
        "",
        f"- verdict: **{payload['verdict']}**",
        f"- window: {payload['test_window'][0]} .. {payload['test_window'][1]}",
        f"- rows: train {payload['rows']['train']}, inner valid {payload['rows']['inner_valid']}, "
        f"test {payload['rows']['test']} over {payload['rows']['test_dates']} dates",
        f"- config hash: `{payload['config_hash'][:16]}`",
        f"- matrix hash: `{payload['matrix_sha256'][:16]}`",
        "",
        "| metric | value |",
        "|---|---|",
        f"| IC | {m['ic']:.4f} |",
        f"| IC 95% CI | [{m['ic_ci95_lo']:.4f}, {m['ic_ci95_hi']:.4f}] |",
        f"| IC t (NW) | {m['ic_t_nw']:.2f} |",
        f"| direction accuracy | {m['dir_acc']:.4f} |",
        f"| AUC | {m['auc']:.4f} |",
        f"| MAE | {m['mae']:.5f} |",
        f"| R2 vs zero | {m['r2_vs_zero']:.5f} |",
        f"| decile rank corr | {payload['decile_rank_corr']:.3f} |",
        f"| DM vs zero p | {payload['dm_vs_zero']['p']:.4f} |",
        "",
        "## Pre-registered criteria",
        "",
        "| criterion | pass |",
        "|---|---|",
    ]
    for name, ok in payload["criteria"].items():
        lines.append(f"| {name} | {'yes' if ok else 'no'} |")
    lines += [
        "",
        "## Decile mean returns",
        "",
        "| decile | return |",
        "|---|---|",
    ]
    for key in sorted(m["deciles"], key=lambda k: int(k)):
        lines.append(f"| {key} | {m['deciles'][key]:.6f} |")
    lines.append("")
    return "\n".join(lines)
