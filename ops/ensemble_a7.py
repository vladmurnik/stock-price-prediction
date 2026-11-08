"""A7: rank-average ensemble of the FinBERT and open-jev sentiment models.

Both inputs are sentiment-only GBDTs trained on the same purged folds, so their predictions
are aligned by ``(fold, date, ticker)`` and averaged *after* converting each to a within-date
percentile rank.  Ranks are used because the two matrices have different scales and the models
are retrained per fold; averaging raw predictions would silently weight one model by variance.

Rows are restricted to the intersection where both backends actually produced a prediction
(``n`` is reported), and the market+sentiment model (A5) is re-scored on exactly that subset so
the comparison is not contaminated by a different row set.

Usage:
    python ops/ensemble_a7.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spp.models.gbdt import _aggregate, _diebold_by_date, _fold_metrics, _pooled

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "reports" / "runs"
OUT_ID = "a7-ensemble"
BASE_MODELS = {
    "finbert": "a2-sentiment-only",
    "openjev": "a3-openjev",
}
CONTROL = "a5-all"


def _load(run_id: str) -> pd.DataFrame:
    path = RUNS / run_id / "predictions.parquet"
    if not path.is_file():
        raise SystemExit(f"missing {path}: run `spp train gbdt --run-id {run_id}` first")
    frame = pd.read_parquet(path)
    keep = ["fold", "date", "ticker", "y_eval", "y_pred_seed_mean"]
    return frame.loc[:, keep].rename(columns={"y_pred_seed_mean": "y_pred"})


def _rank_within_date(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby(["fold", "date"], sort=False)[column]
    return grouped.rank(method="average", pct=True)


def _evaluate(frame: pd.DataFrame, label: str) -> dict[str, object]:
    per_fold = []
    for _fold, block in frame.groupby("fold", sort=True):
        block = block.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
        per_fold.append(_fold_metrics(block))
    aggregate = _aggregate(per_fold)
    pooled = _pooled(frame.assign(y_pred=frame["y_pred"]))
    model_losses = (frame["y_pred"] - frame["y_eval"]).abs().to_numpy()
    zero_losses = frame["y_eval"].abs().to_numpy()
    dm, p_value = _diebold_by_date(frame, model_losses, zero_losses)
    block = aggregate["ic"]
    print(
        f"{label:<22} n={len(frame):5d} folds={len(per_fold)} "
        f"ic={block['mean']:+.4f} std={block['std']:.4f} pooled={pooled['ic']:+.4f} "
        f"dm_p={p_value:.4f}"
    )
    return {
        "label": label,
        "n_rows": int(len(frame)),
        "n_folds": len(per_fold),
        "aggregate": aggregate,
        "pooled": pooled,
        "dm_vs_zero": {"dm": float(dm), "p": float(p_value)},
        "folds": [
            {k: v for k, v in fold.items() if k != "ic_series"} for fold in per_fold
        ],
    }


def main() -> int:
    fin = _load(BASE_MODELS["finbert"])
    jev = _load(BASE_MODELS["openjev"])
    print(f"finbert rows={len(fin)}  openjev rows={len(jev)}")

    merged = fin.merge(
        jev,
        on=["fold", "date", "ticker"],
        how="inner",
        suffixes=("_finbert", "_openjev"),
        validate="one_to_one",
    )
    print(
        f"intersection rows={len(merged)} "
        f"({len(merged) / max(len(fin), 1):.1%} of finbert, "
        f"{len(merged) / max(len(jev), 1):.1%} of openjev)"
    )
    if merged.empty:
        raise SystemExit("no overlapping predictions; nothing to ensemble")

    merged["r_finbert"] = _rank_within_date(merged, "y_pred_finbert")
    merged["r_openjev"] = _rank_within_date(merged, "y_pred_openjev")
    merged["y_pred"] = 0.5 * (merged["r_finbert"] + merged["r_openjev"])
    merged["y_eval"] = merged["y_eval_finbert"]

    ensemble_frame = merged.loc[:, ["fold", "date", "ticker", "y_eval", "y_pred"]].copy()

    results = {
        "a7": _evaluate(ensemble_frame, "A7 ensemble (rank avg)"),
        "finbert": _evaluate(
            merged.assign(y_pred=merged["y_pred_finbert"]).loc[
                :, ["fold", "date", "ticker", "y_eval", "y_pred"]
            ],
            "A2 finbert (subset)",
        ),
        "openjev": _evaluate(
            merged.assign(y_pred=merged["y_pred_openjev"]).loc[
                :, ["fold", "date", "ticker", "y_eval", "y_pred"]
            ],
            "A3 open-jev (subset)",
        ),
    }

    control_path = RUNS / CONTROL / "predictions.parquet"
    if control_path.is_file():
        control = pd.read_parquet(control_path).loc[
            :, ["fold", "date", "ticker", "y_eval", "y_pred_seed_mean"]
        ]
        control = control.rename(columns={"y_pred_seed_mean": "y_pred"})
        subset = control.merge(
            merged.loc[:, ["fold", "date", "ticker"]],
            on=["fold", "date", "ticker"],
            how="inner",
            validate="one_to_one",
        )
        results["control_a5"] = _evaluate(subset, "A5 control (same rows)")

    out_dir = RUNS / OUT_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    ensemble_frame["seed"] = np.nan
    ensemble_frame["y"] = np.nan
    ensemble_frame["y_raw"] = np.nan
    ensemble_frame["score_rank"] = ensemble_frame["y_pred"]
    ensemble_frame["y_pred_seed_mean"] = ensemble_frame["y_pred"]
    ensemble_frame.sort_values(["date", "fold", "ticker"], kind="stable").to_parquet(
        out_dir / "predictions.parquet", index=False
    )

    block = results["a7"]["aggregate"]["ic"]
    payload = {
        "run_id": OUT_ID,
        "schema": "ensemble-v1",
        "method": "0.5 * (within-date percentile rank of finbert + of open-jev)",
        "inputs": {
            "finbert": BASE_MODELS["finbert"],
            "openjev": BASE_MODELS["openjev"],
            "control": CONTROL if "control_a5" in results else None,
        },
        "intersection_rows": int(len(merged)),
        "aggregate": results["a7"]["aggregate"],
        "pooled": results["a7"]["pooled"],
        "dm_vs_zero": results["a7"]["dm_vs_zero"],
        "folds": results["a7"]["folds"],
        "variants": {
            k: {kk: vv for kk, vv in v.items() if kk != "folds"}
            for k, v in results.items()
        },
        "ic_ci95_lo": block["mean"] - 1.96 * block["std"] / max(block["n"], 1) ** 0.5,
        "ic_ci95_hi": block["mean"] + 1.96 * block["std"] / max(block["n"], 1) ** 0.5,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
