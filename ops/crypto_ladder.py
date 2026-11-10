# ruff: noqa: E501  - prints a report table
"""Does more model capacity help? A capacity ladder, measured out of fold.

The proposal under test: more features derived from the old ones, deeper models, longer
training, per-asset specialists and a seed ensemble (a cheap stand-in for a metamodel).
Every rung is evaluated the same way: 5 expanding purged walk-forward folds on the train
split, out-of-fold Spearman IC per fold, and the economic question that actually matters -
what does the top decile of |prediction| earn after the round trip.

If the ladder is flat, the bottleneck is not capacity; it is the signal-to-cost ratio, and
a metamodel over models that cannot predict anything will not manufacture one.

Usage:
    python ops/crypto_ladder.py [--events data/crypto/dataset/events.parquet]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spp.crypto import models
from spp.crypto.dataset import load_bars

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = ROOT / "data" / "crypto" / "dataset" / "events.parquet"
RESULTS = ROOT / "results"
COST_ONE_LEG = 0.0014
COST_ROUND_TRIP_PAIR = 2 * COST_ONE_LEG
TOP_FRACTION = 0.10


def derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Eight features that are pure functions of the causal baseline columns."""
    out = frame.copy()
    eps = 1e-9
    out["ret_z5"] = out["tr_ret_5m"] / (out["tr_vol_30m"] + eps)
    out["ret_z15"] = out["tr_ret_15m"] / (out["tr_vol_60m"] + eps)
    out["pace"] = out["tr_ret_5m"] - out["tr_ret_60m"] / 12.0
    out["accel"] = out["tr_ret_5m"] - out["tr_ret_15m"] / 3.0
    out["vol_ratio"] = out["tr_vol_30m"] / (out["tr_vol_60m"] + eps)
    out["abs_ret60"] = out["tr_ret_60m"].abs()
    out["burst_per_lag"] = out["news_burst_60m"] / (1.0 + out["log_lag"])
    out["btc_rel_5m"] = out["tr_ret_5m"] - out["btc_ret_5m"]
    return out


DERIVED_NAMES = (
    "ret_z5",
    "ret_z15",
    "pace",
    "accel",
    "vol_ratio",
    "abs_ret60",
    "burst_per_lag",
    "btc_rel_5m",
)


def fit_predict(alpha: float, train: pd.DataFrame, test: pd.DataFrame, names: list[str]) -> np.ndarray:
    model = models.fit_quantile_model(alpha, train[names], train["y"])
    return np.asarray(model.predict(test[names]), dtype="float64")


def run_config(
    events: pd.DataFrame,
    names: list[str],
    horizon: int,
    folds: list[models.WalkForwardFold],
    mode: str,
) -> dict[str, Any]:
    """Out-of-fold predictions for one configuration of one horizon."""
    pred = pd.Series(np.nan, index=events.index, dtype="float64")
    per_fold: list[float] = []
    for fold in folds:
        train = events.loc[fold.train_idx].copy()
        test = events.loc[fold.test_idx].copy()
        train["y"] = train[f"excess_{horizon}m"]
        test["y"] = np.nan
        if len(train) < 200 or len(test) < 20:
            continue

        if mode == "per_asset":
            piece = pd.Series(np.nan, index=test.index, dtype="float64")
            for symbol, group in test.groupby("symbol"):
                part = train[train["symbol"] == symbol]
                if len(part) < 300:
                    continue
                piece.loc[group.index] = fit_predict(0.5, part, group, names)
            fallback = piece.isna()
            if fallback.any():
                piece.loc[fallback] = fit_predict(0.5, train, test.loc[fallback], names)
            fold_pred = piece
        elif mode == "ensemble":
            stack = []
            for seed in (1, 7, 13, 42, 99):
                models.MODEL_PARAMS["seed"] = seed
                stack.append(fit_predict(0.5, train, test, names))
            fold_pred = pd.Series(np.mean(stack, axis=0), index=test.index, dtype="float64")
        else:
            fold_pred = pd.Series(fit_predict(0.5, train, test, names), index=test.index, dtype="float64")

        pred.loc[fold_pred.index] = fold_pred.to_numpy()
        ic = models.spearman_ic(test[f"excess_{horizon}m"], fold_pred)
        if ic == ic:
            per_fold.append(float(ic))

    scored = events.loc[pred.notna()]
    truth = scored[f"excess_{horizon}m"]
    oof_ic = models.spearman_ic(truth, pred.loc[scored.index])
    cut = pred.loc[scored.index].abs().quantile(1.0 - TOP_FRACTION)
    top = scored[pred.loc[scored.index].abs() >= cut]
    top_mean_bps = float(top[f"excess_{horizon}m"].mean() * 1e4) if len(top) else float("nan")
    return {
        "mode": mode,
        "horizon": horizon,
        "n_scored": int(len(scored)),
        "oof_ic": float(oof_ic),
        "fold_ic_mean": float(np.mean(per_fold)) if per_fold else float("nan"),
        "fold_ic_sd": float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else float("nan"),
        "folds": len(per_fold),
        "top_n": int(len(top)),
        "top_mean_bps": top_mean_bps,
        "top_net_bps": top_mean_bps - COST_ROUND_TRIP_PAIR * 1e4,
        "population_mean_bps": float(truth.mean() * 1e4),
    }


def render(rows: list[dict[str, Any]]) -> str:
    head = (
        "| конфигурация | горизонт | признаков | OOF IC | IC по фолдам | топ-дециль, бп | net (28 бп) |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['horizon']}m | {row['features']} | {row['oof_ic']:+.4f} | "
            f"{row['fold_ic_mean']:+.4f} ± {row['fold_ic_sd']:.4f} | {row['top_mean_bps']:+.2f} | "
            f"{row['top_net_bps']:+.2f} |"
        )
    return head + "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default=str(DEFAULT_EVENTS))
    parser.add_argument("--out", default=str(RESULTS / "crypto_ladder.json"))
    args = parser.parse_args()

    events = pd.read_parquet(args.events).sort_values("entry_ts").reset_index(drop=True)
    if events.empty:
        print("empty dataset")
        return 2
    # BTCUSDT is the benchmark: its bars feed btc_* even though no BTC event survives
    # the benchmark_asset filter, so the symbol list must add it explicitly.
    symbols = sorted({str(s) for s in events["symbol"]})
    if models.BENCH_SYMBOL not in symbols:
        symbols.append(models.BENCH_SYMBOL)
    bars = load_bars(ROOT / "data" / "crypto" / "bars" / "1m", symbols)
    base, base_names = models.build_features(events, bars)
    wide = derived(pd.concat([base, events[["symbol"]]], axis=1))
    wide_names = base_names + list(DERIVED_NAMES)
    for name in DERIVED_NAMES:
        assert name not in base_names, f"{name} collides with a baseline feature"
    print(f"rows={len(events)} baseline={len(base_names)} derived={len(wide_names)} features")

    folds = models.make_walk_forward_folds(events, split="train")
    print("folds:", [(f.fold_id, len(f.train_idx), len(f.test_idx)) for f in folds])

    baseline_params = dict(models.MODEL_PARAMS)
    deep_params = {
        **baseline_params,
        "n_estimators": 800,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.6,
    }

    rows: list[dict[str, Any]] = []
    for horizon in models.HORIZONS:
        ladder = [
            ("A_base", base_names, "single", baseline_params),
            ("B_derived", wide_names, "single", baseline_params),
            ("C_deep", wide_names, "single", deep_params),
            ("D_per_asset", wide_names, "per_asset", baseline_params),
            ("E_seed_ensemble", wide_names, "ensemble", baseline_params),
        ]
        for label, names, mode, params in ladder:
            models.MODEL_PARAMS.update({**baseline_params, **params})
            frame = events.join(wide[names]) if label != "A_base" else events.join(base[base_names])
            row = run_config(frame, names, horizon, folds, mode)
            row["features"] = len(names)
            row["config"] = label
            rows.append(row)
            print(
                f"{label:16s} {horizon:3d}m  IC {row['oof_ic']:+.4f}  folds {row['fold_ic_mean']:+.4f}"
                f"+-{row['fold_ic_sd']:.4f}  top10% {row['top_mean_bps']:+.2f} bps"
                f"  net {row['top_net_bps']:+.2f}"
            )
    models.MODEL_PARAMS.clear()
    models.MODEL_PARAMS.update(baseline_params)

    table = render(rows)
    Path(args.out).write_text(json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")
    Path(RESULTS / "crypto_ladder.md").write_text(
        "# Лестница ёмкости: помогает ли более сложная модель\n\n" + table, encoding="utf-8"
    )
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
