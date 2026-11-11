# ruff: noqa: E501  - prints a report table
"""Classic market models on the news sample, alone and combined.

The question: do the models people historically use for price prediction - logistic
regression on direction, ridge, random forest, histogram gradient boosting, a tree
quantile model - beat what we already have, and does combining them help?

Everything is measured on the same protocol as the capacity ladder: 5 expanding purged
walk-forward folds on the train split, out-of-fold Spearman IC, and the top decile of
|prediction| net of the 28 bps two-leg round trip. Features are the 15 causal baseline
columns plus six technical indicators computed from bars strictly before the entry:
RSI(60), MACD(12,26) normalised by price, a 20-bar Bollinger z-score, ATR(14)/price,
a 60-bar volume z-score and the 60-bar high/low position.

The combination is deliberately the cheapest possible metamodel: an average of the
cross-sectional ranks of the four base models. If a rank average of models with zero IC
cannot pay 28 bps, a learned metamodel over the same inputs will not either.

Usage:
    python ops/crypto_classic.py [--events data/crypto/dataset/events.parquet]
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from spp.crypto import models

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = ROOT / "data" / "crypto" / "dataset" / "events.parquet"
BARS_DIR = ROOT / "data" / "crypto" / "bars" / "1m"
RESULTS = ROOT / "results"
COST_BPS = 28.0
TOP_FRACTION = 0.10
INDICATORS = ("rsi60", "macd", "boll20", "atr14", "vol_z60", "hl_pos60")


def indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Six classic indicators, each using only bars up to and including the row."""
    close = frame["close"].astype("float64")
    high = frame["high"].astype("float64")
    low = frame["low"].astype("float64")
    volume = frame["volume"].astype("float64")
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(60, min_periods=30).mean()
    loss = (-delta.clip(upper=0.0)).rolling(60, min_periods=30).mean()
    out = pd.DataFrame(index=frame.index)
    out["rsi60"] = 100.0 - 100.0 / (1.0 + gain / (loss + 1e-12))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = (ema12 - ema26) / (close + 1e-12)
    mean20 = close.rolling(20, min_periods=10).mean()
    sd20 = close.rolling(20, min_periods=10).std(ddof=0)
    out["boll20"] = (close - mean20) / (sd20 + 1e-12)
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14, min_periods=7).mean() / (close + 1e-12)
    vmean = volume.rolling(60, min_periods=30).mean()
    vsd = volume.rolling(60, min_periods=30).std(ddof=0)
    out["vol_z60"] = (volume - vmean) / (vsd + 1e-12)
    hi = high.rolling(60, min_periods=30).max()
    lo = low.rolling(60, min_periods=30).min()
    out["hl_pos60"] = (close - lo) / (hi - lo + 1e-12)
    return out


def attach_indicators(events: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Map each event to the indicator row of the last bar strictly before entry."""
    wide = base.copy()
    for name in INDICATORS:
        wide[name] = np.nan
    for symbol, group in events.groupby("symbol"):
        path = BARS_DIR / f"{symbol}.parquet"
        if not path.is_file():
            continue
        bars = pd.read_parquet(path, columns=["ts_open", "high", "low", "close", "volume"])
        ind = indicators(bars)
        stamps = bars["ts_open"].to_numpy(dtype="datetime64[ns]").astype("int64")
        entry = group["entry_ts"].to_numpy(dtype="datetime64[ns]").astype("int64")
        pos = np.searchsorted(stamps, entry, side="left") - 1
        valid = pos >= 0
        rows = np.full(len(group), -1, dtype="int64")
        rows[valid] = pos[valid]
        values = np.full((len(group), len(INDICATORS)), np.nan)
        ok = rows >= 0
        for j, name in enumerate(INDICATORS):
            column = ind[name].to_numpy(dtype="float64")
            values[ok, j] = column[rows[ok]]
        for j, name in enumerate(INDICATORS):
            wide.loc[group.index, name] = values[:, j]
    return wide


def fit_predict(kind: str, train: pd.DataFrame, test: pd.DataFrame, names: list[str]) -> np.ndarray:
    x_train = train[names].to_numpy(dtype="float64")
    x_test = test[names].to_numpy(dtype="float64")
    y = train["y"].to_numpy(dtype="float64")
    # LightGBM and HistGradientBoosting take NaN natively; the linear models and the
    # forest do not. Impute with the TRAIN fold median only, so no test-period
    # information enters the fill values.
    if kind in {"logit", "ridge", "forest"}:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(x_train, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        x_train = np.where(np.isfinite(x_train), x_train, med)
        x_test = np.where(np.isfinite(x_test), x_test, med)
    if kind == "logit":
        model = LogisticRegression(max_iter=2000, C=0.1)
        model.fit(x_train, (y > 0).astype(int))
        return model.predict_proba(x_test)[:, 1]
    if kind == "ridge":
        model = Ridge(alpha=100.0)
        model.fit(x_train, y)
        return model.predict(x_test)
    if kind == "forest":
        model = RandomForestRegressor(
            n_estimators=150, min_samples_leaf=40, max_features="sqrt", n_jobs=-1, random_state=42
        )
        model.fit(x_train, y)
        return model.predict(x_test)
    if kind == "hist":
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=60, random_state=42
        )
        model.fit(x_train, y)
        return model.predict(x_test)
    return np.asarray(models.fit_quantile_model(0.5, train[names], train["y"]).predict(test[names]))


def rank_average(stack: list[np.ndarray]) -> np.ndarray:
    ranks = [pd.Series(p).rank(pct=True).to_numpy() for p in stack]
    return np.mean(ranks, axis=0)


def evaluate(events: pd.DataFrame, pred: np.ndarray, horizon: int) -> dict[str, float]:
    truth = events[f"excess_{horizon}m"]
    ic = models.spearman_ic(truth, pd.Series(pred))
    cut = pd.Series(np.abs(pred)).quantile(1.0 - TOP_FRACTION)
    top = truth[np.abs(pred) >= cut]
    mean_bps = float(top.mean() * 1e4) if len(top) else float("nan")
    return {
        "oof_ic": float(ic),
        "n": int(len(events)),
        "top_n": int(len(top)),
        "top_mean_bps": mean_bps,
        "top_net_bps": mean_bps - COST_BPS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default=str(DEFAULT_EVENTS))
    parser.add_argument("--out", default=str(RESULTS / "crypto_classic.json"))
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
    bars = {
        symbol: pd.read_parquet(BARS_DIR / f"{symbol}.parquet", columns=["ts_open", "open", "close"])
        for symbol in symbols
        if (BARS_DIR / f"{symbol}.parquet").is_file()
    }
    base, base_names = models.build_features(events, bars)
    wide = attach_indicators(events, base)
    names = base_names + list(INDICATORS)
    print(f"rows={len(events)} features={len(names)} indicator_coverage={1.0 - wide[list(INDICATORS)].isna().mean().mean():.3f}")

    folds = models.make_walk_forward_folds(events, split="train")
    kinds = ("logit", "ridge", "forest", "hist", "gbdt", "combo")
    rows: list[dict[str, Any]] = []
    for horizon in models.HORIZONS:
        oof: dict[str, np.ndarray] = {kind: np.full(len(events), np.nan) for kind in kinds}
        fold_ic: dict[str, list[float]] = {kind: [] for kind in kinds}
        frame = events.join(wide[names])
        frame["y"] = frame[f"excess_{horizon}m"]
        for fold in folds:
            train = frame.loc[fold.train_idx]
            test = frame.loc[fold.test_idx]
            # One event can lack an exit bar at the longest horizon; sklearn refuses NaN
            # targets, so the affected rows leave the fit instead of being imputed.
            train = train[np.isfinite(train["y"])]
            stack = []
            for kind in ("logit", "ridge", "forest", "hist", "gbdt"):
                pred = fit_predict(kind, train, test, names)
                oof[kind][test.index] = pred
                ic = models.spearman_ic(test[f"excess_{horizon}m"], pd.Series(pred))
                if ic == ic:
                    fold_ic[kind].append(float(ic))
                stack.append(pred)
            oof["combo"][test.index] = rank_average(stack)
            ic = models.spearman_ic(test[f"excess_{horizon}m"], pd.Series(oof["combo"][test.index]))
            if ic == ic:
                fold_ic["combo"].append(float(ic))
        for kind in kinds:
            mask = ~np.isnan(oof[kind])
            block = frame.loc[mask]
            block = block[np.isfinite(block[f"excess_{horizon}m"])]
            stats = evaluate(block, oof[kind][mask], horizon)
            stats.update(
                {
                    "model": kind,
                    "horizon": horizon,
                    "folds": len(fold_ic[kind]),
                    "fold_ic_mean": float(np.mean(fold_ic[kind])) if fold_ic[kind] else float("nan"),
                    "fold_ic_sd": float(np.std(fold_ic[kind], ddof=1)) if len(fold_ic[kind]) > 1 else float("nan"),
                }
            )
            rows.append(stats)
            print(
                f"{kind:7s} {horizon:3d}m  IC {stats['oof_ic']:+.4f}  folds {stats['fold_ic_mean']:+.4f}"
                f"+-{stats['fold_ic_sd']:.4f}  top10% {stats['top_mean_bps']:+.2f} bps  net {stats['top_net_bps']:+.2f}"
            )

    Path(args.out).write_text(json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")
    table = (
        "| модель | горизонт | OOF IC | IC по фолдам | топ-дециль, бп | net (28 бп) |\n|---|---|---|---|---|---|\n"
    )
    for row in rows:
        table += (
            f"| {row['model']} | {row['horizon']}m | {row['oof_ic']:+.4f} | {row['fold_ic_mean']:+.4f} ± "
            f"{row['fold_ic_sd']:.4f} | {row['top_mean_bps']:+.2f} | {row['top_net_bps']:+.2f} |\n"
        )
    Path(RESULTS / "crypto_classic.md").write_text(
        "# Классические модели и их объединение\n\n" + table, encoding="utf-8"
    )
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
