"""T7 — train the crypto event-study models and write the policy report.

Usage:
    python ops/crypto_train.py
    python ops/crypto_train.py --events data/crypto/dataset/events.parquet

Reads ``events.parquet`` (exit 2 when it is missing, empty or malformed), builds the
causal feature matrix from the 1-minute bars, proves causality with
``assert_no_future_features``, runs purged walk-forward validation on the **train
split only** (5 expanding folds, 60-minute embargo), fits the quantile models
(alpha 0.1 / 0.5 / 0.9, LightGBM with the sklearn fallback) on the full train split
and reads them out honestly on the **val** split. The holdout split is empty by
construction and is never reported; val is never used for fitting or selection.

Outputs ``results/crypto_models.json`` (feature list, fold table, baseline table,
policy table, per-asset table) and ``results/crypto_policy.md`` (Russian summary with
the leverage statement and the "what cannot be claimed" list). The console prints the
OOF IC mean +/- sd and the policy table.
Exit codes: 0 on success, 2 when events.parquet is missing or empty.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spp.crypto.models import (
    HORIZONS,
    LEVERAGES,
    QUANTILE_ALPHAS,
    assert_no_future_features,
    baseline_positions,
    build_features,
    evaluate_policy,
    fit_quantile_model,
    make_walk_forward_folds,
    model_backend,
    policy_net_pnl,
    policy_positions,
    spearman_ic,
)
from spp.logging import get_logger

ROOT = Path(__file__).resolve().parents[1]
LOG = get_logger("ops.crypto_train")

DEFAULT_EVENTS = ROOT / "data" / "crypto" / "dataset" / "events.parquet"
DEFAULT_BARS_DIR = ROOT / "data" / "crypto" / "bars" / "1m"
DEFAULT_JSON = ROOT / "results" / "crypto_models.json"
DEFAULT_MD = ROOT / "results" / "crypto_policy.md"

#: Baseline kinds in report order -> Russian/short label.
BASELINES: tuple[tuple[str, str], ...] = (
    ("flat", "flat (0 trades)"),
    ("long", "long only (+1)"),
    ("short", "short only (-1)"),
    ("momentum", "momentum sign(ret_15m)"),
    ("per_asset_constant", "per-asset constant (in-sample upper bound)"),
)


def read_events(path: Path) -> pd.DataFrame:
    """Load the event dataset; exit 2 with a clear message when missing or empty."""
    if not path.is_file():
        print(f"ERROR: events file not found: {path}", file=sys.stderr)
        raise SystemExit(2)
    events = pd.read_parquet(path)
    if len(events) == 0:
        print(f"ERROR: events file is empty: {path}", file=sys.stderr)
        raise SystemExit(2)
    required = {
        "asset",
        "symbol",
        "entry_ts",
        "lag_minutes",
        "cost",
        "split",
        *(f"excess_{h}m" for h in HORIZONS),
    }
    missing = sorted(required - set(events.columns))
    if missing:
        print(f"ERROR: events file is missing columns: {missing}", file=sys.stderr)
        raise SystemExit(2)
    return events


def read_bars(bars_dir: Path, events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Read ``ts_open``/``open``/``close`` only, and only rows strictly before the
    last ``entry_ts`` (features never touch a bar at or after an entry anyway)."""
    symbols = sorted({str(s) for s in events["symbol"]})
    if "BTCUSDT" not in symbols:
        symbols.append("BTCUSDT")  # market context even when no BTC event exists
    entry = pd.to_datetime(events["entry_ts"], utc=True)
    hi = entry.max()
    lo = entry.min() - pd.Timedelta(minutes=70)
    filters = [[("ts_open", ">=", lo), ("ts_open", "<", hi)]]
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = bars_dir / f"{symbol}.parquet"
        if not path.is_file():
            print(f"ERROR: bars file not found: {path}", file=sys.stderr)
            raise SystemExit(2)
        try:
            frame = pd.read_parquet(path, columns=["ts_open", "open", "close"], filters=filters)
        except Exception:  # noqa: BLE001 - engine without filter pushdown
            frame = pd.read_parquet(path, columns=["ts_open", "open", "close"])
            stamps = pd.to_datetime(frame["ts_open"], utc=True)
            frame = frame.loc[(stamps >= lo) & (stamps < hi)]
        out[symbol] = frame.reset_index(drop=True)
        LOG.info("read_bars: %s -> %d rows", symbol, len(out[symbol]))
    return out


# --------------------------------------------------------------------------- CV
def run_walk_forward(
    events: pd.DataFrame,
    X: pd.DataFrame,
    *,
    n_folds: int,
    embargo_minutes: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], dict[int, dict[float, pd.Series]]]:
    """Purged walk-forward on the train split; returns (fold rows, IC summary, OOF)."""
    folds = make_walk_forward_folds(
        events, n_folds=n_folds, embargo_minutes=embargo_minutes, split="train"
    )
    oof: dict[int, dict[float, pd.Series]] = {
        h: {
            alpha: pd.Series(np.nan, index=events.index, dtype="float64")
            for alpha in QUANTILE_ALPHAS
        }
        for h in HORIZONS
    }
    rows: list[dict[str, Any]] = []
    for fold in folds:
        row = fold.as_dict()
        train_X, test_X = X.loc[fold.train_idx], X.loc[fold.test_idx]
        for h in HORIZONS:
            y_train = events.loc[fold.train_idx, f"excess_{h}m"].to_numpy(dtype="float64")
            fit_ok = np.isfinite(y_train)
            y_test = events.loc[fold.test_idx, f"excess_{h}m"].to_numpy(dtype="float64")
            for alpha in QUANTILE_ALPHAS:
                model = fit_quantile_model(
                    alpha, train_X.iloc[np.flatnonzero(fit_ok)], y_train[fit_ok]
                )
                pred = np.asarray(model.predict(test_X), dtype="float64")
                oof[h][alpha].loc[fold.test_idx] = pred
            valid = np.isfinite(y_test)
            for alpha in QUANTILE_ALPHAS:
                key = f"ic_q{int(round(alpha * 10))}0_{h}m"
                pred = oof[h][alpha].loc[fold.test_idx].to_numpy()[valid]
                row[key] = spearman_ic(y_test[valid], pred)
        rows.append(row)
        LOG.info(
            "fold %d: train=%d test=%d purged=%d",
            fold.fold_id,
            len(fold.train_idx),
            len(fold.test_idx),
            fold.purged_rows,
        )
    return rows, ic_summary(rows), oof


def ic_summary(fold_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean and sd of the OOF q50 Spearman IC across folds, per horizon."""
    out: dict[str, dict[str, float]] = {}
    for h in HORIZONS:
        key = f"ic_q50_{h}m"
        values = np.array([row[key] for row in fold_rows], dtype="float64")
        values = values[np.isfinite(values)]
        out[f"{h}m"] = {
            "mean": float(values.mean()) if values.size else float("nan"),
            "sd": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
            "n_folds": int(values.size),
        }
    return out


def fit_final_models(
    events: pd.DataFrame, X: pd.DataFrame, train_idx: pd.Index
) -> dict[tuple[int, float], Any]:
    """Refit every (horizon, alpha) on the full train split (no val, no holdout)."""
    train_X = X.loc[train_idx]
    fitted: dict[tuple[int, float], Any] = {}
    for h in HORIZONS:
        y = events.loc[train_idx, f"excess_{h}m"].to_numpy(dtype="float64")
        ok = np.isfinite(y)
        for alpha in QUANTILE_ALPHAS:
            fitted[(h, alpha)] = fit_quantile_model(alpha, train_X.iloc[np.flatnonzero(ok)], y[ok])
    return fitted


# --------------------------------------------------------------------------- tables
def baseline_table(
    events: pd.DataFrame, X: pd.DataFrame, val_idx: pd.Index, legs: float = 2.0
) -> list[dict[str, Any]]:
    """Baselines on the val rows (L = 1), one row per (baseline, horizon)."""
    val = events.loc[val_idx]
    trailing = X.loc[val_idx, "tr_ret_15m"].to_numpy(dtype="float64")
    asset = [str(s) for s in val["symbol"]]
    cost = val["cost"].to_numpy(dtype="float64")
    train = events[events["split"] == "train"]
    rows: list[dict[str, Any]] = []
    for kind, label in BASELINES:
        for h in HORIZONS:
            means = (
                train.groupby(train["symbol"].astype(str))[f"excess_{h}m"].mean().to_dict()
                if kind == "per_asset_constant"
                else None
            )
            pos = baseline_positions(kind, trailing, asset, means)
            metrics = evaluate_policy(val["entry_ts"], pos, val[f"excess_{h}m"], cost, legs=legs)
            rows.append(
                {
                    "baseline": kind,
                    "label": label,
                    "horizon": f"{h}m",
                    "in_sample_upper_bound": kind == "per_asset_constant",
                    **metrics,
                }
            )
    return rows


def policy_table(
    events: pd.DataFrame,
    X: pd.DataFrame,
    val_idx: pd.Index,
    pred50: dict[int, np.ndarray],
    legs: float = 2.0,
) -> list[dict[str, Any]]:
    """Policy rows: per horizon and L on the val rows, hurdle = cost (14 bps)."""
    val = events.loc[val_idx]
    cost = val["cost"].to_numpy(dtype="float64")
    hurdle = float(np.nanmedian(cost))
    rows: list[dict[str, Any]] = []
    for h in HORIZONS:
        excess = val[f"excess_{h}m"].to_numpy(dtype="float64")
        for leverage in LEVERAGES:
            pos = policy_positions(pred50[h], hurdle, leverage)
            metrics = evaluate_policy(val["entry_ts"], pos, excess, cost, legs=legs)
            rows.append({"horizon": f"{h}m", "L": leverage, "hurdle_bps": hurdle * 1e4, **metrics})
    return rows


def per_asset_table(
    events: pd.DataFrame,
    X: pd.DataFrame,
    val_idx: pd.Index,
    pred50: dict[int, np.ndarray],
    legs: float = 2.0,
) -> list[dict[str, Any]]:
    """Net PnL per asset for every (horizon, L) of the policy on the val rows."""
    val = events.loc[val_idx]
    cost = val["cost"].to_numpy(dtype="float64")
    hurdle = float(np.nanmedian(cost))
    symbols = np.array([str(s) for s in val["symbol"]])
    rows: list[dict[str, Any]] = []
    for h in HORIZONS:
        excess = val[f"excess_{h}m"].to_numpy(dtype="float64")
        for leverage in LEVERAGES:
            pos = policy_positions(pred50[h], hurdle, leverage)
            valid = np.isfinite(pos) & np.isfinite(excess)
            # the label is a spread: two legs pay
            net = policy_net_pnl(pos, excess, cost, legs=legs)
            frame = pd.DataFrame(
                {
                    "symbol": symbols[valid],
                    "pos": pos[valid],
                    "exc": excess[valid],
                    "net": net[valid],
                }
            )
            for sym, group in frame.groupby("symbol", sort=True):
                traded = group["pos"].to_numpy() != 0.0
                n_trades = int(np.count_nonzero(traded))
                gross_pos = np.sign(group["pos"].to_numpy()[traded])
                gross = (
                    float(1e4 * np.mean(gross_pos * group["exc"].to_numpy()[traded]))
                    if n_trades
                    else float("nan")
                )
                rows.append(
                    {
                        "horizon": f"{h}m",
                        "L": leverage,
                        "symbol": sym,
                        "n": int(len(group)),
                        "n_trades": n_trades,
                        "gross_bps": gross,
                        "net_bps": float(1e4 * group["net"].mean()),
                    }
                )
    return rows


# --------------------------------------------------------------------------- output
def _jsonable(obj: Any) -> Any:
    """Recursively make the payload JSON-safe: NaN/inf -> null, numpy -> python."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    return obj


def atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + os.replace so a crash never leaves a partial report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):  # pragma: no cover - only on a failed replace
            try:
                os.unlink(tmp)
            except OSError:
                pass


def fmt(value: Any, digits: int = 4) -> str:
    """Table cell: fixed decimals, ``n/a`` for missing/non-finite numbers."""
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def render_md(payload: dict[str, Any]) -> str:
    """Russian markdown report from the assembled payload (all numbers are real)."""
    ic = payload["ic_summary"]
    folds = payload["folds"]
    baselines = payload["baselines"]
    policy = payload["policy"]
    per_asset = payload["per_asset"]
    split = payload["splits"]
    lines: list[str] = []
    add = lines.append

    add("# Крипто-событийное исследование: модели и политика")
    add("")
    add(f"Сборка: {payload['generated_at']}. Бэкенд: `{payload['backend']}`.")
    add(
        f"Событий в датасете: {split['all']} (train {split['train']}, val {split['val']}; "
        "holdout пуст по построению — числа holdout не приводятся)."
    )
    add("")
    add(
        "Валидация: purged walk-forward, 5 расширяющихся фолдов **только по train-сплиту**, "
        "эмбарго 60 минут (строки, чей label-окно пересекает границу фолда, вычищены). "
        "Модели: квантильная регрессия LightGBM (alpha 0.1 / 0.5 / 0.9), гиперпараметры "
        "зафиксированы заранее; val не участвовал в выборе — только честное финальное "
        "считывание. Оценка ниже — на val-строках."
    )
    add("")
    add("## OOF IC (Spearman, квантиль 0.5)")
    add("")
    add("| Горизонт | IC средний | IC sd | Фолдов |")
    add("|---|---|---|---|")
    for h in HORIZONS:
        cell = ic[f"{h}m"]
        add(f"| {h}m | {fmt(cell['mean'])} | {fmt(cell['sd'])} | {cell['n_folds']} |")
    add("")
    add("## Фолды")
    add("")
    add("| Фолд | Train | Test | Вычищено | Test с | IC 5m | IC 15m | IC 60m |")
    add("|---|---|---|---|---|---|---|---|")
    for row in folds:
        add(
            f"| {row['fold']} | {row['train_rows']} | {row['test_rows']} | "
            f"{row['purged_rows']} | {row['test_start'][:10]} | {fmt(row['ic_q50_5m'])} | "
            f"{fmt(row['ic_q50_15m'])} | {fmt(row['ic_q50_60m'])} |"
        )
    add("")
    add(f"## Базлайны (val, L=1, hurdle {fmt(payload['hurdle_bps'], 0)} bps)")
    add("")
    add(
        "| Базлайн | Горизонт | Сделок | Muted | Gross bps | Net bps | "
        "t (дни) | Sharpe | MaxDD bps |"
    )
    add("|---|---|---|---|---|---|---|---|---|")
    for row in baselines:
        add(
            f"| {row['label']} | {row['horizon']} | {row['n_trades']} | "
            f"{fmt(100 * row['muted_share'], 1)}% | {fmt(row['gross_bps'], 2)} | "
            f"{fmt(row['net_bps'], 2)} | {fmt(row['t_cluster_days'], 2)} | "
            f"{fmt(row['sharpe_annual'], 2)} | {fmt(row['max_drawdown_bps'], 1)} |"
        )
    add("")
    add(
        f"## Политика (val): position = L·sign(q50) при |q50| ≥ {fmt(payload['hurdle_bps'], 0)} bps"
    )
    add("")
    add("| Горизонт | L | Сделок | Muted | Gross bps | Net bps | t (дни) | Sharpe | MaxDD bps |")
    add("|---|---|---|---|---|---|---|---|---|")
    for row in policy:
        add(
            f"| {row['horizon']} | {row['L']} | {row['n_trades']} | "
            f"{fmt(100 * row['muted_share'], 1)}% | {fmt(row['gross_bps'], 2)} | "
            f"{fmt(row['net_bps'], 2)} | {fmt(row['t_cluster_days'], 2)} | "
            f"{fmt(row['sharpe_annual'], 2)} | {fmt(row['max_drawdown_bps'], 1)} |"
        )
    add("")
    add("## Плечо — не источник alpha")
    add("")
    add(
        "**Плечо масштабирует и edge, и комиссии одним и тем же множителем:** "
        "net = L·(sign·excess − cost) = position·excess − |position|·cost. "
        "При отрицательном ожидании сделки результат при L=5 в 5 раз хуже, чем при L=1, "
        "а не лучше: плечо не может превратить отрицательное ожидание в положительное. "
        "Это не источник alpha, а способ пропорционально увеличить и доходность, и "
        "комиссии/риск. Все числа выше при L≠1 — это просто L× значений при L=1."
    )
    add("")
    add("## По активам (net bps на событие, L=1)")
    add("")
    symbols = sorted({row["symbol"] for row in per_asset})
    add("| Актив | n | 5m | 15m | 60m |")
    add("|---|---|---|---|---|")
    for sym in symbols:
        cells = []
        n_cell = ""
        for h in HORIZONS:
            hit = next(
                (
                    r
                    for r in per_asset
                    if r["symbol"] == sym and r["horizon"] == f"{h}m" and r["L"] == 1
                ),
                None,
            )
            cells.append(fmt(hit["net_bps"], 2) if hit else "n/a")
            if hit and not n_cell:
                n_cell = str(hit["n"])
        add(f"| {sym} | {n_cell} | {cells[0]} | {cells[1]} | {cells[2]} |")
    add("")
    add("## Вердикт по 14 bps")
    add("")
    verdict = payload["verdict"]
    add(
        f"Горизонты с положительным net_bps при L=1: {verdict['positive_horizons']}. "
        f"Максимальный gross_bps (за сделку): {fmt(verdict['max_gross_bps'], 2)} bps при "
        f"комиссии {fmt(payload['hurdle_bps'], 0)} bps. "
        f"Вывод: {verdict['text']}"
    )
    add("")
    add("## Что нельзя утверждать")
    add("")
    for item in payload["cannot_claim"]:
        add(f"- {item}")
    add("")
    return "\n".join(lines)


def print_console(payload: dict[str, Any]) -> None:
    """Required console output: OOF IC mean +/- sd and the policy table."""
    ic = payload["ic_summary"]
    parts = " | ".join(
        f"{h}m {ic[f'{h}m']['mean']:+.4f} +/- {ic[f'{h}m']['sd']:.4f}" for h in HORIZONS
    )
    print(f"OOF IC (Spearman, q50): {parts}")
    print(f"model backend: {payload['backend']}  eval split: val ({payload['splits']['val']} rows)")
    header = (
        f"{'horizon':>7} {'L':>2} {'n_trades':>8} {'muted':>7} "
        f"{'gross_bps':>10} {'net_bps':>9} {'t_days':>8} {'sharpe':>7} {'maxdd_bps':>9}"
    )
    print("policy table (val):")
    print(header)
    for row in payload["policy"]:
        print(
            f"{row['horizon']:>7} {row['L']:>2} {row['n_trades']:>8} "
            f"{100 * row['muted_share']:>6.1f}% {row['gross_bps']:>10.2f} "
            f"{row['net_bps']:>9.2f} {row['t_cluster_days']:>8.2f} "
            f"{row['sharpe_annual']:>7.2f} {row['max_drawdown_bps']:>9.1f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default=str(DEFAULT_EVENTS))
    parser.add_argument("--bars", default=str(DEFAULT_BARS_DIR))
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument("--md", default=str(DEFAULT_MD))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--embargo-minutes", type=int, default=60)
    parser.add_argument(
        "--legs",
        type=float,
        default=2.0,
        help="round trips per event: 2 for an excess label (a spread), 1 for an absolute return",
    )
    args = parser.parse_args(argv)

    events = read_events(Path(args.events))
    split_counts = events["split"].value_counts().to_dict()
    LOG.info("events: rows=%d splits=%s", len(events), split_counts)

    bars = read_bars(Path(args.bars), events)
    X, feature_names = build_features(events, bars)
    assert_no_future_features(events, bars, feature_names)
    print(f"causality guard: PASS ({len(feature_names)} features)")

    fold_rows, ic, oof = run_walk_forward(
        events, X, n_folds=args.folds, embargo_minutes=args.embargo_minutes
    )
    train_idx = events.index[events["split"] == "train"]
    val_idx = events.index[events["split"] == "val"]
    if len(val_idx) == 0:
        print("ERROR: val split is empty — no honest read-out is possible", file=sys.stderr)
        return 2

    fitted = fit_final_models(events, X, train_idx)
    pred50 = {
        h: np.asarray(fitted[(h, 0.5)].predict(X.loc[val_idx]), dtype="float64")
        for h in HORIZONS
    }
    baseline_rows = baseline_table(events, X, val_idx, legs=args.legs)
    policy_rows = policy_table(events, X, val_idx, pred50, legs=args.legs)
    asset_rows = per_asset_table(events, X, val_idx, pred50, legs=args.legs)

    hurdle_bps = float(np.nanmedian(events.loc[val_idx, "cost"])) * 1e4
    positive = [
        f"{h}m"
        for h in HORIZONS
        if next(r for r in policy_rows if r["horizon"] == f"{h}m" and r["L"] == 1)["net_bps"] > 0
    ]
    max_gross = max(r["gross_bps"] for r in policy_rows if np.isfinite(r["gross_bps"]))
    top_symbol = events.loc[val_idx, "symbol"].value_counts()
    top_name, top_share = str(top_symbol.index[0]), float(top_symbol.iloc[0] / len(val_idx))
    val_entry = pd.to_datetime(events.loc[val_idx, "entry_ts"], utc=True)
    val_days = int(val_entry.dt.normalize().nunique())

    payload: dict[str, Any] = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "backend": model_backend(),
        "events_path": str(args.events),
        "splits": {
            "all": int(len(events)),
            "train": int(split_counts.get("train", 0)),
            "val": int(split_counts.get("val", 0)),
            "holdout": "empty by construction - never reported",
        },
        "features": feature_names,
        "folds": fold_rows,
        "ic_summary": ic,
        "hurdle_bps": hurdle_bps,
        "baselines": baseline_rows,
        "policy": policy_rows,
        "per_asset": asset_rows,
        "eval_split": "val",
        "verdict": {
            "positive_horizons_L1": positive,
            "positive_horizons": ", ".join(positive) if positive else "нет",
            "max_gross_bps": max_gross,
            "text": (
                "ни один горизонт не покрывает 14 bps в среднем на сделку — "
                "нет горизонта с положительным net_bps при L=1"
                if not positive
                else "положительный net_bps при L=1 есть, но это одна валидационная "
                "выборка без holdout — как утверждение не принимается"
            ),
        },
        "cannot_claim": [
            f"Мало наблюдений: val — {len(val_idx)} события за {val_days} суток; "
            "эффективная размерность — это кластеры (сутки), а не строки.",
            "Нет holdout: holdout-сплит пуст по построению (0 строк) — внешней, ни разу "
            "не тронутой проверки результатов нет.",
            "Отбор дней на уровне суток: дни выборки отбирались из архива на уровне "
            "суток, это не случайная i.i.d. выборка событий.",
            f"Один актив доминирует: {top_name} — {100 * top_share:.1f}% событий val; "
            "результаты в основном про этот актив.",
            "Плечо не улучшает ожидание: оно масштабирует edge и комиссии одинаково "
            "(см. раздел про плечо).",
            "IC и политика оценены на одних и тех же val-строках; гиперпараметры не "
            "подбирались на val, но и не проверялись на независимой выборке.",
        ],
    }

    blob = json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n"
    atomic_write(Path(args.json), blob)
    atomic_write(Path(args.md), render_md(payload))
    print_console(payload)
    print(f"wrote {args.json}")
    print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
