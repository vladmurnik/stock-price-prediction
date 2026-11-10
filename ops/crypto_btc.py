# ruff: noqa: E501  - prints a report table
"""Absolute BTC returns after BTC headlines: the volume the benchmark filter throws away.

Every pair whose subject *is* the benchmark is dropped from ``events.parquet``, because
``excess = ret - ret`` is identically zero. But those pairs are the largest single block of
the corpus (9,779 of 16,469 in the current harvest), and their label is simply the asset's
own return with a one-leg cost instead of two.

This script builds that alternative dataset with the frozen builder (``allow_benchmark``
keeps the benchmark rows) and measures the event study before any model touches it:

* mean, median and mean absolute return per horizon, in bps;
* the 95 % interval under two clusterings - by headline (one story, one move) and by day;
* what an unconditional long or short earns after one leg (14 bps);
* the same split by the 5-minute move *before* entry: continuation and reversal are
  different bets, and only one of them can be right.

The frame it writes keeps the exact ``events.parquet`` schema, with ``bench_<h> = 0`` and
``excess_<h> = ret_<h>`` so the whole training and policy pipeline runs unchanged, and with
``cost`` set to one leg. The benchmark-identity trick is deliberate and documented: an
absolute-return label is the benchmark-relative label with a zero benchmark.

Usage:
    python ops/crypto_btc.py
    python ops/crypto_train.py --events data/crypto/dataset/events_btc.parquet --legs 1
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spp.crypto import models
from spp.crypto.dataset import (
    CANONICAL_ASSETS,
    DATASET_COLUMNS,
    HORIZONS,
    build_dataset,
    load_bars,
    load_dataset_config,
    run_checks,
)
from spp.data.store import write_parquet_atomic

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEWS: tuple[str, ...] = (
    str(ROOT / "data" / "crypto" / "news" / "wayback.parquet"),
    str(ROOT / "data" / "crypto" / "news" / "recorder.parquet"),
)
BARS_DIR = ROOT / "data" / "crypto" / "bars" / "1m"
DEFAULT_OUT = ROOT / "data" / "crypto" / "dataset" / "events_btc.parquet"
RESULTS = ROOT / "results"
ONE_LEG_BPS = 14.0
MINUTE_NS = 60 * 10**9


def read_news(patterns: list[str]) -> pd.DataFrame:
    """Every matched parquet read once; a file being rewritten is skipped, not trusted."""
    frames: list[pd.DataFrame] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            try:
                frames.append(pd.read_parquet(path))
            except Exception as exc:  # noqa: BLE001 - the harvest may swap the file
                print(f"news: skipping {path}: {exc}")
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True, sort=False)
    print(f"news: {len(frame)} rows from {len(frames)} file(s)")
    return frame


def cluster_ci(frame: pd.DataFrame, value: str, key: str) -> tuple[float, float, int]:
    """Mean, 95 % interval and cluster count for ``value`` clustered on ``key``."""
    work = frame[[value, key]].dropna()
    if len(work) < 3:
        return float("nan"), float("nan"), int(len(work))
    groups = work.groupby(key)[value]
    k = int(groups.ngroups)
    if k < 2:
        return float(work[value].mean()), float("nan"), k
    mean = float(work[value].mean())
    resid = groups.sum() - mean * groups.size()
    variance = float((resid**2).sum()) / len(work) ** 2 * k / (k - 1)
    half = 1.96 * math.sqrt(variance) if variance > 0 else float("nan")
    return mean, half, k


def describe(events: pd.DataFrame) -> list[dict[str, Any]]:
    """One row per horizon: the event study of the absolute return."""
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        column = f"ret_{horizon}m"
        value = events[column].dropna() * 1e4
        if value.empty:
            continue
        mean, half_title, n_title = cluster_ci(events, column, "title")
        _, half_day, n_day = cluster_ci(events, column, "day")
        se_day = half_day / 1.96 if half_day == half_day else float("nan")
        rows.append(
            {
                "horizon": f"{horizon}m",
                "n": int(value.size),
                "mean_bps": float(mean * 1e4),
                "median_bps": float(value.median()),
                "mean_abs_bps": float(value.abs().mean()),
                "share_up": float((value > 0).mean()),
                "ci95_title_bps": float(half_title * 1e4),
                "title_clusters": n_title,
                "ci95_day_bps": float(half_day * 1e4),
                "day_clusters": n_day,
                "mde80_bps": float(2.802 * se_day * 1e4) if se_day == se_day else float("nan"),
                "net_long_bps": float(mean * 1e4 - ONE_LEG_BPS),
                "net_short_bps": float(-mean * 1e4 - ONE_LEG_BPS),
            }
        )
    return rows


def by_prior_move(events: pd.DataFrame, features: pd.DataFrame) -> list[dict[str, Any]]:
    """Absolute return conditioned on the 5-minute move strictly before entry."""
    prior = features["tr_ret_5m"].to_numpy(dtype="float64")
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        column = f"ret_{horizon}m"
        value = events[column].to_numpy(dtype="float64")
        ok = np.isfinite(value) & np.isfinite(prior)
        for label, mask in (("prior up", ok & (prior > 0)), ("prior down", ok & (prior < 0))):
            if not mask.any():
                continue
            bps = value[mask] * 1e4
            rows.append(
                {
                    "horizon": f"{horizon}m",
                    "group": label,
                    "n": int(mask.sum()),
                    "mean_bps": float(bps.mean()),
                    "share_up": float((bps > 0).mean()),
                    "net_long_bps": float(bps.mean() - ONE_LEG_BPS),
                    "net_short_bps": float(-bps.mean() - ONE_LEG_BPS),
                }
            )
    return rows


def render(rows: list[dict[str, Any]], conditional: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = [
        "# BTC-новости абсолютной доходностью: измерено\n\n",
        f"Событий: {meta['n_events']} (это пары, которые фильтр `benchmark_asset` выбрасывает из "
        f"основного датасета). Окно: {meta['first']} … {meta['last']}. Издержки: одна нога, "
        f"{ONE_LEG_BPS:.0f} бп.\n\n",
        "## Безусловное событийное исследование\n\n",
        "| горизонт | n | средний | медиана | средний \\|ret\\| | доля роста | CI95 по заголовкам | CI95 по дням | MDE80 | net long | net short |\n",
        "|---|---|---|---|---|---|---|---|---|---|---|\n",
    ]
    for row in rows:
        lines.append(
            f"| {row['horizon']} | {row['n']} | {row['mean_bps']:+.2f} | {row['median_bps']:+.2f} | "
            f"{row['mean_abs_bps']:.2f} | {row['share_up']:.3f} | ±{row['ci95_title_bps']:.2f} "
            f"({row['title_clusters']}) | ±{row['ci95_day_bps']:.2f} ({row['day_clusters']}) | "
            f"{row['mde80_bps']:.2f} | {row['net_long_bps']:+.2f} | {row['net_short_bps']:+.2f} |\n"
        )
    lines.append(
        "\n## Условие: движение за 5 минут до входа\n\n"
        "| горизонт | группа | n | средний | доля роста | net long | net short |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    for row in conditional:
        lines.append(
            f"| {row['horizon']} | {row['group']} | {row['n']} | {row['mean_bps']:+.2f} | "
            f"{row['share_up']:.3f} | {row['net_long_bps']:+.2f} | {row['net_short_bps']:+.2f} |\n"
        )
    lines.append(
        "\n`net long`/`net short` — что осталось бы после 14 бп, если бы сторону выбирали "
        "безошибочно по этой таблице; это не стратегия, а верхняя граница для правила, "
        "которое знает только знак прошлого движения.\n"
    )
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news", action="append", default=None)
    parser.add_argument("--bars", default=str(BARS_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--json", default=str(RESULTS / "crypto_btc.json"))
    parser.add_argument("--md", default=str(RESULTS / "crypto_btc.md"))
    args = parser.parse_args(argv)

    news = read_news(args.news or list(DEFAULT_NEWS))
    if news.empty:
        print("no news: nothing to build")
        return 2

    config = load_dataset_config()
    config["costs"] = {**config["costs"], "round_trip_perp_bps": ONE_LEG_BPS}
    bench = str(config["labels"]["excess_vs"])
    symbols = sorted(set(CANONICAL_ASSETS.values()) | {bench})
    bars = load_bars(args.bars, symbols)

    frame, funnel = build_dataset(news, bars, config, allow_benchmark=True)
    print("funnel: " + ", ".join(f"{k}={v}" for k, v in funnel.items()))

    btc = frame.loc[frame["symbol"] == bench].copy()
    if btc.empty:
        print("no benchmark rows: nothing to measure")
        return 2
    btc["day"] = pd.to_datetime(btc["entry_ts"], utc=True).dt.date
    btc = btc.sort_values("entry_ts").reset_index(drop=True)

    checks = run_checks(frame, bars)
    failed = [name for name, value in checks.items() if isinstance(value, bool) and not value]
    print("checks: " + ("all PASS" if not failed else f"FAIL {failed}"))
    if failed:
        return 2

    features, feature_names = models.build_features(btc, bars)
    print(f"features: {len(feature_names)} columns on {len(btc)} BTC rows")

    rows = describe(btc)
    conditional = by_prior_move(btc, features)
    for row in rows:
        print(
            f"{row['horizon']:>5} n={row['n']:5d} mean {row['mean_bps']:+.2f} "
            f"median {row['median_bps']:+.2f} |ret| {row['mean_abs_bps']:5.2f} "
            f"CI_title ±{row['ci95_title_bps']:5.2f} CI_day ±{row['ci95_day_bps']:5.2f} "
            f"net long {row['net_long_bps']:+.2f}"
        )

    meta = {
        "n_events": int(len(btc)),
        "first": str(btc["published_at"].min()),
        "last": str(btc["published_at"].max()),
        "sources": btc["source"].value_counts().to_dict(),
        "cost_bps": ONE_LEG_BPS,
        "symbol": bench,
    }

    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "meta": meta,
        "funnel": funnel,
        "event_study": rows,
        "by_prior_move": conditional,
        "features": feature_names,
        "out": str(args.out),
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    Path(args.md).write_text(render(rows, conditional, meta), encoding="utf-8")

    # Absolute-return view of the same rows: the benchmark is zero, so excess == ret.
    out = btc.drop(columns=["day"]).copy()
    for horizon in HORIZONS:
        out[f"bench_{horizon}m"] = 0.0
        out[f"excess_{horizon}m"] = out[f"ret_{horizon}m"]
    out = out[list(DATASET_COLUMNS)]
    write_parquet_atomic(out, Path(args.out))
    print(f"dataset: {len(out)} rows -> {args.out}")
    print(f"wrote {args.json} and {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
