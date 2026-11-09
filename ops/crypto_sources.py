# ruff: noqa: E501  - prints report tables
"""How much is each news source worth?

The question the roadmap asks is whether we can learn which sites to trust. With a
historical corpus and no live first_seen clock, the only honest answer is:

* per-source mean excess, shrunk toward the global mean because a source with 40 events
  has nothing but noise to contribute (James-Stein style, K = 50 events);
* stability: split the corpus by time at its median and ask whether the ranking in the
  first half predicts the ranking in the second half - that is the test "this source is
  reliably better", not the in-sample mean;
* concentration: a source whose whole mean comes from one day, or whose rows are almost
  all BTCUSDT, is measuring a day or the market, not the source.

Latency is deliberately absent: the Wayback corpus has `published_at` but no
`first_seen_at`, so the delay between publication and our awareness cannot be estimated
from it. That number only exists in the live recorder store.

Usage:
    python ops/crypto_sources.py [--events data/crypto/dataset/events.parquet]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = ROOT / "data" / "crypto" / "dataset" / "events.parquet"
RESULTS = ROOT / "results"
SHRINK_K = 50.0
MIN_N = 30
HORIZONS = ("5m", "15m", "60m")
COST_BPS = 14.0


def cluster_t(frame: pd.DataFrame, value: str, cluster: str = "day") -> float:
    """Cluster-robust t of the mean of ``value`` with clusters ``frame[cluster]``."""
    work = frame[[value, cluster]].dropna()
    if len(work) < 3:
        return float("nan")
    grouped = work.groupby(cluster)[value]
    counts = grouped.size()
    if len(counts) < 2:
        return float("nan")
    mean = work[value].mean()
    resid = grouped.sum() - mean * counts
    k = len(counts)
    variance = float((resid**2).sum()) / len(work) ** 2 * k / (k - 1)
    if variance <= 0:
        return float("nan")
    return float(mean / np.sqrt(variance))


def source_stats(events: pd.DataFrame, horizon: str, min_n: int = MIN_N) -> pd.DataFrame:
    """One row per source: counts, raw and shrunk means, concentration, stability."""
    column = f"excess_{horizon}"
    work = events.dropna(subset=[column, "source"]).copy()
    work["day"] = pd.to_datetime(work["entry_ts"]).dt.tz_convert("UTC").dt.date
    global_mean = float(work[column].mean() * 1e4)
    cutoff = work["published_at"].median()
    work["half"] = np.where(work["published_at"] < cutoff, "first", "second")
    btc_share = (work["symbol"] == "BTCUSDT").groupby(work["source"]).mean()

    rows = []
    for source, group in work.groupby("source"):
        n = len(group)
        mean_bps = float(group[column].mean() * 1e4)
        alpha = (n / (n + SHRINK_K)) * mean_bps + (SHRINK_K / (n + SHRINK_K)) * global_mean
        busiest = float(group.groupby("day").size().max() / n)
        halves = group.groupby("half")[column].agg(["size", "mean"])
        first = float(halves["mean"].get("first", np.nan)) * 1e4 if "first" in halves.index else float("nan")
        second = float(halves["mean"].get("second", np.nan)) * 1e4 if "second" in halves.index else float("nan")
        n_first = int(halves["size"].get("first", 0)) if "first" in halves.index else 0
        n_second = int(halves["size"].get("second", 0)) if "second" in halves.index else 0
        rows.append(
            {
                "source": source,
                "n": n,
                "mean_bps": mean_bps,
                "median_bps": float(group[column].median() * 1e4),
                "alpha_bps": alpha,
                "hit_rate": float((group[column] > 0).mean()),
                "sd_bps": float(group[column].std(ddof=1) * 1e4),
                "t": cluster_t(group, column),
                "busiest_day_share": busiest,
                "btc_share": float(btc_share.get(source, float("nan"))),
                "median_lag_min": float(group["lag_minutes"].median()),
                "n_first": n_first,
                "n_second": n_second,
                "first_bps": first,
                "second_bps": second,
                "reliable": bool(n >= min_n and n_first >= 15 and n_second >= 15),
            }
        )
    frame = pd.DataFrame(rows).sort_values("alpha_bps", ascending=False).reset_index(drop=True)
    frame.attrs["global_mean_bps"] = global_mean
    frame.attrs["cutoff"] = str(cutoff)
    return frame


def stability(frame: pd.DataFrame) -> dict[str, object]:
    """Does the first-half ranking predict the second-half ranking?"""
    work = frame[(frame["n_first"] >= 15) & (frame["n_second"] >= 15)]
    if len(work) < 3:
        return {"sources": int(len(work)), "spearman": float("nan"), "same_sign": 0}
    rho = float(work["first_bps"].corr(work["second_bps"], method="spearman"))
    same = int((np.sign(work["first_bps"]) == np.sign(work["second_bps"])).sum())
    return {"sources": int(len(work)), "spearman": rho, "same_sign": same}


def trust_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Trust in [0, 1] as the percentile rank of the shrunk alpha; 0 = worst source."""
    out = frame.copy()
    out["trust"] = out["alpha_bps"].rank(pct=True)
    out.loc[~out["reliable"], "trust"] = np.nan
    return out


def render(frame: pd.DataFrame, horizon: str, cut: str) -> str:
    head = (
        f"### {horizon} (n={int(frame['n'].sum())}, global mean "
        f"{frame.attrs['global_mean_bps']:+.2f} bps, split at {cut[:19]})\n\n"
        "| источник | n | mean бп | alpha бп | hit | t | доля 1 дня | BTC | lag мин | 1-я пол. | 2-я пол. | надёжен |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['source']} | {row['n']} | {row['mean_bps']:+.2f} | {row['alpha_bps']:+.2f} | "
            f"{row['hit_rate']:.3f} | {row['t']:+.2f} | {row['busiest_day_share']:.2f} | "
            f"{row['btc_share']:.2f} | {row['median_lag_min']:.0f} | {row['first_bps']:+.2f} | "
            f"{row['second_bps']:+.2f} | {'да' if row['reliable'] else 'нет'} |"
        )
    return head + "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default=str(DEFAULT_EVENTS))
    parser.add_argument("--out", default=str(RESULTS / "crypto_sources.json"))
    parser.add_argument("--report", default=str(RESULTS / "crypto_sources.md"))
    args = parser.parse_args()

    path = Path(args.events)
    if not path.is_file():
        print(f"missing dataset: {path}")
        return 2
    events = pd.read_parquet(path)
    if events.empty:
        print("empty dataset")
        return 2

    tables = {}
    stability_by_horizon = {}
    blocks = []
    for horizon in HORIZONS:
        frame = source_stats(events, horizon)
        stable = stability(frame)
        trusted = trust_table(frame)
        tables[horizon] = trusted.to_dict(orient="records")
        stability_by_horizon[horizon] = stable
        blocks.append(render(trusted, horizon, frame.attrs["cutoff"]))
        print(render(trusted, horizon, frame.attrs["cutoff"]))

    payload = {
        "events": str(path),
        "rows": int(len(events)),
        "cost_bps": COST_BPS,
        "shrink_k": SHRINK_K,
        "min_n": MIN_N,
        "stability": stability_by_horizon,
        "horizons": tables,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    report = ["# Доверие к источникам: измерено на историческом корпусе\n"]
    report.append(
        f"Корпус: {len(events)} событий, {events['source'].nunique()} источников, "
        f"{events['symbol'].nunique()} инструментов, `published_at` "
        f"{events['published_at'].min()} … {events['published_at'].max()}.\n"
    )
    report.append(
        "Alpha — средняя доходность источника, стянутая к общему среднему с весом "
        f"K={SHRINK_K:.0f} событий: `alpha = n/(n+K)*mean + K/(n+K)*global`. "
        f"Общий ориентир — издержки {COST_BPS:.0f} бп на круг.\n"
    )
    report.extend(blocks)
    report.append("## Устойчивость\n")
    for horizon, stable in stability_by_horizon.items():
        report.append(
            f"* {horizon}: источников с двумя половинами ≥ 15 событий — {stable['sources']}, "
            f"Spearman между половинами {stable['spearman']:+.2f}, одинаковый знак у "
            f"{stable['same_sign']} из {stable['sources']}.\n"
        )
    report.append(
        "\n## Что нельзя утверждать\n\n"
        "* Ни один источник не проверен вне выборки: «доверие» здесь — это ранг in-sample\n"
        "  среднего, а не предсказание будущего качества.\n"
        "* Латентности в таблице нет. У Wayback-корпуса нет времени первого обнаружения, поэтому\n"
        "  задержка измеряется только живым рекордером и только вперёд.\n"
        "* Корпус стратифицирован: захваченные дни — те, что попали в архив, а не случайная\n"
        "  выборка дней; источник с большим n просто чаще попадал в снимки.\n"
        "* «Устойчивость» проверяет только порядок источников, а не то, что лучший источник\n"
        "  окупает 14 бп. Для этого нужен отдельный тест на прибыль, и он отрицателен\n"
        "  (`results/crypto_policy.md`).\n"
    )
    Path(args.report).write_text("".join(report), encoding="utf-8")
    print(f"wrote {args.out} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
