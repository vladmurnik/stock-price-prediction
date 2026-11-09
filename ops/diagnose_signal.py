"""Is the sentiment signal real but small, or absent? One decisive diagnostic."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

F = pd.read_parquet(
    "data/processed/features.parquet",
    columns=["date", "ticker", "y_used_demeaned", "sent_mean", "ewma_24h"],
)
F = F.dropna(subset=["sent_mean", "y_used_demeaned"])


def ic_stats(col: str) -> tuple[float, float, int, float]:
    def one(group: pd.DataFrame) -> float:
        if len(group) <= 5:
            return np.nan
        return spearmanr(group[col], group["y_used_demeaned"]).statistic

    series = F.groupby("date").apply(one, include_groups=False).dropna()
    t_stat = series.mean() / (series.std(ddof=1) / np.sqrt(len(series)))
    return float(series.mean()), float(series.std(ddof=1)), int(len(series)), float(t_stat)


def quintile(col: str) -> tuple[float, float, int] | None:
    frame = F.dropna(subset=[col]).copy()
    counts = frame.groupby("date")[col].transform("count")
    frame = frame.loc[counts >= 20]
    if frame.empty:
        return None
    frame["q"] = frame.groupby("date")[col].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False)
    )
    spread = (
        frame.groupby(["date", "q"])["y_used_demeaned"].mean().unstack()
    )
    if 0 not in spread.columns or 4 not in spread.columns:
        return None
    gross = (spread[4] - spread[0]).dropna()
    if len(gross) < 10:
        return None
    t_stat = gross.mean() / (gross.std(ddof=1) / np.sqrt(len(gross)))
    return float(gross.mean()), float(t_stat), int(len(gross))


print("=" * 74)
print("daily rank IC of the raw feature (no model, 2019-2023, 40 tickers)")
for column in ("sent_mean", "ewma_24h"):
    mean_ic, sd, days, t_stat = ic_stats(column)
    print(
        f"  {column:10s} IC={mean_ic:+.4f}  sd={sd:.3f}  days={days}  "
        f"t={t_stat:+.2f}  ann_ir~{mean_ic / sd * np.sqrt(252):.2f}"
    )

print("=" * 74)
print("within-date quintile spread, equal weight, one-day hold")
for column in ("sent_mean", "ewma_24h"):
    result = quintile(column)
    if result is None:
        print(f"  {column}: not enough cross-section")
        continue
    gross_bps, t_stat, days = result
    gross_pct = gross_bps * 100.0
    net_daily = gross_bps - 0.0010
    print(
        f"  {column:10s} Q5-Q1 = {gross_bps * 1e4:+.1f} bps/day  t={t_stat:+.2f}  days={days}"
    )
    print(
        f"             gross {gross_pct * 252:+.1f}%/yr | net of 10 bps daily "
        f"turnover {net_daily * 252 * 100:+.1f}%/yr | breakeven {gross_bps * 1e4:.1f} bps/day"
    )

print("=" * 74)
print("how often the sign of the spread is right")
for column in ("sent_mean", "ewma_24h"):
    frame = F.dropna(subset=[column]).copy()
    counts = frame.groupby("date")[column].transform("count")
    frame = frame.loc[counts >= 20]
    frame["q"] = frame.groupby("date")[column].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False)
    )
    spread = frame.groupby(["date", "q"])["y_used_demeaned"].mean().unstack()
    gross = (spread[4] - spread[0]).dropna()
    print(
        f"  {column:10s} positive on {(gross > 0).mean():.1%} of {len(gross)} days, "
        f"median {gross.median() * 1e4:+.1f} bps"
    )
