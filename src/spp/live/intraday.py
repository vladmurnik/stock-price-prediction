"""Orchestration for the intraday reaction experiment: bars in, verdict out.

This module exists so that the experiment can be rerun by one command and audited by reading one
report. It deliberately keeps three layers apart:

* :mod:`spp.live.minutes` decides what a bar is and whether the provider actually serves the
  history a claim needs;
* :mod:`spp.live.events` decides what one trade is, including the five-minute lag invariant;
* :mod:`spp.backtest.intraday_eval` decides whether the resulting sample is large enough to answer
  anything and what the answer is net of costs.

None of them may borrow a number from another. In particular the report never states a mean
without the count, the cluster-corrected standard error, the cost and the minimum detectable
effect - because "average excess return of 9 bps" means one thing with n=40,000 and is noise with
n=120.

The trend context the user asked for ("учитывая общий тренд") is implemented as a strictly lagged
daily filter: a session is ``trend_up`` when the previous *completed* session closed above its
50-session mean. It is computed from the daily cache, is shifted by one session, and is therefore
knowable before any bar of the event session exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from spp.backtest.intraday_eval import (
    evaluate_by_horizon,
    minimum_detectable_effect,
)
from spp.live.events import DropStats, build_events
from spp.live.minutes import fetch_minute_bars

__all__ = [
    "DEFAULT_INTRADAY",
    "attach_trend",
    "build_panel",
    "evaluate_panel",
    "load_event_news",
    "load_intraday_config",
    "write_report",
]

DEFAULT_INTRADAY: dict[str, Any] = {
    "interval": "60m",
    "lag_minutes": 5,
    "horizons": [60, 120, 180],
    "cost_bps": 5.0,
    "spread_bps": 5.0,
    "dead_zone": 0.0,
    "dedupe_hours": 6.0,
    "benchmark": "SPY",
    "trend_window": 50,
    "min_events_per_date": 5,
}


@dataclass(frozen=True)
class Panel:
    events: pd.DataFrame
    stats: DropStats
    n_articles: int
    n_dates: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_articles": self.n_articles,
            "n_events": int(len(self.events)),
            "n_dates": self.n_dates,
            "funnel": self.stats.as_dict(),
        }


def load_intraday_config(path: Path | str | None = None) -> dict[str, Any]:
    """Defaults overlaid with a YAML file; unknown keys are kept so a new knob needs no code."""
    config = dict(DEFAULT_INTRADAY)
    if path is None:
        return config
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"intraday config not found: {target}")
    loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{target} must contain a mapping, got {type(loaded).__name__}")
    config.update({k: v for k, v in loaded.items() if not k.startswith("_")})
    config["horizons"] = [int(h) for h in config["horizons"]]
    return config


def build_panel(
    news: pd.DataFrame,
    *,
    cache_dir: Path | str,
    config: dict[str, Any] | None = None,
    refresh_bars: bool = False,
    start: Any = None,
    end: Any = None,
) -> Panel:
    """Fetch bars for exactly the tickers and sessions the news needs, then build events."""
    cfg = {**DEFAULT_INTRADAY, **(config or {})}
    interval = str(cfg["interval"])
    tickers = sorted(set(news["ticker"].dropna().astype(str)))
    benchmark = str(cfg["benchmark"])
    if benchmark not in tickers:
        tickers = [*tickers, benchmark]
    stamps = pd.to_datetime(news["published_at"], utc=True, errors="coerce").dropna()
    if stamps.empty:
        raise ValueError("no usable published_at values; the event clock is undefined")
    window_start = pd.Timestamp(start) if start is not None else stamps.min()
    window_end = pd.Timestamp(end) if end is not None else stamps.max()
    bars = fetch_minute_bars(
        tickers,
        window_start.date(),
        window_end.date(),
        interval=interval,
        cache_dir=cache_dir,
        refresh=refresh_bars,
    )
    events, stats = build_events(
        news,
        bars,
        lag_minutes=int(cfg["lag_minutes"]),
        horizons=tuple(int(h) for h in cfg["horizons"]),
        spy=benchmark,
        dead_zone=float(cfg["dead_zone"]),
        dedupe_hours=float(cfg["dedupe_hours"]),
    )
    dates = int(events["session"].nunique()) if not events.empty else 0
    return Panel(events=events, stats=stats, n_articles=int(len(news)), n_dates=dates)


def load_event_news(
    data_dir: Path | str,
    *,
    backend: str = "finbert",
    start: Any = None,
    end: Any = None,
) -> pd.DataFrame:
    """News rows restricted to a window and joined to one sentiment backend.

    The directional score is ``p_pos - p_neg``: bounded in [-1, 1], no calibration needed because
    the event study only uses a sign and a ranking. A missing sentiment row yields score 0 and the
    event stays directionless rather than being silently dropped or imputed as positive.
    """
    from spp.data.store import read_parquet

    root = Path(data_dir)
    news = read_parquet(root / "interim" / "news_raw.parquet")
    news["published_at"] = pd.to_datetime(news["published_at"], utc=True, errors="coerce")
    if start is not None:
        news = news.loc[news["published_at"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        news = news.loc[news["published_at"] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    path = root / "processed" / f"article_sentiment_{backend}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; score the corpus with the {backend} backend first"
        )
    sentiment = read_parquet(path)
    keep = ["article_id", "ticker", "p_pos", "p_neg"]
    merged = news.merge(
        sentiment.loc[:, [c for c in keep if c in sentiment.columns]],
        on=["article_id", "ticker"],
        how="left",
        validate="many_to_one",
    )
    merged["score"] = merged["p_pos"].fillna(0.0) - merged["p_neg"].fillna(0.0)
    return merged


def attach_trend(
    events: pd.DataFrame, prices: pd.DataFrame, *, window: int = 50
) -> pd.DataFrame:
    """Add ``trend_up`` from *completed* sessions only.

    ``prices`` is the daily frame (``ticker``, ``date``, ``adj_close``/``close``). The average is
    rolled and then shifted by one session, so the flag attached to session *t* uses data up to
    *t-1* and nothing from the session being traded.
    """
    if events.empty:
        return events.assign(trend_up=pd.Series(dtype="boolean"))
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    wide = (
        prices.assign(date=pd.to_datetime(prices["date"]).dt.date)
        .pivot_table(index="date", columns="ticker", values=price_col, aggfunc="last")
        .sort_index()
    )
    mean = wide.rolling(int(window), min_periods=int(window) // 2).mean().shift(1)
    above = (wide > mean).stack(future_stack=True)
    lookup = pd.DataFrame({"trend_up": above}).reset_index()
    lookup.columns = ["session", "ticker", "trend_up"]
    merged = events.copy()
    merged["session"] = pd.to_datetime(merged["session"]).dt.date
    merged = merged.merge(lookup, on=["session", "ticker"], how="left")
    return merged


def evaluate_panel(
    events: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
    prices: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Per-horizon metrics, plus the trend split when the daily filter could be attached."""
    cfg = {**DEFAULT_INTRADAY, **(config or {})}
    if events.empty:
        return {"n_events": 0, "note": "no events; nothing to evaluate"}
    work = events
    if prices is not None:
        work = attach_trend(work, prices, window=int(cfg["trend_window"]))
    report: dict[str, Any] = {
        "config": {
            "lag_minutes": int(cfg["lag_minutes"]),
            "horizons": [int(h) for h in cfg["horizons"]],
            "cost_bps": float(cfg["cost_bps"]),
            "spread_bps": float(cfg["spread_bps"]),
            "interval": str(cfg["interval"]),
        },
        "n_events": int(len(work)),
        "n_dates": int(work["session"].nunique()),
        "funnel": {},
        "horizons": evaluate_by_horizon(
            work,
            cost_bps=float(cfg["cost_bps"]),
            spread_bps=float(cfg["spread_bps"]),
            min_events_per_date=int(cfg["min_events_per_date"]),
        ),
    }
    if "trend_up" in work.columns and work["trend_up"].notna().any():
        split: dict[str, Any] = {}
        for name, mask in (
            ("trend_up", work["trend_up"] == True),  # noqa: E712 - nullable boolean
            ("trend_down", work["trend_up"] == False),  # noqa: E712
        ):
            subset = work.loc[mask.fillna(False)]
            split[name] = evaluate_by_horizon(
                subset,
                cost_bps=float(cfg["cost_bps"]),
                spread_bps=float(cfg["spread_bps"]),
                min_events_per_date=int(cfg["min_events_per_date"]),
            ) if not subset.empty else {"n_events": 0}
        report["by_trend"] = split

    for _horizon_key, block in report["horizons"].items():
        sigma = block.get("std_y")
        if sigma is None or not np.isfinite(sigma):
            continue
        block["mde"] = minimum_detectable_effect(
            float(sigma),
            float(block["n_events"]),
            inflation=float(block.get("cluster", {}).get("inflation") or 1.0),
        )
    entry = pd.to_datetime(work["entry_ts"], utc=True)
    published = pd.to_datetime(work["published_at"], utc=True)
    report["lag_violations"] = int((entry < published).sum())
    report["truncated_share"] = float(work["truncated"].mean()) if "truncated" in work else 0.0
    return report


def write_report(report: dict[str, Any], path: Path | str) -> Path:
    """Markdown that cannot be read as more than the sample supports."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Внутридневная реакция на новости: первый прогон",
        "",
        "Все числа сгенерированы `ops/intraday_build.py` из реальных баров; ни одно не введено",
        "руками. Событие = одна статья, вход по открытию первого бара не раньше",
        "`published_at + "
        f"{report.get('config', {}).get('lag_minutes', 5)} мин`, выход через горизонт",
        "или по закрытию сессии, цель = доходность минус SPY за те же метки.",
        "",
        f"- событий: **{report.get('n_events', 0)}**, сессий: **{report.get('n_dates', 0)}**",
        f"- нарушений лага: **{report.get('lag_violations', 'n/a')}** (должно быть 0)",
        f"- доля усечённых выходов: {report.get('truncated_share', 0.0):.3f}",
        "",
        "| горизонт | событий | mean excess | net | hit rate | cluster t | MDE | IC событий |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, block in sorted(report.get("horizons", {}).items()):
        if not block or "mean_y" not in block:
            lines.append(f"| {key} | {block.get('n_events', 0)} | — | — | — | — | — | — |")
            continue
        cluster = block.get("cluster", {})
        ic = block.get("event_ic", {}).get("mean", float("nan"))
        lines.append(
            "| {h} | {n} | {mean:+.4%} | {net:+.4%} | {hit:.3f} "
            "| {t:+.2f} | {mde:.4%} | {ic:+.4f} |".format(
                h=key,
                n=block["n_events"],
                mean=block["mean_y"],
                net=block["net_mean"],
                hit=block["hit_rate_up"],
                t=cluster.get("t", float("nan")),
                mde=block.get("mde", float("nan")),
                ic=ic,
            )
        )
    lines += [
        "",
        "**Как читать.** `net` — средний результат правила после `2 × cost + spread`; если он",
        "не положителен, правило не торгуется. `MDE` — эффект, который выборка такого размера",
        "вообще способна заметить при 5 % / 80 %; mean меньше MDE означает «данных не хватает», а",
        "не «эффекта нет».",
    ]
    if report.get("by_trend"):
        lines += [
            "",
            "## Разбивка по дневному тренду",
            "",
            "| режим | горизонт | событий | net | t |",
            "|---|---|---|---|---|",
        ]
        for mode, block in report["by_trend"].items():
            for key, sub in sorted(block.items()):
                if not isinstance(sub, dict) or "net_mean" not in sub:
                    continue
                lines.append(
                    f"| {mode} | {key} | {sub['n_events']} | {sub['net_mean']:+.4%} | "
                    f"{sub.get('cluster', {}).get('t', float('nan')):+.2f} |"
                )
    text = "\n".join(lines) + "\n"
    target.write_text(text, encoding="utf-8")
    return target
