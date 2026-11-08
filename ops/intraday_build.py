"""Build and evaluate the intraday event panel from the news corpus and cached bars.

Usage:
    python ops/intraday_build.py --start 2023-12-11 --end 2023-12-29
    python ops/intraday_build.py --start 2023-12-11 --end 2023-12-29 --refresh-bars
    python ops/intraday_build.py --backend finbert            # sentiment that defines direction

The script refuses to publish a report when the sample cannot answer the question it prints: the
minimum detectable effect is computed from the realised dispersion, so a run over six sessions
produces a report that says exactly that instead of a number that looks like a finding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from spp.data.prices import load_prices
from spp.data.store import read_parquet
from spp.live.intraday import build_panel, evaluate_panel, load_intraday_config, write_report
from spp.logging import get_logger

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

LOG = get_logger("ops.intraday")


def _news_with_score(backend: str, start: str | None, end: str | None) -> pd.DataFrame:
    news = read_parquet(DATA / "interim" / "news_raw.parquet")
    news["published_at"] = pd.to_datetime(news["published_at"], utc=True, errors="coerce")
    if start:
        news = news.loc[news["published_at"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        news = news.loc[news["published_at"] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    path = DATA / "processed" / f"article_sentiment_{backend}.parquet"
    if not path.is_file():
        raise SystemExit(f"missing {path}; run `spp score corpus --backend {backend}`")
    sentiment = read_parquet(path)
    keep = ["article_id", "ticker", "p_pos", "p_neg", "p_neu"]
    sentiment = sentiment.loc[:, [c for c in keep if c in sentiment.columns]]
    merged = news.merge(sentiment, on=["article_id", "ticker"], how="left", validate="many_to_one")
    # The directional score is the probability mass difference; it is bounded in [-1, 1] and needs
    # no calibration, because the event study only asks for a sign and a ranking.
    merged["score"] = merged["p_pos"].fillna(0.0) - merged["p_neg"].fillna(0.0)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None, help="First publication timestamp (UTC date).")
    parser.add_argument("--end", default=None, help="Last publication timestamp (UTC date).")
    parser.add_argument("--backend", default="finbert", choices=["finbert", "openjev", "lexicon"])
    parser.add_argument("--config", default="configs/intraday.yaml")
    parser.add_argument("--refresh-bars", action="store_true", help="Re-download bars.")
    parser.add_argument("--no-trend", action="store_true", help="Skip the daily trend split.")
    args = parser.parse_args()

    config = load_intraday_config(ROOT / args.config)
    news = _news_with_score(args.backend, args.start, args.end)
    LOG.info("articles in window: %d over %d tickers", len(news), news["ticker"].nunique())

    panel = build_panel(
        news,
        cache_dir=DATA / "cache" / "minutes",
        config=config,
        refresh_bars=args.refresh_bars,
        start=args.start,
        end=args.end,
    )
    funnel = panel.as_dict()
    print(json.dumps(funnel, indent=2, ensure_ascii=False))
    if panel.events.empty:
        print("no events: the window has no usable bars for these tickers")
        return 2

    prices = None
    if not args.no_trend:
        try:
            prices = load_prices(
                DATA / "cache", pd.Timestamp(args.start), pd.Timestamp(args.end)
            )
        except Exception as exc:  # noqa: BLE001 - the trend split is optional, absence is reported
            LOG.warning("trend filter unavailable: %s", exc)

    report = evaluate_panel(panel.events, config=config, prices=prices)
    report["funnel"] = funnel["funnel"]
    report["backend"] = args.backend
    report["window"] = {"start": args.start, "end": args.end}

    metrics_path = REPORTS / "intraday_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    events_path = DATA / "processed" / "intraday_events.parquet"
    panel.events.to_parquet(events_path, index=False)
    report_path = write_report(report, REPORTS / "intraday_baseline.md")

    for key, block in sorted(report["horizons"].items()):
        if not block.get("mean_y") is not None:
            continue
        print(
            f"{key}: n={block.get('n_events')} mean={block.get('mean_y', float('nan')):+.4%} "
            f"net={block.get('net_mean', float('nan')):+.4%} "
            f"mde={block.get('mde', float('nan')):.4%} "
            f"t={block.get('cluster', {}).get('t', float('nan')):+.2f}"
        )
    print(f"report: {report_path}")
    print(f"events: {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
