"""End-to-end intraday run on the GDELT pilot corpus.

The pilot exists because the two free windows do not overlap: FNSPID ends 2023-12-18 and yfinance
60-minute bars begin 2024-11-11. This script therefore takes news from GDELT, scores it with the
same FinBERT backend the daily study used, builds the event panel and evaluates it.

Two honest constraints are printed next to every number:

* the pilot is small (about a hundred articles), so the minimum detectable effect is large and a
  null result means "not measurable", not "no effect";
* GDELT attributes a headline to a ticker only when the headline names the company, and 84 % of
  raw hits are dropped for that reason - the funnel below reports the real counts.

Usage:
    python ops/intraday_gdelt.py --news data/interim/news_gdelt_pilot.parquet \
        --start 2026-10-01 --end 2026-10-23
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from spp.config import load_config
from spp.data.prices import load_prices
from spp.data.store import read_parquet
from spp.live.intraday import build_panel, evaluate_panel, load_intraday_config, write_report
from spp.logging import get_logger

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

LOG = get_logger("ops.intraday_gdelt")


def _score_missing(news: pd.DataFrame, backend_name: str = "finbert") -> pd.DataFrame:
    """Score exactly the headlines that have no numbers yet, reusing the project's backend."""
    from spp.sentiment.score_corpus import build_backend

    work = news.copy()
    for column in ("p_pos", "p_neg", "p_neu"):
        if column not in work.columns:
            work[column] = float("nan")
    missing = work["p_pos"].isna()
    if not missing.any():
        return work
    texts = work.loc[missing, "title"].astype(str).tolist()
    LOG.info("scoring %d headlines with %s", len(texts), backend_name)
    backend = build_backend(backend_name, config=load_config())
    scored = backend.score(texts)
    index = work.index[missing]
    work.loc[index, "p_pos"] = [float(item.p_pos) for item in scored]
    work.loc[index, "p_neg"] = [float(item.p_neg) for item in scored]
    work.loc[index, "p_neu"] = [float(item.p_neu) for item in scored]
    work["score"] = work["p_pos"].fillna(0.0) - work["p_neg"].fillna(0.0)
    return work


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news", required=True, help="Parquet written by ops/fetch_gdelt.py.")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--backend", default="finbert")
    parser.add_argument("--config", default="configs/intraday.yaml")
    parser.add_argument("--report", default="reports/intraday_gdelt_pilot.md")
    parser.add_argument("--no-trend", action="store_true")
    args = parser.parse_args()

    config = load_intraday_config(ROOT / args.config)
    news = read_parquet(ROOT / args.news)
    news["published_at"] = pd.to_datetime(news["published_at"], utc=True, errors="coerce")
    if args.start:
        news = news.loc[news["published_at"] >= pd.Timestamp(args.start, tz="UTC")]
    if args.end:
        cutoff = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)
        news = news.loc[news["published_at"] < cutoff]
    if news.empty:
        raise SystemExit("no articles in the requested window")
    tickers = sorted(news["ticker"].dropna().astype(str).unique())
    LOG.info("pilot: %d articles over %d tickers", len(news), len(tickers))

    news = _score_missing(news, args.backend)
    panel = build_panel(
        news,
        cache_dir=DATA / "cache" / "minutes",
        config=config,
        start=args.start,
        end=args.end,
    )
    print(json.dumps(panel.as_dict(), indent=2, ensure_ascii=False))
    if panel.events.empty:
        print("no events: the pilot did not survive the funnel")
        return 2

    prices = None
    if not args.no_trend:
        try:
            prices = load_prices(DATA / "cache", pd.Timestamp(args.start), pd.Timestamp(args.end))
        except Exception as exc:  # noqa: BLE001 - the trend split is optional and reported
            LOG.warning("trend filter unavailable: %s", exc)

    report = evaluate_panel(panel.events, config=config, prices=prices)
    report["funnel"] = panel.stats.as_dict()
    report["source"] = {"news": args.news, "start": args.start, "end": args.end}
    (REPORTS / "intraday_gdelt_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    events_path = DATA / "processed" / "intraday_gdelt_events.parquet"
    panel.events.to_parquet(events_path, index=False)
    report_path = write_report(report, ROOT / args.report)

    for key, block in sorted(report["horizons"].items()):
        if block.get("mean_y") is None:
            continue
        print(
            f"{key}: n={block['n_events']} mean={block['mean_y']:+.4%} "
            f"net={block['net_mean']:+.4%} mde={block.get('mde', float('nan')):.4%} "
            f"t={block.get('cluster', {}).get('t', float('nan')):+.2f}"
        )
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
