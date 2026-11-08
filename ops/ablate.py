"""Build ablation feature matrices without touching the frozen ``features.parquet``.

Each ablation gets its own file under ``data/processed/ablation/`` so the frozen matrix
used for the 2023 holdout never changes:

* ``a3-openjev.parquet``   — article sentiment from open-jev instead of FinBERT;
* ``a4-lexicon.parquet``   — lexicon backend (sanity floor);
* ``a6-strict-dates.parquet`` — ``signal_date`` recomputed with ``strict_dates = true``;
* ``a8-openjev-flags.parquet`` — open-jev matrix that also fills ``company_specific_share``
  and ``novelty_mean`` (the two columns FinBERT cannot provide).

Usage:
    python ops/ablate.py --only a4
    python ops/ablate.py --only a6
    python ops/ablate.py --only a3 a8
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

from spp.config import load_config
from spp.data.news_fnspid import _finalize_news
from spp.data.normalize import SignalPolicy
from spp.data.prices import load_prices
from spp.data.store import read_parquet
from spp.features.assemble import MARKET_FEATURES, build_feature_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "processed" / "ablation"

WATCH = (
    "sector_ret_1",
    "sent_mean",
    "pos_share",
    "neg_share",
    "novelty_mean",
    "company_specific_share",
)


def _universe_trading_days(prices: pd.DataFrame) -> list[date]:
    days = pd.to_datetime(prices["date"]).dt.date.unique()
    return sorted(days.tolist())


def _build(
    name: str,
    *,
    sentiment: pd.DataFrame,
    news: pd.DataFrame,
    config,
    prices: pd.DataFrame,
    start: date,
    end: date,
) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"ablate-{name}-"))
    try:
        result = build_feature_matrix(
            prices,
            sentiment=sentiment,
            news=news,
            config=config,
            data_dir=tmp,
            start=start,
            end=end,
            write=True,
        )
        matrix = result.features
    finally:
        produced = tmp / "processed" / "features.parquet"
        if produced.is_file():
            target = OUT / f"{name}.parquet"
            shutil.copyfile(produced, target)
        shutil.rmtree(tmp, ignore_errors=True)

    coverage = {col: float(matrix[col].notna().mean()) for col in WATCH if col in matrix.columns}
    print(f"[{name}] rows={len(matrix)} cols={len(matrix.columns)} -> {target}")
    print(f"[{name}] coverage: {coverage}")
    return target


def run_a4(config, prices, news, start, end) -> Path:
    path = DATA / "processed" / "article_sentiment_lexicon.parquet"
    if not path.is_file():
        raise SystemExit(f"missing {path}; run `spp score corpus --backend lexicon`")
    sentiment = read_parquet(path)
    print(f"[a4] sentiment rows={len(sentiment)} from {path.name}")
    return _build(
        "a4-lexicon",
        sentiment=sentiment,
        news=news,
        config=config,
        prices=prices,
        start=start,
        end=end,
    )


def _strict_news_and_sentiment(config, prices) -> tuple[pd.DataFrame, pd.DataFrame]:
    news = read_parquet(DATA / "interim" / "news_raw.parquet")
    policy = SignalPolicy.from_config(config.news.cutoff_et, True)
    news_strict = _finalize_news(news, policy, _universe_trading_days(prices))
    path = DATA / "processed" / "article_sentiment_finbert.parquet"
    sentiment = read_parquet(path)
    keys = news_strict.loc[:, ["article_id", "ticker", "signal_date"]].drop_duplicates(
        subset=["article_id", "ticker"]
    )
    base = sentiment.drop(columns=["signal_date"], errors="ignore")
    merged = base.merge(keys, on=["article_id", "ticker"], how="left", validate="many_to_one")
    print(
        f"[a6] strict signal_date: {merged['signal_date'].notna().mean():.3f} matched, "
        f"shifted rows={int((merged['signal_date'] != sentiment['signal_date']).sum())}"
    )
    return news_strict, merged


def run_a6(config, prices, start, end) -> Path:
    news_strict, sentiment = _strict_news_and_sentiment(config, prices)
    return _build(
        "a6-strict-dates",
        sentiment=sentiment,
        news=news_strict,
        config=config,
        prices=prices,
        start=start,
        end=end,
    )


def run_openjev(which: str, config, prices, news, start, end) -> Path:
    path = DATA / "processed" / "article_sentiment_openjev.parquet"
    if not path.is_file():
        raise SystemExit(
            f"missing {path}: open-jev scoring has not finished yet "
            "(`spp score corpus --backend openjev`)"
        )
    sentiment = read_parquet(path)
    print(f"[{which}] sentiment rows={len(sentiment)} from {path.name}")
    built = _build(
        f"{which}-openjev",
        sentiment=sentiment,
        news=news,
        config=config,
        prices=prices,
        start=start,
        end=end,
    )
    if which != "a3":
        return built
    # A3 is the sentiment-only ablation, so the market block is dropped from the matrix.
    matrix = read_parquet(built)
    dropped = [c for c in MARKET_FEATURES if c in matrix.columns]
    subset = matrix.drop(columns=dropped)
    target = OUT / "a3-openjev-sent.parquet"
    subset.to_parquet(target, index=False)
    print(f"[a3] dropped {len(dropped)} market columns -> {target} ({len(subset.columns)} cols)")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        default=["a4", "a6"],
        choices=["a4", "a6", "a3", "a8"],
        help="Which ablation matrices to build.",
    )
    args = parser.parse_args()

    config = load_config()
    start = pd.Timestamp(config.dates.train_start).date()
    end = pd.Timestamp(config.dates.test_end).date()
    prices = load_prices(DATA / "cache", start, end)
    print(f"prices rows={len(prices)} tickers={prices['ticker'].nunique()} {start}..{end}")
    news = read_parquet(DATA / "interim" / "news_raw.parquet")

    built: list[Path] = []
    for name in args.only:
        if name == "a4":
            built.append(run_a4(config, prices, news, start, end))
        elif name == "a6":
            built.append(run_a6(config, prices, start, end))
        else:
            built.append(run_openjev(name, config, prices, news, start, end))
    print("built:")
    for path in built:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
