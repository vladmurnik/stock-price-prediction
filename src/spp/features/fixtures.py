"""Synthetic fixtures with the exact T5/T2 schemas (no network, no model runs).

The production ``article_sentiment.parquet`` is produced by a multi-hour FinBERT run;
these fixtures let the feature layer and every leakage check be written and tested
against the *contract* in the meantime. Column names, dtypes and semantics match
``spp.sentiment.score_corpus.OUTPUT_COLUMNS`` and ``spp.data.prices.PRICE_COLUMNS``.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

__all__ = [
    "ARTICLE_SENTIMENT_COLUMNS",
    "NEWS_COLUMNS",
    "make_articles",
    "make_news",
    "make_prices",
    "make_sentiment",
]

#: ``article_sentiment.parquet`` columns in plan order (mirrors T5).
ARTICLE_SENTIMENT_COLUMNS: tuple[str, ...] = (
    "article_id",
    "ticker",
    "signal_date",
    "text_hash",
    "backend",
    "model_version",
    "p_neg",
    "p_neu",
    "p_pos",
    "score_5",
    "score_3",
    "confidence",
    "label_5",
    "is_company_specific",
    "is_novel",
    "skipped",
    "skip_reason",
)

NEWS_COLUMNS: tuple[str, ...] = (
    "article_id",
    "ticker",
    "published_at",
    "date",
    "signal_date",
    "title",
    "publisher",
    "text_hash",
)

_LEVELS = ("very negative", "negative", "neutral", "positive", "very positive")


def make_prices(
    tickers: tuple[str, ...] = ("AAA", "BBB", "CCC"),
    start: str = "2020-01-02",
    periods: int = 120,
    seed: int = 7,
    helpers: bool = True,
) -> pd.DataFrame:
    """Long OHLCV with ``adj_close`` and the exact ``PRICE_COLUMNS`` schema."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=periods)
    rows: list[dict[str, object]] = []
    all_tickers = list(tickers) + (["SPY", "^VIX", "XLK"] if helpers else [])
    for k, ticker in enumerate(all_tickers):
        base = 90.0 + 11.0 * k
        if ticker == "^VIX":
            values = 15.0 + np.cumsum(rng.normal(0, 0.3, size=len(dates)))
            values = np.clip(values, 9.0, 80.0)
        else:
            steps = rng.normal(0.0004, 0.011, size=len(dates))
            values = base * np.exp(np.cumsum(steps))
        for i, day in enumerate(dates):
            adj = float(values[i])
            rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "open": round(adj * 1.001, 4),
                    "high": round(adj * 1.01, 4),
                    "low": round(adj * 0.99, 4),
                    "close": round(adj, 4),
                    "adj_close": round(adj, 4),
                    "volume": int(1_000_000 + rng.integers(0, 500_000)),
                    "source": "synthetic",
                }
            )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["ticker"] = frame["ticker"].astype("string")
    frame["source"] = frame["source"].astype("string")
    return frame


def make_articles(
    tickers: tuple[str, ...] = ("AAA", "BBB", "CCC"),
    start: str = "2020-01-02",
    periods: int = 120,
    seed: int = 11,
    density: float = 0.6,
    published_hour: int | None = None,
) -> pd.DataFrame:
    """``article_sentiment`` fixture; ``published_hour=None`` yields a date-only feed."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start, periods=periods)
    rows: list[dict[str, object]] = []
    counter = 0
    for day in sessions:
        for ticker in tickers:
            n = int(rng.poisson(density))
            for _ in range(n):
                counter += 1
                score3 = float(rng.normal(0, 0.35))
                score5 = float(np.clip(score3 + rng.normal(0, 0.1), -1, 1))
                p_pos = float(np.clip(0.5 + score3 / 2, 0.01, 0.97))
                p_neg = float(np.clip((1 - p_pos) * 0.5, 0.005, 0.99))
                p_neu = float(max(0.005, 1 - p_pos - p_neg))
                label = _LEVELS[2]
                if score5 > 0.5:
                    label = _LEVELS[4]
                elif score5 > 0:
                    label = _LEVELS[3]
                elif score5 < -0.5:
                    label = _LEVELS[0]
                elif score5 < 0:
                    label = _LEVELS[1]
                if published_hour is None:
                    published = pd.Timestamp(day, tz="UTC")
                else:
                    minute = int(rng.integers(0, 60))
                    published = pd.Timestamp(day, tz="UTC") + pd.Timedelta(
                        hours=published_hour, minutes=minute
                    )
                rows.append(
                    {
                        "article_id": f"art{counter:07d}",
                        "ticker": ticker,
                        "signal_date": day.normalize(),
                        "text_hash": f"h{counter:07d}",
                        "backend": "synthetic",
                        "model_version": "fixture-v1",
                        "p_neg": round(p_neg, 6),
                        "p_neu": round(p_neu, 6),
                        "p_pos": round(p_pos, 6),
                        "score_5": round(score5, 6),
                        "score_3": round(score3, 6),
                        "confidence": round(float(max(p_neg, p_neu, p_pos)), 6),
                        "label_5": label,
                        "is_company_specific": bool(rng.random() > 0.4),
                        "is_novel": bool(rng.random() > 0.5),
                        "skipped": bool(rng.random() < 0.05),
                        "skip_reason": None,
                        "published_utc": published,
                        "publisher": f"pub{int(rng.integers(0, 6))}",
                    }
                )
    frame = pd.DataFrame(rows)
    frame["skipped"] = frame["skipped"].astype(bool)
    frame.loc[frame["skipped"], "skip_reason"] = "low_confidence"
    return frame


def make_news(articles: pd.DataFrame) -> pd.DataFrame:
    """``news_raw`` fixture derived from an articles frame (shares ``article_id``)."""
    frame = pd.DataFrame(
        {
            "article_id": articles["article_id"],
            "ticker": articles["ticker"],
            "published_at": articles["published_utc"],
            "date": pd.to_datetime(articles["signal_date"]),
            "signal_date": pd.to_datetime(articles["signal_date"]),
            "title": "fixture headline",
            "publisher": articles["publisher"],
            "text_hash": articles["text_hash"],
        }
    )
    return frame


def make_sentiment(articles: pd.DataFrame) -> pd.DataFrame:
    """Project an articles fixture to ``article_sentiment`` (drop news-only columns)."""
    frame = articles.loc[:, list(ARTICLE_SENTIMENT_COLUMNS)].copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.normalize()
    return frame.reset_index(drop=True)


def fixture_window(periods: int = 120, start: str = "2020-01-02") -> tuple[date, date]:
    sessions = pd.bdate_range(start, periods=periods)
    return sessions[0].date(), sessions[-1].date()
