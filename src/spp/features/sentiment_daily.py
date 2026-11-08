"""Daily sentiment aggregates ``(ticker, signal_date) -> features`` (T6).

Input: ``article_sentiment.parquet`` (T5) joined to ``news_raw.parquet`` on
``article_id`` for the publication timestamp and publisher. Only ``skipped == False``
articles are used (low-confidence / non-English rows were flagged by T5).

Temporal contract — every value attached to row ``(ticker, t)`` is computed from
articles with ``signal_date <= t`` only:

* Same-day features (``sent_*``, ``pos_share``, entropy ...) use ``signal_date == t``.
* The EWMA over publication *time* uses day-``t`` articles only; each article is
  weighted by its age at the end of the day. For a date-only feed all articles share
  a timestamp, so the weights are equal and the EWMA is the plain daily mean — the
  plan's explicit rule (``для date-only — дневное среднее``).
* Cross-day aggregates (``sent_sum_3d/5d/10d``, ``sent_mean_3d/5d/10d``,
  ``n_news_z``, ``sent_surprise``, ``sent_momentum_3d``) are shifted on the **calendar
  spine**, so a row at ``t`` reads ``[t-k, t-1]``. Shifting on a compacted *ticker*
  index would silently read ``t`` itself whenever the ticker had no news on some
  earlier day; the spine makes that impossible.

NaN policy: no news on day ``t`` means ``n_news = 0``, ``log_n_news = 0``, the time
EWMA carries forward, and every ``sent_*`` statistic is **NaN — never 0**. LightGBM
consumes NaN natively; linear baselines impute the training median (T7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spp.logging import get_logger

__all__ = [
    "SENTIMENT_FEATURES",
    "SENTIMENT_DAILY_COLUMNS",
    "SENT_MULTIDAY_FEATURES",
    "aggregate_sentiment_daily",
    "prepare_articles",
]

LOG = get_logger("features.sentiment_daily")

LEVELS: tuple[str, ...] = ("very negative", "negative", "neutral", "positive", "very positive")
MULTI_DAY_WINDOWS: tuple[int, ...] = (3, 5, 10)
ROLLING_DAYS = 30

SENTIMENT_DAILY_COLUMNS: tuple[str, ...] = (
    "ticker",
    "date",
    "n_news",
    "log_n_news",
    "n_news_z",
    "sent_mean",
    "sent_sum",
    "sent_std",
    "sent_min",
    "sent_max",
    "sent_median",
    "pos_share",
    "neg_share",
    "neu_share",
    "very_pos_share",
    "very_neg_share",
    "neg_extreme_share",
    "ewma_6h",
    "ewma_24h",
    "sent_surprise",
    "sent_momentum_3d",
    "company_specific_share",
    "novelty_mean",
    "conf_mean",
    "source_entropy",
    "top_source_share",
    "sent_sum_3d",
    "sent_mean_3d",
    "sent_sum_5d",
    "sent_mean_5d",
    "sent_sum_10d",
    "sent_mean_10d",
)

#: Day-``t`` statistics only. NaN when ``n_news == 0``.
SENT_DAY_FEATURES: tuple[str, ...] = (
    "sent_mean",
    "sent_sum",
    "sent_std",
    "sent_min",
    "sent_max",
    "sent_median",
    "pos_share",
    "neg_share",
    "neu_share",
    "very_pos_share",
    "very_neg_share",
    "neg_extreme_share",
    "sent_surprise",
    "sent_momentum_3d",
    "company_specific_share",
    "novelty_mean",
    "conf_mean",
    "source_entropy",
    "top_source_share",
)
#: Multi-day sums/means over ``[t-w, t-1]``; a quiet window yields exactly 0.
SENT_MULTIDAY_FEATURES: tuple[str, ...] = tuple(
    f"{base}_{w}d" for w in MULTI_DAY_WINDOWS for base in ("sent_sum", "sent_mean")
)
SENTIMENT_FEATURES: tuple[str, ...] = (
    (
        "n_news",
        "log_n_news",
        "n_news_z",
        "ewma_6h",
        "ewma_24h",
    )
    + SENT_DAY_FEATURES
    + SENT_MULTIDAY_FEATURES
)

#: Intermediate per-day aggregates the derivation step consumes.
_NUMERIC_DAY = (
    "sent_sum",
    "sent_min",
    "sent_max",
    "sent_median",
    "score5_sum",
    "sq5_sum",
    "conf_sum",
    "n_extreme_neg",
    "n_very_neg",
    "n_labelled",
    "cs_n",
    "cs_d",
    "nov_n",
    "nov_d",
)


# --------------------------------------------------------------------------- pure helpers
def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _top_share(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return float("nan")
    return float(counts.max() / total)


def _time_ewma(
    times: np.ndarray, values: np.ndarray, halflife_min: float, end: pd.Timestamp
) -> float:
    """Exponentially weighted mean with weights ``0.5 ** (age_min / halflife)``."""
    age_min = (end.to_datetime64() - times) / np.timedelta64(1, "m")
    age_min = np.clip(age_min.astype("float64"), 0.0, None)
    weights = np.power(0.5, age_min / float(halflife_min))
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return float(np.nanmean(values)) if len(values) else float("nan")
    return float((weights * values).sum() / total)


def _flag_stats(frame: pd.DataFrame, flag: str) -> pd.DataFrame:
    """``count(non-null flags)`` and ``count(all articles)`` per (ticker, date)."""
    work = frame.loc[:, ["ticker", "date", flag]].copy()
    work["_flag"] = work[flag].astype("boolean").astype("float64")
    grouped = work.groupby(["ticker", "date"], sort=True)
    return pd.DataFrame(
        {
            f"{flag}_n": grouped["_flag"].sum(),
            f"{flag}_d": grouped["_flag"].count(),
        }
    )


# --------------------------------------------------------------------------- input prep
def prepare_articles(sentiment: pd.DataFrame, news: pd.DataFrame | None) -> pd.DataFrame:
    """Join sentiment to news metadata and normalise dtypes for aggregation.

    ``news`` may be ``None`` when ``sentiment`` already carries ``published_at`` /
    ``publisher``. Missing ``published_at`` falls back to midnight UTC of ``signal_date``
    — a date-only feed, handled exactly as the plan describes.
    """
    required = ("ticker", "signal_date", "score_5", "skipped")
    missing = [c for c in required if c not in sentiment.columns]
    if missing:
        raise ValueError(f"article_sentiment is missing columns {missing}")

    frame = sentiment.loc[~sentiment["skipped"].astype(bool)].copy()
    if news is not None:
        meta_cols = [c for c in ("article_id", "published_at", "publisher") if c in news.columns]
        meta = news.loc[:, meta_cols].drop_duplicates("article_id", keep="first")
        frame = frame.drop(columns=[c for c in ("published_at", "publisher") if c in frame.columns])
        frame = frame.merge(meta, on="article_id", how="left", validate="many_to_one")

    frame["ticker"] = frame["ticker"].astype(str)
    frame["date"] = pd.to_datetime(frame["signal_date"]).dt.tz_localize(None).dt.normalize()
    if "published_at" in frame.columns:
        published = pd.to_datetime(frame["published_at"], utc=True, errors="coerce").dt.tz_localize(
            None
        )
    else:
        published = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    frame["published_at"] = published.fillna(frame["date"])
    if "publisher" not in frame.columns:
        frame["publisher"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame["publisher"] = frame["publisher"].astype("string").fillna("unknown")

    if "score_3" not in frame.columns:
        frame["score_3"] = frame["score_5"]
    for col in ("score_3", "p_neg", "p_neu", "p_pos", "confidence"):
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for flag in ("is_company_specific", "is_novel"):
        if flag not in frame.columns:
            frame[flag] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
        else:
            frame[flag] = frame[flag].astype("boolean")
    if "label_5" not in frame.columns:
        frame["label_5"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame["label_5"] = frame["label_5"].astype("string")
    return frame.loc[frame["score_5"].notna()].reset_index(drop=True)


# --------------------------------------------------------------------------- day panel
def _daily_rows(articles: pd.DataFrame, ewma_halflives: tuple[float, float]) -> pd.DataFrame:
    """One row per ``(ticker, date)`` from same-day articles only (no cross-day reads)."""
    work = articles.copy()
    score5 = work["score_5"].astype("float64")
    work["sq5"] = score5.fillna(0.0) ** 2
    work["score5_sum"] = score5.fillna(0.0)
    work["is_very_neg"] = (work["label_5"] == LEVELS[0]).astype("float64")
    work["n_extreme_neg"] = (work["p_neg"].astype("float64") > 0.8).astype("float64")
    labelled = work["label_5"].notna()

    grouped = work.groupby(["ticker", "date"], sort=True)
    rows = grouped.agg(
        n_news=("score_5", "size"),
        sent_sum=("score_5", lambda s: float(np.nansum(s.to_numpy(dtype="float64")))),
        sent_min=("score_5", "min"),
        sent_max=("score_5", "max"),
        sent_median=("score_5", "median"),
        score5_sum=("score5_sum", "sum"),
        sq5_sum=("sq5", "sum"),
        n_extreme_neg=("n_extreme_neg", "sum"),
        n_very_neg=("is_very_neg", "sum"),
        conf_sum=("confidence", "sum"),
        n_labelled=("label_5", lambda s: float(s.notna().sum())),
    )

    # Day std over articles (ddof=1); 0 when the day has a single article (the plan's rule).
    def _std(s: pd.Series) -> float:
        return float(s.std(ddof=1)) if len(s) > 1 else 0.0

    rows["sent_std"] = grouped["score_5"].agg(_std)
    rows["source_entropy"] = grouped["publisher"].agg(
        lambda s: _entropy(s.value_counts().to_numpy(dtype="float64"))
    )
    rows["top_source_share"] = grouped["publisher"].agg(
        lambda s: _top_share(s.value_counts().to_numpy(dtype="float64"))
    )
    rows = rows.join(_flag_stats(work, "is_company_specific")).join(_flag_stats(work, "is_novel"))

    # label_5 shares: NaN for a day where the backend produced no labels at all.
    counts = (
        work.loc[labelled].groupby(["ticker", "date", "label_5"]).size().rename("n").reset_index()
    )
    for level in LEVELS:
        part = counts.loc[counts["label_5"] == level].set_index(["ticker", "date"])["n"]
        # A day with news but no article of this level has a share of exactly 0, not NaN:
        # leaving it NaN also discards the neighbouring days that do carry the level.
        rows[f"share_{level}"] = (part.reindex(rows.index, fill_value=0.0) / rows["n_labelled"])
    rows = rows.rename(
        columns={
            "share_very negative": "very_neg_share",
            "share_very positive": "very_pos_share",
            "share_negative": "neg_share",
            "share_neutral": "neu_share",
            "share_positive": "pos_share",
        }
    )
    rows["neg_extreme_share"] = rows["n_extreme_neg"] / rows["n_news"]

    for name, halflife in zip(("ewma_6h", "ewma_24h"), ewma_halflives, strict=True):
        rows[name] = _ewma_by_group(work, halflife)
    return rows.reset_index()


def _ewma_by_group(articles: pd.DataFrame, halflife_min: float) -> pd.Series:
    """Time-decayed mean of ``score_5`` per ``(ticker, date)``; age to 23:59 of that day."""
    values: dict[tuple, float] = {}
    for (ticker, day), group in articles.groupby(["ticker", "date"], sort=True):
        published = pd.to_datetime(group["published_at"]).to_numpy(dtype="datetime64[ns]")
        order = np.argsort(published, kind="stable")
        day_end = pd.Timestamp(day) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        values[(ticker, day)] = _time_ewma(
            published[order],
            group["score_5"].to_numpy(dtype="float64")[order],
            halflife_min,
            day_end,
        )
    series = pd.Series(values, dtype="float64")
    series.index = pd.MultiIndex.from_tuples(series.index, names=["ticker", "date"])
    return series


# --------------------------------------------------------------------------- aggregation
def aggregate_sentiment_daily(
    articles: pd.DataFrame,
    spine: pd.DatetimeIndex,
    tickers: list[str],
    *,
    ewma_halflives_min: tuple[int, int] = (360, 1440),
    rolling_days: int = ROLLING_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate articles into the ``(ticker, date)`` panel on the shared calendar spine.

    Returns ``(panel, inventory)``. ``inventory`` is the per-article provenance log
    (``ticker, date, max_signal_date``, ``n_articles``) that leakage check 1 verifies:
    every aggregate can be traced back to the newest article it read.
    """
    if len(ewma_halflives_min) != 2:
        raise ValueError(f"exactly two EWMA halflives expected, got {ewma_halflives_min}")
    daily = _daily_rows(articles, (float(ewma_halflives_min[0]), float(ewma_halflives_min[1])))
    calendar = pd.DatetimeIndex(spine).normalize().sort_values()
    index = pd.MultiIndex.from_product([list(tickers), calendar], names=["ticker", "date"])
    if daily.empty:
        panel = pd.DataFrame(index=index).reset_index()
    else:
        panel = daily.set_index(["ticker", "date"]).reindex(index).reset_index()

    def numeric(name: str, default: float = np.nan) -> pd.Series:
        if name not in panel.columns:
            return pd.Series(default, index=panel.index, dtype="float64")
        return pd.to_numeric(panel[name], errors="coerce").astype("float64")

    panel["n_news"] = numeric("n_news").fillna(0.0)
    for name in _NUMERIC_DAY:
        panel[name] = numeric(name).fillna(0.0)
    for name in (
        "sent_min",
        "sent_max",
        "sent_median",
        "sent_std",
        "source_entropy",
        "top_source_share",
        "pos_share",
        "neg_share",
        "neu_share",
        "very_pos_share",
        "very_neg_share",
    ):
        panel[name] = numeric(name)

    return _derive(panel, rolling_days), _inventory(articles)


def _inventory(articles: pd.DataFrame) -> pd.DataFrame:
    """Newest ``signal_date`` read by an aggregate: ``max(signal_date)`` per (ticker, date)."""
    if articles.empty:
        return pd.DataFrame(
            columns=["ticker", "date", "max_signal_date", "n_articles"], dtype="object"
        )
    frame = articles.loc[:, ["ticker", "date", "signal_date"]].copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.tz_localize(None).dt.normalize()
    grouped = frame.groupby(["ticker", "date"], sort=True)["signal_date"]
    out = pd.DataFrame({"max_signal_date": grouped.max(), "n_articles": grouped.size()})
    return out.reset_index()


def _derive(panel: pd.DataFrame, rolling_days: int) -> pd.DataFrame:
    """Cross-day derivations: each reads ``[t-k, t-1]`` through the calendar spine."""
    panel = panel.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    n = panel["n_news"].to_numpy(dtype="float64")
    is_news = n > 0
    ticker = panel["ticker"]

    # --- day-t statistics -------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        sent_mean = np.where(
            is_news, panel["sent_sum"].to_numpy() / np.where(is_news, n, 1.0), np.nan
        )
    panel["sent_mean"] = sent_mean
    panel["neg_extreme_share"] = np.where(
        is_news, panel["n_extreme_neg"].to_numpy() / np.where(is_news, n, 1.0), np.nan
    )
    panel["very_neg_share"] = np.where(
        is_news & (panel["n_labelled"].to_numpy() > 0),
        panel["n_very_neg"].to_numpy() / np.maximum(panel["n_labelled"].to_numpy(), 1.0),
        np.nan,
    )
    for share in ("pos_share", "neg_share", "neu_share", "very_pos_share"):
        panel[share] = np.where(is_news, panel[share].to_numpy(), np.nan)
    panel["conf_mean"] = np.where(
        is_news, panel["conf_sum"].to_numpy() / np.where(is_news, n, 1.0), np.nan
    )
    panel["company_specific_share"] = np.where(
        panel["cs_d"].to_numpy() > 0,
        panel["cs_n"].to_numpy() / np.maximum(panel["cs_d"].to_numpy(), 1.0),
        np.nan,
    )
    panel["novelty_mean"] = np.where(
        panel["nov_d"].to_numpy() > 0,
        panel["nov_n"].to_numpy() / np.maximum(panel["nov_d"].to_numpy(), 1.0),
        np.nan,
    )
    for name in (
        "sent_min",
        "sent_max",
        "sent_median",
        "sent_std",
        "source_entropy",
        "top_source_share",
    ):
        panel[name] = np.where(is_news, panel[name].to_numpy(), np.nan)

    # ``n_news_z`` / ``sent_surprise`` use the trailing ``rolling_days`` window over
    # ``[t-w, t-1]`` (config: features.rolling_days = 30) — never the full history.
    cnt = pd.Series(_rolling_prev(ticker, pd.Series(np.ones_like(n)), rolling_days))
    win_n = pd.Series(_rolling_prev(ticker, pd.Series(n), rolling_days))
    win_n_sq = pd.Series(_rolling_prev(ticker, pd.Series(n**2), rolling_days))
    win_sent = pd.Series(
        _rolling_prev(ticker, pd.Series(panel["sent_sum"].to_numpy()), rolling_days)
    )
    win_sq = pd.Series(_rolling_prev(ticker, pd.Series(panel["sq5_sum"].to_numpy()), rolling_days))
    with np.errstate(divide="ignore", invalid="ignore"):
        win_n_std = np.sqrt(np.clip(_window_var(win_n, win_n_sq, cnt), 0.0, None))
        win_sent_std = np.sqrt(np.clip(_window_var(win_sent, win_sq, win_n), 0.0, None))
        panel["n_news_z"] = (n - _window_mean(win_n, cnt)) / (win_n_std + 1e-9)
        panel["sent_surprise"] = (sent_mean - _window_mean(win_sent, win_n)) / (win_sent_std + 1e-9)

    sent_shift3 = _shift_sessions(ticker, pd.Series(sent_mean), 3).to_numpy()
    panel["sent_momentum_3d"] = sent_mean - sent_shift3

    for w in MULTI_DAY_WINDOWS:
        rolling_sum = _rolling_prev(ticker, pd.Series(panel["sent_sum"].to_numpy()), w)
        rolling_cnt = _rolling_prev(ticker, pd.Series(n), w)
        panel[f"sent_sum_{w}d"] = rolling_sum
        with np.errstate(divide="ignore", invalid="ignore"):
            panel[f"sent_mean_{w}d"] = np.where(
                rolling_cnt > 0, rolling_sum / np.maximum(rolling_cnt, 1.0), np.nan
            )

    panel["ewma_6h"] = _carry_forward(ticker, panel["ewma_6h"])
    panel["ewma_24h"] = _carry_forward(ticker, panel["ewma_24h"])
    panel["log_n_news"] = np.log1p(n)

    ordered = [c for c in SENTIMENT_DAILY_COLUMNS if c in panel.columns]
    extra = [c for c in panel.columns if c not in ordered]
    return panel.loc[:, ordered + extra]


# --------------------------------------------------------------------------- spine maths
def _cum_sum(panel: pd.DataFrame, series: pd.Series) -> np.ndarray:
    return series.groupby(panel["ticker"], sort=False).cumsum().to_numpy(dtype="float64")


def _window_mean(total: pd.Series, count: pd.Series) -> np.ndarray:
    """Rolling sum divided by its rolling count; NaN where the count is zero."""
    c = count.to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(c > 0, total.to_numpy(dtype="float64") / np.maximum(c, 1.0), np.nan)


def _window_var(total: pd.Series, total_sq: pd.Series, count: pd.Series) -> np.ndarray:
    """Sample variance from rolling sums; NaN where n < 2."""
    c = count.to_numpy(dtype="float64")
    s = total.to_numpy(dtype="float64")
    sq = total_sq.to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(c > 1, (sq - s**2 / np.maximum(c, 1.0)) / np.maximum(c - 1.0, 1.0), np.nan)


def _shift_sessions(ticker: pd.Series, series: pd.Series, k: int) -> pd.Series:
    """``series`` shifted by +k sessions inside each ticker (``series`` must be on the spine)."""
    return series.groupby(ticker, sort=False).shift(k).astype("float64")


def _rolling_prev(ticker: pd.Series, series: pd.Series, w: int) -> np.ndarray:
    """Sum of ``series`` over ``[t-w, t-1]``: shift(1) then a w-row rolling sum."""
    shifted = series.groupby(ticker, sort=False).shift(1)
    rolling = shifted.groupby(ticker, sort=False).rolling(w, min_periods=0).sum()
    rolling.index = rolling.index.droplevel(0)
    return rolling.sort_index().to_numpy(dtype="float64")


def _carry_forward(ticker: pd.Series, series: pd.Series) -> np.ndarray:
    """Forward-fill inside a ticker: the time EWMA carries the last news day's value."""
    return series.groupby(ticker, sort=False).ffill().to_numpy(dtype="float64")
