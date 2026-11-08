"""Feature matrix assembly (T6).

Produces ``data/processed/features.parquet`` — one row per ``(ticker, date)`` on the
calendar spine with market + sentiment features, the label block and the ``split``.

Every step is causal by construction:

1. **Spine** — NYSE sessions ∩ price dates over the requested window.
2. **Market** — wide prices sliced to the spine, trailing windows only, ``shift(1)``.
3. **Sentiment** — aggregated per ``(ticker, signal_date)``, reindexed onto the spine,
   cross-day aggregates shifted on the spine.
4. **Labels** — forward returns, ghost flags, per-date demean.
5. **Join** — ``left`` on ``(ticker, date)``; a missing *price* row is tolerated as NaN
   (a ticker listed later), a missing *news* row is normal and yields NaNs.
6. **Split** — ``train``/``valid``/``test`` by date from the config.

``features.parquet`` carries everything (including ``y``) for training;
``features_no_sentiment.parquet`` is the same table with the sentiment columns
dropped (B1 baseline) — a projection, not a second copy of the data.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spp.config import Config
from spp.data.prices import DEFAULT_HELPERS as HELPER_TICKERS
from spp.data.store import write_manifest, write_parquet_atomic
from spp.features.calendar import calendar_spine
from spp.features.labels import LABEL_COLUMNS, build_labels
from spp.features.market import MARKET_FEATURES, build_market_features
from spp.features.sentiment_daily import (
    SENTIMENT_FEATURES,
    aggregate_sentiment_daily,
    prepare_articles,
)
from spp.logging import get_logger

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "BuildResult",
    "FeatureAssemblyError",
    "SplitCfg",
    "assemble_features",
    "build_feature_matrix",
    "feature_columns",
    "load_split",
    "split_label",
    "write_feature_schema",
]

LOG = get_logger("features.assemble")

FEATURE_SCHEMA_VERSION = "features-v1"
SPLITS: tuple[str, ...] = ("train", "valid", "test")


class FeatureAssemblyError(RuntimeError):
    """The price data cannot support a feature matrix (empty window, tiny spine)."""


@dataclass(frozen=True)
class SplitCfg:
    """``train`` = ``[start, train_end]``, ``valid`` = ``(train_end, valid_end]``, then ``test``."""

    train_start: date
    train_end: date
    valid_end: date
    test_end: date

    @classmethod
    def from_config(cls, cfg: Config) -> SplitCfg:
        d = cfg.dates
        return cls(d.train_start, d.train_end, d.valid_end, d.test_end)


@dataclass
class BuildResult:
    """Everything the leakage report needs, plus the written paths."""

    features: pd.DataFrame
    panel: pd.DataFrame
    inventory: pd.DataFrame
    labels: pd.DataFrame
    articles: pd.DataFrame
    spine: pd.DatetimeIndex
    adj_close: pd.DataFrame
    split: SplitCfg
    schema: dict[str, Any]
    paths: dict[str, Path]


# --------------------------------------------------------------------------- helpers
def feature_columns(include_sentiment: bool = True) -> list[str]:
    """Ordered feature list: market first, then sentiment (stable, schema-pinned)."""
    cols = list(MARKET_FEATURES)
    if include_sentiment:
        cols += list(SENTIMENT_FEATURES)
    return cols


def split_label(dates: pd.Series, split: SplitCfg) -> pd.Series:
    """Map each date to ``train`` / ``valid`` / ``test`` / ``unused``."""
    d = pd.to_datetime(dates)
    out = pd.Series("unused", index=dates.index, dtype="object")
    out[(d >= pd.Timestamp(split.train_start)) & (d <= pd.Timestamp(split.train_end))] = "train"
    out[(d > pd.Timestamp(split.train_end)) & (d <= pd.Timestamp(split.valid_end))] = "valid"
    out[(d > pd.Timestamp(split.valid_end)) & (d <= pd.Timestamp(split.test_end))] = "test"
    return out


def load_split(cfg: Config) -> SplitCfg:
    return SplitCfg.from_config(cfg)


def _wide(prices: pd.DataFrame, field: str, tickers: Sequence[str]) -> pd.DataFrame:
    frame = prices.loc[:, ["date", "ticker", field]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    wide = frame.pivot(index="date", columns="ticker", values=field)
    wide.index = pd.DatetimeIndex(wide.index).normalize()
    return wide.reindex(columns=list(tickers)).sort_index(kind="stable")


def _one_series(prices: pd.DataFrame, ticker: str) -> pd.Series:
    frame = prices.loc[prices["ticker"].astype(str) == ticker, ["date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    series = frame.set_index("date")["close"].astype("float64")
    return series[~series.index.duplicated(keep="last")].sort_index(kind="stable")


def _mask_ghosts(matrix: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """A ghost (repeated close) row is never trained on: mask its features and labels."""
    if "ghost" not in matrix.columns:
        return matrix
    ghost = matrix["ghost"].astype(bool).to_numpy()
    if not bool(ghost.any()):
        return matrix
    labels = [c for c in LABEL_COLUMNS if c in matrix.columns and c not in {"ghost", "label_ok"}]
    matrix.loc[ghost, [*feature_cols, *labels]] = np.nan
    matrix.loc[ghost, "label_ok"] = False
    LOG.info("masked %d ghost rows (repeated close) from features and labels", int(ghost.sum()))
    return matrix


def _sector_etf(config: Config) -> dict[str, str]:
    from spp.config import load_universe_config

    return dict(load_universe_config(config.universe).sector_etf)


def _sector_of(config: Config) -> dict[str, str]:
    from spp.config import load_universe

    return {t: info.sector for t, info in load_universe(config.universe).items()}


def _schema(matrix: pd.DataFrame, *, sentiment_available: bool) -> dict[str, Any]:
    market = list(MARKET_FEATURES)
    sentiment = list(SENTIMENT_FEATURES)
    kinds = {name: ("category" if name == "dow" else "float64") for name in (*market, *sentiment)}
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "n_features": len(market) + len(sentiment),
        "features": [*market, *sentiment],
        "market_features": market,
        "sentiment_features": sentiment,
        "label_columns": list(LABEL_COLUMNS),
        "dtypes": kinds,
        "sentiment_available": bool(sentiment_available),
        "rows": int(len(matrix)),
        "tickers": int(matrix["ticker"].nunique()),
        "target": {
            "name": "y",
            "definition": "log(adj_close[t+1]/adj_close[t]) on the NYSE spine",
            "demeaned_column": "y_used_demeaned",
            "raw_column": "y_raw",
            "horizons_sessions": [1, 2, 3, 5],
        },
        "nan_policy": {
            "no_news": "sent_* = NaN, n_news = 0, log_n_news = 0, ewma carried forward",
            "ghost": "features and labels masked",
            "linear_baselines": "median of the training slice",
        },
        "splits": list(SPLITS),
    }


# --------------------------------------------------------------------------- assembly
def assemble_features(
    prices: pd.DataFrame,
    *,
    sentiment: pd.DataFrame | None,
    news: pd.DataFrame | None,
    config: Config,
    start: date | None = None,
    end: date | None = None,
) -> BuildResult:
    """Build the feature matrix in memory (nothing written)."""
    split = SplitCfg.from_config(config)
    start = start or split.train_start
    end = end or split.test_end

    universe = _sector_of(config)
    prices = prices.loc[:, ["date", "ticker", "close", "adj_close", "volume"]].copy()
    prices["ticker"] = prices["ticker"].astype(str)
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices = prices.loc[
        (prices["date"] >= pd.Timestamp(start)) & (prices["date"] <= pd.Timestamp(end))
    ]
    if prices.empty:
        raise FeatureAssemblyError(f"no prices between {start} and {end}")

    # The universe actually traded: configured members found in the prices, else every
    # non-helper ticker present (fixtures). SPY / ^VIX / sector ETFs are never modelled.
    helpers = set(HELPER_TICKERS)
    traded = sorted(set(prices["ticker"].unique()) - helpers)
    universe_tickers = [t for t in universe if t in traded] or traded
    if not universe_tickers:
        raise FeatureAssemblyError("no modellable tickers in the price window")
    LOG.info("modelling %d tickers", len(universe_tickers))

    spine = calendar_spine(pd.DatetimeIndex(prices["date"].unique()), start, end)
    if len(spine) < 10:
        raise FeatureAssemblyError(f"calendar spine too short: {len(spine)} sessions")

    adj_close = _wide(prices, "adj_close", universe_tickers).reindex(spine)
    volume = _wide(prices, "volume", universe_tickers).reindex(spine)
    # Sector ETFs are helpers, not modelled names, so they need their own wide matrix.
    sector_close = _wide(
        prices, "adj_close", sorted(set(_sector_etf(config).values()))
    ).reindex(spine)
    LOG.info(
        "spine sessions=%d, tickers with prices=%d/%d",
        len(spine),
        int((adj_close.notna().sum() > 0).sum()),
        len(universe_tickers),
    )

    market = build_market_features(
        adj_close,
        volume,
        universe_tickers=universe_tickers,
        sector_of=universe,
        sector_etf=_sector_etf(config),
        sector_close=sector_close,
        spy=_one_series(prices, "SPY").reindex(spine),
        vix=pd.to_numeric(_one_series(prices, "^VIX"), errors="coerce").reindex(spine),
    )
    labels = build_labels(adj_close, demean_target=config.features.demean_target)

    if sentiment is not None and not sentiment.empty:
        articles = prepare_articles(sentiment, news)
        panel, inventory = aggregate_sentiment_daily(
            articles,
            spine,
            universe_tickers,
            ewma_halflives_min=tuple(config.features.ewma_halflives_min),  # type: ignore[arg-type]
            rolling_days=config.features.rolling_days,
        )
        sentiment_block: pd.DataFrame | None = panel.loc[
            :, ["ticker", "date", *[c for c in SENTIMENT_FEATURES if c in panel.columns]]
        ]
    else:
        LOG.warning("no sentiment supplied: sentiment columns will be all-NaN")
        articles = pd.DataFrame(columns=["ticker", "date", "signal_date"])
        inventory = pd.DataFrame(columns=["ticker", "date", "max_signal_date", "n_articles"])
        panel = pd.DataFrame(
            {"ticker": pd.Series(dtype="string"), "date": pd.Series(dtype="datetime64[ns]")}
        )
        sentiment_block = None

    matrix = market.merge(labels, on=["ticker", "date"], how="left", validate="one_to_one")
    if sentiment_block is not None:
        matrix = matrix.merge(
            sentiment_block, on=["ticker", "date"], how="left", validate="one_to_one"
        )
    else:
        for name in SENTIMENT_FEATURES:
            matrix[name] = np.nan

    matrix = _mask_ghosts(matrix, feature_columns(include_sentiment=True))
    matrix["split"] = split_label(matrix["date"], split).to_numpy()

    cols = ["ticker", "date", *feature_columns(), *LABEL_COLUMNS, "split"]
    matrix = (
        matrix.loc[:, cols].sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    )
    matrix["dow"] = matrix["dow"].astype("Int8")

    schema = _schema(matrix, sentiment_available=sentiment_block is not None)
    return BuildResult(
        features=matrix,
        panel=panel,
        inventory=inventory,
        labels=labels,
        articles=articles,
        spine=spine,
        adj_close=adj_close,
        split=split,
        schema=schema,
        paths={},
    )


def build_feature_matrix(
    prices: pd.DataFrame,
    *,
    sentiment: pd.DataFrame | None,
    news: pd.DataFrame | None,
    config: Config,
    data_dir: str | Path,
    start: date | None = None,
    end: date | None = None,
    write: bool = True,
) -> BuildResult:
    """Assemble, and (by default) write ``features.parquet`` + schema + manifest."""
    result = assemble_features(
        prices, sentiment=sentiment, news=news, config=config, start=start, end=end
    )
    if not write:
        return result
    processed = Path(data_dir) / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    features_path = processed / "features.parquet"
    no_sent_path = processed / "features_no_sentiment.parquet"
    schema_path = processed / "features_schema.json"
    matrix = result.features
    write_parquet_atomic(matrix, features_path, date32=("date",))
    without = matrix.drop(columns=[c for c in SENTIMENT_FEATURES if c in matrix.columns])
    write_parquet_atomic(without, no_sent_path, date32=("date",))
    write_feature_schema(result.schema, schema_path)
    write_manifest(
        features_path.with_name(features_path.name + ".manifest.json"),
        {
            "kind": "features",
            "schema_version": FEATURE_SCHEMA_VERSION,
            "rows": int(len(matrix)),
            "tickers": int(matrix["ticker"].nunique()),
            "n_features": result.schema["n_features"],
            "split_counts": matrix["split"].value_counts().to_dict(),
            "sentiment_available": result.schema["sentiment_available"],
        },
    )
    result.paths = {
        "features": features_path,
        "features_no_sentiment": no_sent_path,
        "schema": schema_path,
    }
    LOG.info(
        "features written: %s (%d rows, %d features)",
        features_path,
        len(matrix),
        result.schema["n_features"],
    )
    return result


def write_feature_schema(schema: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
