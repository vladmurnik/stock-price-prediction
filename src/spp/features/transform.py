"""Fitted feature transforms for the GBDT pipeline (T9).

Two transforms, both deliberately boring and both **fitted on the training slice
only**:

``CrossSectionalRank``
    For every date, rank each feature across tickers and map the rank onto a
    standard normal (``rank_gauss``). A model that lives on the cross-section wants
    every row expressed *relative to its peers that day*, which also neutralises
    the large per-day market move that the demeaned target has already removed.
    Ranking is a per-date operation: a row's value at ``t`` depends on the tickers
    present at ``t`` and nothing else — no future, no full-sample statistic.

``FeaturePrep``
    The full, serialisable pre-processing contract used by the offline trainer and
    by live inference, in this fixed order:

    1. reorder columns to the saved feature list (live may send any order);
    2. cast ``dow``-style categorical columns to float (LightGBM sees numbers);
    3. per-date ``rank_gauss`` on the requested columns on days with enough tickers;
       on thin days the raw value is kept and flagged in ``rank_status``;
    4. *no imputation*. LightGBM handles NaN natively and ``NaN == no news`` must
       stay distinguishable from ``0 == neutral news``. Medians are stored in the
       artifact for diagnostics and for linear consumers, never used to fill.

The transform is a plain dataclass with JSON-able fields, so ``preprocess.joblib``
round-trips without importing this module's version of sklearn.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_MIN_RANK_TICKERS",
    "FeaturePrep",
    "RankStatus",
]

#: A date needs at least this many tickers for a cross-sectional rank to mean anything.
DEFAULT_MIN_RANK_TICKERS = 10

RankStatus = Literal["ranked", "raw", "missing"]


def _normal_quantiles(count: int) -> np.ndarray:
    """Expected order statistics of N(0,1) for ``count`` sorted values (rank_gauss map)."""
    from scipy.stats import norm

    return norm.ppf((np.arange(1, count + 1) - 0.5) / count)


@dataclass
class FeaturePrep:
    """Fitted pre-processing state; everything needed to reproduce training inputs."""

    feature_columns: list[str]
    categorical_columns: list[str] = field(default_factory=list)
    rank_columns: list[str] = field(default_factory=list)
    medians: dict[str, float] = field(default_factory=dict)
    nan_share_train: dict[str, float] = field(default_factory=dict)
    min_rank_tickers: int = DEFAULT_MIN_RANK_TICKERS
    version: str = "featureprep-v1"

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        feature_columns: list[str],
        *,
        rank_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
        min_rank_tickers: int = DEFAULT_MIN_RANK_TICKERS,
    ) -> FeaturePrep:
        """Derive the column order, medians and NaN shares from the **train** rows only."""
        missing = [c for c in feature_columns if c not in train.columns]
        if missing:
            raise ValueError(f"training frame lacks feature columns: {missing}")
        numeric = pd.DataFrame(
            {c: pd.to_numeric(train[c], errors="coerce").astype("float64") for c in feature_columns}
        )
        medians = {
            c: float(v) for c, v in numeric.median(skipna=True).items() if np.isfinite(v)
        }
        nan_share = {c: float(numeric[c].isna().mean()) for c in feature_columns}
        cats = [c for c in (categorical_columns or []) if c in feature_columns]
        rank_source = rank_columns if rank_columns is not None else feature_columns
        ranked = [c for c in rank_source if c in feature_columns]
        if not ranked:
            raise ValueError("no rank-able feature columns: cross-sectional ranking impossible")
        return cls(
            feature_columns=list(feature_columns),
            categorical_columns=cats,
            rank_columns=list(ranked),
            medians=medians,
            nan_share_train=nan_share,
            min_rank_tickers=int(min_rank_tickers),
        )

    # ------------------------------------------------------------------ transform
    def transform(
        self, frame: pd.DataFrame, *, dates: pd.Series | np.ndarray | None = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Apply the saved contract; returns ``(X, rank_status)``.

        ``frame`` may contain extra columns and any column order. ``dates`` defaults
        to ``frame["date"]``; without a date column ranking is skipped and every row
        is flagged ``missing`` (the caller decides whether that is acceptable).
        """
        if dates is None:
            if "date" not in frame.columns:
                raise ValueError("transform needs a 'date' column to rank cross-sectionally")
            dates = frame["date"]
        day = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates))).normalize().to_numpy()

        block = pd.DataFrame(index=frame.index)
        for name in self.feature_columns:
            if name in frame.columns:
                block[name] = pd.to_numeric(frame[name], errors="coerce").astype("float64")
            else:  # missing column: explicitly NaN, never zero
                block[name] = np.nan

        status = pd.Series("missing", index=frame.index, dtype="object")
        ranked = [c for c in self.rank_columns if c in self.feature_columns]
        if ranked:
            ranked_values, ranked_status = _rank_gauss_by_date(
                block, ranked, day, self.min_rank_tickers
            )
            block.loc[:, ranked] = ranked_values
            status = ranked_status
        return block, status

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rank_gauss_by_date(
    block: pd.DataFrame, columns: list[str], days: np.ndarray, min_tickers: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Per-date cross-sectional ``rank_gauss``; rows on thin dates keep the raw value."""
    out = block.copy()
    status = pd.Series("missing", index=block.index, dtype="object")
    values = block.loc[:, columns].to_numpy(dtype="float64")
    ranked = values.copy()
    for day in pd.unique(days):
        mask = days == day
        n_rows = int(mask.sum())
        if n_rows < min_tickers:
            status.loc[mask] = "raw"
            continue
        status.loc[mask] = "ranked"
        _normal_quantiles(n_rows)
        sub = values[mask]
        for j in range(sub.shape[1]):
            column = sub[:, j]
            finite = np.isfinite(column)
            if finite.sum() < 2:
                # Nothing to rank against: leave NaN as NaN (no opinion, not zero).
                ranked[np.flatnonzero(mask)[~finite], j] = np.nan
                continue
            order = np.argsort(column[finite], kind="stable")
            # Quantiles must match the number of *finite* values in this column, otherwise a
            # column with missing entries (no news that day) cannot be filled.
            local = np.empty(int(finite.sum()), dtype="float64")
            local[order] = _normal_quantiles(int(finite.sum()))
            dest = np.flatnonzero(mask)
            ranked[dest[finite], j] = local
            ranked[dest[~finite], j] = np.nan
    out.loc[:, columns] = ranked
    return out, status
