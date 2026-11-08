"""Targets on ``adj_close`` (T6).

The target is a **return**, never a price level::

    y[k]  = log(adj_close[t+k] / adj_close[t])     k = 1, 2, 3, 5
    y     = y[1]                                    primary target
    y_dir = sign(y)
    y_raw = y                                       kept separately for the backtest

All series are built on the shared calendar spine so ``t+k`` means "``k`` NYSE
sessions later", matching ``features.calendar.horizon_returns``.

Masking policy (documented, not silent):

* ``HORIZON_MASK_SESSIONS`` (5 = ``max(HORIZONS)``) trailing sessions of the history
  have an incomplete forward window. Their ``y`` / ``y_dir`` are masked to NaN so a
  frame sliced at ``2022-12-31`` never trains on a label that peeked into 2023.
* "Ghost" rows (the feed repeats yesterday's identical close) get ``ghost=True``;
  their features are zeroed by :mod:`spp.features.assemble` and they are expected to
  be dropped before training. The flag is preserved so the leakage audit can see them.

Cross-sectional demean (``config.features.demean_target``): ``y_used = y − median(y)``
per date, computed per date from *raw* ``y`` (a per-date operation, causal by
construction). ``y_raw`` always keeps the un-demeaned value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spp.features.calendar import HORIZONS, ghost_mask, horizon_returns
from spp.logging import get_logger

__all__ = [
    "HORIZON_MASK_SESSIONS",
    "LABEL_COLUMNS",
    "build_labels",
    "demean_by_date",
]

LOG = get_logger("features.labels")

HORIZON_MASK_SESSIONS = max(HORIZONS)
LABEL_COLUMNS: tuple[str, ...] = (
    "y_1d",
    "y_2d",
    "y_3d",
    "y_5d",
    "y",
    "y_dir",
    "y_raw",
    "y_used_demeaned",
    "ghost",
    "label_ok",
)


def demean_by_date(y: pd.DataFrame, by: pd.Series) -> pd.Series:
    """Per-date cross-sectional demean: ``y − median(y)`` on each ``date``.

    Rows where ``date`` is missing stay NaN. This is a per-date operation, so it
    never mixes information across dates.
    """
    frame = pd.DataFrame({"y": y.to_numpy(dtype="float64"), "date": by.to_numpy()})
    medians = frame.groupby("date")["y"].transform("median")
    return (frame["y"] - medians).astype("float64")


def build_labels(
    adj_close: pd.DataFrame,
    *,
    demean_target: bool = True,
) -> pd.DataFrame:
    """Long ``(ticker, date)`` labels on the spine (``adj_close`` is the spine-aligned wide frame).

    Returns the label columns of :data:`LABEL_COLUMNS` plus ``ticker``/``date``.
    """
    forward = horizon_returns(adj_close)
    ghost = ghost_mask(adj_close)
    frames = [_to_long(forward[h], f"y_{h}d") for h in HORIZONS]
    long = frames[0]
    for frame in frames[1:]:
        long = long.merge(frame, on=["ticker", "date"], how="left", validate="one_to_one")
    long["y"] = long["y_1d"]
    long["y_dir"] = np.sign(long["y"]).astype("float64")
    long["y_raw"] = long["y"]
    long["ghost"] = _to_long(ghost.astype("float64"), "_ghost")["_ghost"].fillna(0.0).astype(bool)

    # Rows whose k=5 window would reach past the data keep no usable label at all:
    # a train/valid slice must never carry a label that peeked into the holdout.
    dates = pd.DatetimeIndex(adj_close.index)
    tail_dates = set(dates[-HORIZON_MASK_SESSIONS:]) if len(dates) else set()
    in_tail = long["date"].isin(tail_dates)
    long["label_ok"] = long["y"].notna() & ~in_tail
    long.loc[in_tail, ["y_1d", "y_2d", "y_3d", "y_5d", "y", "y_dir"]] = np.nan

    if demean_target:
        long["y_used_demeaned"] = demean_by_date(long["y"], long["date"])
    else:
        long["y_used_demeaned"] = long["y"]
    long = long.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    return long.loc[:, ["ticker", "date", *LABEL_COLUMNS]]


def _to_long(wide: pd.DataFrame, name: str) -> pd.DataFrame:
    long = wide.stack(future_stack=True).rename(name).reset_index()
    long.columns = ["date", "ticker", name]
    return long
