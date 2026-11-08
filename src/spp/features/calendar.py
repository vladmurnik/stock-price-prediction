"""NYSE trading calendar helpers shared by the feature, screening and label modules.

The feature layer never invents a trading day: the calendar is the *intersection* of
NYSE open days and the days present in the price cache. Everything downstream (target
``t+1``, purged CV embargo, future-shift test) indexes into that single spine, so
"the next trading day" has exactly one definition in the project.

``pandas_market_calendars`` is the authority when available; the business-day fallback
exists only for machines without the package and is logged loudly because it treats
US market holidays as trading days.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from spp.logging import get_logger

__all__ = [
    "HORIZONS",
    "calendar_spine",
    "ghost_mask",
    "horizon_returns",
    "nyse_sessions",
    "shift_by_sessions",
    "step_calendar",
]

LOG = get_logger("features.calendar")

#: Forecast horizons (trading sessions). ``1`` is the primary target ``y``.
HORIZONS: tuple[int, ...] = (1, 2, 3, 5)
#: Rows whose ``t + max(HORIZONS)`` lies beyond the data get their *labels* masked.
LABEL_MASK_SESSIONS = max(HORIZONS)


def _normalize_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index).tz_localize(None).normalize()


def nyse_sessions(start: date, end: date) -> pd.DatetimeIndex:
    """NYSE open days in ``[start, end]``, timezone-free and normalised to midnight."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    try:
        import pandas_market_calendars as mcal

        cal = mcal.get_calendar("NYSE")
        sessions = cal.valid_days(start_date=start_ts, end_date=end_ts + pd.Timedelta(days=1))
        return _normalize_index(pd.DatetimeIndex(sessions).unique()).sort_values()
    except Exception as exc:  # pragma: no cover - only without the package
        LOG.warning(
            "NYSE calendar unavailable (%s); falling back to business days, "
            "which treats US market holidays as sessions",
            exc,
        )
        return pd.bdate_range(start_ts, end_ts)


def step_calendar(index: pd.DatetimeIndex, n: int) -> pd.DatetimeIndex:
    """Session index shifted by ``n`` sessions: ``out[i] = index[i + n]`` (NaN past the end)."""
    idx = _normalize_index(index)
    if n == 0:
        return idx.copy()
    out = np.full(len(idx), np.datetime64("NaT"), dtype="datetime64[ns]")
    if n > 0 and len(idx) > n:
        out[:-n] = idx.to_numpy()[n:]
    elif n < 0 and len(idx) > -n:
        out[-n:] = idx.to_numpy()[:n]
    return pd.DatetimeIndex(out)


def shift_by_sessions(sessions: pd.DatetimeIndex, target: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map each session in ``sessions`` to the session ``target`` steps later on the spine.

    Used to build the ``t+1`` / ``t+5`` target *dates* (leakage check 3) and the
    embargo boundary of the purged CV (T8). Dates not on the spine map to ``NaT``.
    """
    spine = _normalize_index(target)
    lookup = pd.Series(spine, index=spine)
    return pd.DatetimeIndex(lookup.reindex(_normalize_index(sessions)).to_numpy())


def calendar_spine(price_dates: pd.DatetimeIndex, start: date, end: date) -> pd.DatetimeIndex:
    """Trading sessions in the window: NYSE open days that also carry prices.

    * A day present in prices but missing from NYSE (data glitch) is dropped.
    * A NYSE day missing from prices everywhere is dropped — features must not be
      computed on a day the model can never trade.
    """
    nyse = nyse_sessions(start, end)
    present = _normalize_index(pd.DatetimeIndex(price_dates).unique())
    window = present[(present >= pd.Timestamp(start)) & (present <= pd.Timestamp(end))]
    spine = nyse.intersection(window)
    return pd.DatetimeIndex(spine).sort_values()


def ghost_mask(adj_close: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Flat-price rows: identical ``adj_close`` to the previous *present* price, or zero volume.

    yfinance occasionally repeats a stale close after a holiday / half session. A row
    whose return is exactly 0.0 because the feed copied yesterday's price cannot be
    distinguished from a genuine flat session by price alone, so the row is flagged
    and its *features* are dropped by the assemble step. ``False`` means "usable".
    The first observation of a ticker is usable (no comparison).
    """
    del lookback  # reserved for a future volatility-scaled threshold
    values = adj_close.to_numpy(dtype="float64")
    mask = np.zeros(values.shape, dtype=bool)
    if values.shape[0] == 0:
        return pd.DataFrame(mask, index=adj_close.index, columns=adj_close.columns)
    if values.shape[0] > 1:
        prev = values[:-1]
        both = np.isfinite(prev) & np.isfinite(values[1:])
        mask[1:] = both & (values[1:] == prev)
    return pd.DataFrame(mask, index=adj_close.index, columns=adj_close.columns)


def horizon_returns(adj_close: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """``h``-session forward log returns ``log(P[t+h] / P[t])`` on the given spine.

    The result at row ``t`` depends only on data *after* ``t`` — it is a label, never
    a feature. Rows beyond the end become NaN (callers mask them).
    """
    values = adj_close.to_numpy(dtype="float64")
    out: dict[int, pd.DataFrame] = {}
    for h in HORIZONS:
        fwd = np.full_like(values, np.nan)
        if len(values) > h:
            with np.errstate(divide="ignore", invalid="ignore"):
                fwd[:-h] = np.log(values[h:] / values[:-h])
        out[h] = pd.DataFrame(fwd, index=adj_close.index, columns=adj_close.columns)
    return out
