"""Market features on the shared calendar spine (T6).

All inputs are wide ``index=date, columns=ticker`` frames built from the price cache.
Every feature at row ``t`` uses prices with date ``<= t`` only:

* ``ret_k`` = ``log(P_t / P_{t-k})`` — the newest observation in the window is ``t``
  itself, which is fine: the target is ``t+1``. Nothing reads ``t+1``.
* Rolling statistics (``vol_20``, ``px_sma20``, ``rsi_14``, ``volume_z``) are shifted
  by **one session** before use, so a row at ``t`` describes the window ending at
  ``t-1``. That keeps the "feature knowable at the close of t" contract without
  letting a same-day extreme leak in through a full-sample rolling window, and it is
  what the plan's ``rolling-фичи только через shift(1)`` asks for.
* ``spy_*`` / ``vix_*`` come from the same cache (``SPY``, ``^VIX``); ``sector_ret_1``
  maps each ticker's sector to its ETF via the universe YAML.

No statistic is ever fitted over the whole history (no full-sample mean/std for
scaling); everything is a trailing window over the spine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spp.logging import get_logger

__all__ = [
    "MARKET_FEATURES",
    "SPY_TICKER",
    "VIX_TICKER",
    "build_market_features",
    "sector_map_from_universe",
]

LOG = get_logger("features.market")

SPY_TICKER = "SPY"
VIX_TICKER = "^VIX"

RET_LAGS: tuple[int, ...] = (1, 2, 3, 4, 5)

MARKET_FEATURES: tuple[str, ...] = (
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_4",
    "ret_5",
    "vol_5",
    "vol_20",
    "vol_60",
    "vol_ratio",
    "volume_z",
    "rsi_14",
    "px_sma20",
    "px_sma50",
    "sector_ret_1",
    "spy_ret_1",
    "spy_ret_5",
    "vix_level",
    "vix_chg_1",
    "dow",
)


def sector_map_from_universe(universe: dict[str, object]) -> dict[str, str]:
    """``ticker -> sector`` from the loaded universe (``TickerInfo.sector``)."""
    out: dict[str, str] = {}
    for ticker, info in universe.items():
        sector = getattr(info, "sector", None)
        if sector is None and isinstance(info, dict):
            sector = info.get("sector")
        if sector:
            out[str(ticker)] = str(sector)
    return out


def _log_returns(close: pd.DataFrame) -> pd.DataFrame:
    values = close.to_numpy(dtype="float64")
    out = np.full_like(values, np.nan)
    if len(values) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = values[1:] / values[:-1]
            ratio[~(ratio > 0)] = np.nan
            out[1:] = np.log(ratio)
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def _shift(wide: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    """Shift rows down by ``k`` sessions: row ``t`` gets the value computed at ``t-k``."""
    return wide.shift(k)


def _rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder RSI on the *shifted* price level (row ``t`` uses prices up to ``t-1``)."""
    shifted = _shift(close, 1)
    delta = shifted.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 with positive gains is a straight-line up move: RSI = 100.
    rsi = rsi.where(~(avg_loss.eq(0.0) & avg_gain.gt(0.0)), 100.0)
    return rsi


def _volume_z(volume: pd.DataFrame, window: int) -> pd.DataFrame:
    shifted = _shift(volume, 1)
    mean = shifted.rolling(window, min_periods=window).mean()
    std = shifted.rolling(window, min_periods=2).std(ddof=1)
    return (shifted - mean) / (std + 1e-9)


def _stack_wide(wide: pd.DataFrame, name: str) -> pd.DataFrame:
    """Long ``(ticker, date, <name>)`` from a date x ticker matrix."""
    long = wide.stack(future_stack=True).rename(name).reset_index()
    long.columns = ["date", "ticker", name]
    return long


def build_market_features(
    adj_close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    universe_tickers: list[str],
    sector_of: dict[str, str],
    sector_etf: dict[str, str],
    spy: pd.Series,
    vix: pd.Series,
    vol_long_window: int = 60,
    sector_close: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long ``(ticker, date)`` frame with :data:`MARKET_FEATURES` for universe tickers.

    ``adj_close`` / ``volume`` are wide frames already sliced to the calendar spine;
    ``spy`` / ``vix`` are 1-d series on the same index. Odd tickers missing from
    ``adj_close`` simply get NaN rows downstream (never fabricated prices).
    """
    index = adj_close.index
    close = adj_close.loc[:, [t for t in universe_tickers if t in adj_close.columns]]
    volume = volume.reindex(index=index, columns=close.columns)
    rets = _log_returns(close)

    frames: list[pd.DataFrame] = []
    for lag in RET_LAGS:
        frames.append(_stack_wide(rets.shift(lag - 1), f"ret_{lag}"))

    vol_20 = _shift(rets.rolling(20, min_periods=10).std(ddof=1), 1)
    vol_5 = _shift(rets.rolling(5, min_periods=3).std(ddof=1), 1)
    vol_60 = _shift(rets.rolling(vol_long_window, min_periods=20).std(ddof=1), 1)
    frames.append(_stack_wide(vol_5, "vol_5"))
    frames.append(_stack_wide(vol_20, "vol_20"))
    frames.append(_stack_wide(vol_60, "vol_60"))
    frames.append(_stack_wide(vol_5 / vol_60.replace(0.0, np.nan), "vol_ratio"))

    frames.append(_stack_wide(_volume_z(volume, 20), "volume_z"))
    frames.append(_stack_wide(_rsi(close, 14), "rsi_14"))

    shifted_close = _shift(close, 1)
    sma20 = shifted_close.rolling(20, min_periods=10).mean()
    sma50 = shifted_close.rolling(50, min_periods=25).mean()
    frames.append(_stack_wide(shifted_close / sma20 - 1.0, "px_sma20"))
    frames.append(_stack_wide(shifted_close / sma50 - 1.0, "px_sma50"))

    long = frames[0]
    for frame in frames[1:]:
        long = long.merge(frame, on=["ticker", "date"], how="left", validate="one_to_one")

    # Sector ETF return, lag 1 (etf[t-1]/etf[t-2]); rows before the second session are NaN.
    # ``sector_close`` carries the ETF columns when ``adj_close`` holds only modelled names.
    sector_source = sector_close if sector_close is not None else adj_close
    etf_rets = _log_returns(
        sector_source[[c for c in dict.fromkeys(sector_etf.values()) if c in sector_source.columns]]
    )
    etf_lag = etf_rets.shift(1)
    sectors = long["ticker"].map(sector_of)
    sector_series = pd.Series(np.nan, index=long.index, dtype="float64")
    for sector, etf in sector_etf.items():
        if etf not in etf_lag.columns:
            continue
        mask = sectors == sector
        if not bool(mask.any()):
            continue
        values = etf_lag[etf].reindex(long.loc[mask, "date"].to_numpy())
        sector_series.loc[mask] = values.to_numpy()
    long["sector_ret_1"] = sector_series

    spy_rets = _log_returns(spy.to_frame(name=SPY_TICKER))[SPY_TICKER]
    long["spy_ret_1"] = long["date"].map(spy_rets.shift(1))
    long["spy_ret_5"] = long["date"].map(spy_rets.shift(1).rolling(5, min_periods=1).sum())
    long["vix_level"] = long["date"].map(vix)
    long["vix_chg_1"] = long["date"].map(vix.diff().shift(1))
    long["dow"] = pd.to_datetime(long["date"]).dt.day_of_week.astype("int8")

    long = long.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    missing = [c for c in MARKET_FEATURES if c not in long.columns]
    if missing:
        raise AssertionError(f"market features missing: {missing}")
    return long.loc[:, ["ticker", "date", *MARKET_FEATURES]]
