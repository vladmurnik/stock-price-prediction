"""Portfolio construction and daily P&L with realistic frictions (T11).

The engine turns ``(date, ticker, score)`` rows into a dollar-neutral long/short
book and measures what survives costs. Two rules are non-negotiable:

* The signal of day ``t`` may only use information available at ``t``: the score
  (built from news with ``signal_date <= t``) and prices up to ``t``. Entry and
  exit prices are taken from ``t+1`` only, never earlier.
* The realised return is computed from **raw** prices. ``adj_close`` in the cache
  is retroactively adjusted for every dividend after the fact, so an
  adjusted-price return realised *within* a session cannot be traded; it is
  kept as a diagnostic only (see :func:`execution_return_columns`).

Execution modes (``BacktestCfg.execution``):

``next_open_hold_close``
    Buy at ``open[t+1]``, sell at ``close[t+1]`` — the plan's primary mode and
    the only one that is implementable with a signal formed after ``t``'s close.
``close_to_close``
    Buy at ``close[t]``, sell at ``close[t+1]`` — comparison variant; assumes the
    book can still be traded at ``t``'s closing print after the signal is known.
``next_open``
    Buy at ``open[t+1]``, sell at ``close[t+2]`` — overnight hold variant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "COST_PER_SIDE",
    "DEFAULT_MIN_LEG_NAMES",
    "DEFAULT_MIN_NAMES",
    "TARGETING_NO_OBS",
    "ExecutionFrame",
    "PortfolioResult",
    "build_day_weights",
    "build_execution_frame",
    "compute_turnover",
    "execution_return_columns",
    "leg_holdings",
    "midrank_scores",
    "null_distribution",
    "random_scores",
    "run_backtest",
    "summarise_returns",
]

#: Plan minimums: a day trades only with enough names, a leg needs diversification.
DEFAULT_MIN_NAMES = 5
DEFAULT_MIN_LEG_NAMES = 3
#: Plan formula: cost_per_side = (commission_bps + slippage_bps) / 1e4.
COST_PER_SIDE = 0.0010
TRADING_DAYS = 252
#: Equity, not price, is targeted: zero train-to-date mean after warm-up, |ρ| <= 0.10.
TARGETING_NO_OBS = 252
TARGETING_MIN_DATES = 126
TARGETING_MAX_ABS_RHO = 0.10


# --------------------------------------------------------------------------- execution frame
@dataclass(frozen=True)
class ExecutionFrame:
    """Per ``(date, ticker)`` prices and realised returns for one execution mode."""

    mode: str
    frame: pd.DataFrame  # date, ticker, score, ret, adj_ret, halted, entry_px, exit_px
    notes: list[str] = field(default_factory=list)

    @property
    def dates(self) -> np.ndarray:
        return np.sort(self.frame["date"].unique())

    @property
    def n_halted(self) -> int:
        return int(self.frame["halted"].sum())

    @property
    def n_nan_ret(self) -> int:
        return int(self.frame["ret"].isna().sum())


def execution_return_columns(mode: str) -> tuple[str, str]:
    """``(ret_column, adj_column)`` used by ``mode``.

    ``ret`` is the raw-price, tradable return; ``adj_ret`` is the same horizon on
    ``adj_close`` and is reported for comparison only.
    """
    if mode == "next_open_hold_close":
        return "ret_oc_raw", "ret_oc_adj"
    if mode == "close_to_close":
        return "ret_cc_raw", "ret_cc_adj"
    if mode == "next_open":
        return "ret_o2c_raw", "ret_o2c_adj"
    raise ValueError(f"unknown execution mode: {mode!r}")


def _price_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Attach forward returns per ticker (each ticker uses its own next session)."""
    frame = prices.loc[:, ["date", "ticker", "open", "close", "adj_close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    grouped = frame.groupby("ticker", sort=False)
    # Forward-looking legs; every column is used only for the action that follows.
    frame["close_t1"] = grouped["close"].shift(-1)
    frame["open_t1"] = grouped["open"].shift(-1)
    frame["adj_t1"] = grouped["adj_close"].shift(-1)
    frame["close_t2"] = grouped["close"].shift(-2)
    frame["adj_t2"] = grouped["adj_close"].shift(-2)
    frame["adj_t0"] = frame["adj_close"]

    with np.errstate(divide="ignore", invalid="ignore"):
        frame["ret_oc_raw"] = np.log(frame["close_t1"] / frame["open_t1"])
        frame["ret_oc_adj"] = np.log(frame["adj_t1"] / frame["open_t1"])
        frame["ret_cc_raw"] = np.log(frame["close_t1"] / frame["close"])
        frame["ret_cc_adj"] = np.log(frame["adj_t1"] / frame["adj_t0"])
        frame["ret_o2c_raw"] = np.log(frame["close_t2"] / frame["open_t1"])
        frame["ret_o2c_adj"] = np.log(frame["adj_t2"] / frame["open_t1"])
    return frame.drop(columns=["adj_t1", "adj_t2", "adj_t0"])


def build_execution_frame(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    mode: str,
    sector_map: dict[str, str] | None = None,
    score_col: str = "y_pred_seed_mean",
    max_date: pd.Timestamp | str | None = None,
) -> ExecutionFrame:
    """Join scores with forward returns and flag untradeable rows.

    A row is ``halted`` when the implied session did not trade (``open == close``
    exactly, i.e. a zero intraday range stub) or when the exit price is missing.
    Nothing is filled forward: a missing return stays NaN and the day is skipped
    downstream by the leg-size rule.
    """
    if score_col not in predictions.columns and "y_pred" in predictions.columns:
        # Holdout runs publish a single ``y_pred`` column (seed mean under that name).
        score_col = "y_pred"
    pred = predictions.loc[:, ["date", "ticker", score_col]].copy()
    pred["date"] = pd.to_datetime(pred["date"])
    pred = pred.rename(columns={score_col: "score"})
    if max_date is not None:
        pred = pred.loc[pred["date"] <= pd.Timestamp(max_date)]

    ret, adj = execution_return_columns(mode)
    px = _price_returns(prices)
    joined = pred.merge(px, on=["date", "ticker"], how="left", validate="many_to_one")

    if mode == "next_open_hold_close":
        entry_px, exit_px = joined["open_t1"], joined["close_t1"]
        halt = entry_px.eq(exit_px) & entry_px.notna() & joined["close_t1"].notna()
    elif mode == "close_to_close":
        entry_px, exit_px = joined["close"], joined["close_t1"]
        halt = entry_px.eq(exit_px) & entry_px.notna() & joined["close_t1"].notna()
    else:  # next_open
        entry_px, exit_px = joined["open_t1"], joined["close_t2"]
        halt = entry_px.eq(exit_px) & entry_px.notna() & joined["close_t2"].notna()

    out = pd.DataFrame(
        {
            "date": joined["date"],
            "ticker": joined["ticker"].astype(str),
            "score": joined["score"].astype(float),
            "ret": joined[ret].astype(float),
            "adj_ret": joined[adj].astype(float),
            "halted": halt.fillna(False).astype(bool),
            "entry_px": entry_px.astype(float),
            "exit_px": exit_px.astype(float),
        }
    )
    if sector_map:
        out["sector"] = out["ticker"].map(sector_map).fillna("unknown")
    else:
        out["sector"] = "unknown"

    notes = [f"execution={mode} ret={ret} adj={adj}"]
    n_missing_px = int(joined["close_t1"].isna().sum()) if mode != "next_open" else int(
        joined["close_t2"].isna().sum()
    )
    if n_missing_px:
        notes.append(f"{n_missing_px} rows without a forward price (kept NaN, not filled)")
    notes.append(f"{int(out['halted'].sum())} rows flagged halted/zero-range")
    return ExecutionFrame(mode=mode, frame=out, notes=notes)


# --------------------------------------------------------------------------- weighting
def midrank_scores(frame: pd.DataFrame) -> pd.Series:
    """Cross-sectional percentile rank within each date (in ``[0, 1]``).

    Ranks only describe the current cross-section, so they cannot move weights
    across days; a marginal score change therefore changes ranks only at ties.
    """
    grouped = frame.groupby("date", sort=False)["score"]
    return grouped.rank(method="average", pct=True)


def leg_holdings(frame: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """Assign each usable row to the top / bottom score bucket or to ``mid``."""
    usable = frame.loc[frame["score"].notna() & frame["ret"].notna()].copy()
    usable["rank_pct"] = midrank_scores(usable)
    upper = 1.0 - 1.0 / n_quantiles
    lower = 1.0 / n_quantiles
    usable["leg"] = "mid"
    usable.loc[usable["rank_pct"] > upper + 1e-12, "leg"] = "long"
    usable.loc[usable["rank_pct"] <= lower + 1e-12, "leg"] = "short"
    return usable


@dataclass
class DayBook:
    date: pd.Timestamp
    n_long: int
    n_short: int
    traded: bool
    reason: str
    weights: pd.Series  # ticker -> signed dollar weight
    returns: pd.Series  # ticker -> realised return


def build_day_weights(
    leg_frame: pd.DataFrame,
    n_quantiles: int = 5,
    gross: float = 1.0,
    min_names: int = DEFAULT_MIN_NAMES,
    min_leg_names: int = DEFAULT_MIN_LEG_NAMES,
    sector_cap: float | None = None,
) -> DayBook:
    """Turn one day's leg assignment into a dollar-neutral book.

    Returns an empty (untraded) book when the day has too few names or either leg
    is under-diversified. ``sector_cap`` caps each sector's gross exposure and is
    split evenly between the long and the short side.
    """
    day = leg_frame["date"].iloc[0]
    longs = leg_frame.loc[leg_frame["leg"] == "long", "ticker"].tolist()
    shorts = leg_frame.loc[leg_frame["leg"] == "short", "ticker"].tolist()
    usable = leg_frame.loc[leg_frame["leg"] != "mid"]
    n_names = len(usable)

    def _empty(reason: str) -> DayBook:
        return DayBook(
            date=day,
            n_long=len(longs),
            n_short=len(shorts),
            traded=False,
            reason=reason,
            weights=pd.Series(dtype=float),
            returns=pd.Series(dtype=float),
        )

    if n_names < min_names:
        return _empty("few_names")
    if len(longs) < min_leg_names or len(shorts) < min_leg_names:
        return _empty("thin_leg")

    half = gross / 2.0
    weights = pd.Series(0.0, index=usable["ticker"].to_numpy())
    weights.loc[longs] = half / len(longs)
    weights.loc[shorts] = -half / len(shorts)

    if sector_cap is not None and sector_cap > 0:
        weights = _apply_sector_cap(weights, usable, sector_cap, half)

    longest = leg_frame.set_index("ticker")["ret"]
    returns = longest.reindex(weights.index)
    return DayBook(
        date=day,
        n_long=len(longs),
        n_short=len(shorts),
        traded=True,
        reason="traded",
        weights=weights,
        returns=returns,
    )


def _apply_sector_cap(
    weights: pd.Series, usable: pd.DataFrame, cap: float, half: float
) -> pd.Series:
    """Scale weights so no sector exceeds ``cap`` of gross on either side."""
    sectors = usable.set_index("ticker")["sector"].reindex(weights.index)
    limit_long = cap * half
    out = weights.copy()
    for side, mask in (("long", weights > 0), ("short", weights < 0)):
        side_w = weights.loc[mask]
        if side_w.empty:
            continue
        exposure = side_w.abs().groupby(sectors.loc[mask]).sum()
        scale = (limit_long / exposure).clip(upper=1.0)
        for sector, factor in scale.items():
            if factor >= 1.0:
                continue
            members = sectors.loc[mask][sectors.loc[mask] == sector].index
            out.loc[members] = side_w.loc[members] * factor
        _ = side
    return out


# --------------------------------------------------------------------------- returns
def compute_turnover(
    weights_by_day: list[pd.Series], cost_per_side: float = COST_PER_SIDE
) -> pd.DataFrame:
    """Daily turnover and costs exactly as the plan specifies.

    ``turnover = 0.5 * sum|w_t - w_{t-1}|`` and ``cost = turnover * 2 * cost_per_side``;
    the first day is charged as a full open from cash (``w_{-1} = 0``), matching a
    book that starts flat.
    """
    rows = []
    previous = pd.Series(dtype=float)
    for weights in weights_by_day:
        current = weights.reindex(weights.index.union(previous.index)).fillna(0.0)
        prior = previous.reindex(current.index).fillna(0.0)
        turnover = 0.5 * float((current - prior).abs().sum())
        rows.append(
            {
                "turnover": turnover,
                "cost": turnover * 2.0 * cost_per_side,
                "gross_long": float(current.clip(lower=0).sum()),
                "gross_short": float(-current.clip(upper=0).sum()),
                "net_weight": float(current.sum()),
            }
        )
        previous = weights
    return pd.DataFrame(rows)


@dataclass
class PortfolioResult:
    daily: pd.DataFrame
    lagged_weights: pd.Series
    summary: dict[str, Any]
    books: list[DayBook]
    notes: list[str]


def run_backtest(
    frame: pd.DataFrame,
    n_quantiles: int = 5,
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
    borrow_annual: float = 0.05,
    gross: float = 1.0,
    min_names: int = DEFAULT_MIN_NAMES,
    min_leg_names: int = DEFAULT_MIN_LEG_NAMES,
    sector_cap: float | None = None,
) -> PortfolioResult:
    """Run the daily long/short backtest over ``frame`` (an :class:`ExecutionFrame`).

    Returns one row per calendar date in the sample. ``gross_ret`` is the
    dollar-weighted realised return; ``net_ret = gross_ret - cost - borrow`` holds
    to machine precision by construction (checked in :func:`summarise_returns`).
    """
    cost_per_side = (commission_bps + slippage_bps) / 1e4
    borrow_rate = borrow_annual / TRADING_DAYS
    ranked = leg_holdings(frame, n_quantiles=n_quantiles)

    books: list[DayBook] = []
    for _, day_frame in ranked.groupby("date", sort=True):
        books.append(
            build_day_weights(
                day_frame,
                n_quantiles=n_quantiles,
                gross=gross,
                min_names=min_names,
                min_leg_names=min_leg_names,
                sector_cap=sector_cap,
            )
        )

    # Positions are held until the *next* signal; the book of day t earns day t's return.
    weights = [b.weights if b.traded else pd.Series(dtype=float) for b in books]
    costs = compute_turnover(weights, cost_per_side=cost_per_side)
    gross_ret = np.array([float((b.weights * b.returns).sum()) for b in books], dtype=float)
    # A skipped day cannot earn a return: NaN, not zero (zero would bias Sharpe down).
    traded = np.array([b.traded for b in books], dtype=bool)
    gross_ret = np.where(traded, gross_ret, np.nan)

    daily = pd.DataFrame(
        {
            "date": [b.date for b in books],
            "gross_ret": gross_ret,
            "cost": costs["cost"].to_numpy(),
            "borrow": borrow_rate * costs["gross_short"].to_numpy(),
            "turnover": costs["turnover"].to_numpy(),
            "gross_long": costs["gross_long"].to_numpy(),
            "gross_short": costs["gross_short"].to_numpy(),
            "net_weight": costs["net_weight"].to_numpy(),
            "n_long": [b.n_long for b in books],
            "n_short": [b.n_short for b in books],
            "traded": traded,
            "reason": [b.reason for b in books],
        }
    )
    daily["trade_ret"] = daily["gross_ret"].fillna(0.0)
    daily["net_ret"] = daily["trade_ret"] - daily["cost"] - daily["borrow"]
    daily["equity"] = (1.0 + daily["net_ret"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0

    summary = summarise_returns(daily)
    lagged = books[-1].weights if books else pd.Series(dtype=float)
    notes = [f"{int((~traded).sum())} of {len(books)} dates not traded"]
    if sector_cap:
        notes.append(f"sector_cap={sector_cap}")
    return PortfolioResult(
        daily=daily, lagged_weights=lagged, summary=summary, books=books, notes=notes
    )


def _sharpe(values: pd.Series, periods: int = TRADING_DAYS) -> float:
    clean = values.dropna()
    if len(clean) < 2:
        return float("nan")
    std = float(clean.std(ddof=1))
    if std == 0.0:
        return float("nan")
    return float(clean.mean()) / std * np.sqrt(periods)


def summarise_returns(daily: pd.DataFrame, periods: int = TRADING_DAYS) -> dict[str, Any]:
    """Headline metrics from a ``daily`` frame (also used to re-derive from CSV)."""
    if daily.empty:
        return {}
    net = daily["net_ret"]
    gross = daily["gross_ret"]
    investable = daily["trade_ret"]
    equity = (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    ann_return = float(equity.iloc[-1] ** (periods / max(len(daily), 1)) - 1.0)
    ann_vol = float(investable.std(ddof=1) * np.sqrt(periods)) if len(daily) > 1 else float("nan")
    downside = net.loc[net < 0]
    sortino = (
        float(net.mean()) / float(np.sqrt((downside**2).mean())) * np.sqrt(periods)
        if len(downside) > 1
        else float("nan")
    )
    max_dd = float(drawdown.min())
    traded = daily["traded"].to_numpy(dtype=bool)
    return {
        "n_dates": int(len(daily)),
        "n_dates_traded": int(traded.sum()),
        "n_dates_skipped": int((~traded).sum()),
        "sharpe_gross": _sharpe(gross, periods),
        "sharpe_net": _sharpe(net, periods),
        "sortino": sortino,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_dd": max_dd,
        "calmar": float(ann_return / abs(max_dd)) if max_dd < 0 else float("nan"),
        "hit_rate_daily": float((net > 0).mean()),
        "mean_net_ret": float(net.mean()),
        "mean_gross_ret": float(gross.mean()),
        "turnover_ann": float(daily["turnover"].mean() * periods),
        "avg_n_long": float(daily["n_long"].mean()),
        "avg_n_short": float(daily["n_short"].mean()),
        "cost_drag_ann": float(daily["cost"].mean() * periods),
        "borrow_drag_ann": float(daily["borrow"].mean() * periods),
        "pnl_by_year": {
            str(int(year)): float(group["net_ret"].sum())
            for year, group in daily.groupby(pd.to_datetime(daily["date"]).dt.year)
        },
    }


# --------------------------------------------------------------------------- null models
def random_scores(frame: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """Per-date random permutation of the scores: the no-skill null."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(frame), dtype=float)
    for position, index in enumerate(frame.groupby("date", sort=False).groups.values()):
        values = frame.loc[index, "score"].to_numpy()
        out[frame.index.get_indexer(index)] = rng.permutation(values)
        _ = position
    return out


def null_distribution(
    frame: pd.DataFrame,
    n_draws: int = 200,
    seed: int = 42,
    n_quantiles: int = 5,
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
    borrow_annual: float = 0.05,
) -> pd.DataFrame:
    """Sharpe of a no-skill strategy with identical names, costs and book sizes.

    Each draw permutes the scores inside every date, so the number of names, the
    legs' cardinality and the turnover distribution stay as observed; only the
    prediction→return link is destroyed. This is the reference for "does the
    observed Sharpe exceed what this universe and these costs produce by luck".
    """
    rng = np.random.default_rng(seed)
    rows = []
    base = frame.copy()
    for draw in range(n_draws):
        shuffled = []
        for _, day_frame in base.groupby("date", sort=True):
            scores = day_frame["score"].to_numpy()
            shuffled.append(
                day_frame.assign(score=rng.permutation(scores)).reset_index(drop=True)
            )
        shuffled_frame = pd.concat(shuffled, ignore_index=True)
        result = run_backtest(
            shuffled_frame,
            n_quantiles=n_quantiles,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            borrow_annual=borrow_annual,
        )
        rows.append(
            {
                "draw": draw,
                "sharpe_net": result.summary["sharpe_net"],
                "sharpe_gross": result.summary["sharpe_gross"],
                "ann_return": result.summary["ann_return"],
            }
        )
    return pd.DataFrame(rows)
