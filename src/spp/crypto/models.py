"""T7 — crypto event-study model layer: causal features, quantile GBDTs, purged
walk-forward validation, baselines and the hurdle policy.

Feature causality contract
--------------------------
Every feature is computed from data strictly known at ``entry_ts``:

* calendar/identity features (``hour_sin``/``hour_cos``, ``dow``, ``asset_code``,
  ``log_lag``) come from the event row itself;
* ``tr_ret_*`` / ``tr_vol_*`` use only bars of the traded symbol with
  ``ts_open < entry_ts``; ``btc_*`` uses only BTCUSDT bars with ``ts_open < entry_ts``;
* ``news_burst_60m`` counts events of the same asset whose ``entry_ts`` lies in
  ``[entry_ts - 60min, entry_ts)`` — the current event and simultaneous ones are
  excluded, so the count is causal over the sorted event list.

**Forbidden as features** (a contract, enforced by :func:`assert_no_future_features`):

* ``quote_volume_entry`` and ``trades_entry`` — they describe the *entry bar*, which
  opens at ``entry_ts`` and closes one minute AFTER it, so both are a one-minute
  look-ahead at decision time;
* anything derived from ``ret_*`` / ``bench_*`` / ``excess_*`` of the current event —
  those are the labels (``excess_h`` is exactly the regression target), not features.

Validation
----------
Purged walk-forward: ``N_FOLDS`` expanding folds over the **train split only**, with an
``EMBARGO_MINUTES`` embargo — a train row is purged when its label window
``[entry_ts, entry_ts + label_minutes]`` touches the fold's test boundary or when it
sits inside the embargo before it. Out-of-fold predictions never train on their own
test rows. The val split is never used for fitting or selection, only for the final
honest read-out; the holdout split is empty by construction and is never reported.

Policy
------
``position = L * sign(pred_q50)`` when ``|pred_q50| >= hurdle`` else ``0``, with
``hurdle = cost`` (14 bps). Net PnL per event is
``L * (sign * excess_h - cost) = position * excess_h - |position| * cost``:
leverage scales the edge and the fees by the same factor, so it can never turn a
negative expectancy positive — leverage is not a source of alpha.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from spp.logging import get_logger

try:  # pragma: no cover - the sklearn fallback is only used when lightgbm is absent
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None  # type: ignore[assignment]

__all__ = [
    "BENCH_SYMBOL",
    "BURST_WINDOW_MINUTES",
    "DEFAULT_COST",
    "EMBARGO_MINUTES",
    "FEATURE_NAMES",
    "FORBIDDEN_FEATURES",
    "HORIZONS",
    "LEVERAGES",
    "MODEL_PARAMS",
    "N_FOLDS",
    "QUANTILE_ALPHAS",
    "SklearnQuantileModel",
    "WalkForwardFold",
    "assert_no_future_features",
    "baseline_positions",
    "build_features",
    "cluster_t_days",
    "daily_pnl",
    "evaluate_policy",
    "fit_quantile_model",
    "make_quantile_model",
    "make_walk_forward_folds",
    "max_drawdown_bps",
    "model_backend",
    "policy_net_pnl",
    "policy_positions",
    "sharpe_annual",
    "spearman_ic",
]

LOG = get_logger("spp.crypto.models")

#: Forward horizons in minutes; one regression target per horizon.
HORIZONS: Final[tuple[int, ...]] = (5, 15, 60)
#: Quantile levels of the models (q10 / q50 / q90).
QUANTILE_ALPHAS: Final[tuple[float, ...]] = (0.1, 0.5, 0.9)
#: Leverage ladder of the policy.
LEVERAGES: Final[tuple[int, ...]] = (1, 2, 3, 5)
#: Round-trip cost of a perpetual-futures trade (14 bps), same units as the returns.
DEFAULT_COST: Final[float] = 0.0014
#: Benchmark symbol for the market-context features.
BENCH_SYMBOL: Final[str] = "BTCUSDT"
#: Walk-forward configuration over the train split.
N_FOLDS: Final[int] = 5
EMBARGO_MINUTES: Final[int] = 60
#: Look-back of the ``news_burst_60m`` feature.
BURST_WINDOW_MINUTES: Final[int] = 60

#: Nanoseconds in one minute.
NS_PER_MIN: Final[int] = 60 * 10**9

#: Exact feature column order returned by :func:`build_features`.
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "hour_sin",
    "hour_cos",
    "dow",
    "asset_code",
    "log_lag",
    "tr_ret_5m",
    "tr_ret_15m",
    "tr_ret_60m",
    "tr_vol_30m",
    "tr_vol_60m",
    "btc_ret_5m",
    "btc_ret_15m",
    "btc_ret_60m",
    "btc_vol_30m",
    "news_burst_60m",
)

#: Integer-typed features (everything else is float64).
_INT_FEATURES: Final[frozenset[str]] = frozenset({"dow", "asset_code", "news_burst_60m"})

#: Columns that must never appear as features (see the module docstring).
FORBIDDEN_FEATURES: Final[tuple[str, ...]] = (
    "quote_volume_entry",
    "trades_entry",
    "ret_5m",
    "ret_15m",
    "ret_60m",
    "bench_5m",
    "bench_15m",
    "bench_60m",
    "excess_5m",
    "excess_15m",
    "excess_60m",
)

#: Hard-regularised LightGBM hyperparameters for ~7k rows (fixed, never tuned on val).
MODEL_PARAMS: Final[dict[str, Any]] = {
    "num_leaves": 15,
    "min_data_in_leaf": 100,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "random_state": 42,
}


# --------------------------------------------------------------------------- #
# Bar helpers
# --------------------------------------------------------------------------- #
def _ns_values(values: Any) -> np.ndarray:
    """UTC nanosecond int64 view of any datetime-like input (unit-safe)."""
    return pd.to_datetime(values, utc=True).to_numpy(dtype="datetime64[ns]").astype("int64")


def _bar_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(stamps_ns, opens, closes)`` of a bar frame, sorted ascending by stamp."""
    stamps = _ns_values(frame["ts_open"])
    opens = frame["open"].to_numpy(dtype="float64")
    closes = frame["close"].to_numpy(dtype="float64")
    if stamps.size > 1 and np.any(np.diff(stamps) < 0):
        order = np.argsort(stamps, kind="stable")
        stamps, opens, closes = stamps[order], opens[order], closes[order]
    return stamps, opens, closes


def _last_at_or_before(stamps: np.ndarray, target_ns: int) -> int:
    """Index of the last bar with ``ts_open <= target_ns`` (``-1`` when absent)."""
    return int(np.searchsorted(stamps, target_ns, side="right")) - 1


def _trailing_return(
    stamps: np.ndarray, opens: np.ndarray, closes: np.ndarray, entry_ns: int, horizon: int
) -> float:
    """Return of the ``horizon`` bars ending at ``entry_ts`` (open of the first bar
    ``entry - horizon`` to close of the bar ``entry - 1``, i.e. exactly ``horizon``
    minutes of exposure, every stamp strictly before ``entry_ns``). A missing bar makes
    the value stale but never future — causality wins over freshness.
    """
    end_i = _last_at_or_before(stamps, entry_ns - NS_PER_MIN)
    start_i = _last_at_or_before(stamps, entry_ns - horizon * NS_PER_MIN)
    if end_i < 0 or start_i < 0:
        return float("nan")
    start, end = float(opens[start_i]), float(closes[end_i])
    if not np.isfinite(start) or not np.isfinite(end) or start <= 0.0:
        return float("nan")
    return end / start - 1.0


def _trailing_vol(stamps: np.ndarray, closes: np.ndarray, entry_ns: int, window: int) -> float:
    """Sample std (ddof=1) of 1-minute log returns inside ``[entry - window, entry)``.

    Only bars strictly before ``entry_ns`` enter the window; fewer than three closes
    (fewer than two returns) yields NaN.
    """
    lo = int(np.searchsorted(stamps, entry_ns - window * NS_PER_MIN, side="left"))
    hi = int(np.searchsorted(stamps, entry_ns, side="left"))
    if hi - lo < 3:
        return float("nan")
    seg = np.asarray(closes[lo:hi], dtype="float64")
    seg = seg[np.isfinite(seg) & (seg > 0.0)]
    if seg.size < 3:
        return float("nan")
    logret = np.diff(np.log(seg))
    if logret.size < 2:
        return float("nan")
    return float(np.std(logret, ddof=1))


def _truncate_bars(bars: Mapping[str, pd.DataFrame], cut_ns: int) -> dict[str, pd.DataFrame]:
    """Copy of ``bars`` with every bar ``ts_open >= cut_ns`` removed."""
    out: dict[str, pd.DataFrame] = {}
    for symbol, frame in bars.items():
        if frame is None or len(frame) == 0:
            out[symbol] = frame
            continue
        stamps = _ns_values(frame["ts_open"])
        if np.all(np.diff(stamps) >= 0):
            k = int(np.searchsorted(stamps, cut_ns, side="left"))
            out[symbol] = frame if k >= len(frame) else frame.iloc[:k]
        else:
            out[symbol] = frame[stamps < cut_ns]
    return out


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def build_features(
    events: pd.DataFrame, bars: Mapping[str, pd.DataFrame]
) -> tuple[pd.DataFrame, list[str]]:
    """Causal feature matrix aligned to ``events.index`` plus the feature-name list.

    ``bars`` maps symbol -> 1-minute frame with ``ts_open``/``open``/``close``; only
    rows strictly before ``entry_ts`` are ever read (see the module docstring for the
    full causality contract and the forbidden columns). Empty ``events`` yields an
    empty frame with the exact :data:`FEATURE_NAMES` schema.
    """
    names = list(FEATURE_NAMES)
    if len(events) == 0:
        empty = pd.DataFrame(index=events.index)
        for name in names:
            empty[name] = pd.Series(dtype="int64" if name in _INT_FEATURES else "float64")
        return empty, names

    entry = pd.to_datetime(events["entry_ts"], utc=True)
    entry_ns = _ns_values(entry)
    hour = entry.dt.hour.to_numpy(dtype="float64")
    hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
    dow = entry.dt.day_of_week.to_numpy(dtype="int64")

    symbols = events["symbol"]
    # Codes are the sorted categories of THIS frame: build train and eval in one call
    # so the coding is identical on both sides of the split.
    asset_code = np.asarray(pd.Categorical(symbols).codes, dtype="int64")
    sym_list = [str(s) for s in symbols.tolist()]
    log_lag = np.log1p(events["lag_minutes"].to_numpy(dtype="float64"))

    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}

    def arrays(symbol: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if symbol not in cache:
            frame = bars.get(symbol) if hasattr(bars, "get") else None
            cache[symbol] = _bar_arrays(frame) if frame is not None and len(frame) else None
        return cache[symbol]

    bench = arrays(BENCH_SYMBOL)

    # Causal burst index: entry stamps per asset, sorted; strictly-earlier only.
    burst_index: dict[str, np.ndarray] = {}
    for sym in pd.unique(symbols):
        mask = (symbols == sym).to_numpy()
        burst_index[str(sym)] = np.sort(entry_ns[mask])
    burst_ns = BURST_WINDOW_MINUTES * NS_PER_MIN

    n = len(events)
    tr_ret = {h: np.full(n, np.nan, dtype="float64") for h in HORIZONS}
    tr_vol = {30: np.full(n, np.nan), 60: np.full(n, np.nan)}
    btc_ret = {h: np.full(n, np.nan, dtype="float64") for h in HORIZONS}
    btc_vol30 = np.full(n, np.nan)
    burst = np.zeros(n, dtype="int64")

    for i in range(n):
        t = int(entry_ns[i])
        arr = arrays(sym_list[i])
        if arr is not None:
            stamps, opens, closes = arr
            for h in HORIZONS:
                tr_ret[h][i] = _trailing_return(stamps, opens, closes, t, h)
            tr_vol[30][i] = _trailing_vol(stamps, closes, t, 30)
            tr_vol[60][i] = _trailing_vol(stamps, closes, t, 60)
        if bench is not None:
            b_stamps, b_opens, b_closes = bench
            for h in HORIZONS:
                btc_ret[h][i] = _trailing_return(b_stamps, b_opens, b_closes, t, h)
            btc_vol30[i] = _trailing_vol(b_stamps, b_closes, t, 30)
        idx = burst_index[sym_list[i]]
        lo = int(np.searchsorted(idx, t - burst_ns, side="left"))
        hi = int(np.searchsorted(idx, t, side="left"))
        burst[i] = hi - lo

    data: dict[str, np.ndarray] = {
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow": dow,
        "asset_code": asset_code,
        "log_lag": log_lag,
        "tr_ret_5m": tr_ret[5],
        "tr_ret_15m": tr_ret[15],
        "tr_ret_60m": tr_ret[60],
        "tr_vol_30m": tr_vol[30],
        "tr_vol_60m": tr_vol[60],
        "btc_ret_5m": btc_ret[5],
        "btc_ret_15m": btc_ret[15],
        "btc_ret_60m": btc_ret[60],
        "btc_vol_30m": btc_vol30,
        "news_burst_60m": burst,
    }
    X = pd.DataFrame({name: data[name] for name in names}, index=events.index)
    return X, names


def assert_no_future_features(
    events: pd.DataFrame,
    bars: Mapping[str, pd.DataFrame],
    feature_names: list[str],
    *,
    max_cuts: int = 12,
) -> None:
    """Raise :class:`AssertionError` when a feature reads a bar at/after ``entry_ts``.

    Three checks, in order:

    1. static: a forbidden column (``quote_volume_entry``, ``trades_entry``, any
       ``ret_*``/``bench_*``/``excess_*`` of the current event) is rejected outright;
    2. certification: every requested name must be produced by :func:`build_features`
       — an unknown name cannot be certified causal, so it is rejected;
    3. numeric: for up to ``max_cuts`` distinct ``entry_ts`` values (first, last and
       evenly spaced), the features are recomputed from a copy of ``bars`` with every
       bar ``ts_open >= cut`` removed. For the rows at that cut a causal feature is
       bit-identical (it never touched those bars); a future-looking feature changes
       and the offending names are reported.
    """
    names = list(feature_names)
    for name in names:
        if name in FORBIDDEN_FEATURES or name.startswith(("excess_", "bench_")):
            raise AssertionError(
                f"feature {name!r} is forbidden: it is a label or a look-ahead column "
                "(quote_volume_entry/trades_entry close one minute after entry_ts)"
            )

    X_ref, ref_names = build_features(events, bars)
    unknown = [name for name in names if name not in ref_names]
    if unknown:
        raise AssertionError(
            f"feature(s) {unknown} are not produced by build_features and cannot be "
            "certified as causal"
        )
    if len(events) == 0:
        return

    entry_ns = _ns_values(events["entry_ts"])
    cuts = np.unique(entry_ns)
    if cuts.size > max_cuts:
        picks = np.unique(np.linspace(0, cuts.size - 1, max_cuts).round().astype(int))
        cuts = cuts[picks]

    for cut in cuts:
        pos = np.flatnonzero(entry_ns == int(cut))
        if pos.size == 0:  # pragma: no cover - unique() guarantees a hit
            continue
        truncated = _truncate_bars(bars, int(cut))
        X_cut, _ = build_features(events, truncated)
        left = X_ref[names].iloc[pos].to_numpy(dtype="float64")
        right = X_cut[names].iloc[pos].to_numpy(dtype="float64")
        bad = ~np.isclose(left, right, rtol=1e-9, atol=1e-12, equal_nan=True)
        if bad.any():
            offenders = [names[j] for j in np.flatnonzero(bad.any(axis=0))]
            raise AssertionError(
                f"feature(s) {offenders} change when bars at/after "
                f"{pd.Timestamp(int(cut))} are removed: they read bar data at or after "
                "entry_ts"
            )
    LOG.debug(
        "assert_no_future_features: %d feature(s) causal over %d cut(s)", len(names), cuts.size
    )


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class SklearnQuantileModel:
    """Fallback quantile model: sklearn ``GradientBoostingRegressor(loss="quantile")``.

    Used only when ``lightgbm`` is not importable. Gradient boosting cannot consume
    NaN, so columns are median-imputed at fit time (medians frozen from the training
    fold) and reused at predict time.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.model = GradientBoostingRegressor(
            loss="quantile",
            alpha=self.alpha,
            n_estimators=MODEL_PARAMS["n_estimators"],
            learning_rate=MODEL_PARAMS["learning_rate"],
            max_depth=4,  # ~ num_leaves 15
            min_samples_leaf=MODEL_PARAMS["min_data_in_leaf"],
            subsample=MODEL_PARAMS["bagging_fraction"],
            max_features=MODEL_PARAMS["feature_fraction"],
            random_state=MODEL_PARAMS["random_state"],
        )
        self.medians: pd.Series | None = None

    @staticmethod
    def _frame(X: Any) -> pd.DataFrame:
        return pd.DataFrame(np.asarray(X, dtype="float64"), columns=getattr(X, "columns", None))

    def fit(self, X: Any, y: Any) -> SklearnQuantileModel:
        frame = self._frame(X)
        self.medians = frame.median().fillna(0.0)
        self.model.fit(frame.fillna(self.medians), np.asarray(y, dtype="float64"))
        return self

    def predict(self, X: Any) -> np.ndarray:
        frame = self._frame(X)
        assert self.medians is not None, "predict() before fit()"
        return self.model.predict(frame.fillna(self.medians))


def model_backend() -> str:
    """``"lightgbm"`` normally, ``"sklearn"`` when lightgbm is not importable."""
    return "lightgbm" if LGBMRegressor is not None else "sklearn"


def make_quantile_model(alpha: float) -> Any:
    """Unfitted quantile model at ``alpha`` (LightGBM, sklearn fallback)."""
    if LGBMRegressor is not None:
        return LGBMRegressor(objective="quantile", alpha=float(alpha), **MODEL_PARAMS)
    return SklearnQuantileModel(alpha)


def fit_quantile_model(alpha: float, X: Any, y: Any) -> Any:
    """Fit and return a quantile model at ``alpha`` on ``(X, y)``."""
    model = make_quantile_model(alpha)
    model.fit(X, y)
    return model


# --------------------------------------------------------------------------- #
# Purged walk-forward folds
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WalkForwardFold:
    """One expanding purged fold over the train split.

    ``train_idx``/``test_idx`` are index labels of ``events``; ``purged_rows`` counts
    train rows dropped because their label window touched the boundary or they sat
    inside the embargo.
    """

    fold_id: int
    train_idx: pd.Index
    test_idx: pd.Index
    purged_rows: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold_id,
            "train_rows": int(len(self.train_idx)),
            "test_rows": int(len(self.test_idx)),
            "purged_rows": self.purged_rows,
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


def make_walk_forward_folds(
    events: pd.DataFrame,
    *,
    n_folds: int = N_FOLDS,
    embargo_minutes: int = EMBARGO_MINUTES,
    label_minutes: int = max(HORIZONS),
    split: str | None = "train",
) -> list[WalkForwardFold]:
    """``n_folds`` expanding purged walk-forward folds over one split of ``events``.

    The split is sorted by ``entry_ts`` and cut into ``n_folds + 1`` contiguous
    segments: segment 0 is the initial training block, segment ``k + 1`` is the test
    block of fold ``k`` and segments ``0..k`` its training block. A train row is
    purged when ``entry_ts + label_minutes >= test_start`` (its label window touches
    the boundary) or ``entry_ts >= test_start - embargo_minutes``. Train and test are
    index-disjoint by construction and the folds are expanding over time.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    if split is None or "split" not in events.columns:
        sub = events
    else:
        sub = events[events["split"] == split]
    if len(sub) < n_folds + 1:
        raise ValueError(
            f"walk-forward needs at least n_folds+1 = {n_folds + 1} rows, got {len(sub)}"
        )
    entry_ns = _ns_values(sub["entry_ts"])
    order = np.argsort(entry_ns, kind="stable")
    labels = sub.index.to_numpy()[order]
    stamps = entry_ns[order]

    chunks = np.array_split(np.arange(len(sub)), n_folds + 1)
    folds: list[WalkForwardFold] = []
    embargo_ns = embargo_minutes * NS_PER_MIN
    label_ns = label_minutes * NS_PER_MIN
    for k in range(n_folds):
        test_pos = chunks[k + 1]
        if test_pos.size == 0:  # pragma: no cover - guarded by the row count above
            raise ValueError(f"fold {k}: empty test segment (rows={len(sub)}, n_folds={n_folds})")
        train_pos = np.concatenate(chunks[: k + 1])
        test_start_ns = int(stamps[test_pos[0]])
        purge = (stamps[train_pos] + label_ns >= test_start_ns) | (
            stamps[train_pos] >= test_start_ns - embargo_ns
        )
        kept = train_pos[~purge]
        train_idx = pd.Index(labels[kept])
        test_idx = pd.Index(labels[test_pos])
        shared = train_idx.intersection(test_idx)
        if len(shared):  # pragma: no cover - segments are disjoint by construction
            raise AssertionError(f"fold {k}: train and test share {len(shared)} row(s)")
        folds.append(
            WalkForwardFold(
                fold_id=k,
                train_idx=train_idx,
                test_idx=test_idx,
                purged_rows=int(np.count_nonzero(purge)),
                test_start=pd.to_datetime(int(stamps[test_pos[0]]), unit="ns", utc=True),
                test_end=pd.to_datetime(int(stamps[test_pos[-1]]), unit="ns", utc=True),
            )
        )
    return folds


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def spearman_ic(y_true: Any, y_pred: Any) -> float:
    """Spearman rank correlation; NaN with < 3 finite pairs or a constant side."""
    a = pd.Series(np.asarray(y_true, dtype="float64")).reset_index(drop=True)
    b = pd.Series(np.asarray(y_pred, dtype="float64")).reset_index(drop=True)
    if len(a) != len(b):
        raise ValueError(f"length mismatch: y_true {len(a)} vs y_pred {len(b)}")
    keep = np.isfinite(a.to_numpy()) & np.isfinite(b.to_numpy())
    if int(keep.sum()) < 3:
        return float("nan")
    return float(a[keep].corr(b[keep], method="spearman"))


def daily_pnl(entry_ts: Any, per_event_pnl: Any) -> pd.Series:
    """Total PnL per UTC day, reindexed over the full day range (missing days = 0).

    The zero-filled days matter: a day with no trade contributes exactly nothing to
    the Sharpe ratio, the cluster t and the drawdown.
    """
    values = np.asarray(per_event_pnl, dtype="float64")
    if values.size == 0:
        return pd.Series(dtype="float64")
    days = pd.DatetimeIndex(pd.to_datetime(entry_ts, utc=True)).normalize()
    if len(days) != values.size:
        raise ValueError(f"length mismatch: entry_ts {len(days)} vs pnl {values.size}")
    totals = pd.Series(values, index=days).groupby(level=0).sum()
    totals = totals.reindex(pd.date_range(totals.index[0], totals.index[-1], freq="D"))
    return totals.fillna(0.0).astype("float64")


def sharpe_annual(daily: Any, periods_per_year: float = 365.0) -> float:
    """Annualised Sharpe from daily PnL (24/7 market -> sqrt(365))."""
    values = np.asarray(daily, dtype="float64")
    if values.size < 2:
        return float("nan")
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    if not np.isfinite(std) or std == 0.0:
        return 0.0 if mean == 0.0 else float("nan")
    return mean / std * float(np.sqrt(periods_per_year))


def cluster_t_days(daily: Any) -> float:
    """Cluster-robust t over days: day clusters collapse to daily PnL totals."""
    values = np.asarray(daily, dtype="float64")
    n = values.size
    if n < 2:
        return float("nan")
    std = float(values.std(ddof=1))
    if not np.isfinite(std) or std == 0.0:
        return float("nan")
    return float(values.mean() / (std / np.sqrt(n)))


def max_drawdown_bps(daily: Any) -> float:
    """Max peak-to-trough drawdown of the cumulative daily PnL, in bps."""
    values = np.asarray(daily, dtype="float64")
    if values.size == 0:
        return 0.0
    cum = np.cumsum(values)
    peak = np.maximum.accumulate(cum)
    return float(np.max(peak - cum)) * 1e4


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
def policy_positions(pred_q50: Any, hurdle: float, leverage: int | float) -> np.ndarray:
    """``L * sign(pred)`` where ``|pred| >= hurdle``, else 0 (NaN predictions mute)."""
    pred = np.asarray(pred_q50, dtype="float64")
    pos = np.zeros_like(pred)
    active = np.isfinite(pred) & (np.abs(pred) >= float(hurdle))
    pos[active] = np.sign(pred[active])
    return float(leverage) * pos


def policy_net_pnl(positions: Any, excess: Any, cost: Any, legs: float = 1.0) -> np.ndarray:
    """Net PnL per event: ``position * excess - |position| * cost * legs``.

    Equal to ``L * (sign * excess - cost * legs)`` — leverage scales the edge and the
    fees by the same factor and therefore cannot flip the sign of the expectancy.

    ``legs`` is the number of cost-paying legs of the traded position. It is 1 for a
    single instrument, and **2 when the label is an excess return**: ``excess = ret -
    bench`` can only be earned by holding the asset and shorting the benchmark, which
    pays a round trip twice. Passing 1 there understates the fees by exactly one leg.
    """
    pos = np.asarray(positions, dtype="float64")
    exc = np.asarray(excess, dtype="float64")
    fees = np.broadcast_to(np.asarray(cost, dtype="float64") * float(legs), pos.shape)
    return pos * exc - np.abs(pos) * fees


def evaluate_policy(
    entry_ts: Any, positions: Any, excess: Any, cost: Any, legs: float = 1.0
) -> dict[str, Any]:
    """Policy metrics on one horizon: trade counts, bps, day-clustered t, Sharpe, DD.

    ``gross_bps`` is per *traded* event (1e4 * mean(sign * excess)); ``net_bps`` is
    per *evaluated* event including muted zeros (1e4 * mean(net)); NaN labels are
    excluded before anything is computed. ``sharpe_annual`` uses daily PnL with
    sqrt(365); ``t_cluster_days`` treats each UTC day as one cluster.
    """
    pos = np.asarray(positions, dtype="float64")
    exc = np.asarray(excess, dtype="float64")
    if not (len(pos) == len(exc)):
        raise ValueError(f"length mismatch: positions {len(pos)} vs excess {len(exc)}")
    cost_arr = np.broadcast_to(np.asarray(cost, dtype="float64"), pos.shape)
    valid = np.isfinite(pos) & np.isfinite(exc)
    n = int(np.count_nonzero(valid))
    if n == 0:
        return {
            "n_eval": 0,
            "n_trades": 0,
            "muted_share": 0.0,
            "gross_bps": float("nan"),
            "net_bps": 0.0,
            "net_bps_per_trade": float("nan"),
            "t_cluster_days": float("nan"),
            "sharpe_annual": float("nan"),
            "max_drawdown_bps": 0.0,
            "n_days": 0,
        }
    pos, exc = pos[valid], exc[valid]
    fees = cost_arr[valid]
    entries = pd.DatetimeIndex(pd.to_datetime(entry_ts, utc=True))[valid]
    traded = pos != 0.0
    n_trades = int(np.count_nonzero(traded))
    net = policy_net_pnl(pos, exc, fees, legs=legs)
    daily = daily_pnl(entries, net)
    return {
        "n_eval": n,
        "n_trades": n_trades,
        "muted_share": float(1.0 - n_trades / n),
        "gross_bps": float(1e4 * np.mean(np.sign(pos[traded]) * exc[traded]))
        if n_trades
        else float("nan"),
        "net_bps": float(1e4 * net.mean()),
        "net_bps_per_trade": float(1e4 * net[traded].mean()) if n_trades else float("nan"),
        "t_cluster_days": cluster_t_days(daily),
        "sharpe_annual": sharpe_annual(daily),
        "max_drawdown_bps": max_drawdown_bps(daily),
        "n_days": int(len(daily)),
    }


def baseline_positions(
    kind: str,
    trailing_15m: Any,
    asset: Any,
    train_asset_mean_excess: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Model-free baseline directions (L = 1) on the same rows as the policy.

    ``flat`` trades nothing; ``long``/``short`` take a fixed direction on every event;
    ``momentum`` uses ``sign(trailing 15-minute return)`` (NaN or exactly 0 mutes);
    ``per_asset_constant`` uses ``sign(train mean excess)`` per asset — the direction
    is selected **in-sample** from the training split, so it is an upper bound, not a
    tradeable strategy.
    """
    n = len(trailing_15m)
    if kind == "flat":
        return np.zeros(n, dtype="float64")
    if kind == "long":
        return np.ones(n, dtype="float64")
    if kind == "short":
        return -np.ones(n, dtype="float64")
    if kind == "momentum":
        ret = np.asarray(trailing_15m, dtype="float64")
        pos = np.zeros(n, dtype="float64")
        finite = np.isfinite(ret)
        pos[finite] = np.sign(ret[finite])
        return pos
    if kind == "per_asset_constant":
        if train_asset_mean_excess is None:
            raise ValueError("per_asset_constant needs train_asset_mean_excess")
        symbols = [str(s) for s in asset]
        means = train_asset_mean_excess
        pos = np.zeros(n, dtype="float64")
        for i, sym in enumerate(symbols):
            value = means.get(sym, float("nan"))
            if np.isfinite(value):
                pos[i] = np.sign(value)
        return pos
    raise ValueError(f"unknown baseline kind: {kind!r}")
