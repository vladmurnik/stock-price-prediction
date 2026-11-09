"""T5 — crypto event dataset: every headline joined to the first tradeable 1m bar after it.

Contract (frozen with ``configs/crypto_dataset.yaml`` v1):

* One row per ``(news row, ticker)`` pair. ``target = published_at + news_lag_minutes``;
  ``entry_ts`` is the ``ts_open`` of the **first bar with** ``ts_open >= target`` — a binary
  search (bisect) over the symbol's stamps, never a scan per row. The pair is dropped as
  ``no_bar_at_or_after`` when no such bar exists (the headline is after the last bar) or
  when the eligible moment precedes the symbol's very first bar (a later listing such as
  TONUSDT-2024-08, or a window clipped at the data start): entering there would join the
  headline to a distant first bar with no observed market in between.
* ``entry_price`` is the open of that bar. For horizon ``h`` the exit bar starts at
  ``entry_ts + (h - 1)`` minutes and its **close** is used: open of the entry bar to close
  of the last bar inside the window is exactly ``h`` minutes of exposure. The bar starting
  at ``entry + h`` (one extra minute) is never used; a missing exit bar makes that
  horizon's return NaN and increments ``missing_exit_<h>``.
* ``bench_<h>`` is BTCUSDT over the identical stamps (open at ``entry_ts`` -> close at the
  same exit stamp), computed once per distinct ``entry_ts`` and reused; ``excess_<h> =
  ret_<h> - bench_<h>``. ``cost = round_trip_perp_bps / 10000`` is stored once per event
  and never subtracted here — the evaluator subtracts it once per event.
* ``lag_minutes`` is the realised lag ``(entry_ts - published_at)`` in minutes, which is
  always ``>= news_lag_minutes`` (a 12:00:30 headline enters at the 12:02 open with
  ``lag_minutes == 1.5``).
* Splits: ``train``/``val``/``holdout`` from the config cut dates, ``outside`` beyond the
  holdout, and ``embargo`` for any row within ``embargo_minutes`` of a split boundary
  (boundaries: train end, holdout start, holdout end — midnight UTC instants).
* Funnel balance is enforced: ``kept + duplicate + na_published_at + no_symbol +
  no_bar_at_or_after == total_pairs`` or :class:`ValueError` is raised with the counts.
  ``missing_exit_*`` counts rows that are kept, so they never enter the balance.
* ``tickers`` are resolved through :data:`SYMBOL_MAP`, built at import from the config
  ``aliases`` keys/values plus the suffix rule ``"name" -> "NAMEUSDT"`` and validated so
  that all twelve canonical config names (bitcoin, ethereum, ...) resolve. An unknown
  ticker is dropped as ``no_symbol``, never silently.

The news harvest rewrites ``wayback.parquet`` while this module runs: callers must read
the file once and hand the in-memory frame to :func:`build_dataset`, which never touches
the filesystem.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import yaml

from spp.logging import get_logger

__all__ = [
    "CANONICAL_ASSETS",
    "DATASET_COLUMNS",
    "DEFAULT_CONFIG_PATH",
    "FUNNEL_DROP_KEYS",
    "FUNNEL_KEYS",
    "HORIZONS",
    "LazyBars",
    "asset_symbol_map",
    "build_dataset",
    "empty_dataset",
    "empty_funnel",
    "load_bars",
    "load_dataset_config",
    "resolve_symbol",
    "run_checks",
]

LOG = get_logger("spp.crypto.dataset")

#: Repo root resolved from this file (``src/spp/crypto/dataset.py``), never from the cwd.
ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH: Final[Path] = ROOT / "configs" / "crypto_dataset.yaml"

#: Forward horizons in minutes. The output schema is frozen around them.
HORIZONS: Final[tuple[int, ...]] = (5, 15, 60)

_NS_PER_MINUTE: Final[int] = 60 * 10**9

#: Exact output column order of every frame :func:`build_dataset` returns.
DATASET_COLUMNS: Final[tuple[str, ...]] = (
    "asset",
    "symbol",
    "source",
    "title",
    "url",
    "published_at",
    "entry_ts",
    "entry_price",
    "lag_minutes",
    "ret_5m",
    "bench_5m",
    "excess_5m",
    "ret_15m",
    "bench_15m",
    "excess_15m",
    "ret_60m",
    "bench_60m",
    "excess_60m",
    "cost",
    "quote_volume_entry",
    "trades_entry",
    "snapshot_ts",
    "split",
    "event_kind",
)

#: dtypes of :func:`empty_dataset` (and of every built frame, after finalisation).
_COLUMN_DTYPES: Final[dict[str, str]] = {
    "asset": "str",
    "symbol": "str",
    "source": "str",
    "title": "str",
    "url": "str",
    "published_at": "datetime64[ns, UTC]",
    "entry_ts": "datetime64[ns, UTC]",
    "entry_price": "float64",
    "lag_minutes": "float64",
    "ret_5m": "float64",
    "bench_5m": "float64",
    "excess_5m": "float64",
    "ret_15m": "float64",
    "bench_15m": "float64",
    "excess_15m": "float64",
    "ret_60m": "float64",
    "bench_60m": "float64",
    "excess_60m": "float64",
    "cost": "float64",
    "quote_volume_entry": "float64",
    "trades_entry": "int64",
    "snapshot_ts": "datetime64[ns, UTC]",
    "split": "str",
    "event_kind": "str",
}

#: Funnel buckets that *drop* a pair; their sum plus ``kept`` must equal ``total_pairs``.
FUNNEL_DROP_KEYS: Final[tuple[str, ...]] = (
    "duplicate",
    "na_published_at",
    "no_symbol",
    "benchmark_asset",
    "no_bar_at_or_after",
)

#: Every funnel key. ``missing_exit_*`` count kept rows, not drops.
FUNNEL_KEYS: Final[tuple[str, ...]] = (
    "total_pairs",
    "kept",
    *FUNNEL_DROP_KEYS,
    *(f"missing_exit_{h}m" for h in HORIZONS),
)

#: The twelve config names from ``archive.query_terms`` mapped to their Binance symbol.
CANONICAL_ASSETS: Final[dict[str, str]] = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "solana": "SOLUSDT",
    "binance": "BNBUSDT",
    "xrp": "XRPUSDT",
    "dogecoin": "DOGEUSDT",
    "avalanche": "AVAXUSDT",
    "chainlink": "LINKUSDT",
    "cardano": "ADAUSDT",
    "toncoin": "TONUSDT",
    "arbitrum": "ARBUSDT",
    "optimism": "OPUSDT",
}

#: Fallbacks used verbatim when a config key is absent (the file itself must exist).
_CONFIG_DEFAULTS: Final[dict[str, dict[str, Any]]] = {
    "labels": {
        "horizons": [5, 15, 60],
        "news_lag_minutes": 1,
        "excess_vs": "BTCUSDT",
        "dead_zone": 0.05,
    },
    "costs": {"round_trip_perp_bps": 14.0},
    "splits": {
        "train_end": "2025-12-31",
        "val_end": "2026-09-14",
        "holdout_start": "2026-09-15",
        "holdout_end": "2026-11-08",
        "embargo_minutes": 60,
    },
}

#: Names of the boolean leakage checks returned by :func:`run_checks`.
_CHECK_NAMES: Final[tuple[str, ...]] = (
    "entry_not_before_news_lag",
    "bench_same_window",
    "no_bar_earlier_than_entry",
    "split_boundaries_disjoint",
    "no_duplicate_url_symbol",
    "entry_price_matches_bar",
    "realised_lag_consistent",
)

#: Columns :func:`load_bars` reads from each parquet (the assembler's contract fields).
_BAR_READ_COLUMNS: Final[tuple[str, ...]] = (
    "ts_open",
    "open",
    "close",
    "quote_volume",
    "trades",
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _utc(value: Any) -> pd.Timestamp:
    """Timestamp in UTC; naive input is localised, aware input converted."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _day_end(value: Any) -> pd.Timestamp:
    """Last second of the config date: ``2025-12-31`` -> ``2025-12-31 23:59:59`` UTC."""
    return _utc(value) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


def _resolve_config(config: Mapping[str, Any], source: str) -> dict[str, Any]:
    """Return a copy of ``config`` with every dataset key resolved to a concrete value.

    ``labels.horizons`` also accepts the config's alternate spelling
    ``labels.horizons_minutes`` (no warning when only that is present); any other missing
    key falls back to :data:`_CONFIG_DEFAULTS` and emits a :class:`RuntimeWarning`.
    ``source`` only labels the warning.
    """
    out: dict[str, Any] = dict(config)
    for section, defaults in _CONFIG_DEFAULTS.items():
        node = dict(config.get(section) or {})
        for key, default in defaults.items():
            if node.get(key) is not None:
                continue
            is_horizons = (section, key) == ("labels", "horizons")
            alternate = node.get("horizons_minutes") if is_horizons else None
            if alternate is not None:
                node["horizons"] = alternate
                continue
            warnings.warn(
                f"crypto config {source}: missing {section}.{key}; using default {default!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            node[key] = default
        out[section] = node
    labels = out["labels"]
    labels["horizons"] = [int(h) for h in labels["horizons"]]
    labels["news_lag_minutes"] = float(labels["news_lag_minutes"])
    labels["excess_vs"] = str(labels["excess_vs"])
    labels["dead_zone"] = float(labels["dead_zone"])
    out["costs"]["round_trip_perp_bps"] = float(out["costs"]["round_trip_perp_bps"])
    out["splits"]["embargo_minutes"] = int(out["splits"]["embargo_minutes"])
    return out


def load_dataset_config(path: Path | str | None = None) -> dict[str, Any]:
    """Read ``configs/crypto_dataset.yaml`` with every dataset key resolved.

    The YAML file is never modified. A missing *file* raises :class:`FileNotFoundError`
    (same contract as :func:`spp.crypto.bars.load_frozen_config`); a missing *key* is
    filled from the frozen defaults with a :class:`RuntimeWarning`.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"crypto config not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {cfg_path}")
    return _resolve_config(raw, str(cfg_path))


# --------------------------------------------------------------------------- #
# Asset mapping
# --------------------------------------------------------------------------- #
def asset_symbol_map(config: Mapping[str, Any]) -> dict[str, str]:
    """Ticker name -> Binance symbol, built from the config ``aliases`` section.

    Three sources are merged: the ``aliases`` keys (symbols) and values (names such as
    ``bitcoin``), the hardcoded suffix rule ``"name" -> "NAMEUSDT"`` (a candidate is kept
    only when it is one of the twelve known symbols, so ``btc`` -> ``BTCUSDT`` while
    ``shib`` -> ``SHIBAUSDT`` is rejected), and :data:`CANONICAL_ASSETS` for names the
    suffix rule cannot form (``binance`` -> ``BNBUSDT``). Weak tickers like ``link`` or
    ``op`` resolve through the suffix rule applied to the symbol base.
    """
    known = {str(s).strip().upper() for s in (config.get("symbols") or [])}
    known |= set(CANONICAL_ASSETS.values())
    mapping: dict[str, str] = {}
    aliases = config.get("aliases") or {}
    for symbol, names in aliases.items():
        sym = str(symbol).strip().upper()
        mapping[sym.lower()] = sym
        for name in names or []:
            mapping.setdefault(str(name).strip().lower(), sym)
    # Suffix rule: "name" -> "NAMEUSDT", kept only when the candidate is a known symbol.
    for name in [*mapping, *CANONICAL_ASSETS]:
        upper = name.upper()
        candidate = upper if upper.endswith("USDT") else f"{upper}USDT"
        if candidate in known:
            mapping.setdefault(name, candidate)
    for sym in known:
        mapping.setdefault(sym[: -len("USDT")].lower(), sym)
    for name, sym in CANONICAL_ASSETS.items():
        mapping.setdefault(name, sym)
    return mapping


def resolve_symbol(ticker: object, mapping: Mapping[str, str] | None = None) -> str | None:
    """Symbol for ``ticker`` or ``None`` when it is unknown (counted as ``no_symbol``)."""
    table = SYMBOL_MAP if mapping is None else mapping
    if ticker is None:
        return None
    key = str(ticker).strip().lower()
    if not key:
        return None
    hit = table.get(key)
    if hit:
        return hit
    upper = key.upper()
    candidate = upper if upper.endswith("USDT") else f"{upper}USDT"
    return candidate if candidate in set(table.values()) else None


#: Module-level map, validated at import so all twelve canonical names resolve.
SYMBOL_MAP: Final[dict[str, str]] = asset_symbol_map(load_dataset_config())
for _name, _expected in CANONICAL_ASSETS.items():
    if SYMBOL_MAP.get(_name) != _expected:
        raise ValueError(
            f"asset map: {_name!r} resolved to {SYMBOL_MAP.get(_name)!r}, expected {_expected!r}"
        )
for _sym in CANONICAL_ASSETS.values():
    if SYMBOL_MAP.get(_sym.lower()) != _sym:
        raise ValueError(f"asset map: symbol {_sym} does not map to itself")


# --------------------------------------------------------------------------- #
# News normalisation
# --------------------------------------------------------------------------- #
_NEWS_TEXT_COLUMNS: Final[tuple[str, ...]] = ("source", "title", "url", "guid")


def _empty_news() -> pd.DataFrame:
    """Zero-row frame with the canonical news columns."""
    return pd.DataFrame(
        {
            "source": pd.Series(dtype="str"),
            "title": pd.Series(dtype="str"),
            "url": pd.Series(dtype="str"),
            "guid": pd.Series(dtype="str"),
            "published_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "snapshot_ts": pd.Series(dtype="datetime64[ns, UTC]"),
            "tickers": pd.Series(dtype="object"),
        }
    )


def _normalize_news(news: pd.DataFrame | None) -> pd.DataFrame:
    """One in-memory copy with canonical columns and UTC timestamps.

    Tolerates the recorder store (no ``snapshot_ts`` but ``first_seen_at``) and files
    grown while they were being read: only columns are touched, never rows.
    """
    if news is None or len(news) == 0:
        return _empty_news()
    frame = news.copy()
    if "snapshot_ts" not in frame and "first_seen_at" in frame:
        frame["snapshot_ts"] = frame["first_seen_at"]
    elif "snapshot_ts" in frame and "first_seen_at" in frame:
        snapshot = pd.to_datetime(frame["snapshot_ts"], errors="coerce", utc=True)
        fallback = pd.to_datetime(frame["first_seen_at"], errors="coerce", utc=True)
        frame["snapshot_ts"] = snapshot.fillna(fallback)
    if "snapshot_ts" not in frame:
        frame["snapshot_ts"] = pd.NaT
    for column in _NEWS_TEXT_COLUMNS:
        if column not in frame:
            frame[column] = ""
        else:
            frame[column] = frame[column].fillna("")
    if "tickers" not in frame:
        frame["tickers"] = [[] for _ in range(len(frame))]
    if "published_at" not in frame:
        frame["published_at"] = pd.NaT
    frame["published_at"] = pd.to_datetime(
        frame["published_at"], errors="coerce", utc=True
    ).astype("datetime64[ns, UTC]")
    frame["snapshot_ts"] = pd.to_datetime(
        frame["snapshot_ts"], errors="coerce", utc=True
    ).astype("datetime64[ns, UTC]")
    return frame.reset_index(drop=True)


def _ns(series: pd.Series) -> np.ndarray:
    """UTC nanosecond int64 view of a datetime column (NaT -> ``INT64_MIN``)."""
    naive = series.astype("datetime64[ns, UTC]").dt.tz_localize(None)
    return naive.to_numpy().astype("int64")


def _window_mask(published: pd.Series, since: Any, until: Any) -> np.ndarray:
    """Rows inside ``[since, until]`` on ``published_at``; NaT rows survive to be counted."""
    mask = np.ones(len(published), dtype=bool)
    if since is None and until is None:
        return mask
    pub_ns = _ns(published)
    is_na = published.isna().to_numpy()
    if since is not None:
        mask &= (pub_ns >= _utc(since).value) | is_na
    if until is not None:
        mask &= (pub_ns <= _utc(until).value) | is_na
    return mask


# --------------------------------------------------------------------------- #
# Bar index
# --------------------------------------------------------------------------- #
class _BarIndex:
    """Sorted int64-ns stamps plus aligned columns; one binary search per lookup."""

    __slots__ = ("symbol", "stamps", "opens", "closes", "quote_volume", "trades")

    def __init__(self, symbol: str, frame: pd.DataFrame | None) -> None:
        self.symbol = symbol
        empty_f = np.empty(0, dtype="float64")
        empty_i = np.empty(0, dtype="int64")
        if frame is None or len(frame) == 0:
            self.stamps = empty_i
            self.opens = empty_f
            self.closes = empty_f
            self.quote_volume = empty_f
            self.trades = empty_i
            return
        work = frame
        ts = pd.to_datetime(work["ts_open"], utc=True)
        if not ts.is_monotonic_increasing:
            work = work.iloc[np.argsort(ts.to_numpy(), kind="stable")]
            ts = pd.to_datetime(work["ts_open"], utc=True)
        stamps = ts.astype("datetime64[ns, UTC]").dt.tz_localize(None).to_numpy().astype("int64")
        if stamps.size and np.any(np.diff(stamps) == 0):
            keep = np.concatenate(([True], np.diff(stamps) != 0))
            work = work.iloc[keep]
            stamps = stamps[keep]
        self.stamps = stamps
        self.opens = work["open"].to_numpy(dtype="float64")
        self.closes = work["close"].to_numpy(dtype="float64")
        self.quote_volume = (
            work["quote_volume"].to_numpy(dtype="float64")
            if "quote_volume" in work
            else np.full(stamps.size, np.nan)
        )
        self.trades = (
            work["trades"].to_numpy(dtype="int64")
            if "trades" in work
            else np.zeros(stamps.size, dtype="int64")
        )

    @property
    def n(self) -> int:
        return int(self.stamps.size)

    def bar_at_or_after(self, target_ns: int) -> int:
        """Index of the first bar with ``ts_open >= target_ns`` (bisect), or ``self.n``."""
        return int(np.searchsorted(self.stamps, target_ns))

    def index_of(self, stamp_ns: int) -> int:
        """Index of the bar opening exactly at ``stamp_ns``, else ``-1``."""
        i = int(np.searchsorted(self.stamps, stamp_ns))
        if i < self.n and self.stamps[i] == stamp_ns:
            return i
        return -1

    def open_at(self, stamp_ns: int) -> float:
        i = self.index_of(stamp_ns)
        return float("nan") if i < 0 else float(self.opens[i])

    def close_at(self, stamp_ns: int) -> float:
        i = self.index_of(stamp_ns)
        return float("nan") if i < 0 else float(self.closes[i])


def _index_for(
    bars: Mapping[str, pd.DataFrame], symbol: str, cache: dict[str, _BarIndex | None]
) -> _BarIndex | None:
    """Cached :class:`_BarIndex` for ``symbol``; ``None`` when the mapping has no frame."""
    if symbol not in cache:
        frame = bars.get(symbol) if hasattr(bars, "get") else None
        cache[symbol] = _BarIndex(symbol, frame) if frame is not None else None
    return cache[symbol]


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def _split_bounds(splits: Mapping[str, Any]) -> dict[str, Any]:
    """Cut instants, embargo boundaries and embargo width from the ``splits`` section."""
    train_cut = _day_end(splits["train_end"])
    val_cut = _day_end(splits["val_end"])
    hold_start = _utc(splits["holdout_start"])
    hold_end = _day_end(splits["holdout_end"])
    boundaries = sorted(
        {
            (train_cut + pd.Timedelta(seconds=1)).value,
            (val_cut + pd.Timedelta(seconds=1)).value,
            hold_start.value,
            (hold_end + pd.Timedelta(seconds=1)).value,
        }
    )
    return {
        "train_cut_ns": train_cut.value,
        "val_cut_ns": val_cut.value,
        "hold_start_ns": hold_start.value,
        "hold_end_ns": hold_end.value,
        "boundary_ns": boundaries,
        "embargo_ns": int(pd.Timedelta(minutes=int(splits["embargo_minutes"])).value),
    }


def _boundary_violations(bounds: Mapping[str, Any]) -> int:
    """Failed relations: train before val, val before holdout, holdout window ordered."""
    violations = 0
    violations += bounds["train_cut_ns"] >= bounds["val_cut_ns"]
    violations += bounds["val_cut_ns"] >= bounds["hold_start_ns"]
    violations += bounds["hold_start_ns"] > bounds["hold_end_ns"]
    boundaries = list(bounds["boundary_ns"])
    violations += any(a >= b for a, b in zip(boundaries, boundaries[1:], strict=False))
    return int(violations)


def _split_labels(pub_ns: np.ndarray, bounds: Mapping[str, Any]) -> np.ndarray:
    """``train``/``val``/``holdout``/``outside``, overridden by ``embargo`` near a boundary.

    Inputs must be NaT-free (``build_dataset`` drops those rows first); the embargo window
    is the closed interval ``boundary +/- embargo_minutes`` around each midnight boundary.
    """
    n = pub_ns.size
    labels = np.full(n, "outside", dtype=object)
    train = pub_ns <= bounds["train_cut_ns"]
    val = (~train) & (pub_ns <= bounds["val_cut_ns"])
    holdout = (
        (~train)
        & (~val)
        & (pub_ns >= bounds["hold_start_ns"])
        & (pub_ns <= bounds["hold_end_ns"])
    )
    labels[train] = "train"
    labels[val] = "val"
    labels[holdout] = "holdout"
    embargo = np.zeros(n, dtype=bool)
    for boundary in bounds["boundary_ns"]:
        embargo |= np.abs(pub_ns - boundary) <= bounds["embargo_ns"]
    labels[embargo] = "embargo"
    return labels


# --------------------------------------------------------------------------- #
# Funnel and empty frame
# --------------------------------------------------------------------------- #
def empty_funnel() -> dict[str, int]:
    """Zeroed funnel with every key of :data:`FUNNEL_KEYS`."""
    return {key: 0 for key in FUNNEL_KEYS}


def empty_dataset() -> pd.DataFrame:
    """Zero-row frame with the exact :data:`DATASET_COLUMNS` schema and dtypes."""
    return pd.DataFrame(
        {col: pd.Series(dtype=dtype) for col, dtype in _COLUMN_DTYPES.items()},
        columns=list(DATASET_COLUMNS),
    )


def _check_balance(funnel: dict[str, int]) -> None:
    """Raise :class:`ValueError` with the counts unless the funnel balances."""
    dropped = sum(funnel[key] for key in FUNNEL_DROP_KEYS)
    if funnel["kept"] + dropped != funnel["total_pairs"]:
        raise ValueError(
            "funnel does not balance: "
            f"kept={funnel['kept']} dropped={dropped} "
            f"total_pairs={funnel['total_pairs']} funnel={funnel}"
        )


def _ticker_items(value: object) -> list[object]:
    """``tickers`` cell -> list of tickers; strings are one ticker, nulls are none."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return list(value)
    if value is None:
        return []
    return [] if bool(pd.isna(value)) else [value]


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_dataset(
    news: pd.DataFrame,
    bars_by_symbol: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    since: Any = None,
    until: Any = None,
    allow_benchmark: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Join every ``(news row, ticker)`` pair to its entry bar; return frame + funnel.

    ``since``/``until`` restrict the news ``published_at`` window (inclusive, NaT rows
    still counted); pairs outside it are out of scope and never enter ``total_pairs``.
    ``bars_by_symbol`` maps symbol -> bar frame; a missing key behaves like an empty
    frame, so a later listing (TONUSDT from 2024-08) drops its early headlines instead of
    crashing. The input frames are only read, never written or re-read.
    """
    cfg = _resolve_config(config, "build_dataset(config)")
    labels = cfg["labels"]
    horizons = tuple(int(h) for h in labels["horizons"])
    if horizons != HORIZONS:
        raise ValueError(
            f"the frozen output schema is fixed to horizons {HORIZONS}, got {horizons}"
        )
    lag_ns = int(pd.Timedelta(minutes=float(labels["news_lag_minutes"])).value)
    bench_symbol = str(labels["excess_vs"])
    cost = float(cfg["costs"]["round_trip_perp_bps"]) / 10000.0
    bounds = _split_bounds(cfg["splits"])
    funnel = empty_funnel()

    source = _normalize_news(news)
    sub = source.loc[_window_mask(source["published_at"], since, until)].reset_index(drop=True)
    if len(sub) == 0:
        _check_balance(funnel)
        return empty_dataset(), funnel

    # 1. expand (news row, ticker) pairs and resolve the symbol.
    pub_ns = _ns(sub["published_at"])
    is_na = sub["published_at"].isna().to_numpy()
    urls = sub["url"].to_numpy(dtype=object)
    pairs: list[tuple[int, str, str]] = []  # (row, asset, symbol)
    for row, cell in enumerate(sub["tickers"].tolist()):
        for ticker in _ticker_items(cell):
            funnel["total_pairs"] += 1
            symbol = resolve_symbol(ticker, SYMBOL_MAP)
            if symbol is None:
                funnel["no_symbol"] += 1
                LOG.debug("build_dataset: unknown ticker %r dropped as no_symbol", ticker)
                continue
            pairs.append((row, str(ticker), symbol))

    # 2. NaT publication times.
    dated: list[tuple[int, str, str]] = []
    for row, asset, symbol in pairs:
        if is_na[row]:
            funnel["na_published_at"] += 1
            continue
        dated.append((row, asset, symbol))

    # 3. dedupe on (url, symbol) keeping the first row in published_at order.
    ordered = sorted(dated, key=lambda item: int(pub_ns[item[0]]))
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[int, str, str]] = []
    for row, asset, symbol in ordered:
        key = (str(urls[row]), symbol)
        if key in seen:
            funnel["duplicate"] += 1
            continue
        seen.add(key)
        unique.append((row, asset, symbol))

    # 4. bar join: first bar at or after published_at + lag, then the exit windows.
    cache: dict[str, _BarIndex | None] = {}
    bench_index = _index_for(bars_by_symbol, bench_symbol, cache)
    bench_cache: dict[tuple[int, int], float] = {}

    def bench_value(entry_ns: int, horizon: int) -> float:
        key = (entry_ns, horizon)
        if key in bench_cache:
            return bench_cache[key]
        value = float("nan")
        if bench_index is not None and bench_index.n:
            entry_open = bench_index.open_at(entry_ns)
            exit_close = bench_index.close_at(entry_ns + (horizon - 1) * _NS_PER_MINUTE)
            if entry_open == entry_open and exit_close == exit_close and entry_open != 0.0:
                value = exit_close / entry_open - 1.0
        bench_cache[key] = value
        return value

    records: dict[str, list[Any]] = {col: [] for col in DATASET_COLUMNS}
    published_kept: list[int] = []
    for row, asset, symbol in unique:
        if symbol == bench_symbol and not allow_benchmark:
            # The benchmark cannot be its own subject: excess = ret - bench is identically
            # zero, which silently turns the majority of the labels into constants and makes
            # every IC and every policy number meaningless.
            funnel["benchmark_asset"] += 1
            continue
        bars_index = _index_for(bars_by_symbol, symbol, cache)
        if bars_index is None or bars_index.n == 0:
            funnel["no_bar_at_or_after"] += 1
            continue
        pub = int(pub_ns[row])
        target = pub + lag_ns
        if bars_index.stamps[0] > target:
            # The eligible moment precedes the symbol's first bar: the join would use the
            # series' very first bar while the headline became tradeable earlier (a later
            # listing such as TONUSDT-2024-08, or a window clipped at the data start).
            funnel["no_bar_at_or_after"] += 1
            continue
        bar_i = bars_index.bar_at_or_after(target)
        if bar_i >= bars_index.n:
            funnel["no_bar_at_or_after"] += 1
            continue
        entry_ns = int(bars_index.stamps[bar_i])
        entry_price = float(bars_index.opens[bar_i])

        returns: dict[int, float] = {}
        benches: dict[int, float] = {}
        excesses: dict[int, float] = {}
        for horizon in horizons:
            exit_close = bars_index.close_at(entry_ns + (horizon - 1) * _NS_PER_MINUTE)
            if exit_close != exit_close:
                funnel[f"missing_exit_{horizon}m"] += 1
                ret = float("nan")
            else:
                ret = exit_close / entry_price - 1.0
            bench = bench_value(entry_ns, horizon)
            returns[horizon] = ret
            benches[horizon] = bench
            excesses[horizon] = ret - bench

        records["asset"].append(asset)
        records["symbol"].append(symbol)
        records["source"].append(sub["source"].iat[row])
        records["title"].append(sub["title"].iat[row])
        records["url"].append(str(urls[row]))
        records["published_at"].append(pub)
        records["entry_ts"].append(entry_ns)
        records["entry_price"].append(entry_price)
        records["lag_minutes"].append((entry_ns - pub) / _NS_PER_MINUTE)
        for horizon in horizons:
            records[f"ret_{horizon}m"].append(returns[horizon])
            records[f"bench_{horizon}m"].append(benches[horizon])
            records[f"excess_{horizon}m"].append(excesses[horizon])
        records["cost"].append(cost)
        records["quote_volume_entry"].append(float(bars_index.quote_volume[bar_i]))
        records["trades_entry"].append(int(bars_index.trades[bar_i]))
        records["snapshot_ts"].append(sub["snapshot_ts"].iat[row])
        records["event_kind"].append("")
        published_kept.append(pub)

    funnel["kept"] = len(records["asset"])
    _check_balance(funnel)
    if funnel["kept"] == 0:
        return empty_dataset(), funnel

    records["split"] = list(_split_labels(np.asarray(published_kept, dtype="int64"), bounds))
    frame = pd.DataFrame(records, columns=list(DATASET_COLUMNS))
    return _finalize(frame), funnel


def _finalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Force the frozen dtypes and the exact :data:`DATASET_COLUMNS` order."""
    for col in ("asset", "symbol", "source", "title", "url", "split", "event_kind"):
        frame[col] = frame[col].fillna("").astype("str")
    frame["published_at"] = pd.to_datetime(
        frame["published_at"], unit="ns", utc=True
    ).astype("datetime64[ns, UTC]")
    frame["entry_ts"] = pd.to_datetime(frame["entry_ts"], unit="ns", utc=True).astype(
        "datetime64[ns, UTC]"
    )
    frame["snapshot_ts"] = pd.to_datetime(
        frame["snapshot_ts"], errors="coerce", utc=True
    ).astype("datetime64[ns, UTC]")
    for col, dtype in _COLUMN_DTYPES.items():
        if dtype == "float64":
            frame[col] = frame[col].astype("float64")
    frame["trades_entry"] = frame["trades_entry"].astype("int64")
    return frame[list(DATASET_COLUMNS)]


# --------------------------------------------------------------------------- #
# Leakage checks
# --------------------------------------------------------------------------- #
def run_checks(frame: pd.DataFrame, bars: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Leakage checks: ``{name: bool, "<name>_count": int}`` for the checks in the spec.

    (a) ``entry_ts >= published_at + news_lag_minutes`` for every kept row; (b) every
    ``bench_<h>`` recomputes from ``bars`` over the identical ``entry_ts``/exit stamps
    (``bars`` must contain the benchmark symbol to verify it); (c) the entry bar is a real
    bar of the row's symbol, opens at/after the lag target with no earlier eligible bar;
    (d) split boundaries are disjoint and holdout comes last; (e) no duplicate
    ``(url, symbol)``. Two extra invariants: the stored ``entry_price`` equals the entry
    bar's open, and the stored ``lag_minutes`` equals ``entry_ts - published_at``.
    """
    try:
        cfg = load_dataset_config()
    except FileNotFoundError:
        cfg = _resolve_config({}, "built-in defaults")
    labels = cfg["labels"]
    lag_ns = int(pd.Timedelta(minutes=float(labels["news_lag_minutes"])).value)
    bench_symbol = str(labels["excess_vs"])
    bounds = _split_bounds(cfg["splits"])

    violations = {name: 0 for name in _CHECK_NAMES}
    violations["split_boundaries_disjoint"] = _boundary_violations(bounds)
    if len(frame):
        pub_ns = _ns(frame["published_at"])
        entry_ns = _ns(frame["entry_ts"])
        target = pub_ns + lag_ns
        violations["entry_not_before_news_lag"] = int(np.count_nonzero(entry_ns < target))
        lag_minutes = frame["lag_minutes"].to_numpy(dtype="float64")
        realised = (entry_ns - pub_ns) / _NS_PER_MINUTE
        violations["realised_lag_consistent"] = int(
            np.count_nonzero(np.abs(realised - lag_minutes) > 1e-9)
        )

        dup = int(frame.duplicated(["url", "symbol"]).sum())
        violations["no_duplicate_url_symbol"] = dup

        # (b) recompute every benchmark return from the same stamps.
        cache: dict[str, _BarIndex | None] = {}
        bench_index = _index_for(bars, bench_symbol, cache)
        for horizon in HORIZONS:
            stored = frame[f"bench_{horizon}m"].to_numpy(dtype="float64")
            for i in range(len(frame)):
                expected = float("nan")
                if bench_index is not None and bench_index.n:
                    entry_open = bench_index.open_at(int(entry_ns[i]))
                    exit_close = bench_index.close_at(
                        int(entry_ns[i]) + (horizon - 1) * _NS_PER_MINUTE
                    )
                    if entry_open == entry_open and exit_close == exit_close and entry_open != 0.0:
                        expected = exit_close / entry_open - 1.0
                if not np.isclose(stored[i], expected, rtol=1e-9, atol=1e-12, equal_nan=True):
                    violations["bench_same_window"] += 1

        # (c) entry bar integrity: a real bar, not earlier than the target, first eligible.
        entry_price = frame["entry_price"].to_numpy(dtype="float64")
        for i in range(len(frame)):
            bars_index = _index_for(bars, str(frame["symbol"].iat[i]), cache)
            if bars_index is None or bars_index.n == 0:
                violations["no_bar_earlier_than_entry"] += 1
                continue
            bar_i = bars_index.index_of(int(entry_ns[i]))
            if bar_i < 0:
                violations["no_bar_earlier_than_entry"] += 1
                continue
            if int(entry_ns[i]) < int(target[i]):
                violations["no_bar_earlier_than_entry"] += 1
            if bar_i > 0 and int(bars_index.stamps[bar_i - 1]) >= int(target[i]):
                violations["no_bar_earlier_than_entry"] += 1
            if bars_index.opens[bar_i] != entry_price[i]:
                violations["entry_price_matches_bar"] += 1

    checks: dict[str, Any] = {}
    for name in _CHECK_NAMES:
        checks[name] = violations[name] == 0
        checks[f"{name}_count"] = int(violations[name])
    return checks


# --------------------------------------------------------------------------- #
# Lazy bar loading
# --------------------------------------------------------------------------- #
class LazyBars(Mapping[str, pd.DataFrame]):
    """``symbol -> frame`` view over a directory of parquets, read on first access.

    Only the five columns the assembler contracts on (:data:`_BAR_READ_COLUMNS`) are read,
    ``since``/``until`` are padded by one day so exits near the window edge still resolve,
    and a symbol whose file is absent behaves like an unknown key (``in`` -> False,
    ``.get`` -> ``None``) instead of raising.
    """

    def __init__(
        self,
        bars_dir: Path | str,
        symbols: Iterable[str],
        since: Any = None,
        until: Any = None,
    ) -> None:
        directory = Path(bars_dir)
        self._paths = {str(sym): directory / f"{str(sym)}.parquet" for sym in symbols}
        self._since = _utc(since) - pd.Timedelta(days=1) if since is not None else None
        self._until = _utc(until) + pd.Timedelta(days=1) if until is not None else None
        self._cache: dict[str, pd.DataFrame] = {}

    def __getitem__(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._paths:
            raise KeyError(symbol)
        if symbol not in self._cache:
            path = self._paths[symbol]
            if not path.is_file():
                raise KeyError(symbol)
            frame = pd.read_parquet(path, columns=list(_BAR_READ_COLUMNS))
            ts = pd.to_datetime(frame["ts_open"], utc=True)
            mask = np.ones(len(frame), dtype=bool)
            if self._since is not None:
                mask &= (ts >= self._since).to_numpy()
            if self._until is not None:
                mask &= (ts <= self._until).to_numpy()
            self._cache[symbol] = frame.loc[mask].reset_index(drop=True)
            LOG.debug("load_bars: %s -> %d rows", symbol, len(self._cache[symbol]))
        return self._cache[symbol]

    def __contains__(self, symbol: object) -> bool:
        return isinstance(symbol, str) and symbol in self._paths and self._paths[symbol].is_file()

    def __iter__(self):
        return (symbol for symbol in self._paths if self._paths[symbol].is_file())

    def __len__(self) -> int:
        return sum(1 for symbol in self._paths if self._paths[symbol].is_file())

    @property
    def loaded(self) -> set[str]:
        """Symbols whose parquet has actually been read so far."""
        return set(self._cache)


def load_bars(
    bars_dir: Path | str,
    symbols: Iterable[str],
    since: Any = None,
    until: Any = None,
) -> LazyBars:
    """Lazy view over ``{bars_dir}/{SYMBOL}.parquet`` for ``symbols``.

    Nothing is read until a symbol is accessed; the returned mapping satisfies the
    ``bars_by_symbol`` contract of :func:`build_dataset` and :func:`run_checks`.
    """
    return LazyBars(bars_dir, symbols, since=since, until=until)
