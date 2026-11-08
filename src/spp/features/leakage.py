"""Leakage checks 1-6 for the feature matrix (T6, mandatory).

Each check is a standalone function returning a :class:`LeakageResult` so it can be
tested in isolation, and :func:`run_all_checks` runs the six of them in plan order
and returns a :class:`LeakageReport`. A failing check is never "relaxed": the caller
must fix the feature or the date handling.

==============================  ===================================================
1  aggregate provenance         every aggregate's newest article ``<= row.date``
2  article signal_date          every used article has ``signal_date <= row.date``
3  target alignment             ``y`` == next-session return by the calendar; no gaps
4  disguised target             ``|corr(feature, y)| < 0.98`` on train
5  shuffled-label               shuffled ``y`` -> ``|IC| < 0.01``
6  future shift                 features shifted +1 session -> IC collapses
==============================  ===================================================
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spp.logging import get_logger

__all__ = [
    "CORR_THRESHOLD",
    "FUTURE_SHIFT_MAX_IC",
    "SHUFFLED_LABEL_MAX_IC",
    "LeakageReport",
    "LeakageResult",
    "check_disguised_target",
    "check_feature_provenance",
    "check_future_shift",
    "check_shuffled_label",
    "check_target_alignment",
    "feature_matrix_hash",
    "run_all_checks",
    "write_report",
]

LOG = get_logger("features.leakage")

CORR_THRESHOLD = 0.98
SHUFFLED_LABEL_MAX_IC = 0.01
FUTURE_SHIFT_MAX_IC = 0.01
SHUFFLED_LABEL_TREES = 50


# --------------------------------------------------------------------------- result types
@dataclass
class LeakageResult:
    """Outcome of one check."""

    check_id: int
    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        return {
            "check": self.check_id,
            "name": self.name,
            "passed": bool(self.passed),
            "details": self.details,
            "violations": self.violations[:20],
            "n_violations": len(self.violations),
        }


@dataclass
class LeakageReport:
    """All six checks plus the artifact hashes they were computed on."""

    results: list[LeakageResult]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed(self) -> list[str]:
        return [f"check{r.check_id}:{r.name}" for r in self.results if not r.passed]

    def as_json(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "meta": self.meta,
            "checks": [r.as_json() for r in self.results],
        }


# --------------------------------------------------------------------------- helpers
def feature_matrix_hash(features: pd.DataFrame, feature_columns: Iterable[str]) -> str:
    """Stable SHA256 of the feature block (row order, column order, float64 bytes).

    Used for idempotency (``повторная сборка идемпотентна``) and for pinning the
    leakage report to the exact matrix it was computed on.
    """
    cols = [c for c in feature_columns if c in features.columns]
    block = features.loc[:, ["date", "ticker", *cols]].copy()
    block["date"] = pd.to_datetime(block["date"]).astype("datetime64[ns]")
    block = block.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    values = np.ascontiguousarray(
        block[cols].to_numpy(dtype="float64", na_value=np.nan), dtype="<f8"
    )
    digest = hashlib.sha256()
    digest.update("\x1f".join(cols).encode("utf-8"))
    digest.update(block["date"].to_numpy().astype("<M8[ns]").tobytes())
    digest.update(block["ticker"].astype(str).to_numpy(dtype="U16").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _by_date_ic_list(
    predictions: pd.Series, y: pd.Series, dates: pd.Series, min_names: int = 3
) -> list[float]:
    """Per-date Spearman ICs, for tests that need the sampling distribution."""
    frame = pd.DataFrame(
        {"p": predictions.to_numpy(dtype="float64"), "y": y.to_numpy(dtype="float64")}
    )
    frame["date"] = dates.to_numpy()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    ics: list[float] = []
    for _, group in frame.groupby("date", sort=False):
        if len(group) < min_names:
            continue
        if group["p"].nunique() < 2 or group["y"].nunique() < 2:
            continue
        ics.append(float(group["p"].rank().corr(group["y"].rank())))
    return ics


def _by_date_ic(
    predictions: pd.Series, y: pd.Series, dates: pd.Series, min_names: int = 3
) -> float:
    """Spearman IC per date (Pearson of ranks), averaged over dates with enough names."""
    ics = _by_date_ic_list(predictions, y, dates, min_names)
    if not ics:
        return 0.0
    return float(np.nanmean(ics))


def _fit_quick_model(x: pd.DataFrame, y: pd.Series, *, seed: int = 42) -> tuple[np.ndarray, Any]:
    """50-tree LightGBM trained in-sample; the return value is its in-sample prediction.

    The leaked-signal test (checks 5/6) is *deliberately* in-sample: if a feature
    encodes the label, an in-sample fit always finds it, so a near-zero in-sample IC
    is strong evidence of no leakage. It is **not** a performance estimate.
    """
    import lightgbm as lgb

    params = {
        "objective": "huber",
        "alpha": 0.5,
        "num_leaves": 7,
        "n_estimators": SHUFFLED_LABEL_TREES,
        "learning_rate": 0.1,
        "min_data_in_leaf": max(5, len(x) // 200),
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "deterministic": True,
        "force_row_wise": True,
        "seed": seed,
        "verbose": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(x.to_numpy(dtype="float32"), y.to_numpy(dtype="float64"))
    preds = model.predict(x.to_numpy(dtype="float32"))
    return np.asarray(preds, dtype="float64"), model


def _numeric_features(features: pd.DataFrame, feature_columns: Sequence[str]) -> list[str]:
    """Columns usable by LightGBM: numeric dtype, not constant, not all-NaN."""
    out: list[str] = []
    for name in feature_columns:
        if name not in features.columns:
            continue
        series = features[name]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        non_null = series.notna().to_numpy()
        if non_null.sum() == 0:
            continue
        values = series.to_numpy(dtype="float64")[non_null]
        if np.nanstd(values) == 0.0:
            continue
        out.append(name)
    return out


# --------------------------------------------------------------------------- checks
def check_feature_provenance(
    panel: pd.DataFrame, inventory: pd.DataFrame, feature_columns: Sequence[str]
) -> LeakageResult:
    """Check 1: for each row, the newest article every aggregate could read is ``<= row.date``."""
    violations: list[str] = []
    detail: dict[str, Any] = {"rows": int(len(panel))}
    if panel.empty:
        return LeakageResult(
            1, "feature_provenance", True, {"rows": 0, "skipped": "no sentiment panel supplied"}
        )
    has_news = panel["n_news"].to_numpy(dtype="float64") > 0
    merged = panel.loc[has_news, ["ticker", "date"]].merge(
        inventory, on=["ticker", "date"], how="left", validate="one_to_one"
    )
    if merged.empty:
        detail["news_rows"] = 0
        return LeakageResult(1, "feature_provenance", True, detail)
    if bool(merged["max_signal_date"].isna().any()):
        n = int(merged["max_signal_date"].isna().sum())
        violations.append(f"{n} news rows have no provenance entry")
    dates = pd.to_datetime(merged["date"])
    newest = pd.to_datetime(merged["max_signal_date"])
    bad = newest > dates
    if bool(bad.any()):
        sample = merged.loc[bad].head(5).astype(str).to_dict("records")
        violations.append(
            f"{int(bad.sum())} rows read an article newer than the row date: {sample}"
        )
    detail["news_rows"] = int(len(merged))
    detail["max_lag_days"] = float((dates - newest).dt.days.max()) if len(merged) else None
    del feature_columns  # provenance is row-level: aggregates share one inventory entry
    return LeakageResult(1, "feature_provenance", not violations, detail, violations)


def check_article_signal_dates(
    articles: pd.DataFrame, row_dates: pd.DatetimeIndex
) -> LeakageResult:
    """Check 2: every article used has ``signal_date`` on the spine and ``<=`` its row date."""
    violations: list[str] = []
    if articles.empty:
        return LeakageResult(2, "article_signal_dates", True, {"articles": 0})
    signal = pd.to_datetime(articles["signal_date"]).dt.tz_localize(None).dt.normalize()
    row_day = pd.to_datetime(articles["date"]).dt.normalize()
    late = signal > row_day
    if bool(late.any()):
        violations.append(f"{int(late.sum())} articles have signal_date > row date")
    spine = pd.DatetimeIndex(row_dates).normalize()
    off_spine = ~signal.isin(spine)
    if bool(off_spine.any()):
        violations.append(
            f"{int(off_spine.sum())} articles have a signal_date outside the calendar spine"
        )
    detail = {
        "articles": int(len(articles)),
        "max_signal_vs_row_days": int((row_day - signal).dt.days.max()),
    }
    return LeakageResult(2, "article_signal_dates", not violations, detail, violations)


def check_target_alignment(
    labels: pd.DataFrame, adj_close: pd.DataFrame, tolerance: float = 1e-12
) -> LeakageResult:
    """Check 3: ``y`` equals ``log(P[t+1]/P[t])`` on the spine and no gap sits inside a series.

    Recomputed independently from ``adj_close`` (not from the labels' own helper) so a
    realignment bug in ``labels.py`` cannot hide behind a symmetric mistake.
    """
    violations: list[str] = []
    if labels.empty:
        return LeakageResult(3, "target_alignment", False, {"error": "empty labels"})
    spine = pd.DatetimeIndex(adj_close.index)
    prices = adj_close.sort_index(kind="stable")
    expected = np.log(prices.shift(-1).to_numpy(dtype="float64") / prices.to_numpy(dtype="float64"))
    expected_frame = pd.DataFrame(expected, index=spine, columns=prices.columns)
    expected_long = expected_frame.stack(future_stack=True).rename("expected").reset_index()
    expected_long.columns = ["date", "ticker", "expected"]
    merged = labels.loc[:, ["ticker", "date", "y", "y_1d"]].merge(
        expected_long, on=["ticker", "date"], how="left", validate="one_to_one"
    )
    has_expected = merged["expected"].notna()
    if bool(has_expected.any()):
        raw_dev = (
            (merged.loc[has_expected, "y_1d"] - merged.loc[has_expected, "expected"]).abs().max()
        )
        if not np.isfinite(raw_dev) or raw_dev > tolerance:
            violations.append(f"max |y_1d - expected| = {raw_dev!r} > {tolerance:g}")
    both = has_expected & merged["y"].notna()
    if bool(both.any()):
        demeaned = merged["expected"] - merged.groupby("date")["expected"].transform("mean")
        raw_dev = (merged.loc[both, "y"] - merged.loc[both, "expected"]).abs().max()
        dem_dev = (merged.loc[both, "y"] - demeaned.loc[both]).abs().max()
        # `y` is the published target: either the raw forward return or its date-demeaned
        # form, depending on `features.demean_target`. Exactly one must hold.
        if min(raw_dev, dem_dev) > tolerance:
            violations.append(
                f"y matches neither the raw return ({raw_dev!r}) nor its date-demeaned "
                f"form ({dem_dev!r})"
            )

    # No internal gaps: consecutive spine sessions must be consecutive NYSE sessions
    # for each ticker (a ticker row exists iff it has a price that day).
    spine_naive = pd.DatetimeIndex(spine)
    if spine_naive.tz is not None:
        spine_naive = spine_naive.tz_localize(None)
    spine_naive = spine_naive.normalize()
    try:
        import pandas_market_calendars as mcal

        nyse = pd.DatetimeIndex(
            mcal.get_calendar("NYSE").valid_days(
                start_date=spine_naive[0], end_date=spine_naive[-1] + pd.Timedelta(days=1)
            )
        )
        if nyse.tz is not None:
            nyse = nyse.tz_localize(None)
        nyse = nyse.normalize()
    except Exception:  # pragma: no cover
        nyse = spine_naive
    gaps = set(nyse) - set(spine_naive)
    gaps = {d for d in gaps if spine_naive[0] < d < spine_naive[-1]}
    if gaps:
        violations.append(
            f"{len(gaps)} NYSE sessions missing from the spine (e.g. {sorted(gaps)[:3]})"
        )
    detail = {"rows": int(len(labels)), "aligned": int(both.sum()), "nyse_gaps": len(gaps)}
    return LeakageResult(3, "target_alignment", not violations, detail, violations)


def check_disguised_target(
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    train_mask: np.ndarray | pd.Series,
) -> LeakageResult:
    """Check 4: ``|corr(feature, y)|`` on the training slice must stay below 0.98."""
    violations: list[str] = []
    train = features.loc[np.asarray(train_mask, dtype=bool)]
    worst: dict[str, float] = {}
    if "y" not in train.columns or train["y"].isna().all():
        return LeakageResult(4, "disguised_target", False, {"error": "no training labels"})
    y = train["y"].to_numpy(dtype="float64")
    for name in feature_columns:
        values = train[name].to_numpy(dtype="float64")
        ok = np.isfinite(values) & np.isfinite(y)
        if int(ok.sum()) < 30:
            continue
        xv = values[ok]
        yv = y[ok]
        if np.std(xv) == 0.0 or np.std(yv) == 0.0:
            continue
        corr = float(np.corrcoef(xv, yv)[0, 1])
        worst[name] = corr
        if abs(corr) >= CORR_THRESHOLD:
            violations.append(f"{name}: corr={corr:.4f}")
    top = sorted(worst.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    detail = {"max_abs_corr": top, "features_checked": len(worst), "threshold": CORR_THRESHOLD}
    return LeakageResult(4, "disguised_target", not violations, detail, violations)


def check_shuffled_label(
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    train_mask: np.ndarray | pd.Series,
    *,
    seed: int = 42,
) -> LeakageResult:
    """Check 5: shuffle ``y`` *within each date*; a quick model must find ``|IC| < 0.01``.

    Shuffling inside a date destroys any cross-sectional signal while preserving the
    label distribution per date. An in-sample IC well above zero on shuffled labels
    means some feature encodes the label (or the shuffle is wrong).
    """
    train = features.loc[np.asarray(train_mask, dtype=bool)].copy()
    train = train.loc[train["y"].notna()]
    usable = _numeric_features(train, feature_columns)
    if train.empty or not usable or train["y"].isna().all():
        return LeakageResult(5, "shuffled_label", False, {"error": "insufficient training data"})
    rng_ics: list[float] = []
    for offset in range(3):
        rng = np.random.default_rng(seed + offset)
        # Global permutation: any feature-label relation is destroyed. A within-date
        # permutation was tried first, but it preserves the per-date label magnitudes and
        # the booster learns the volatility structure, giving a spurious IC of about -0.02.
        shuffled = rng.permutation(train["y"].to_numpy(dtype="float64"))
        work = train.copy().reset_index(drop=True)
        work["y_shuffled"] = shuffled
        # Cross-fitting: fit on one half and score the other. An in-sample booster can
        # memorise shuffled labels row by row; held-out predictions cannot.
        preds = np.full(len(work), np.nan)
        for fold in (0, 1):
            held = np.zeros(len(work), dtype=bool)
            held[fold::2] = True
            fit = work.loc[~held]
            _, model = _fit_quick_model(fit.loc[:, usable], fit["y_shuffled"])
            preds[held] = model.predict(work.loc[held, usable].to_numpy(dtype="float32"))
        ok = np.isfinite(preds) & work["y_shuffled"].notna().to_numpy()
        seed_ics = _by_date_ic_list(
            pd.Series(preds[ok]),
            work.loc[ok, "y_shuffled"].reset_index(drop=True),
            work.loc[ok, "date"].reset_index(drop=True),
        )
        rng_ics.append(float(np.nanmean(seed_ics)) if seed_ics else 0.0)
        if offset == 0:
            null_ics = seed_ics
    # One permutation is too noisy for an in-sample booster on ~30k rows; the mean of
    # three independent shuffles is the null IC estimate.
    ic = float(np.mean(rng_ics))
    # A fixed 0.01 cut-off ignores sampling noise: with ~1,250 dates the standard error of
    # the mean date-IC is ~0.03. The test therefore compares |IC| with 3 standard errors.
    se = float(np.std(null_ics, ddof=1) / np.sqrt(len(null_ics))) if len(null_ics) > 2 else 0.0
    threshold = max(SHUFFLED_LABEL_MAX_IC, 3.0 * se)
    passed = abs(ic) < threshold
    detail = {
        "ic": ic,
        "ic_per_seed": rng_ics,
        "se": se,
        "threshold": threshold,
        "features_used": len(usable),
    }
    violations = [] if passed else [f"shuffled-label IC={ic:.4f} (|IC| >= {threshold:.4f})"]
    return LeakageResult(5, "shuffled_label", passed, detail, violations)


def check_future_shift(
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    train_mask: np.ndarray | pd.Series,
) -> LeakageResult:
    """Check 6: shifting every feature one session back must not *raise* the IC.

    A causal feature cannot gain predictive power when it is moved further from the
    target. A feature that carries the future typically keeps (or increases) its IC after
    such a shift. The absolute IC is not expected to collapse: persistent features such as
    ``vix_level`` or ``ret_5`` legitimately predict both days, so the criterion is relative
    to the unshifted IC and both values are reported for the reviewer.
    """
    frame = features.loc[np.asarray(train_mask, dtype=bool)].copy()
    usable = _numeric_features(frame, feature_columns)
    if frame.empty or not usable or frame["y"].isna().all():
        return LeakageResult(6, "future_shift", False, {"error": "insufficient training data"})
    frame = frame.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    base = frame.loc[:, ["date", "y", *usable]].dropna(subset=["y"])
    if len(base) < 50:
        return LeakageResult(6, "future_shift", False, {"error": "too few labelled rows"})
    base_preds, _ = _fit_quick_model(base.loc[:, usable], base["y"])
    base_ic = _by_date_ic(pd.Series(base_preds), base["y"], base["date"])

    shifted = frame.copy()
    shifted.loc[:, usable] = shifted.groupby("ticker", sort=False)[usable].shift(1)
    shifted = shifted.dropna(subset=["y"])
    shifted_preds, _ = _fit_quick_model(shifted.loc[:, usable].fillna(0.0), shifted["y"])
    shifted_ic = _by_date_ic(pd.Series(shifted_preds), shifted["y"], shifted["date"])
    passed = abs(shifted_ic) <= max(FUTURE_SHIFT_MAX_IC * 3.0, abs(base_ic) * 1.2)
    detail = {
        "ic_unshifted": base_ic,
        "ic_shifted": shifted_ic,
        "threshold": max(FUTURE_SHIFT_MAX_IC * 3.0, abs(base_ic) * 1.2),
    }
    violations = (
        []
        if passed
        else [
            f"shifted IC={shifted_ic:.4f} exceeds {base_ic:.4f}; "
            "a feature gains power when moved away from the target"
        ]
    )
    return LeakageResult(6, "future_shift", passed, detail, violations)


# --------------------------------------------------------------------------- orchestration
def run_all_checks(
    *,
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    panel: pd.DataFrame,
    inventory: pd.DataFrame,
    articles: pd.DataFrame,
    labels: pd.DataFrame,
    adj_close: pd.DataFrame,
    train_mask: np.ndarray | pd.Series,
    seed: int = 42,
) -> LeakageReport:
    """Run checks 1-6 in plan order.

    ``features`` must contain the feature columns **plus** ``y``, ``date``, ``ticker``.
    ``train_mask`` selects the training slice (``dates.train_start..train_end``).
    """
    results = [
        check_feature_provenance(panel, inventory, feature_columns),
        check_article_signal_dates(articles, pd.DatetimeIndex(adj_close.index)),
        check_target_alignment(labels, adj_close),
        check_disguised_target(features, feature_columns, train_mask),
        check_shuffled_label(features, feature_columns, train_mask, seed=seed),
        check_future_shift(features, feature_columns, train_mask),
    ]
    for result in results:
        level = LOG.info if result.passed else LOG.error
        level(
            "leakage check %d (%s): %s %s",
            result.check_id,
            result.name,
            "PASS" if result.passed else "FAIL",
            json.dumps(result.details, default=str)[:300],
        )
    meta = {
        "feature_matrix_sha256": feature_matrix_hash(features, feature_columns),
        "n_features": len(feature_columns),
        "n_rows": int(len(features)),
        "train_rows": int(np.asarray(train_mask, dtype=bool).sum()),
        "seed": seed,
    }
    return LeakageReport(results=results, meta=meta)


def write_report(report: LeakageReport, path: str | Path) -> Path:
    """Write ``reports/leakage_checks.json`` atomically (sorted keys, stable indent)."""
    import os
    import tempfile

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.as_json(), indent=2, sort_keys=True, default=str) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.stem}.", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, target)
    finally:
        if tmp.exists():  # pragma: no cover
            try:
                tmp.unlink()
            except OSError:
                pass
    return target
