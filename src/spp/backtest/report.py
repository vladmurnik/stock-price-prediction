"""Backtest statistics, diagnostics and the markdown/JSON/PNG report (T11).

Everything here is a pure function of two inputs: the stored ``predictions``
parquet and the stored ``prices`` parquet. No network access, no 2023 rows, and
no tuning: the quintile count, cost model and execution mode all come from the
config, and the report states the verdict against pre-registered thresholds
rather than against an arbitrary Sharpe target.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spp.backtest.portfolio import (
    ExecutionFrame,
    PortfolioResult,
    build_execution_frame,
    leg_holdings,
    run_backtest,
)
from spp.models.metrics import bootstrap_ic, diebold_mariano, ic

__all__ = [
    "BlockBootstrap",
    "BacktestReport",
    "annual_returns",
    "quintile_returns",
    "block_bootstrap_sharpe",
    "cost_decomposition",
    "ic_diagnostics",
    "render_markdown",
    "run_backtest_suite",
    "write_artifacts",
]

TRADING_DAYS = 252
BOOTSTRAP_BLOCK = 20
BOOTSTRAP_ITER = 1000


# --------------------------------------------------------------------------- helpers
def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return pd.Timestamp(value).date().isoformat()
    return value


def _write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- statistics
@dataclass
class BlockBootstrap:
    """Circular block bootstrap of a daily return series (block 20, 1000 draws)."""

    point: float
    lo: float
    hi: float
    p_value: float
    n_draws: int
    block: int
    draws: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "sharpe": self.point,
            "ci95": [self.lo, self.hi],
            "p_value_le_zero": self.p_value,
            "n_draws": self.n_draws,
            "block": self.block,
        }


def block_bootstrap_sharpe(
    returns: np.ndarray,
    block: int = BOOTSTRAP_BLOCK,
    n_draws: int = BOOTSTRAP_ITER,
    seed: int = 42,
    periods: int = TRADING_DAYS,
) -> BlockBootstrap:
    """Sharpe CI by circular block bootstrap, preserving autocorrelation of the P&L."""
    series = np.asarray(returns, dtype=float)
    series = series[np.isfinite(series)]
    n = len(series)
    if n < 2:
        nan = float("nan")
        return BlockBootstrap(nan, nan, nan, nan, 0, block)
    point = float(series.mean() / series.std(ddof=1) * np.sqrt(periods))
    rng = np.random.default_rng(seed)
    block = max(1, min(int(block), n))
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    starts = rng.integers(0, n, size=(n_draws, n_blocks))
    index = (starts[:, :, None] + offsets[None, None, :]) % n
    samples = series[index.reshape(n_draws, -1)[:, :n]]
    means = samples.mean(axis=1)
    stds = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = means / stds * np.sqrt(periods)
    sharpes = sharpes[np.isfinite(sharpes)]
    return BlockBootstrap(
        point=point,
        lo=float(np.quantile(sharpes, 0.025)),
        hi=float(np.quantile(sharpes, 0.975)),
        p_value=float((sharpes <= 0).mean()),
        n_draws=int(len(sharpes)),
        block=block,
        draws=sharpes,
    )


def quintile_returns(leg_frame: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """Mean realised return per predicted bucket, per date, then averaged.

    Uses the prediction ranking only; the return is the tradable one from the
    execution frame. Monotonicity is the Spearman correlation between bucket
    index and mean return across buckets.
    """
    ranked = leg_holdings(leg_frame, n_quantiles=n_quantiles)
    usable = ranked.loc[ranked["rank_pct"].notna()].copy()
    # 1 = lowest score, n_quantiles = highest score (rank_pct is ascending).
    usable["bucket"] = np.minimum(
        np.floor(usable["rank_pct"] * n_quantiles + 1e-12).astype(int) + 1, n_quantiles
    )
    per_date = usable.pivot_table(index="date", columns="bucket", values="ret", aggfunc="mean")
    per_date = per_date.dropna(how="any")
    summary = per_date.mean().rename("mean_ret").to_frame()
    summary["ann_ret"] = summary["mean_ret"] * TRADING_DAYS
    summary["n_dates"] = len(per_date)
    summary["std_ret"] = per_date.std(ddof=1)
    summary["t_stat"] = (summary["mean_ret"] / summary["std_ret"] * np.sqrt(len(per_date))).replace(
        [np.inf, -np.inf], np.nan
    )
    bucket_index = summary.index.to_numpy(dtype=float)
    if len(summary) >= 2:
        rho = pd.Series(bucket_index).corr(
            pd.Series(summary["mean_ret"].to_numpy()), method="spearman"
        )
    else:
        rho = float("nan")
    summary.attrs["monotonicity_spearman"] = float(rho)
    mean_ret = summary["mean_ret"]
    summary.attrs["top_minus_bottom"] = float(mean_ret.iloc[-1] - mean_ret.iloc[0])
    return summary


def ic_diagnostics(leg_frame: pd.DataFrame) -> dict[str, Any]:
    """Pooled IC with CI/t-stat, per-year IC and IC without the top-1% return rows."""
    clean = leg_frame.loc[leg_frame["score"].notna() & leg_frame["ret"].notna()]
    result: dict[str, Any] = {}
    if clean.empty:
        return result
    pooled = ic(clean["ret"], clean["score"], clean["date"])
    point, lo, hi = bootstrap_ic(
        clean["ret"], clean["score"], clean["date"], n=500, seed=42
    )
    result["pooled"] = {
        "ic": pooled.get("ic"),
        "ic_se": pooled.get("ic_se"),
        "t": pooled.get("ic_t"),
        "t_nw": pooled.get("ic_t_nw"),
        "ci95": [pooled.get("ic_ci95_lo"), pooled.get("ic_ci95_hi")],
        "bootstrap_ci95": [lo, hi],
        "n_dates": pooled.get("n_dates"),
        "n_rows": pooled.get("n_rows"),
    }
    by_year = {}
    for year, group in clean.groupby(clean["date"].dt.year):
        block = ic(group["ret"], group["score"], group["date"])
        by_year[str(int(year))] = {
            "ic": block.get("ic"),
            "t": block.get("ic_t"),
            "n_dates": block.get("n_dates"),
        }
    result["by_year"] = by_year

    # Trim the largest |ret| rows cross-sectionally (1% of rows) and recompute IC.
    threshold = clean["ret"].abs().quantile(0.99)
    trimmed = clean.loc[clean["ret"].abs() <= threshold]
    thin = ic(trimmed["ret"], trimmed["score"], trimmed["date"])
    result["trimmed_1pct"] = {
        "threshold_abs_ret": float(threshold),
        "ic": thin.get("ic"),
        "t": thin.get("ic_t"),
        "n_rows": thin.get("n_rows"),
        "n_rows_dropped": int(len(clean) - len(trimmed)),
    }
    # Same statistic against the adjusted-price return: the data-quality diagnostic.
    if "adj_ret" in clean.columns:
        adj_clean = clean.loc[clean["adj_ret"].notna()]
        adj_block = ic(adj_clean["adj_ret"], adj_clean["score"], adj_clean["date"])
        result["adj_close_ret"] = {
            "ic": adj_block.get("ic"),
            "t": adj_block.get("ic_t"),
            "n_dates": adj_block.get("n_dates"),
        }
        diff = (adj_clean["adj_ret"] - adj_clean["ret"]).rename("adj_minus_raw")
        result["adj_minus_raw_mean"] = float(diff.mean())
    return result


def annual_returns(daily: pd.DataFrame, column: str = "net_ret") -> dict[str, dict[str, float]]:
    """Compounded return and per-year Sharpe for a return column."""
    out: dict[str, dict[str, float]] = {}
    years = pd.to_datetime(daily["date"]).dt.year
    for year, group in daily.groupby(years):
        values = group[column].dropna()
        if values.empty:
            continue
        equity = float((1.0 + values).prod())
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        out[str(int(year))] = {
            "ret": equity - 1.0,
            "sharpe": (
                float(values.mean() / std * np.sqrt(TRADING_DAYS)) if std == std else float("nan")
            ),
            "n_dates": int(len(values)),
        }
    return out


def cost_decomposition(daily: pd.DataFrame) -> dict[str, float]:
    """Gross P&L, cost drag and borrow drag in annualised terms."""
    n = max(len(daily), 1)
    gross = float(daily["trade_ret"].mean() * TRADING_DAYS)
    cost = float(daily["cost"].mean() * TRADING_DAYS)
    borrow = float(daily["borrow"].mean() * TRADING_DAYS)
    return {
        "gross_ann": gross,
        "cost_ann": cost,
        "borrow_ann": borrow,
        "net_ann_arith": gross - cost - borrow,
        "cost_share_of_gross": cost / gross if gross else float("nan"),
        "turnover_ann": float(daily["turnover"].mean() * TRADING_DAYS),
        "n_dates": n,
    }


def _dm_block(
    daily: pd.DataFrame, benchmark: pd.Series, name: str, target: str = "net_ret"
) -> dict[str, Any]:
    """Diebold–Mariano test on squared errors of ``target`` vs ``benchmark``.

    ``dm > 0`` means the strategy has the larger loss (i.e. it is worse). Both
    series must be aligned on dates; missing values are dropped, never filled.
    """
    bench = benchmark
    if isinstance(bench, pd.DataFrame):
        bench = bench.set_index("date") if "date" in bench.columns else bench
        bench = bench.iloc[:, 0]
    aligned = pd.DataFrame(
        {"strat": daily.set_index("date")[target], "bench": pd.Series(bench)}
    ).dropna()
    if len(aligned) < 30:
        return {"benchmark": name, "n": int(len(aligned)), "dm": None, "p": None}
    loss_a = (aligned["strat"].to_numpy() - 0.0) ** 2
    loss_b = (aligned["bench"].to_numpy() - 0.0) ** 2
    dm, p = diebold_mariano(loss_a, loss_b, h=1)
    return {"benchmark": name, "n": int(len(aligned)), "dm": float(dm), "p": float(p)}


# --------------------------------------------------------------------------- suite
@dataclass
class BacktestReport:
    run_id: str
    mode: str
    result: PortfolioResult
    frame: ExecutionFrame
    daily: pd.DataFrame
    ic_block: dict[str, Any]
    quintiles: pd.DataFrame
    bootstrap: BlockBootstrap
    null: pd.DataFrame
    dm: dict[str, Any]
    headline: dict[str, Any]
    notes: list[str]

    def as_dict(self) -> dict[str, Any]:
        key = self.ic_block.get("pooled", {})
        q_rho = float(self.quintiles.attrs.get("monotonicity_spearman", float("nan")))
        q_spread = float(self.quintiles.attrs.get("top_minus_bottom", float("nan")))
        summary = self.result.summary
        return {
            "run_id": self.run_id,
            "execution": self.mode,
            "window": {
                "start": str(pd.to_datetime(self.daily["date"]).min().date()),
                "end": str(pd.to_datetime(self.daily["date"]).max().date()),
            },
            "notes": self.notes,
            "metrics": {
                "sharpe_gross": summary.get("sharpe_gross"),
                "sharpe_net": summary.get("sharpe_net"),
                "sortino": summary.get("sortino"),
                "ann_return": summary.get("ann_return"),
                "ann_vol": summary.get("ann_vol"),
                "max_dd": summary.get("max_dd"),
                "calmar": summary.get("calmar"),
                "hit_rate_daily": summary.get("hit_rate_daily"),
                "turnover_ann": summary.get("turnover_ann"),
                "avg_n_long": summary.get("avg_n_long"),
                "avg_n_short": summary.get("avg_n_short"),
                "pnl_by_year": summary.get("pnl_by_year"),
                "cost_drag_ann": summary.get("cost_drag_ann"),
                "borrow_drag_ann": summary.get("borrow_drag_ann"),
                "n_dates": summary.get("n_dates"),
                "n_dates_traded": summary.get("n_dates_traded"),
                "n_dates_skipped": summary.get("n_dates_skipped"),
            },
            "ic": {
                "pooled": key,
                "by_year": self.ic_block.get("by_year"),
                "trimmed_1pct": self.ic_block.get("trimmed_1pct"),
                "adj_close_ret": self.ic_block.get("adj_close_ret"),
                "adj_minus_raw_mean": self.ic_block.get("adj_minus_raw_mean"),
            },
            "quintiles": {
                "mean_ret": {str(k): float(v) for k, v in self.quintiles["mean_ret"].items()},
                "monotonicity_spearman": q_rho,
                "top_minus_bottom": q_spread,
            },
            "annual": annual_returns(self.daily),
            "annual_gross": annual_returns(self.daily, "gross_ret"),
            "costs": cost_decomposition(self.daily),
            "bootstrap_sharpe": self.bootstrap.as_dict(),
            "null_distribution": self.headline.get("null"),
            "dm": self.dm,
            "headline": self.headline,
            "data_quality": {
                "n_halted_rows": self.frame.n_halted,
                "n_nan_ret_rows": self.frame.n_nan_ret,
                "flagged_days": int((self.daily["reason"] != "traded").sum()),
                "skipped_reasons": {
                    str(k): int(v)
                    for k, v in self.daily.loc[self.daily["reason"] != "traded", "reason"]
                    .value_counts()
                    .items()
                },
            },
        }


def run_backtest_suite(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    mode: str,
    sector_map: dict[str, str] | None = None,
    n_quantiles: int = 5,
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
    borrow_annual: float = 0.05,
    sector_cap: float | None = None,
    max_date: str | pd.Timestamp | None = None,
    run_id: str = "local",
    null_draws: int = 50,
    bootstrap_draws: int = BOOTSTRAP_ITER,
    seed: int = 42,
) -> BacktestReport:
    """Full T11 pipeline for one execution mode."""
    frame = build_execution_frame(
        predictions, prices, mode, sector_map=sector_map, max_date=max_date
    )
    result = run_backtest(
        frame.frame,
        n_quantiles=n_quantiles,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        borrow_annual=borrow_annual,
        sector_cap=sector_cap,
    )
    daily = result.daily
    ranked = leg_holdings(frame.frame, n_quantiles=n_quantiles)
    ic_block = ic_diagnostics(ranked)
    quintiles = quintile_returns(frame.frame, n_quantiles=n_quantiles)
    traded_net = daily["net_ret"].dropna().to_numpy()
    bootstrap = block_bootstrap_sharpe(traded_net, n_draws=bootstrap_draws, seed=seed)

    # Null model: same universe, same costs, scores permuted inside each date.
    null = _null_draws(
        frame.frame,
        n_draws=null_draws,
        n_quantiles=n_quantiles,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        borrow_annual=borrow_annual,
        seed=seed,
    )
    null_summary = _null_summary(null, result.summary.get("sharpe_net"))

    spy_daily = _spy_returns(prices, daily["date"], mode)
    spy_sharpe = float(spy_daily.mean() / spy_daily.std(ddof=1) * np.sqrt(TRADING_DAYS))
    zero = pd.Series(0.0, index=pd.to_datetime(daily["date"]))
    dm = {
        "vs_zero": _dm_block(daily, zero, "zero"),
        "vs_spy_long_only": _dm_block(daily, spy_daily, "spy_long_only"),
    }

    headline = {
        "verdict": _verdict(result.summary, bootstrap, ic_block, quintiles, null_summary),
        "null": null_summary,
        "spy_long_only": {
            "ann_ret": float((1.0 + spy_daily).prod() ** (TRADING_DAYS / len(spy_daily)) - 1.0),
            "sharpe": spy_sharpe,
            "n_dates": int(len(spy_daily)),
        },
        "gates": _gates(ic_block, bootstrap, quintiles),
    }
    notes = list(frame.notes) + result.notes
    return BacktestReport(
        run_id=run_id,
        mode=mode,
        result=result,
        frame=frame,
        daily=daily,
        ic_block=ic_block,
        quintiles=quintiles,
        bootstrap=bootstrap,
        null=null,
        dm=dm,
        headline=headline,
        notes=notes,
    )


def _null_draws(
    frame: pd.DataFrame,
    n_draws: int,
    seed: int,
    **kwargs: Any,
) -> pd.DataFrame:
    from spp.backtest.portfolio import null_distribution

    if n_draws <= 0:
        return pd.DataFrame(columns=["draw", "sharpe_net", "sharpe_gross", "ann_return"])
    return null_distribution(frame, n_draws=n_draws, seed=seed, **kwargs)


def _null_summary(null: pd.DataFrame, observed: float | None) -> dict[str, float | None]:
    if null.empty:
        return {"n_draws": 0, "mean_sharpe": None, "p_value": None}
    values = null["sharpe_net"].dropna().to_numpy()
    if len(values) == 0:
        return {"n_draws": 0, "mean_sharpe": None, "p_value": None}
    p_value = float((values >= (observed if observed is not None else np.inf)).mean())
    return {
        "n_draws": int(len(values)),
        "mean_sharpe": float(values.mean()),
        "std_sharpe": float(values.std(ddof=1)) if len(values) > 1 else None,
        "q05_sharpe": float(np.quantile(values, 0.05)),
        "q95_sharpe": float(np.quantile(values, 0.95)),
        "p_value": p_value,
    }


def _spy_returns(prices: pd.DataFrame, dates: pd.Series, mode: str) -> pd.Series:
    """SPY long-only return over the strategy's exact holding horizon."""
    stub = pd.DataFrame(
        {"date": pd.to_datetime(dates).drop_duplicates(), "ticker": "SPY", "y_pred_seed_mean": 1.0}
    )
    frame = build_execution_frame(stub, prices, mode).frame
    return frame.set_index("date")["ret"].fillna(0.0).rename("spy_long_only")


def _gates(
    ic_block: dict[str, Any], bootstrap: BlockBootstrap, quintiles: pd.DataFrame
) -> dict[str, Any]:
    """Pre-registered T10 signal criteria, evaluated on the backtest window."""
    pooled = ic_block.get("pooled", {}) or {}
    ic_value = pooled.get("ic")
    lo = (pooled.get("bootstrap_ci95") or [None, None])[0]
    rho = float(quintiles.attrs.get("monotonicity_spearman", float("nan")))
    return {
        "ic_gt_0.02": bool(ic_value is not None and ic_value > 0.02),
        "ic_ci_lo_gt_0": bool(lo is not None and lo > 0.0),
        "quintile_monotonicity_gt_0.5": bool(rho == rho and rho > 0.5),
        "bootstrap_sharpe_ci_lo_gt_0": bool(
            bootstrap.lo == bootstrap.lo and bootstrap.lo > 0.0
        ),
    }


def _verdict(
    summary: dict[str, Any],
    bootstrap: BlockBootstrap,
    ic_block: dict[str, Any],
    quintiles: pd.DataFrame,
    null_summary: dict[str, Any],
) -> str:
    gates = _gates(ic_block, bootstrap, quintiles)
    passed = sum(1 for v in gates.values() if v)
    if passed == len(gates):
        return "signal_confirmed"
    if passed == 0:
        return "not_confirmed"
    return "insufficient_evidence"


# --------------------------------------------------------------------------- artefacts
def write_artifacts(
    reports: dict[str, BacktestReport],
    out_dir: Path,
    figures_dir: Path,
    config_snapshot: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write per-mode artifacts plus a primary-mode alias used by later stages.

    The primary mode (``next_open_hold_close``) also lands as
    ``backtest_daily_returns.csv`` / ``backtest_metrics.json`` because acceptance
    check 4 re-derives Sharpe and max drawdown from that exact CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for mode, report in reports.items():
        run_id = report.run_id
        payload = report.as_dict()
        if config_snapshot is not None:
            payload["config_snapshot"] = config_snapshot
        json_path = out_dir / f"backtest_{run_id}.json"
        written[f"{mode}:json"] = _write_json(payload, json_path)
        md_path = out_dir / f"backtest_{run_id}.md"
        md_path.write_text(render_markdown(report), encoding="utf-8")
        written[f"{mode}:md"] = md_path
        daily_path = out_dir / f"backtest_{run_id}_daily_returns.csv"
        report.daily.to_csv(daily_path, index=False)
        written[f"{mode}:daily"] = daily_path
        figure_paths = write_figures(report, figures_dir)
        written.update({f"{mode}:{k}": v for k, v in figure_paths.items()})

        if mode == "next_open_hold_close":
            alias_daily = out_dir / "backtest_daily_returns.csv"
            report.daily.to_csv(alias_daily, index=False)
            written["primary:daily"] = alias_daily
            written["primary:json"] = _write_json(payload, out_dir / "backtest_metrics.json")
    return written


def write_figures(report: BacktestReport, figures_dir: Path) -> dict[str, Path]:
    """Equity curve (net) and daily-return histogram, both from the saved CSV frame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: dict[str, Path] = {}
    daily = report.daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    equity = (1.0 + daily["net_ret"].fillna(0.0)).cumprod()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(daily["date"], equity, color="#1f5fa8", linewidth=1.2, label="net equity")
    ax.plot(
        daily["date"],
        (1.0 + daily["gross_ret"].fillna(0.0)).cumprod(),
        color="#b0b0b0",
        linewidth=1.0,
        linestyle="--",
        label="gross (no costs)",
    )
    ax.axhline(1.0, color="#444444", linewidth=0.6)
    ax.set_title(f"{report.mode}: net equity (run {report.run_id})")
    ax.set_ylabel("growth of 1.0")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = figures_dir / f"equity_{report.run_id}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written["equity"] = path

    net = daily["net_ret"].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(net.to_numpy(), bins=60, color="#1f5fa8", alpha=0.85)
    ax.axvline(float(net.mean()), color="#b00020", linewidth=1.0, label=f"mean {net.mean():.4%}")
    ax.axvline(0.0, color="#444444", linewidth=0.6)
    ax.set_title(f"{report.mode}: daily net returns (n={len(net)})")
    ax.set_xlabel("net return")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = figures_dir / f"hist_{report.run_id}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written["hist"] = path

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.fill_between(
        daily["date"], daily["drawdown"].to_numpy() * 100, 0, color="#b00020", alpha=0.35
    )
    ax.set_title(f"{report.mode}: drawdown (net, %)")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = figures_dir / f"drawdown_{report.run_id}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written["drawdown"] = path
    return written


def _fmt(value: Any, digits: int = 4, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return "n/a"
    return f"{f * 100:.2f}%" if percent else f"{f:.{digits}f}"


def render_markdown(report: BacktestReport) -> str:
    """Render the plan's report layout: table, figures, quintiles, costs, verdict."""
    payload = report.as_dict()
    m = payload["metrics"]
    key = payload["ic"]["pooled"]
    boot = payload["bootstrap_sharpe"]
    costs = payload["costs"]
    lines: list[str] = []
    lines.append(f"# Backtest report — {report.run_id}")
    lines.append("")
    lines.append(
        f"Execution: `{report.mode}` · window "
        f"{payload['window']['start']} .. {payload['window']['end']}"
    )
    lines.append("")
    lines.append("> Costs: 5 bps commission + 5 bps slippage per side, 5% annual borrow on shorts,")
    lines.append("> dollar-neutral, gross 1.0. Returns are raw-price (tradable), not `adj_close`.")
    lines.append("")
    lines.append("## 1. Metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for name in (
        "sharpe_gross",
        "sharpe_net",
        "sortino",
        "ann_return",
        "ann_vol",
        "max_dd",
        "calmar",
        "hit_rate_daily",
        "turnover_ann",
        "avg_n_long",
        "avg_n_short",
        "cost_drag_ann",
        "borrow_drag_ann",
    ):
        percent = name in {
            "ann_return",
            "ann_vol",
            "max_dd",
            "hit_rate_daily",
            "cost_drag_ann",
            "borrow_drag_ann",
        }
        lines.append(f"| {name} | {_fmt(m.get(name), percent=percent)} |")
    lines.append(
        f"| dates | {m['n_dates']} ({m['n_dates_traded']} traded, {m['n_dates_skipped']} skipped) |"
    )
    lo = key.get("bootstrap_ci95") or [None, None]
    lines.append(
        f"| IC (pooled, tradable ret) | {_fmt(key.get('ic'))} "
        f"| t={_fmt(key.get('t'))} | NW t={_fmt(key.get('t_nw'))} | "
        f"CI95 [{_fmt(lo[0])}, {_fmt(lo[1])}] |"
    )
    lines.append("")
    lines.append(
        f"Sharpe net block-bootstrap CI95 [{_fmt(boot['ci95'][0])}, {_fmt(boot['ci95'][1])}], "
        f"P(Sharpe<=0)={_fmt(boot['p_value_le_zero'])} "
        f"(block={boot['block']}, n={boot['n_draws']})."
    )
    null = payload["null_distribution"] or {}
    if null.get("n_draws"):
        lines.append(
            f"Null (scores permuted per date, same book and costs, n={null['n_draws']}): "
            f"mean Sharpe {_fmt(null['mean_sharpe'])}, "
            f"90% range [{_fmt(null['q05_sharpe'])}, {_fmt(null['q95_sharpe'])}], "
            f"p(observed)={_fmt(null['p_value'])}."
        )
    lines.append("")
    lines.append("## 2. Equity curve")
    lines.append("")
    lines.append(f"![equity](../figures/equity_{report.run_id}.png)")
    lines.append(f"![drawdown](../figures/drawdown_{report.run_id}.png)")
    lines.append("")
    lines.append("## 3. Daily return distribution")
    lines.append("")
    lines.append(f"![hist](../figures/hist_{report.run_id}.png)")
    lines.append("")
    lines.append("## 4. Returns by predicted quintile")
    lines.append("")
    lines.append("| bucket (1 = lowest score) | mean daily ret | ann. ret | t | n dates |")
    lines.append("|---|---|---|---|---|")
    for bucket, row in report.quintiles.iterrows():
        lines.append(
            f"| {int(bucket)} | {_fmt(row['mean_ret'], percent=True)} | "
            f"{_fmt(row['ann_ret'], percent=True)} | {_fmt(row['t_stat'], 2)} | "
            f"{int(row['n_dates'])} |"
        )
    lines.append("")
    lines.append(
        f"Monotonicity (Spearman bucket→return): "
        f"{_fmt(report.quintiles.attrs.get('monotonicity_spearman'), 3)}; "
        f"top−bottom {_fmt(report.quintiles.attrs.get('top_minus_bottom'), percent=True)} per day."
    )
    lines.append("")
    lines.append("## 5. Cost decomposition (annualised)")
    lines.append("")
    lines.append("| item | value |")
    lines.append("|---|---|")
    lines.append(f"| gross P&L | {_fmt(costs['gross_ann'], percent=True)} |")
    lines.append(f"| trading cost drag | {_fmt(costs['cost_ann'], percent=True)} |")
    lines.append(f"| borrow drag | {_fmt(costs['borrow_ann'], percent=True)} |")
    lines.append(f"| net (arithmetic) | {_fmt(costs['net_ann_arith'], percent=True)} |")
    lines.append(f"| annual turnover | {_fmt(costs['turnover_ann'], 1)}x |")
    lines.append("")
    lines.append("## 6. Yearly breakdown and DM tests")
    lines.append("")
    lines.append("| year | net ret | gross ret | sharpe_net | n |")
    lines.append("|---|---|---|---|---|")
    for year, row in payload["annual"].items():
        gross_row = payload["annual_gross"].get(year, {})
        lines.append(
            f"| {year} | {_fmt(row['ret'], percent=True)} | "
            f"{_fmt(gross_row.get('ret'), percent=True)} | "
            f"{_fmt(row['sharpe'], 2)} | {row['n_dates']} |"
        )
    lines.append("")
    for name, block in payload["dm"].items():
        if isinstance(block, dict):
            lines.append(
                f"- DM {name}: dm={_fmt(block.get('dm'), 2)} p={_fmt(block.get('p'), 3)} "
                f"(n={block.get('n')}); dm>0 means the strategy has larger squared loss."
            )
    lines.append("")
    lines.append("## 7. Data quality")
    dq = payload["data_quality"]
    lines.append("")
    lines.append(f"- halted / zero-range rows: {dq['n_halted_rows']}")
    lines.append(f"- rows without a forward price (kept NaN): {dq['n_nan_ret_rows']}")
    lines.append(f"- dates flagged (not traded): {dq['flagged_days']} {dq['skipped_reasons']}")
    adj_block = payload["ic"].get("adj_close_ret") or {}
    lines.append(
        f"- IC on `adj_close` returns: {_fmt(adj_block.get('ic'))} "
        f"(t={_fmt(adj_block.get('t'))}) vs "
        f"tradable IC {_fmt(key.get('ic'))}; mean(adj−raw) per row "
        f"{_fmt(payload['ic'].get('adj_minus_raw_mean'))}. `adj_close` is back-adjusted ex-post, "
        "so its within-session return is not tradable."
    )
    trimmed = payload["ic"].get("trimmed_1pct") or {}
    lines.append(
        f"- IC without the top-1% |return| rows: {_fmt(trimmed.get('ic'))} "
        f"(t={_fmt(trimmed.get('t'))}, dropped {trimmed.get('n_rows_dropped')} rows)."
    )
    lines.append("")
    lines.append("## 8. Verdict")
    lines.append("")
    gates = payload["headline"]["gates"]
    for name, passed in gates.items():
        lines.append(f"- gate `{name}`: **{'PASS' if passed else 'FAIL'}**")
    lines.append("")
    verdict = payload["headline"]["verdict"]
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append(
        "Criteria are the pre-registered T10 gates, not a Sharpe target. A negative verdict is a "
        "publishable result: it says this signal does not pay for its own costs on this universe."
    )
    lines.append("")
    lines.append("## 9. Reproduce")
    lines.append("")
    lines.append("```powershell")
    lines.append(f"python -m spp.cli backtest --run-id {report.run_id} --execution {report.mode}")
    lines.append("```")
    return "\n".join(lines) + "\n"


def reproducible_stats(csv_path: Path) -> dict[str, float]:
    """Re-derive Sharpe (gross/net) and max drawdown from a saved daily CSV.

    Acceptance check 4 compares these numbers with the JSON payload; it also
    re-validates ``net = gross - cost - borrow`` to 1e-12 on every row.
    """
    daily = pd.read_csv(csv_path)
    net = daily["net_ret"]
    gross = daily["trade_ret"]
    equity = (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    residual = (daily["trade_ret"] - daily["cost"] - daily["borrow"] - daily["net_ret"]).abs().max()
    return {
        "sharpe_net": float(net.mean() / net.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe_gross": float(gross.mean() / gross.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "max_dd": float(drawdown.min()),
        "identity_residual_max": float(residual),
        "max_abs_net_weight": float(daily["net_weight"].abs().max()),
    }
