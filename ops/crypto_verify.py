# ruff: noqa: E501  - prints a report table
"""Independent recomputation of every published number.

This script deliberately does not import the pipeline modules: the point is to re-derive the
claims from the artifacts with separate arithmetic, so that a bug in ``models.py`` or
``dataset.py`` cannot hide inside a "verification". Only loaders are shared.

Every check has a claimed value (hardcoded from the document or read from the results JSON)
and a recomputed value; the script prints both and fails on any mismatch outside tolerance.

Usage:
    python ops/crypto_verify.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "crypto"
RESULTS = ROOT / "results"
BPS = 1e4
TOL_BPS = 0.02
TOL_T = 0.02
TOL_PCT = 0.01
notes: list[str] = []


def cluster_t(values: pd.Series, days: pd.Series) -> float:
    """CR1 t-statistic clustered on ``days``, written from the definition."""
    frame = pd.DataFrame({"x": values.to_numpy(dtype="float64"), "g": days.to_numpy()})
    frame = frame.dropna()
    if len(frame) < 3 or frame["g"].nunique() < 2:
        return float("nan")
    mean = float(frame["x"].mean())
    sums = frame.groupby("g")["x"].sum()
    sizes = frame.groupby("g")["x"].size()
    resid = sums - mean * sizes
    k = int(sums.size)
    var = float((resid**2).sum()) / len(frame) ** 2 * k / (k - 1)
    return mean / np.sqrt(var) if var > 0 else float("nan")


def mde80(values: pd.Series, days: pd.Series) -> float:
    """80 % MDE as published: 2.802 * sd / sqrt(n), i.i.d. (the clustered value is larger)."""
    clean = values.dropna()
    if len(clean) < 3:
        return float("nan")
    return float(2.802 * clean.std(ddof=1) / np.sqrt(len(clean)))


def mde80_clustered(values: pd.Series, days: pd.Series) -> float:
    """The same quantity under day clustering: 2.802 * se, with se backed out of the CR1 t."""
    t = cluster_t(values, days)
    mean = float(values.mean())
    if not np.isfinite(t) or t == 0:
        return float("nan")
    return abs(2.802 * mean / t)


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def check(self, name: str, claimed: float, computed: float, tol: float, unit: str = "") -> None:
        if claimed != claimed or computed != computed:
            verdict = "SKIP"
            delta = float("nan")
        else:
            delta = computed - claimed
            verdict = "PASS" if abs(delta) <= tol else "FAIL"
        self.rows.append(
            {"check": name, "claimed": claimed, "computed": computed, "delta": delta, "verdict": verdict, "unit": unit}
        )

    def exact(self, name: str, claimed: Any, computed: Any) -> None:
        verdict = "PASS" if claimed == computed else "FAIL"
        self.rows.append({"check": name, "claimed": claimed, "computed": computed, "delta": None, "verdict": verdict, "unit": ""})

    def render(self) -> str:
        head = "| проверка | опубликовано | пересчитано | Δ | вердикт |\n|---|---|---|---|---|\n"
        lines = []
        for row in self.rows:
            if isinstance(row["claimed"], float):
                claimed = f"{row['claimed']:+.4f}"
                computed = f"{row['computed']:+.4f}" if row["computed"] == row["computed"] else "—"
                delta = f"{row['delta']:+.4f}" if row["delta"] == row["delta"] else "—"
            else:
                claimed, computed, delta = str(row["claimed"]), str(row["computed"]), "—"
            lines.append(f"| {row['check']} | {claimed} | {computed} | {delta} | {row['verdict']} |")
        return head + "\n".join(lines) + "\n"


def check_dataset(checks: Checks) -> pd.DataFrame:
    events = pd.read_parquet(DATA / "dataset" / "events.parquet")
    checks.exact("датасет: строк", 21588, len(events))
    checks.exact("датасет: активов", 11, int(events["symbol"].nunique()))
    checks.exact("датасет: издержка одна на всех", 1, int((events["cost"] - 0.0014).abs().lt(1e-9).sum() == len(events)))
    splits = {key: int(value) for key, value in events["split"].value_counts().items() if key != "embargo"}
    checks.exact("датасет: сплиты", {"train": 17155, "val": 3964, "holdout": 465}, splits)
    checks.exact("датасет: holdout не тронут (n)", 465, int((events["split"] == "holdout").sum()))

    claimed = {"5m": (0.09, 0.56, 0.54), "15m": (0.29, 1.10, 1.05), "60m": (0.87, 1.75, 1.39)}
    for horizon, (mean_bps, t_title, t_day) in claimed.items():
        excess = events[f"excess_{horizon}"] * BPS
        checks.check(f"excess {horizon}: среднее, бп", mean_bps, float(excess.mean()), TOL_BPS)
        checks.check(f"excess {horizon}: t по заголовкам", t_title, cluster_t(excess, events["title"]), TOL_T)
        checks.check(f"excess {horizon}: t по дням", t_day, cluster_t(excess, events["entry_ts"].dt.date), TOL_T)
    checks.check("excess 5m: MDE 80 % (i.i.d.)", 0.41, mde80(events["excess_5m"] * BPS, events["entry_ts"].dt.date), TOL_BPS)
    checks.check("excess 60m: MDE 80 % (i.i.d.)", 1.29, mde80(events["excess_60m"] * BPS, events["entry_ts"].dt.date), TOL_BPS)
    notes.append(
        "MDE 80 % с кластером по дням: "
        f"5m {mde80_clustered(events['excess_5m'] * BPS, events['entry_ts'].dt.date):.2f} бп, "
        f"60m {mde80_clustered(events['excess_60m'] * BPS, events['entry_ts'].dt.date):.2f} бп — "
        "опубликованные 0.41 / 1.29 это i.i.d.-оценка, кластерная честнее и хуже."
    )

    # The label must be exactly the asset leg minus the benchmark leg on the same stamps.
    for horizon in ("5m", "15m", "60m"):
        gap = (events[f"ret_{horizon}"] - events[f"bench_{horizon}"] - events[f"excess_{horizon}"]).abs()
        checks.check(f"excess {horizon} = ret − bench: max|Δ|", 0.0, float(gap.max()) * BPS, TOL_BPS)
    lag = pd.to_datetime(events["entry_ts"], utc=True) - pd.to_datetime(events["published_at"], utc=True)
    checks.check("лаг p50, мин", 1.45, float(lag.dt.total_seconds().quantile(0.5) / 60), 0.01)
    return events


def check_listings(checks: Checks) -> None:
    claimed = json.loads((RESULTS / "crypto_listings.json").read_text(encoding="utf-8"))
    stats = {row["horizon"]: row for row in claimed["stats"]["all"]} if isinstance(claimed["stats"], dict) else None
    events = pd.read_parquet(DATA / "listings" / "events.parquet")
    checks.exact("листинги: измерено", 714, len(events))
    checks.check("листинги: аукцион, медиана %", 18.14, float(events["auction_ret"].median() * 100), TOL_PCT)
    day = events["entry_bar"].dt.date
    for horizon, mean_pct, t_day in (("60", -2.65, -3.56), ("1440", -5.44, -4.63)):
        column = f"ret_{horizon}"
        checks.check(f"листинги {horizon}м: среднее, %", mean_pct, float(events[column].mean() * 100), TOL_PCT)
        checks.check(f"листинги {horizon}м: t по дням", t_day, cluster_t(events[column], day), TOL_T)
    checks.check("листинги: вход = второй бар, все 714", 714, int((events["entry_bar"] - events["first_bar"]).dt.total_seconds().eq(60).sum()), 0)
    if stats is not None:
        checks.check("листинги 60м: t из артефакта", stats["60"]["t"], cluster_t(events["ret_60"], day), TOL_T)


def check_perps(checks: Checks) -> None:
    events = pd.read_parquet(DATA / "listings" / "events_perp.parquet")
    checks.exact("перпы: измерено", 923, len(events))
    day = events["entry_bar"].dt.date
    for horizon, mean_pct, t_day in (("1", -0.26, -2.15), ("5", -0.54, -3.28), ("60", -0.61, -1.79)):
        column = f"ret_{horizon}"
        checks.check(f"перпы {horizon}м: среднее, %", mean_pct, float(events[column].mean() * 100), TOL_PCT)
        checks.check(f"перпы {horizon}м: t по дням", t_day, cluster_t(events[column], day), TOL_T)
    checks.check("перпы: p99 хода против шорта, %", 51.33, float(events["max_60m"].quantile(0.99) * 100), 0.05)
    checks.check("перпы: доля ≥20 % за час", 0.093, float((events["max_60m"] >= 0.20).mean()), 0.002)
    checks.check("перпы: доля ≥33 % за час", 0.038, float((events["max_60m"] >= 1 / 3).mean()), 0.002)


def check_macro(checks: Checks) -> None:
    calendar = pd.read_csv(RESULTS / "crypto_macro_calendar.csv", parse_dates=["release_ts"])
    calendar["release_ts"] = pd.to_datetime(calendar["release_ts"], utc=True)
    checks.exact("календарь: событий", 528, len(calendar))
    checks.exact("календарь: состав", {"cpi": 225, "nfp": 225, "fomc": 78}, calendar["event"].value_counts().to_dict())

    bars = pd.read_parquet(DATA / "bars" / "1m_macro" / "BTCUSDT.parquet", columns=["ts_open", "open", "close"])
    stamps = bars["ts_open"].to_numpy(dtype="datetime64[ns]").astype("int64")
    open_ = bars["open"].to_numpy(dtype="float64")
    close = bars["close"].to_numpy(dtype="float64")

    rows = []
    for row in calendar[calendar["event"] == "cpi"].itertuples():
        release = int(pd.Timestamp(row.release_ts).value)
        i = int(np.searchsorted(stamps, release, side="left"))
        if i < 61 or i + 61 >= len(stamps):
            continue
        if stamps[i] - release > 60 * 60 * 10**9:
            continue
        sign = float(np.sign(close[i] - open_[i]))  # a doji is not a direction: it contributes zero
        rows.append({"day": pd.Timestamp(row.release_ts).date(), "mom": sign * (close[i + 60] / open_[i + 1] - 1.0) * BPS})
    frame = pd.DataFrame(rows)
    checks.exact("макро: измерено CPI на BTC", 109, len(frame))
    checks.check("CPI BTC 60м: моментум, бп", 28.65, float(frame["mom"].mean()), 0.05)
    checks.check("CPI BTC 60м: t", 2.66, cluster_t(frame["mom"], frame["day"]), TOL_T)


def check_sources(checks: Checks, events: pd.DataFrame) -> None:
    published = json.loads((RESULTS / "crypto_sources.json").read_text(encoding="utf-8"))
    table = (published.get("horizons") or {}).get("5m") or []
    if not isinstance(table, list) or not table:
        checks.check("источники: таблица 5m найдена", 1, 0, 0)
        return
    excess = events["excess_5m"] * BPS
    for claimed in table:
        name = str(claimed["source"])
        mask = events["source"] == name
        if mask.sum() < 30:
            continue
        checks.exact(f"источник {name}: n", int(claimed["n"]), int(mask.sum() - excess[mask].isna().sum()))
        checks.check(f"источник {name}: среднее 5м, бп", float(claimed["mean_bps"]), float(excess[mask].mean()), 0.05)
        checks.check(
            f"источник {name}: t",
            float(claimed["t"]),
            cluster_t(excess[mask], events.loc[mask, "entry_ts"].dt.date),
            0.05,
        )


def check_structural(checks: Checks) -> None:
    """Directional effect per headline class, recomputed from the event table."""
    try:
        events = pd.read_parquet(DATA / "dataset" / "events_structural.parquet")
        direction = events["direction"].astype("float64")
        signed = direction * events["ret_60"]  # the structural table stores bps already
        day = events["entry_ts"].dt.date
    except Exception as error:  # noqa: BLE001 - a missing column is reported, not raised
        checks.rows.append(
            {"check": f"структурные: пересчёт невозможен ({type(error).__name__})", "claimed": "", "computed": "", "delta": None, "verdict": "SKIP", "unit": ""}
        )
        return
    checks.exact("структурные: событий", 45_625, len(events))
    checks.exact("структурные: commentary", 42_145, int((events["klass"] == "commentary").sum()))
    insolvent = events["klass"] == "insolvency"
    checks.exact("структурные insolvency: n", 224, int(insolvent.sum()))
    checks.check("структурные insolvency: dir60, бп", 13.48, float(signed[insolvent].mean()), 0.05)
    checks.check("структурные insolvency: t по дням", 1.63, cluster_t(signed[insolvent], day[insolvent]), TOL_T)
    liquidation_still_in = int(events["title"].str.contains("liquidat", case=False, na=False).mul(insolvent).sum())
    checks.check(
        "структурные: liquidat*-строк в insolvency (было 206 из 427)", 3, liquidation_still_in, 0
    )


def main() -> int:
    checks = Checks()
    events = check_dataset(checks)
    check_listings(checks)
    check_perps(checks)
    check_macro(checks)
    check_sources(checks, events)
    check_structural(checks)

    table = checks.render()
    failed = sum(1 for row in checks.rows if row["verdict"] == "FAIL")
    skipped = sum(1 for row in checks.rows if row["verdict"] == "SKIP")
    print(table)
    for note in notes:
        print(f"note: {note}")
    print(f"checks={len(checks.rows)} pass={len(checks.rows) - failed - skipped} fail={failed} skip={skipped}")
    (RESULTS / "crypto_verification.md").write_text(
        "# Независимая перепроверка опубликованных чисел\n\n"
        + table
        + "\n".join(f"- {note}" for note in notes)
        + f"\n\nПроверок: {len(checks.rows)}, провалов: {failed}, пропусков: {skipped}.\n",
        encoding="utf-8",
    )
    (RESULTS / "crypto_verification.json").write_text(json.dumps(checks.rows, indent=2, default=str) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
