# ruff: noqa: E501  - prints report tables
"""Where the amplitude is: structural events measured out to one trading day.

The published cross-section says the average headline is worth 0.09-0.87 bps against a
28 bps round trip. That verdict is about *ordinary* headlines. This script asks the
different question: do the headlines that describe a structural event - a listing, a
delisting, an exploit, an ETF decision, an insolvency - move the named asset by enough
to clear the cost, and how much of that move is still available after the headline is
public?

Four measurements per class and horizon:

* ``jump`` - the move from the close of the bar that contains ``published_at`` to the
  open of the first bar after it. That part is unreachable: it happens before an order
  can be routed.
* ``pre60`` - the move in the hour strictly before entry. A class that has already run
  100 bps is a class where the news is late.
* ``dir`` - the direction-adjusted absolute return from the entry open to the exit close.
  This is what a one-leg trade earns; the cost is 14 bps.
* ``excess`` - the same minus BTC over identical stamps. The cross-section trade is a
  spread and pays two legs, 28 bps.

Horizons beyond the dataset's 5/15/60 minutes are recomputed from the 1-minute bars.
The recomputed 5/15/60 numbers are compared against the stored ``ret_*`` columns, so the
extended horizons rest on an implementation that is verified against the published one.

Usage:
    python ops/crypto_structural.py [--bars-dir data/crypto/bars/1m] [--out results/crypto_structural.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spp.crypto import taxonomy

ROOT = Path(__file__).resolve().parents[1]
CROSS_EVENTS = ROOT / "data" / "crypto" / "dataset" / "events.parquet"
BTC_EVENTS = ROOT / "data" / "crypto" / "dataset" / "events_btc.parquet"
BARS_DIR = ROOT / "data" / "crypto" / "bars" / "1m"
RESULTS = ROOT / "results"

HORIZONS: tuple[int, ...] = (5, 15, 60, 240, 1440)
CHECK_HORIZONS: tuple[int, ...] = (5, 15, 60)
COST_ONE_LEG_BPS = 14.0
COST_TWO_LEGS_BPS = 28.0
MINUTE_NS = 60 * 10**9
BENCH = "BTCUSDT"
MIN_EVENTS = 30


def _ns(stamps: pd.Series) -> np.ndarray:
    return pd.to_datetime(stamps, utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()


def _lookup(stamps: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the exact stamp and a mask of the targets that exist."""
    idx = np.searchsorted(stamps, targets, side="left")
    safe = np.clip(idx, 0, max(0, len(stamps) - 1))
    return safe, (idx < len(stamps)) & (stamps[safe] == targets)


def load_events() -> pd.DataFrame:
    frames = []
    for path in (CROSS_EVENTS, BTC_EVENTS):
        if path.is_file():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise SystemExit(f"no event files under {CROSS_EVENTS.parent}")
    events = pd.concat(frames, ignore_index=True)
    events = events.drop_duplicates(subset=["url", "symbol"]).reset_index(drop=True)
    events["published_at"] = pd.to_datetime(events["published_at"], utc=True)
    events["entry_ts"] = pd.to_datetime(events["entry_ts"], utc=True)
    events["day"] = events["entry_ts"].dt.date
    events["klass"] = [taxonomy.classify(title) for title in events["title"]]
    events["direction"] = [taxonomy.DIRECTIONS.get(k, 0) for k in events["klass"]]
    events["klass"] = events["klass"].fillna("commentary")
    return events


def forward_returns(events: pd.DataFrame, bars_dir: Path) -> pd.DataFrame:
    """Direction-free forward measurements, aligned to ``events.index``."""
    out = pd.DataFrame(index=events.index, dtype="float64")
    out["jump_bps"] = np.nan
    out["pre60_bps"] = np.nan
    for horizon in HORIZONS:
        out[f"ret_{horizon}"] = np.nan
        out[f"bench_{horizon}"] = np.nan
        out[f"excess_{horizon}"] = np.nan

    bench_bars = pd.read_parquet(bars_dir / f"{BENCH}.parquet", columns=["ts_open", "open", "close"])
    b_stamps = _ns(bench_bars["ts_open"])
    b_open = bench_bars["open"].to_numpy(dtype="float64")
    b_close = bench_bars["close"].to_numpy(dtype="float64")

    for symbol, group in events.groupby("symbol", sort=False):
        path = bars_dir / f"{symbol}.parquet"
        if not path.is_file():
            continue
        bars = pd.read_parquet(path, columns=["ts_open", "open", "close"])
        stamps = _ns(bars["ts_open"])
        open_ = bars["open"].to_numpy(dtype="float64")
        close = bars["close"].to_numpy(dtype="float64")
        n = len(stamps)

        entry = _ns(group["entry_ts"])
        published = _ns(group["published_at"])
        idx, ok = _lookup(stamps, entry)

        jump = np.full(len(group), np.nan)
        ref, ok_ref = _lookup(stamps, published)
        # The reference bar is the last one that *opened* at or before the publication.
        ref = np.clip(np.searchsorted(stamps, published, side="right") - 1, 0, n - 1)
        ok_ref = np.searchsorted(stamps, published, side="right") > 0
        usable = ok & ok_ref & (ref < idx)
        jump[usable] = open_[idx[usable]] / close[ref[usable]] - 1.0
        out.loc[group.index, "jump_bps"] = jump * 1e4

        pre_target = entry - 60 * MINUTE_NS
        pre, ok_pre = _lookup(stamps, pre_target)
        pre_move = np.full(len(group), np.nan)
        usable = ok & ok_pre & (idx >= 1)
        pre_move[usable] = close[idx[usable] - 1] / open_[pre[usable]] - 1.0
        out.loc[group.index, "pre60_bps"] = pre_move * 1e4

        b_entry, ok_b_entry = _lookup(b_stamps, entry)
        for horizon in HORIZONS:
            target = entry + (horizon - 1) * MINUTE_NS
            exit_i, ok_exit = _lookup(stamps, target)
            ret = np.full(len(group), np.nan)
            usable = ok & ok_exit
            ret[usable] = close[exit_i[usable]] / open_[idx[usable]] - 1.0

            b_exit, ok_b_exit = _lookup(b_stamps, target)
            bench = np.full(len(group), np.nan)
            b_usable = ok_b_entry & ok_b_exit
            bench[b_usable] = b_close[b_exit[b_usable]] / b_open[b_entry[b_usable]] - 1.0

            out.loc[group.index, f"ret_{horizon}"] = ret * 1e4
            out.loc[group.index, f"bench_{horizon}"] = bench * 1e4
            out.loc[group.index, f"excess_{horizon}"] = (ret - bench) * 1e4
    return out


def cluster_t(frame: pd.DataFrame, value: str, group: str = "day") -> float:
    """One-way cluster-robust t of the mean with the ``k / (k - 1)`` factor."""
    work = frame[[value, group]].dropna()
    if len(work) < 3 or work[group].nunique() < 2:
        return float("nan")
    grouped = work.groupby(group)[value]
    k = grouped.ngroups
    mean = float(work[value].mean())
    resid = grouped.sum() - mean * grouped.size()
    variance = float((resid**2).sum()) / len(work) ** 2 * k / (k - 1)
    if variance <= 0:
        return float("nan")
    return float(mean / np.sqrt(variance))


def cluster_se(frame: pd.DataFrame, value: str, group: str = "day") -> float:
    work = frame[[value, group]].dropna()
    if len(work) < 3 or work[group].nunique() < 2:
        return float("nan")
    grouped = work.groupby(group)[value]
    k = grouped.ngroups
    mean = float(work[value].mean())
    resid = grouped.sum() - mean * grouped.size()
    variance = float((resid**2).sum()) / len(work) ** 2 * k / (k - 1)
    return float(np.sqrt(variance)) if variance > 0 else float("nan")


def summarise(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for klass, block in frame.groupby("klass", sort=False):
        direction = int(block["direction"].iloc[0])
        row: dict[str, Any] = {
            "klass": klass,
            "direction": direction,
            "n": int(len(block)),
            "n_days": int(block["day"].nunique()),
            "n_titles": int(block["title"].nunique()),
            "symbols": block["symbol"].value_counts().head(3).index.tolist(),
            "jump_bps": float(block["jump_bps"].mean()),
            "pre60_bps": float(block["pre60_bps"].mean()),
        }
        for horizon in HORIZONS:
            row[f"mean_abs_{horizon}m_bps"] = float(block[f"ret_{horizon}"].abs().mean())
            ret = block[f"ret_{horizon}"]
            excess = block[f"excess_{horizon}"]
            if direction:
                signed = direction * ret
                signed_excess = direction * excess
                row[f"dir_{horizon}_bps"] = float(signed.mean())
                row[f"t_dir_{horizon}"] = cluster_t(pd.DataFrame({"v": signed, "day": block["day"]}), "v")
                row[f"mde_{horizon}"] = 2.802 * cluster_se(
                    pd.DataFrame({"v": signed, "day": block["day"]}), "v"
                )
                row[f"hit_{horizon}"] = float((signed.dropna() > 0).mean())
                row[f"dir_excess_{horizon}_bps"] = float(signed_excess.mean())
                row[f"t_dir_excess_{horizon}"] = cluster_t(
                    pd.DataFrame({"v": signed_excess, "day": block["day"]}), "v"
                )
                row[f"net_abs_{horizon}_bps"] = row[f"dir_{horizon}_bps"] - COST_ONE_LEG_BPS
                row[f"net_excess_{horizon}_bps"] = row[f"dir_excess_{horizon}_bps"] - COST_TWO_LEGS_BPS
            else:
                row[f"dir_{horizon}_bps"] = float("nan")
                row[f"t_dir_{horizon}"] = float("nan")
                row[f"mde_{horizon}"] = float("nan")
                row[f"hit_{horizon}"] = float("nan")
                row[f"dir_excess_{horizon}_bps"] = float("nan")
                row[f"t_dir_excess_{horizon}"] = float("nan")
                row[f"net_abs_{horizon}_bps"] = float("nan")
                row[f"net_excess_{horizon}_bps"] = float("nan")
        rows.append(row)
    return rows


def _fmt(value: float, digits: int = 2, sign: bool = True) -> str:
    if value != value:
        return "-"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def render(rows: list[dict[str, Any]], coverage: dict[str, int], validation: dict[str, float]) -> str:
    ordered = sorted(rows, key=lambda r: (r["klass"] == "commentary", -r["n"]))
    lines = [
        "# Структурные события: где амплитуда\n\n",
        "Класс определяется по заголовку детерминированными правилами "
        "(`src/spp/crypto/taxonomy.py`), знак — ожидаемое направление актива. "
        f"Абсолютная доходность — одна нога, {COST_ONE_LEG_BPS:.0f} бп; "
        f"excess против BTC — две ноги, {COST_TWO_LEGS_BPS:.0f} бп. "
        "`jump` — движение от закрытия бара с `published_at` до открытия нашего входа "
        "(недостижимо), `pre60` — движение за час до входа (опоздание).\n\n",
    ]

    depth = "| класс | n | дней | jump | pre60 | " + " | ".join(
        f"{h} мин" for h in (60, 240, 1440)
    )
    depth += " |\n|---|---|---|---|---|" + "---|" * 3 + "\n"
    for row in ordered:
        depth += (
            f"| {row['klass']} | {row['n']} | {row['n_days']} | {_fmt(row['jump_bps'])} | "
            f"{_fmt(row['pre60_bps'])} | "
            + " | ".join(
                f"{_fmt(row[f'dir_{h}_bps'])} (t {_fmt(row[f't_dir_{h}'])})"
                for h in (60, 240, 1440)
            )
            + " |\n"
        )
    lines.append("## Направленный эффект по абсолютной доходности, бп\n\n" + depth)

    amp = "| класс | n | 5 мин | 15 мин | 60 мин | 240 мин | 1440 мин |\n|---|---|---|---|---|---|---|\n"
    for row in ordered:
        amp += (
            f"| {row['klass']} | {row['n']} | "
            + " | ".join(_fmt(row[f"mean_abs_{h}m_bps"], sign=False) for h in (5, 15, 60, 240, 1440))
            + " |\n"
        )
    lines.append("\n## Амплитуда: среднее |доходность|, бп\n\n" + amp)

    net = (
        "| класс | направление | n | 60 мин dir | net (14 бп) | t | 240 мин dir | net | t | 1440 мин dir | net | t |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    for row in ordered:
        direction = {1: "+1", -1: "−1", 0: "—"}.get(row["direction"], "—")
        net += (
            f"| {row['klass']} | {direction} | {row['n']} | "
            + " | ".join(
                f"{_fmt(row[f'dir_{h}_bps'])} | {_fmt(row[f'net_abs_{h}_bps'])} | {_fmt(row[f't_dir_{h}'])}"
                for h in (60, 240, 1440)
            )
            + " |\n"
        )
    lines.append("\n## Экономика одной ноги (14 бп)\n\n" + net)

    excess = "| класс | n | 60 мин excess | net (28 бп) | t | 1440 мин excess | net | t |\n|---|---|---|---|---|---|---|---|\n"
    for row in ordered:
        excess += (
            f"| {row['klass']} | {row['n']} | {_fmt(row['dir_excess_60_bps'])} | "
            f"{_fmt(row['net_excess_60_bps'])} | {_fmt(row['t_dir_excess_60'])} | "
            f"{_fmt(row['dir_excess_1440_bps'])} | {_fmt(row['net_excess_1440_bps'])} | "
            f"{_fmt(row['t_dir_excess_1440'])} |\n"
        )
    lines.append("\n## Кросс-секция: excess против BTC (28 бп)\n\n" + excess)

    lines.append("\n## Покрытие и проверка реализации\n\n")
    lines.append(
        "Событий с измеримым горизонтом: "
        + ", ".join(f"{h} мин — {coverage[str(h)]}" for h in HORIZONS)
        + ".\n\n"
    )
    lines.append(
        "Пересчёт из баров против сохранённых колонок датасета (кросс-секция): "
        + ", ".join(f"{h} мин — max|Δ| {validation[str(h)]:.2e}" for h in CHECK_HORIZONS)
        + ".\n"
    )
    lines.append(
        "\nКлассы с n < 30 в таблицы попадают, но читать их нельзя: "
        "кластерная ошибка среднего там больше самого среднего.\n"
    )
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars-dir", default=str(BARS_DIR))
    parser.add_argument("--out", default=str(RESULTS / "crypto_structural.json"))
    args = parser.parse_args()

    events = load_events()
    bars_dir = Path(args.bars_dir)
    forward = forward_returns(events, bars_dir)
    frame = pd.concat([events, forward], axis=1)
    print(
        f"events={len(frame)} structural={int((frame['klass'] != 'commentary').sum())} "
        f"classes={frame.loc[frame['klass'] != 'commentary', 'klass'].nunique()}"
    )

    validation: dict[str, float] = {}
    cross = frame[frame["symbol"] != BENCH]
    for horizon in CHECK_HORIZONS:
        # the stored labels are fractions, the recomputed ones are basis points
        stored = cross[f"ret_{horizon}m"].to_numpy(dtype="float64") * 1e4
        computed = cross[f"ret_{horizon}"].to_numpy(dtype="float64")
        both = ~np.isnan(stored) & ~np.isnan(computed)
        validation[str(horizon)] = float(np.max(np.abs(stored[both] - computed[both]))) if both.any() else float("nan")
    print("validation max|delta| bps:", validation)

    coverage = {str(h): int(frame[f"ret_{h}"].notna().sum()) for h in HORIZONS}
    print("coverage:", coverage)

    rows = summarise(frame)
    frame.to_parquet(ROOT / "data" / "crypto" / "dataset" / "events_structural.parquet", index=False)

    for row in sorted(rows, key=lambda r: (r["klass"] == "commentary", -r["n"])):
        print(
            f"{row['klass']:14s} n={row['n']:6d} days={row['n_days']:5d} "
            f"|60m| {row['mean_abs_60m_bps']:6.1f}  "
            f"dir60 {_fmt(row['dir_60_bps'])} (t {_fmt(row['t_dir_60'])})  "
            f"dir240 {_fmt(row['dir_240_bps'])} (t {_fmt(row['t_dir_240'])})  "
            f"dir1440 {_fmt(row['dir_1440_bps'])} (t {_fmt(row['t_dir_1440'])})"
        )

    Path(args.out).write_text(
        json.dumps(
            {
                "rows": rows,
                "coverage": coverage,
                "validation_max_delta_bps": validation,
                "costs": {"one_leg_bps": COST_ONE_LEG_BPS, "two_legs_bps": COST_TWO_LEGS_BPS},
                "horizons": list(HORIZONS),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    report = render(rows, coverage, validation)
    (RESULTS / "crypto_structural.md").write_text(report, encoding="utf-8")
    print(f"wrote {args.out} and results/crypto_structural.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
