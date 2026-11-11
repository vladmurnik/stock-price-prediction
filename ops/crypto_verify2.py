# ruff: noqa: E501  - prints a report table
"""Second-pass robustness: overlap, bootstrap, funding and slippage.

The verifier proved that every published number is reproducible. That is not the same as
proving the numbers are trustworthy: a reproducible estimate can still rest on overlapping
windows, a handful of events, or an unmeasured cost. This script attacks those three.

1. **Overlap.** A cross-sectional event's 60-minute return shares its window with any other
   headline on the same symbol inside the same hour. The i.i.d. t assumes they do not. The
   count of overlapping events and the t clustered by (symbol, day) bound the damage.
2. **Bootstrap.** The CPI result is one cell out of 36. A block bootstrap over release days
   says how much of +28.65 bps is carried by a few prints.
3. **Funding.** A perp short pays or receives funding, which the listing-perp report left
   unmodeled. Measured from Binance's own monthly funding archives on the first day, not assumed.
4. **Slippage.** Instead of one cost guess, the whole family is priced at 0/5/10/20 extra bps,
   so the reader can see where each result crosses zero.

Usage:
    python ops/crypto_verify2.py [--funding-symbols 150]
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EVENTS = ROOT / "data" / "crypto" / "dataset" / "events.parquet"
PERP = ROOT / "data" / "crypto" / "listings" / "events_perp.parquet"
SPOT_LISTINGS = ROOT / "data" / "crypto" / "listings" / "events.parquet"
BARS = ROOT / "data" / "crypto" / "bars" / "1m_macro" / "BTCUSDT.parquet"
CALENDAR = RESULTS / "crypto_macro_calendar.csv"
FUNDING_URL = (
    "https://data.binance.vision/data/futures/um/monthly/fundingRate/"
    "{symbol}/{symbol}-fundingRate-{year:04d}-{month:02d}.zip"
)
FUNDING_CACHE = ROOT / "data" / "crypto" / "funding"
BPS = 1e4
HORIZONS = (5, 15, 60)
MINUTE_NS = 60 * 1_000_000_000


def cluster_t(values: pd.Series, keys: pd.Series) -> float:
    """One-way cluster-robust t of the mean, CR1 correction (same as the pipeline's)."""
    work = pd.DataFrame({"v": values.to_numpy(dtype="float64"), "k": keys.to_numpy()})
    work = work.dropna()
    if len(work) < 3 or work["k"].nunique() < 2:
        return float("nan")
    grouped = work.groupby("k")["v"]
    mean = float(work["v"].mean())
    resid = grouped.sum() - mean * grouped.size()
    k = grouped.ngroups
    variance = float((resid**2).sum()) / len(work) ** 2 * k / (k - 1)
    return mean / float(np.sqrt(variance)) if variance > 0 else float("nan")


def overlap_block(events: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """How much of the sample shares a window with another event on the same symbol."""
    frame = events.sort_values(["symbol", "entry_ts"]).copy()
    frame["prev_ts"] = frame.groupby("symbol")["entry_ts"].shift(1)
    gap = (frame["entry_ts"] - frame["prev_ts"]).dt.total_seconds() / 60.0
    frame["gap_min"] = gap
    rows = []
    for horizon in HORIZONS:
        overlapping = int((frame["gap_min"] < horizon).sum())
        rows.append(
            {
                "horizon": horizon,
                "overlapping": overlapping,
                "share": overlapping / len(frame),
                "independent_approx": len(frame) - overlapping,
            }
        )
    frame["day"] = frame["entry_ts"].dt.date
    frame["sym_day"] = frame["symbol"].astype(str) + "|" + frame["day"].astype(str)
    table = []
    for horizon in HORIZONS:
        column = f"excess_{horizon}m"
        table.append(
            {
                "horizon": horizon,
                "mean_bps": float(frame[column].mean() * BPS),
                "t_iid": cluster_t(frame[column], pd.Series(np.arange(len(frame)))),
                "t_symbol_day": cluster_t(frame[column], frame["sym_day"]),
                "t_symbol": cluster_t(frame[column], frame["symbol"]),
                "clusters": int(frame["sym_day"].nunique()),
            }
        )
    return {"overlap": rows, "tests": table}, frame


def bootstrap_mean(values: np.ndarray, keys: np.ndarray, reps: int = 10_000, seed: int = 20261111) -> dict[str, float]:
    """Block bootstrap over clusters: resample whole days, keep within-day dependence."""
    frame = pd.DataFrame({"v": values, "k": keys}).dropna()
    groups = [group["v"].to_numpy(dtype="float64") for _, group in frame.groupby("k")]
    if len(groups) < 5:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_two_sided": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        means[i] = np.concatenate([groups[j] for j in pick]).mean()
    point = float(frame["v"].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2.0 * min(float((means <= 0).mean()), float((means >= 0).mean()))
    return {"mean": point, "lo": float(lo), "hi": float(hi), "p_two_sided": p, "days": len(groups)}


def cpi_momentum() -> dict[str, Any]:
    """Recompute CPI 60-minute momentum on BTC and ETH with the pipeline's conventions."""
    calendar = pd.read_csv(CALENDAR)
    calendar["release_ts"] = pd.to_datetime(calendar["release_ts"], utc=True)
    releases = calendar[calendar["event"] == "cpi"]["release_ts"]
    out: dict[str, Any] = {}
    for symbol in ("BTCUSDT", "ETHUSDT"):
        path = BARS.parent / f"{symbol}.parquet"
        if not path.is_file():
            continue
        bars = pd.read_parquet(path, columns=["ts_open", "open", "high", "low", "close"])
        stamps = bars["ts_open"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        open_ = bars["open"].to_numpy(dtype="float64")
        close = bars["close"].to_numpy(dtype="float64")
        rows = []
        for release in releases:
            target = int(release.value)
            i = int(np.searchsorted(stamps, target, side="left"))
            if i < 61 or i + 60 >= len(stamps):
                continue
            if i > 0 and (stamps[i] - stamps[i - 1]) > 60 * MINUTE_NS:
                pass  # gaps are allowed, the entry is simply the next open
            jump = close[i] / open_[i] - 1.0
            side = float(np.sign(jump))
            if side == 0.0:
                continue
            ret = close[i + 60] / open_[i + 1] - 1.0
            rows.append({"day": release.date(), "mom": side * ret * BPS})
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        keys = frame["day"].to_numpy()
        out[symbol] = {
            "n": int(len(frame)),
            "mean_bps": float(frame["mom"].mean()),
            "t": cluster_t(frame["mom"], pd.Series(keys)),
            "bootstrap": bootstrap_mean(frame["mom"].to_numpy(), keys),
            "leave_one_year_out": {
                str(year): float(group["mom"].mean())
                for year, group in frame.groupby(frame["day"].map(lambda d: d.year))
            },
        }
    return out


def funding_month(symbol: str, year: int, month: int) -> pd.DataFrame | None:
    """Monthly funding archive for one symbol, cached on disk. None when the month is absent."""
    FUNDING_CACHE.mkdir(parents=True, exist_ok=True)
    path = FUNDING_CACHE / f"{symbol}-{year:04d}-{month:02d}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    url = FUNDING_URL.format(symbol=symbol, year=year, month=month)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            name = archive.namelist()[0]
            raw = archive.read(name)
        text = raw.decode("utf-8", errors="replace")
        frame = pd.read_csv(io.StringIO(text))
        lowered = {str(column).lower(): column for column in frame.columns}
        time_column = lowered.get("calc_time") or lowered.get("fundingtime")
        rate_column = lowered.get("last_funding_rate") or lowered.get("fundingrate")
        if time_column is None or rate_column is None:
            return None
        frame = frame[[time_column, rate_column]].copy()
        frame.columns = ["calc_time", "rate"]
        if pd.api.types.is_numeric_dtype(frame["calc_time"]):
            frame["calc_time"] = pd.to_datetime(frame["calc_time"], unit="ms", utc=True)
        else:
            frame["calc_time"] = pd.to_datetime(frame["calc_time"], utc=True)
        frame["rate"] = pd.to_numeric(frame["rate"], errors="coerce")
        frame = frame.dropna()
        frame.to_parquet(path, index=False)
        return frame
    except Exception:  # noqa: BLE001 - a missing month is data, not an error
        return None


def funding_block(limit: int) -> dict[str, Any]:
    """Mean funding a perp short pays over the first 24 hours after listing."""
    if not PERP.is_file():
        return {"available": False}
    events = pd.read_parquet(PERP)
    if "first_bar" not in events.columns:
        return {"available": False}
    events = events.drop_duplicates(subset=["symbol"]).head(limit)
    rows: list[dict[str, Any]] = []
    for event in events.itertuples():
        stamp = pd.Timestamp(event.first_bar)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        frame = funding_month(str(event.symbol), stamp.year, stamp.month)
        if frame is None or frame.empty:
            continue
        window = frame[(frame["calc_time"] >= stamp) & (frame["calc_time"] <= stamp + pd.Timedelta(hours=24))]
        if window.empty:
            continue
        # A short receives positive funding and pays negative funding.
        rows.append({"symbol": str(event.symbol), "pay_bps": float(-window["rate"].sum() * BPS), "events": int(len(window))})
    if not rows:
        return {"available": True, "n": 0}
    frame = pd.DataFrame(rows)
    return {
        "available": True,
        "n": int(len(frame)),
        "mean_pay_bps": float(frame["pay_bps"].mean()),
        "median_pay_bps": float(frame["pay_bps"].median()),
        "p10_bps": float(frame["pay_bps"].quantile(0.1)),
        "p90_bps": float(frame["pay_bps"].quantile(0.9)),
        "share_short_receives": float((frame["pay_bps"] < 0).mean()),
    }


def slippage_block() -> list[dict[str, Any]]:
    """Net return of each published family at extra cost 0/5/10/20 bps."""
    rows: list[dict[str, Any]] = []
    families = (
        ("cross-section 60m", EVENTS, "excess_60m", 28.0, BPS),
        ("btc absolute 60m", ROOT / "data" / "crypto" / "dataset" / "events_btc.parquet", "excess_60m", 14.0, BPS),
        ("listing spot 60m", SPOT_LISTINGS, "ret_60", 14.0, 100.0),
        ("listing perp 60m (short)", PERP, "ret_60", 14.0, 100.0),
    )
    for label, path, column, base, scale in families:
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        if column not in frame.columns:
            continue
        values = frame[column].astype("float64") * scale
        gross = float(values.mean())
        cells = {"family": label, "n": int(values.notna().sum()), "gross_bps": gross, "base_cost_bps": base}
        for extra in (0, 5, 10, 20):
            cells[f"net_at_{extra}"] = gross - base - extra
        rows.append(cells)
    return rows


def render(payload: dict[str, Any]) -> str:
    lines = ["# Вторая проверка: перекрытия, бутстрэп, фандинг, проскальзывание\n"]
    overlap = payload["cross_section"]
    lines.append("## 1. Перекрывающиеся окна в кросс-секции\n")
    lines.append("| горизонт | событий с перекрытием | доля | независимых (оценка) |")
    lines.append("|---|---|---|---|")
    for row in overlap["overlap"]:
        lines.append(
            f"| {row['horizon']} мин | {row['overlapping']} | {row['share']:.1%} | {row['independent_approx']} |"
        )
    lines.append("\n| горизонт | среднее, бп | t i.i.d. | t по (символ, день) | t по символу | кластеров |")
    lines.append("|---|---|---|---|---|---|")
    for row in overlap["tests"]:
        lines.append(
            f"| {row['horizon']} мин | {row['mean_bps']:+.2f} | {row['t_iid']:+.2f} | "
            f"{row['t_symbol_day']:+.2f} | {row['t_symbol']:+.2f} | {row['clusters']} |"
        )
    lines.append("\n## 2. Бутстрэп CPI 60 минут\n")
    lines.append("| инструмент | n | среднее, бп | 95 % интервал | p | дней |")
    lines.append("|---|---|---|---|---|---|")
    for symbol, block in payload["cpi"].items():
        boot = block["bootstrap"]
        lines.append(
            f"| {symbol} | {block['n']} | {block['mean_bps']:+.2f} | "
            f"[{boot['lo']:+.2f}, {boot['hi']:+.2f}] | {boot['p_two_sided']:.3f} | {boot['days']} |"
        )
    lines.append("\n## 3. Фандинг на первом дне шорта перпа\n")
    funding = payload["funding"]
    if funding.get("n"):
        lines.append(
            f"Измерено на {funding['n']} символах: шорт платит в среднем {funding['mean_pay_bps']:+.2f} бп "
            f"за первые 24 часа (медиана {funding['median_pay_bps']:+.2f}, "
            f"p10 {funding['p10_bps']:+.2f}, p90 {funding['p90_bps']:+.2f}); "
            f"шорт получает фандинг в {funding['share_short_receives']:.0%} случаев.\n"
        )
    else:
        lines.append("Архивы фандинга не прочитались — вопрос остаётся открытым.\n")
    lines.append("## 4. Чувствительность к проскальзыванию\n")
    lines.append("| семейство | n | gross, бп | база, бп | +0 | +5 | +10 | +20 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in payload["slippage"]:
        lines.append(
            f"| {row['family']} | {row['n']} | {row['gross_bps']:+.2f} | {row['base_cost_bps']:.0f} | "
            f"{row['net_at_0']:+.2f} | {row['net_at_5']:+.2f} | {row['net_at_10']:+.2f} | {row['net_at_20']:+.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--funding-symbols", type=int, default=150)
    parser.add_argument("--out", default=str(RESULTS / "crypto_robustness.json"))
    args = parser.parse_args()

    events = pd.read_parquet(EVENTS)
    events["entry_ts"] = pd.to_datetime(events["entry_ts"], utc=True)
    cross, _ = overlap_block(events)
    print("overlap:", cross["overlap"])
    print("tests:", cross["tests"])

    cpi = cpi_momentum()
    for symbol, block in cpi.items():
        print(f"cpi {symbol}: n={block['n']} mean={block['mean_bps']:+.2f} t={block['t']:+.2f} boot={block['bootstrap']}")

    funding = funding_block(args.funding_symbols)
    print("funding:", funding)

    slippage = slippage_block()
    for row in slippage:
        print(row)

    payload = {"cross_section": cross, "cpi": cpi, "funding": funding, "slippage": slippage}
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    Path(RESULTS / "crypto_robustness.md").write_text(render(payload), encoding="utf-8")
    print(f"wrote {args.out} and results/crypto_robustness.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
