# ruff: noqa: E501  - prints report tables
"""Where the amplitude actually is: the first hours of a newly listed token.

The structural classes measured on the twelve majors move 30-60 bps in an hour - the same
as ordinary commentary, because a hack of some DeFi protocol does not reprice SOL by a
percent. The event with a genuinely different amplitude is a **new listing**: the token
opens for trading on a large venue and the first hour is not a 40 bps move but a 4-40 %
one.

The announcement feed turned out to be unusable as a clock (`publishDate` comes back as
``None``), and it is not needed: **the first bar of a symbol in the Binance archive is the
moment trading opened**, which is exactly the entry a participant gets. So the event sample
is built from the archive alone:

1. enumerate every spot symbol in the S3 bucket (``delimiter=/`` on
   ``data/spot/monthly/klines/``);
2. list the months of each symbol (one request per symbol, the same bucket) and take the
   earliest one - that month contains the first bar ever, that is the listing;
3. download that month and measure forward returns from the open of the first bar;
4. repeat with a 15-minute delay - the lag this project measured on its own news recorder
   (p50 15.9 min).

Entry at the open is the best case: it assumes the participant knew the token would open.
The delayed entry is the realistic case. The gap between them is the price of not knowing.

One leg, 14 bps. Long only: there is no borrow for a token that just opened.

Usage:
    python ops/crypto_listings.py [--workers 8] [--max-symbols 2000]
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spp.crypto.bars import download_month, read_month

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "crypto" / "bars" / "listings"
RESULTS = ROOT / "results"

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
KLINE_PREFIX = "data/spot/monthly/klines/"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; spp-research/1.0)"}

HORIZONS: tuple[int, ...] = (1, 5, 15, 60, 240, 1440)
DELAY_MINUTES = 15
COST_BPS = 14.0
MINUTE_NS = 60 * 10**9
LEVERAGED = re.compile(r"(UP|DOWN|BULL|BEAR)USDT$")
STABLE = {"USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT", "DAIUSDT", "USDPUSDT", "EURUSDT", "GBPUSDT", "TRYUSDT", "BRLUSDT", "AUDUSDT", "RUBUSDT", "UAHUSDT", "ZARUSDT", "PLNUSDT", "RONUSDT", "ARSUSDT", "JPYUSDT"}


def _s3(params: dict[str, str]) -> ET.Element:
    """One bucket listing call, with a small retry."""
    query = urllib.parse.urlencode(params)
    last: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(f"{S3}?{query}", headers=HEADERS)
            with urllib.request.urlopen(request, timeout=30) as response:
                return ET.fromstring(response.read())
        except Exception as exc:  # noqa: BLE001 - a transient network shape must retry
            last = exc
            time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"s3 listing failed: {last}")


def list_symbols(limit: int | None = None) -> list[str]:
    """Every spot symbol that has at least one monthly file in the bucket."""
    symbols: list[str] = []
    marker = ""
    while True:
        params = {"delimiter": "/", "prefix": KLINE_PREFIX}
        if marker:
            params["marker"] = marker
        root = _s3(params)
        for node in root.findall("s3:CommonPrefixes/s3:Prefix", NS):
            text = node.text or ""
            name = text.rstrip("/").rsplit("/", 1)[-1]
            if name:
                symbols.append(name)
        truncated = (root.findtext("s3:IsTruncated", "", NS) or "").strip().lower() == "true"
        if not truncated:
            break
        marker = root.findtext("s3:NextMarker", "", NS) or symbols[-1]
        print(f"  symbols: {len(symbols)} (paging)")
    symbols = sorted(set(symbols))
    print(f"symbols in the archive: {len(symbols)}")
    if limit:
        symbols = symbols[:limit]
    return symbols


def first_month(symbol: str) -> tuple[int, int] | None:
    """The earliest month that has a kline file for ``symbol``."""
    # The interval must be pinned: without it the keys of every interval are mixed in and
    # `min()` picks the 1M bucket, whose first month need not match the 1m archive.
    root = _s3({"prefix": f"{KLINE_PREFIX}{symbol}/1m/"})
    keys = [node.text or "" for node in root.findall("s3:Contents/s3:Key", NS)]
    months = [key for key in keys if key.endswith(".zip")]
    if not months:
        return None
    first = min(months)
    match = re.search(r"-(\d{4})-(\d{2})\.zip$", first)
    return (int(match.group(1)), int(match.group(2))) if match else None


def measure(symbol: str, bars: pd.DataFrame) -> dict[str, Any]:
    """Forward returns from the first *executable* price of the symbol.

    The first bar is the opening auction: 1INCH opened with ``open 0.20`` and ``high 2.90``
    in the same minute, so its open is a nominal print nobody could trade. The entry is
    therefore the open of the second bar - one minute after trading started, once price
    discovery has cleared. The auction itself is reported separately as ``auction_ret``
    and is not tradable.
    """
    bars = bars.sort_values("ts_open").drop_duplicates(subset=["ts_open"]).reset_index(drop=True)
    stamps = bars["ts_open"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    open_ = bars["open"].to_numpy(dtype="float64")
    close = bars["close"].to_numpy(dtype="float64")
    high = bars["high"].to_numpy(dtype="float64")
    low = bars["low"].to_numpy(dtype="float64")
    entry_i = 1 if len(bars) > 1 else 0
    entry_ns = int(stamps[entry_i])
    row: dict[str, Any] = {
        "symbol": symbol,
        "first_bar": pd.Timestamp(bars["ts_open"].iloc[0]),
        "entry_bar": pd.Timestamp(bars["ts_open"].iloc[entry_i]),
        "n_bars": len(bars),
        "entry_price": float(open_[entry_i]),
        "auction_ret": float(close[0] / open_[0] - 1.0) if open_[0] > 0 else float("nan"),
    }
    for horizon in HORIZONS:
        target = entry_ns + (horizon - 1) * MINUTE_NS
        idx = int(np.searchsorted(stamps, target, side="left"))
        ok = idx < len(stamps) and int(stamps[idx]) == target
        row[f"ret_{horizon}"] = float(close[idx] / open_[entry_i] - 1.0) if ok else float("nan")
    window = min(60, len(high) - entry_i)
    row["max_60m"] = float(high[entry_i:entry_i + window].max() / open_[entry_i] - 1.0) if window > 0 else float("nan")
    row["min_60m"] = float(low[entry_i:entry_i + window].min() / open_[entry_i] - 1.0) if window > 0 else float("nan")
    late = int(np.searchsorted(stamps, entry_ns + DELAY_MINUTES * MINUTE_NS, side="left"))
    if late < len(stamps):
        for horizon in HORIZONS:
            target = int(stamps[late]) + (horizon - 1) * MINUTE_NS
            idx = int(np.searchsorted(stamps, target, side="left"))
            ok = idx < len(stamps) and int(stamps[idx]) == target
            row[f"late_ret_{horizon}"] = float(close[idx] / open_[late] - 1.0) if ok else float("nan")
    else:
        for horizon in HORIZONS:
            row[f"late_ret_{horizon}"] = float("nan")
    return row


def process(symbol: str) -> dict[str, Any] | None:
    """Listing month -> first bar -> measured row. Runs inside a worker thread."""
    month = first_month(symbol)
    if month is None:
        return {"symbol": symbol, "status": "no_archive"}
    year, mon = month
    try:
        path = download_month(symbol, year, mon, CACHE)
    except FileNotFoundError:
        return {"symbol": symbol, "status": "no_archive"}
    except Exception as exc:  # noqa: BLE001 - one bad symbol must not kill the run
        return {"symbol": symbol, "status": f"error:{type(exc).__name__}"}
    bars = read_month(path, symbol)
    if not len(bars):
        return {"symbol": symbol, "status": "empty"}
    row = measure(symbol, bars)
    row["status"] = "measured"
    row["month"] = f"{year:04d}-{mon:02d}"
    return row


def cluster_t(values: pd.Series, groups: pd.Series) -> float:
    work = pd.DataFrame({"v": values, "g": groups}).dropna()
    if len(work) < 3 or work["g"].nunique() < 2:
        return float("nan")
    grouped = work.groupby("g")["v"]
    k = grouped.ngroups
    mean = float(work["v"].mean())
    resid = grouped.sum() - mean * grouped.size()
    variance = float((resid**2).sum()) / len(work) ** 2 * k / (k - 1)
    return float(mean / np.sqrt(variance)) if variance > 0 else float("nan")


def summarise(frame: pd.DataFrame, prefix: str, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        column = frame[f"{prefix}ret_{horizon}"]
        valid = column.dropna()
        if not len(valid):
            continue
        groups = frame.loc[valid.index, "first_bar"].dt.date
        rows.append(
            {
                "group": label,
                "prefix": prefix,
                "horizon": horizon,
                "n": int(len(valid)),
                "mean_pct": float(valid.mean() * 100.0),
                "median_pct": float(valid.median() * 100.0),
                "p10_pct": float(valid.quantile(0.10) * 100.0),
                "p90_pct": float(valid.quantile(0.90) * 100.0),
                "hit": float((valid > 0).mean()),
                "t": cluster_t(valid, groups),
                "net_pct": float(valid.mean() * 100.0) - COST_BPS / 100.0,
            }
        )
    return rows


def render(rows: list[dict[str, Any]], funnel: dict[str, Any], frame: pd.DataFrame) -> str:
    lines = [
        "# Листинги: где амплитуда\n\n",
        "Событие — **первая минута торгов** нового спот-символа Binance. Точка отсчёта взята "
        "из самого архива: первый бар символа и есть момент открытия торгов, объявление не нужно "
        "(в фиде объявлений Binance `publishDate` приходит `None`).\n\n",
        f"Вход: **открытие второго бара** — через минуту после начала торгов, когда аукцион "
        "открытия уже прошёл. Первый бар не торгуется: у 1INCH он открылся по 0.20 при максимуме "
        "2.90 в ту же минуту, то есть `open` первого бара — номинальный принт. Сам аукцион "
        f"показан отдельно: медиана {frame['auction_ret'].median() * 100.0:+.2f} %, "
        f"среднее {frame['auction_ret'].mean() * 100.0:+.2f} % — это недостижимый потолок.\n\n",
        f"Издержки {COST_BPS:.0f} бп, одна нога, только лонг — занимать токен, который только "
        "что открылся, негде.\n\n",
        f"`late` — вход через {DELAY_MINUTES} мин после открытия: задержка, которую проект "
        "измерил на своём новостном рекордере (p50 15.9 мин). Разница между двумя блоками — "
        "цена того, что вход не мгновенный.\n\n",
        "## Воронка\n\n",
        f"- символов в архиве: {funnel['symbols']}\n",
        f"- проверено: {funnel['checked']}\n",
        f"- найдено первое событие: {funnel['measured']}\n",
        f"- без месячного файла: {funnel['no_archive']}\n",
        f"- ошибок загрузки: {funnel['errors']}\n\n",
    ]
    overall = [row for row in rows if row["group"] == "all"]
    for prefix, title in (("", "Вход на открытии"), ("late_", f"Вход через {DELAY_MINUTES} мин")):
        block = [row for row in overall if row["prefix"] == prefix]
        if not block:
            continue
        table = (
            "| горизонт | n | среднее, % | медиана, % | p10, % | p90, % | доля > 0 | t | net (14 бп), % |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
        )
        for row in block:
            table += (
                f"| {row['horizon']} мин | {row['n']} | {row['mean_pct']:+.2f} | {row['median_pct']:+.2f} | "
                f"{row['p10_pct']:+.2f} | {row['p90_pct']:+.2f} | {row['hit']:.3f} | "
                f"{row['t']:+.2f} | {row['net_pct']:+.2f} |\n"
            )
        lines.append(f"## {title}\n\n{table}\n")

    years = sorted({row["group"] for row in rows if row["group"].startswith("y")})
    if years:
        lines.append("## По годам (вход на открытии)\n\n")
        table = "| год | n | 60 мин, % | медиана 60, % | t | net 60 (14 бп), % |\n|---|---|---|---|---|---|\n"
        for year in years:
            row = next(
                (r for r in rows if r["group"] == year and r["prefix"] == "" and r["horizon"] == 60), None
            )
            if row is None:
                continue
            table += (
                f"| {year[1:]} | {row['n']} | {row['mean_pct']:+.2f} | {row['median_pct']:+.2f} | "
                f"{row['t']:+.2f} | {row['net_pct']:+.2f} |\n"
            )
        lines.append(table + "\n")

    lines.append(
        "`max_60m` / `min_60m` — максимум и минимум первого часа относительно входа: "
        f"медиана максимума {frame['max_60m'].median() * 100.0:+.2f} %, "
        f"медиана минимума {frame['min_60m'].median() * 100.0:+.2f} %. "
        "Это потолок идеального таймера; разница с средним — цена того, что вход не идеален.\n"
    )
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--out", default=str(RESULTS / "crypto_listings.json"))
    args = parser.parse_args()

    symbols = list_symbols(args.max_symbols or None)
    symbols = [s for s in symbols if s.endswith("USDT") and s not in STABLE and not LEVERAGED.search(s)]
    print(f"usdt spot symbols under test: {len(symbols)}")

    rows: list[dict[str, Any]] = []
    statuses: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, symbol): symbol for symbol in symbols}
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result is None:
                statuses["none"] = statuses.get("none", 0) + 1
            elif result.get("status") == "measured":
                rows.append(result)
            else:
                statuses[result["status"]] = statuses.get(result["status"], 0) + 1
            if done % 100 == 0:
                print(f"  {done}/{len(symbols)} measured={len(rows)} {statuses}")

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("no listings measured")
    frame["first_bar"] = pd.to_datetime(frame["first_bar"], utc=True)
    frame = frame.sort_values("first_bar").reset_index(drop=True)
    frame["year"] = frame["first_bar"].dt.year

    funnel = {
        "symbols": len(symbols),
        "checked": len(symbols),
        "measured": int(len(frame)),
        "no_archive": statuses.get("no_archive", 0),
        "errors": sum(v for k, v in statuses.items() if k.startswith("error")),
    }
    stats = summarise(frame, "", "all") + summarise(frame, "late_", "all")
    for year, block in frame.groupby("year"):
        if len(block) < 20:
            continue
        stats += summarise(block, "", f"y{year}")

    for row in stats:
        if row["group"] != "all":
            continue
        print(
            f"{row['prefix']:5s} {row['horizon']:5d}m n={row['n']:4d} mean {row['mean_pct']:+7.2f}% "
            f"median {row['median_pct']:+6.2f}% hit {row['hit']:.3f} t {row['t']:+.2f} "
            f"net {row['net_pct']:+7.2f}%"
        )

    payload = {
        "funnel": funnel,
        "statuses": statuses,
        "stats": stats,
        "max_60m_median_pct": float(frame["max_60m"].median() * 100.0),
        "max_60m_mean_pct": float(frame["max_60m"].mean() * 100.0),
        "min_60m_median_pct": float(frame["min_60m"].median() * 100.0),
        "cost_bps": COST_BPS,
        "delay_minutes": DELAY_MINUTES,
        "first_bar_min": str(frame["first_bar"].min()),
        "first_bar_max": str(frame["first_bar"].max()),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    (ROOT / "data" / "crypto" / "listings").mkdir(parents=True, exist_ok=True)
    frame.to_parquet(ROOT / "data" / "crypto" / "listings" / "events.parquet", index=False)
    (RESULTS / "crypto_listings.md").write_text(render(stats, funnel, frame), encoding="utf-8")
    print(f"wrote {args.out} and results/crypto_listings.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
