# ruff: noqa: E501  - prints report tables
"""Can the listing amplitude be captured? Only by a short, and only on the perp.

The spot study shows the amplitude exists but is unreachable: a new Binance spot pair
opens with an auction that is already 80 % above its first print (median), and everything
after the auction drifts *down* - 11 % over the first hour. Going long loses; going short
spot is impossible, because a token that just opened has no borrow.

The perpetual contract is where the short becomes financeable. Many new listings get a UM
perp on the same day, and a perp can be shorted with margin. So the event here is the first
minute of the **perpetual**, and the trade is a short from the second bar:

* entry at the open of the second 1m bar (the first bar of a perp is also an opening
  auction, so its open is a nominal print);
* exit at the close ``h`` minutes later, horizons 1 / 5 / 15 / 60 / 240 / 1440;
* cost 14 bps per round trip, one leg;
* funding is **not** modelled - it is small over hours and it is not what decides this.

The number that decides is the left tail. A short of a token that opens up 40 % is not a
trade until the margin survives the move against it, so ``max_60m`` (the highest price in
the first hour relative to the entry) is reported next to the return, together with the
share of events where a 5x or a 3x short would have been liquidated.

Usage:
    python ops/crypto_listing_perp.py [--workers 8] [--max-symbols 2000]
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
import requests

from spp.crypto.bars import read_month

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "crypto" / "bars" / "listings_perp"
RESULTS = ROOT / "results"

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
PERP_ROOT = "data/futures/um/monthly/klines/"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; spp-research/1.0)"}

HORIZONS: tuple[int, ...] = (1, 5, 15, 60, 240, 1440)
COST_BPS = 14.0
MINUTE_NS = 60 * 10**9
LIQUIDATION_5X = 0.20
LIQUIDATION_3X = 0.33


def _s3(params: dict[str, str]) -> ET.Element:
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


def list_perp_symbols(limit: int | None = None) -> list[str]:
    symbols: list[str] = []
    marker = ""
    while True:
        params = {"delimiter": "/", "prefix": PERP_ROOT}
        if marker:
            params["marker"] = marker
        root = _s3(params)
        for node in root.findall("s3:CommonPrefixes/s3:Prefix", NS):
            name = (node.text or "").rstrip("/").rsplit("/", 1)[-1]
            if name:
                symbols.append(name)
        truncated = (root.findtext("s3:IsTruncated", "", NS) or "").strip().lower() == "true"
        if not truncated:
            break
        marker = root.findtext("s3:NextMarker", "", NS) or symbols[-1]
    symbols = sorted(set(symbols))
    print(f"perp symbols in the archive: {len(symbols)}")
    return symbols[:limit] if limit else symbols


def first_month(symbol: str) -> tuple[int, int] | None:
    root = _s3({"prefix": f"{PERP_ROOT}{symbol}/1m/"})
    keys = [node.text or "" for node in root.findall("s3:Contents/s3:Key", NS)]
    months = [key for key in keys if key.endswith(".zip")]
    if not months:
        return None
    match = re.search(r"-(\d{4})-(\d{2})\.zip$", min(months))
    return (int(match.group(1)), int(match.group(2))) if match else None


def download_perp_month(symbol: str, year: int, month: int) -> Path:
    """Cash the perp month; 404 means the perp did not trade yet."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{symbol}-1m-{year:04d}-{month:02d}.zip"
    if path.is_file() and path.stat().st_size > 0:
        return path
    url = f"https://data.binance.vision/{PERP_ROOT}{symbol}/1m/{symbol}-1m-{year:04d}-{month:02d}.zip"
    response = requests.get(url, headers=HEADERS, timeout=60)
    if response.status_code == 404:
        raise FileNotFoundError(f"HTTP 404: {url}")
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def measure(symbol: str, bars: pd.DataFrame) -> dict[str, Any]:
    """Short returns from the first executable perp price."""
    bars = bars.sort_values("ts_open").drop_duplicates(subset=["ts_open"]).reset_index(drop=True)
    stamps = bars["ts_open"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    open_ = bars["open"].to_numpy(dtype="float64")
    high = bars["high"].to_numpy(dtype="float64")
    close = bars["close"].to_numpy(dtype="float64")
    entry_i = 1 if len(bars) > 1 else 0
    entry_ns = int(stamps[entry_i])
    row: dict[str, Any] = {
        "symbol": symbol,
        "first_bar": pd.Timestamp(bars["ts_open"].iloc[0]),
        "entry_bar": pd.Timestamp(bars["ts_open"].iloc[entry_i]),
        "n_bars": len(bars),
        "entry_price": float(open_[entry_i]),
        "auction_ret": float(close[0] / open_[0] - 1.0),
    }
    for horizon in HORIZONS:
        target = entry_ns + (horizon - 1) * MINUTE_NS
        idx = int(np.searchsorted(stamps, target, side="left"))
        ok = idx < len(stamps) and int(stamps[idx]) == target
        # short: profit when the price falls
        row[f"ret_{horizon}"] = float(1.0 - close[idx] / open_[entry_i]) if ok else float("nan")
    window = min(60, len(high) - entry_i)
    row["max_60m"] = float(high[entry_i:entry_i + window].max() / open_[entry_i] - 1.0) if window > 0 else float("nan")
    return row


def process(symbol: str) -> dict[str, Any] | None:
    month = first_month(symbol)
    if month is None:
        return {"symbol": symbol, "status": "no_archive"}
    year, mon = month
    try:
        path = download_perp_month(symbol, year, mon)
    except FileNotFoundError:
        return {"symbol": symbol, "status": "no_archive"}
    except Exception as exc:  # noqa: BLE001 - one bad symbol must not kill the run
        return {"symbol": symbol, "status": f"error:{type(exc).__name__}"}
    bars = read_month(path, symbol)
    if len(bars) < 3:
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


def summarise(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        column = frame[f"ret_{horizon}"]
        valid = column.dropna()
        if not len(valid):
            continue
        groups = frame.loc[valid.index, "first_bar"].dt.date
        rows.append(
            {
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


def render(rows: list[dict[str, Any]], funnel: dict[str, Any], frame: pd.DataFrame) -> tuple[str, dict[str, float]]:
    table = (
        "| горизонт | n | среднее, % | медиана, % | p10, % | p90, % | доля > 0 | t | net (14 бп), % |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    for row in rows:
        table += (
            f"| {row['horizon']} мин | {row['n']} | {row['mean_pct']:+.2f} | {row['median_pct']:+.2f} | "
            f"{row['p10_pct']:+.2f} | {row['p90_pct']:+.2f} | {row['hit']:.3f} | "
            f"{row['t']:+.2f} | {row['net_pct']:+.2f} |\n"
        )
    worst = frame["max_60m"].dropna()
    liquidated_5x = float((worst >= LIQUIDATION_5X).mean()) if len(worst) else float("nan")
    liquidated_3x = float((worst >= LIQUIDATION_3X).mean()) if len(worst) else float("nan")
    return (
        "# Шорт нового перпетуала: можно ли забрать амплитуду\n\n"
        "Событие — первая минута торгов **бессрочного фьючерса** Binance (UM). Вход: открытие "
        "второго бара (первый бар — аукцион, его `open` неисполним). Выход: закрытие бара через "
        f"h минут. Издержки {COST_BPS:.0f} бп на круг, одна нога. Фандинг не моделируется — "
        "на часах он мал и не решает.\n\n"
        "## Воронка\n\n"
        f"- символов с перпетуалом в архиве: {funnel['symbols']}\n"
        f"- измерено: {funnel['measured']}\n"
        f"- без архива: {funnel['no_archive']}\n"
        f"- ошибок: {funnel['errors']}\n\n"
        f"## Шорт от второй минуты\n\n{table}\n"
        "## Риск шорта (движение против позиции в первый час)\n\n"
        f"- медиана максимума: {worst.median() * 100.0:+.2f} %\n"
        f"- p90 максимума: {worst.quantile(0.90) * 100.0:+.2f} %\n"
        f"- p99 максимума: {worst.quantile(0.99) * 100.0:+.2f} %\n"
        f"- доля событий, где шорт с плечом 5x был бы ликвидирован (рост ≥ {LIQUIDATION_5X * 100:.0f} %): {liquidated_5x:.3f}\n"
        f"- доля событий, где шорт с плечом 3x был бы ликвидирован (рост ≥ {LIQUIDATION_3X * 100:.0f} %): {liquidated_3x:.3f}\n\n"
        "Средний доход шорта без этой колонки был бы неполной картиной: положительное среднее "
        "при 20-процентном хвосте против позиции — это не стратегия, а лотерея с маржин-коллом.\n"
    ), {
        "liquidation_5x": liquidated_5x,
        "liquidation_3x": liquidated_3x,
        "max_60m_median_pct": float(worst.median() * 100.0) if len(worst) else float("nan"),
        "max_60m_p90_pct": float(worst.quantile(0.90) * 100.0) if len(worst) else float("nan"),
        "max_60m_p99_pct": float(worst.quantile(0.99) * 100.0) if len(worst) else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--out", default=str(RESULTS / "crypto_listing_perp.json"))
    args = parser.parse_args()

    symbols = list_perp_symbols(args.max_symbols or None)
    symbols = [s for s in symbols if s.endswith("USDT")]
    print(f"usdt perps under test: {len(symbols)}")

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
        raise SystemExit("no perps measured")
    frame["first_bar"] = pd.to_datetime(frame["first_bar"], utc=True)
    frame = frame.sort_values("first_bar").reset_index(drop=True)

    funnel = {
        "symbols": len(symbols),
        "measured": int(len(frame)),
        "no_archive": statuses.get("no_archive", 0),
        "errors": sum(v for k, v in statuses.items() if k.startswith("error")),
    }
    stats = summarise(frame)
    for row in stats:
        print(
            f"{row['horizon']:5d}m n={row['n']:4d} mean {row['mean_pct']:+7.2f}% "
            f"median {row['median_pct']:+6.2f}% hit {row['hit']:.3f} t {row['t']:+.2f} "
            f"net {row['net_pct']:+7.2f}%"
        )

    report, risk = render(stats, funnel, frame)
    payload = {
        "funnel": funnel,
        "stats": stats,
        "risk": risk,
        "statuses": statuses,
        "cost_bps": COST_BPS,
        "first_bar_min": str(frame["first_bar"].min()),
        "first_bar_max": str(frame["first_bar"].max()),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    (ROOT / "data" / "crypto" / "listings").mkdir(parents=True, exist_ok=True)
    frame.to_parquet(ROOT / "data" / "crypto" / "listings" / "events_perp.parquet", index=False)
    (RESULTS / "crypto_listing_perp.md").write_text(report, encoding="utf-8")
    print(f"wrote {args.out} and results/crypto_listing_perp.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
