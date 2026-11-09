# ruff: noqa: E501  - these scripts print reports
"""Measure which free crypto data endpoints are reachable from here, and how fresh the news feeds are.

The equity branch died on data supply: hourly bars start 2024-11-11, the news corpus ends 2023-12-16,
and GDELT answers 429 on a fifth of requests. Before designing anything for crypto, this script
answers the only question that matters: is the data actually obtainable, at what granularity, and
how old is the freshest item in each news feed?

Usage:
    python ops/crypto_probe.py [--out data/crypto_probe.json]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import requests

TIMEOUT = 20
HEADERS = {"User-Agent": "Mozilla/5.0 (research; stock-price-prediction)"}

PRICE_ENDPOINTS = [
    ("binance_spot_time", "https://api.binance.com/api/v3/time"),
    (
        "binance_klines_1m",
        "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=5",
    ),
    (
        "binance_vision_zip",
        "https://data.binance.vision/data/spot/monthly/klines/"
        "BTCUSDT/1m/BTCUSDT-1m-2024-01.zip",
    ),
    (
        "bybit_kline_1m",
        "https://api.bybit.com/v5/market/kline?category=spot&symbol=BTCUSDT&interval=1&limit=5",
    ),
    (
        "okx_candles_1m",
        "https://www.okx.com/api/v5/market/history-candles?instId=BTC-USDT&bar=1m&limit=5",
    ),
    (
        "kraken_ohlc_1m",
        "https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1",
    ),
    (
        "coinbase_candles_1m",
        "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=60",
    ),
    (
        "binance_funding",
        "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=5",
    ),
]

NEWS_FEEDS = [
    ("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("cointelegraph", "https://cointelegraph.com/rss"),
    ("theblock", "https://www.theblock.co/rss.xml"),
    ("decrypt", "https://decrypt.co/feed"),
    ("cryptoslate", "https://cryptoslate.com/feed/"),
    ("bitcoinmagazine", "https://bitcoinmagazine.com/feed"),
    ("newsbtc", "https://newsbtc.com/feed/"),
    ("bitcoinist", "https://bitcoinist.com/feed/"),
    ("ambcrypto", "https://ambcrypto.com/feed/"),
    ("utoday", "https://u.today/rss"),
    ("beincrypto", "https://beincrypto.com/feed/"),
    ("cryptopotato", "https://cryptopotato.com/feed/"),
    ("thedefiant", "https://thedefiant.io/api/feed"),
    ("blockworks", "https://blockworks.co/feed"),
    ("coingape", "https://coingape.com/feed/"),
    ("protos", "https://protos.com/feed/"),
    ("cryptopanic", "https://cryptopanic.com/api/v1/posts/?public=true"),
    ("cryptopanic_rss", "https://cryptopanic.com/news/rss/"),
]


def _snippet(response: requests.Response, limit: int = 120) -> str:
    text = " ".join(response.text.split())
    return text[:limit]


def probe(url: str, method: str = "GET") -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {"url": url, "method": method}
    try:
        response = requests.request(method, url, timeout=TIMEOUT, headers=HEADERS)
        row["status"] = response.status_code
        row["seconds"] = round(time.perf_counter() - started, 2)
        row["bytes"] = len(response.content)
        row["content_type"] = response.headers.get("content-type", "")
        row["snippet"] = _snippet(response)
        row["_text"] = response.text
    except Exception as exc:  # noqa: BLE001 - a dead endpoint is the measurement
        row["status"] = None
        row["seconds"] = round(time.perf_counter() - started, 2)
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        row["_text"] = ""
    return row


def _parse_rss(text: str) -> list[dict[str, str]]:
    """Minimal RSS/Atom reader: no dependency, no namespace fuss."""
    import xml.etree.ElementTree as ET

    items: list[dict[str, str]] = []
    try:
        root = ET.fromstring(text.strip())
    except ET.ParseError:
        return items
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag not in {"item", "entry"}:
            continue
        row: dict[str, str] = {}
        for child in node:
            name = child.tag.rsplit("}", 1)[-1]
            if name in {"title", "pubDate", "published", "updated", "link"}:
                row[name] = (child.text or child.attrib.get("href") or "").strip()
        items.append(row)
    return items


def _rss_age(items: list[dict[str, str]]) -> dict[str, Any]:
    stamps: list[datetime] = []
    for item in items:
        raw = item.get("pubDate") or item.get("published") or item.get("updated") or ""
        raw = raw.strip()
        if not raw:
            continue
        for parser in (
            parsedate_to_datetime,
            lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")),
        ):
            try:
                stamp = parser(raw)
            except (TypeError, ValueError):
                continue
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=UTC)
            stamps.append(stamp.astimezone(UTC))
            break
    if not stamps:
        return {"n_with_time": 0, "n_items": len(items)}
    stamps.sort()
    now = datetime.now(UTC)
    return {
        "n_items": len(items),
        "n_with_time": len(stamps),
        "newest_utc": stamps[-1].isoformat(),
        "age_minutes": round((now - stamps[-1]).total_seconds() / 60, 1),
        "oldest_utc": stamps[0].isoformat(),
        "median_gap_minutes": round(
            sorted((b - a).total_seconds() / 60 for a, b in zip(stamps, stamps[1:], strict=False))[
                len(stamps) // 2
            ],
            1,
        )
        if len(stamps) > 1
        else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/crypto_probe.json")
    args = parser.parse_args()

    report: dict[str, Any] = {"generated_utc": datetime.now(UTC).isoformat()}

    print("=" * 78)
    print("PRICE ENDPOINTS")
    print("=" * 78)
    report["price"] = []
    for name, url in PRICE_ENDPOINTS:
        row = probe(url, method="HEAD" if name.endswith("zip") else "GET")
        row.pop("_text", None)
        row["name"] = name
        report["price"].append(row)
        status = row.get("status")
        print(f"{name:22s} {str(status):5s} {row['seconds']:6.2f}s {row.get('bytes', 0):>9} B  {row.get('snippet', row.get('error', ''))[:70]}")

    print()
    print("=" * 78)
    print("NEWS FEEDS (freshness matters more than volume: we want to be early)")
    print("=" * 78)
    report["news"] = []
    for name, url in NEWS_FEEDS:
        row = probe(url)
        items = _parse_rss(row.get("_text", ""))
        age = _rss_age(items) if items else {}
        entry = {"name": name, **{k: v for k, v in row.items() if k != "_text"}, **{f"rss_{k}": v for k, v in age.items()}}
        report["news"].append(entry)
        head = items[0].get("title", "")[:52] if items else ""
        print(
            f"{name:16s} {str(row.get('status')):5s} {row['seconds']:6.2f}s  "
            f"items={age.get('n_items', 0):3d} age={str(age.get('age_minutes', 'n/a')):>6s} min  {head}"
        )

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
