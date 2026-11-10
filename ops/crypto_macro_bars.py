# ruff: noqa: E501  - prints a progress line per symbol
"""Long-history 1-minute bars for the macro instruments.

The crypto branch was built on 2023-01 onward, which leaves only ~46 CPI, ~46 payrolls
and ~31 FOMC meetings - a thin sample for a study whose whole point is timing that is
known in advance. BTCUSDT and ETHUSDT have klines back to 2017-08, and the macro
calendar reaches back just as far, so the honest sample is ten years, not four.

This script only *extends* the archive: it writes ``data/crypto/bars/1m_macro/{SYMBOL}.parquet``
and never touches ``data/crypto/bars/1m/``, so the published cross-sectional results keep
the exact bars they were computed on.

Usage:
    python ops/crypto_macro_bars.py [--symbols BTCUSDT ETHUSDT] [--start 2017-08] [--end 2026-10]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from spp.crypto.bars import DEFAULT_CACHE_DIR, download_month, read_month

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "crypto" / "bars" / "1m_macro"


def months(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Every ``(year, month)`` from ``start`` to ``end`` inclusive."""
    y, m = start
    out: list[tuple[int, int]] = []
    while (y, m) <= end:
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def parse_month(text: str) -> tuple[int, int]:
    year, month = text.split("-")
    return int(year), int(month)


def build(symbol: str, start: tuple[int, int], end: tuple[int, int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for year, month in months(start, end):
        try:
            path = download_month(symbol, year, month, DEFAULT_CACHE_DIR)
        except FileNotFoundError:
            missing.append(f"{year:04d}-{month:02d}")
            continue
        frame = read_month(path, symbol)
        if len(frame):
            frames.append(frame)
    if missing:
        print(f"{symbol}: archive has no {missing[0]}..{missing[-1]} ({len(missing)} months)")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["ts_open"]).sort_values("ts_open")
    return combined.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--start", default="2017-08")
    parser.add_argument("--end", default="2026-10")
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()

    start = parse_month(args.start)
    end = parse_month(args.end)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol in args.symbols:
        frame = build(symbol, start, end)
        target = out_dir / f"{symbol}.parquet"
        frame.to_parquet(target, index=False)
        print(
            f"{symbol}: rows={len(frame):,} {frame['ts_open'].min()} .. {frame['ts_open'].max()}"
            f" -> {target}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
