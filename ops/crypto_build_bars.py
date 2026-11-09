"""T1 — build 1-minute crypto bars from the Binance Vision archive (resumable).

Usage:
    python ops/crypto_build_bars.py --symbols BTCUSDT,ETHUSDT --start 2024-01 --end 2024-02
    python ops/crypto_build_bars.py --symbols BTCUSDT --start 2024-01 --end 2024-01 --refresh

The manifest is printed as a table; the per-symbol quality report is written to
``reports/crypto_bars_quality.json`` (gitignored) and copied to
``results/crypto_bars_quality.json`` (where quoted numbers live). Exit code 0 when every
requested symbol produced rows, 2 when any symbol is empty — a month the archive does not
have is reported with its exact URL in the log, never filled with invented bars.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spp.crypto.bars import (
    DEFAULT_OUT_DIR,
    build_all,
    empty_bars,
    load_frozen_config,
    quality_report,
    resolve_window,
)
from spp.data.store import read_parquet
from spp.logging import get_logger

ROOT = Path(__file__).resolve().parents[1]
LOG = get_logger("ops.crypto_bars")

DEFAULT_REPORT = ROOT / "reports" / "crypto_bars_quality.json"
RESULTS_REPORT = ROOT / "results" / "crypto_bars_quality.json"


def _parse_symbols(raw: str | None) -> list[str] | None:
    """``"BTCUSDT,ethusdt"`` -> ``["BTCUSDT", "ETHUSDT"]``; ``None`` means "use the config"."""
    if raw is None:
        return None
    symbols = [part.strip().upper() for part in raw.split(",") if part.strip()]
    if not symbols:
        raise SystemExit("--symbols was given but contains no symbols (expected BTCUSDT,ETHUSDT)")
    return symbols


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build 1m crypto bars from data.binance.vision")
    parser.add_argument(
        "--symbols", default=None, help="Comma-separated symbols (default: config symbols)"
    )
    parser.add_argument(
        "--start", default=None, help="First month YYYY-MM (default: config period.start)"
    )
    parser.add_argument(
        "--end", default=None, help="Last month YYYY-MM (default: config period.end)"
    )
    parser.add_argument(
        "--out", default=None, help=f"Parquet directory (default: {DEFAULT_OUT_DIR})"
    )
    parser.add_argument("--refresh", action="store_true", help="Re-download cached monthly zips")
    parser.add_argument(
        "--report", default=None, help=f"Quality JSON path (default: {DEFAULT_REPORT})"
    )
    args = parser.parse_args(argv)

    symbols = _parse_symbols(args.symbols)
    try:
        config = load_frozen_config()
    except FileNotFoundError:
        config = {}
        LOG.warning("frozen config not found; relying on explicit --symbols/--start/--end")
    if symbols is None:
        symbols = [str(item) for item in config.get("symbols") or []]
    if not symbols:
        parser.error("no symbols: pass --symbols or declare them in configs/crypto_dataset.yaml")
    start, end = resolve_window(config, args.start, args.end)

    out_dir = Path(args.out) if args.out else DEFAULT_OUT_DIR
    report_path = Path(args.report) if args.report else DEFAULT_REPORT

    manifest = build_all(
        config=config,
        out_dir=out_dir,
        symbols=symbols,
        start=start,
        end=end,
        refresh=args.refresh,
    )
    print(manifest.to_string(index=False))

    quality: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for row in manifest.to_dict(orient="records"):
        symbol = str(row["symbol"])
        parquet = out_dir / f"{symbol}.parquet"
        if int(row["rows"]) > 0 and parquet.is_file():
            quality[symbol] = quality_report(read_parquet(parquet), symbol)
            continue
        quality[symbol] = quality_report(empty_bars(), symbol)
        if int(row["bytes"]) == 0:
            errors[symbol] = "download failed (missing archive month or bad HTTP status); see log"
        else:
            errors[symbol] = "archive file contained no data rows"

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "window": {"start": start, "end": end},
        "refresh": bool(args.refresh),
        "symbols": quality,
        "errors": errors,
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(payload, encoding="utf-8")
    RESULTS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_REPORT.write_text(payload, encoding="utf-8")
    print(f"quality report: {report_path}")
    print(f"quality report (copy): {RESULTS_REPORT}")
    if errors:
        print("problems:")
        for symbol, message in errors.items():
            print(f"  {symbol}: {message}")

    if len(manifest) == 0:
        print("FAIL: no symbols were built")
        return 2
    empty_symbols = [str(item) for item in manifest.loc[manifest["rows"] == 0, "symbol"]]
    if empty_symbols:
        print(f"FAIL: {len(empty_symbols)} symbol(s) produced no rows: {', '.join(empty_symbols)}")
        return 2
    print(f"OK: {len(manifest)} symbol(s) produced rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
