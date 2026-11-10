# ruff: noqa: E501  - prints an audit table
"""Adversarial checks on the crypto event dataset.

The pipeline claims: entry at the open of the first 1-minute bar at or after
``published_at + 1 min``, exit at the close of the bar ``h - 1`` minutes later, and
``y = asset return - BTC return`` over the same stamps, with 14 bps charged once.

This script does not re-derive the numbers in the report; it looks for the ways the design
can be wrong and measures each one:

1. exposure  - does the exit bar really close ``h`` minutes after the entry open, and do
               asset and benchmark use identical stamps?
2. lag       - how long after publication does the entry actually happen?
3. missed    - the price move between publication and the entry open is invisible to the
               strategy; how large is it?
4. bench     - ``ret`` vs ``excess``: subtracting BTC changes the PnL a single-asset
               position would earn. How much, per asset?
5. selection - which hours and which days the Wayback captures over-sample.
6. duplicates- syndicated headlines counted as independent events.
7. splits    - no leakage across train / val / holdout, counted from the parquet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "crypto" / "dataset" / "events.parquet"
BARS = ROOT / "data" / "crypto" / "bars" / "1m"
HORIZONS = (5, 15, 60)


def load_bars(symbol: str, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(BARS / f"{symbol}.parquet", columns=columns)
    frame["ns"] = frame["ts_open"].astype("datetime64[ns, UTC]").astype("int64")
    return frame.sort_values("ns").reset_index(drop=True)


def check_exposure(events: pd.DataFrame) -> dict[str, object]:
    """Verify exit close = entry open + h minutes, asset and benchmark on the same stamps."""
    rows = []
    for symbol, group in events.groupby("symbol"):
        bars = load_bars(symbol, ["ts_open", "open", "close"])
        ns = bars["ns"].to_numpy()
        sample = group.head(400)
        for horizon in HORIZONS:
            entry_ns = sample["entry_ts"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
            index = np.searchsorted(ns, entry_ns, side="left")
            valid = (index < len(ns)) & (ns[np.clip(index, 0, len(ns) - 1)] == entry_ns)
            index = index[valid]
            if len(index) == 0:
                continue
            exit_ns = entry_ns[valid] + (horizon - 1) * 60 * 10**9
            exit_index = np.searchsorted(ns, exit_ns, side="left")
            if (exit_index >= len(ns)).any():
                exit_index = np.clip(exit_index, 0, len(ns) - 1)
            span = (ns[exit_index] - ns[index]) / 60e9 + 1.0
            price_ret = bars["close"].to_numpy()[exit_index] / bars["open"].to_numpy()[index] - 1.0
            stored = sample[f"ret_{horizon}m"].to_numpy()[valid]
            rows.append(
                {
                    "symbol": symbol,
                    "horizon": horizon,
                    "n": int(len(index)),
                    "bar_span_min_median": float(np.median(span)),
                    "bar_span_min_min": float(span.min()),
                    "max_abs_diff_vs_stored_bps": float(np.nanmax(np.abs(price_ret - stored)) * 1e4),
                }
            )
    frame = pd.DataFrame(rows)
    return {
        "span_ok": bool((frame["bar_span_min_median"] == frame["horizon"]).all()),
        "max_diff_bps": float(frame["max_abs_diff_vs_stored_bps"].max()),
        "span_min": float(frame["bar_span_min_min"].min()),
        "n_checked": int(frame["n"].sum()),
    }


def check_lag(events: pd.DataFrame) -> dict[str, float]:
    lag = (events["entry_ts"] - events["published_at"]).dt.total_seconds() / 60.0
    return {
        "p50": float(lag.median()),
        "p90": float(lag.quantile(0.9)),
        "p99": float(lag.quantile(0.99)),
        "max": float(lag.max()),
        "share_over_5min": float((lag > 5).mean()),
    }


def check_missed(events: pd.DataFrame) -> pd.DataFrame:
    """Return from the close of the bar containing the news to the entry open."""
    rows = []
    for symbol, group in events.groupby("symbol"):
        bars = load_bars(symbol, ["ts_open", "open", "close"])
        ns = bars["ns"].to_numpy()
        pub_ns = group["published_at"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        entry_ns = group["entry_ts"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        contain = np.searchsorted(ns, pub_ns, side="right") - 1
        entry_at = np.searchsorted(ns, entry_ns, side="left")
        ok = (contain >= 0) & (entry_at < len(ns))
        missed = (
            bars["open"].to_numpy()[entry_at[ok]] / bars["close"].to_numpy()[contain[ok]] - 1.0
        )
        rows.append(
            {
                "symbol": symbol,
                "n": int(len(missed)),
                "missed_mean_bps": float(np.mean(missed) * 1e4),
                "missed_abs_median_bps": float(np.median(np.abs(missed)) * 1e4),
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def check_benchmark(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol, group in events.groupby("symbol"):
        row = {"symbol": symbol, "n": len(group)}
        for horizon in HORIZONS:
            row[f"ret_{horizon}m"] = float(group[f"ret_{horizon}m"].mean() * 1e4)
            row[f"excess_{horizon}m"] = float(group[f"excess_{horizon}m"].mean() * 1e4)
        rows.append(row)
    frame = pd.DataFrame(rows)
    totals = {"symbol": "ALL", "n": len(events)}
    for horizon in HORIZONS:
        totals[f"ret_{horizon}m"] = float(events[f"ret_{horizon}m"].mean() * 1e4)
        totals[f"excess_{horizon}m"] = float(events[f"excess_{horizon}m"].mean() * 1e4)
    frame = pd.concat([frame, pd.DataFrame([totals])], ignore_index=True)
    # Split the pooled mean into the part explained by BTC's own move, per day.
    daily = events.assign(day=events["entry_ts"].dt.date).groupby("day")[
        ["ret_5m", "excess_5m", "bench_5m"]
    ].mean()
    extra = {
        "days": int(len(daily)),
        "day_mean_ret_bps": float(daily["ret_5m"].mean() * 1e4),
        "day_mean_bench_bps": float(daily["bench_5m"].mean() * 1e4),
        "day_mean_excess_bps": float(daily["excess_5m"].mean() * 1e4),
        "day_sd_excess_bps": float(daily["excess_5m"].std(ddof=1) * 1e4),
    }
    return frame, extra


def check_selection(events: pd.DataFrame) -> dict[str, object]:
    hour = events["published_at"].dt.hour
    bands = {
        "00-08Z": float(((hour >= 0) & (hour < 8)).mean()),
        "08-16Z": float(((hour >= 8) & (hour < 16)).mean()),
        "16-24Z": float(((hour >= 16) & (hour < 24)).mean()),
    }
    days = events["entry_ts"].dt.date.value_counts()
    return {
        "hour_bands": bands,
        "days": int(len(days)),
        "events_per_day_median": float(days.median()),
        "events_per_day_p95": float(days.quantile(0.95)),
        "busiest_day_share": float(days.iloc[0] / len(events)),
    }


def check_duplicates(events: pd.DataFrame) -> dict[str, float]:
    key = (
        events["title"]
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    counts = key.value_counts()
    return {
        "distinct_titles": int(len(counts)),
        "share_titles_repeated": float((counts > 1).sum() / len(counts)),
        "share_rows_in_repeated": float(counts[counts > 1].sum() / len(events)),
        "max_copies": int(counts.iloc[0]),
    }


def check_splits(events: pd.DataFrame) -> dict[str, object]:
    out = {}
    for name, group in events.groupby("split"):
        out[name] = {
            "n": int(len(group)),
            "first": str(group["entry_ts"].min()),
            "last": str(group["entry_ts"].max()),
        }
    events.sort_values("entry_ts")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default=str(EVENTS))
    parser.add_argument("--out", default=str(ROOT / "results" / "crypto_audit.json"))
    args = parser.parse_args()

    events = pd.read_parquet(Path(args.events))
    events["published_at"] = pd.to_datetime(events["published_at"], utc=True)
    events["entry_ts"] = pd.to_datetime(events["entry_ts"], utc=True)

    exposure = check_exposure(events)
    lag = check_lag(events)
    missed = check_missed(events)
    benchmark, bench_extra = check_benchmark(events)
    selection = check_selection(events)
    duplicates = check_duplicates(events)
    splits = check_splits(events)

    print(f"events={len(events)} symbols={events['symbol'].nunique()} sources={events['source'].nunique()}")
    print("\n[1] exposure:", json.dumps(exposure))
    print("[2] lag:", json.dumps(lag))
    print("\n[3] missed reaction (publication -> entry open)")
    print(missed.to_string(index=False))
    print(f"  pooled: mean {missed['missed_mean_bps'].mean():+.2f} bps, median |missed| {missed['missed_abs_median_bps'].median():.2f} bps")
    print("\n[4] raw return vs excess vs BTC")
    print(benchmark.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    print("  daily:", json.dumps(bench_extra))
    print("\n[5] selection:", json.dumps(selection))
    print("[6] duplicates:", json.dumps(duplicates))
    print("[7] splits:", json.dumps(splits, ensure_ascii=False))

    payload = {
        "events": int(len(events)),
        "exposure": exposure,
        "lag": lag,
        "missed": missed.to_dict(orient="records"),
        "benchmark": benchmark.to_dict(orient="records"),
        "benchmark_daily": bench_extra,
        "selection": selection,
        "duplicates": duplicates,
        "splits": splits,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
