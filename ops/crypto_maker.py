# ruff: noqa: E501  - prints a report table
"""Lever 1: would maker execution rescue the edge?

Taker execution costs 28 bps round trip for a hedged pair, and the measured edge is 0.5-1.2
bps. Maker execution is the only lever that can change the sign, so it must be measured,
not assumed.

The rule under test is the one a trader would actually use:

* after a headline, place a limit order at ``p_ref`` - the close of the bar that contains
  ``published_at``. That is the last price printed before the news could be read, so the
  limit can only fill if the price comes back to it;
* the order lives for ``fill_window`` minutes and fills if any bar in the window trades
  through it (``low <= p_ref`` for a buy, ``high >= p_ref`` for a sell); the fill price is
  exactly ``p_ref``, with no queue simulation and no partial fills;
* the exit is at market ``horizon`` minutes after the fill, as a taker;
* cost = 2 bps maker + 5 bps taker + 2 bps slippage on the exit leg = 9 bps round trip.

Two numbers decide everything: the **fill rate** (a limit that never fills is not a
strategy) and the **adverse selection** (the average price move between ``p_ref`` and the
fill, which is negative for a buy by construction). If filled trades lose the same amount
that the maker fee saved, the lever is empty.

Usage:
    python ops/crypto_maker.py [--events data/crypto/dataset/events.parquet]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = ROOT / "data" / "crypto" / "dataset" / "events.parquet"
BARS_DIR = ROOT / "data" / "crypto" / "bars" / "1m"
RESULTS = ROOT / "results"
MAKER_BPS = 2.0
TAKER_BPS = 5.0
SLIPPAGE_BPS = 2.0
COST_BPS = MAKER_BPS + TAKER_BPS + SLIPPAGE_BPS
LAG_MINUTES = 1
FILL_WINDOW = 5
HORIZONS = (5, 15, 60)
MINUTE_NS = 60 * 1_000_000_000


def _ns(stamps: pd.Series) -> np.ndarray:
    return stamps.astype("datetime64[ns, UTC]").astype("int64").to_numpy()


def cluster_t(frame: pd.DataFrame, value: str) -> float:
    work = frame[[value, "day"]].dropna()
    if len(work) < 3 or work["day"].nunique() < 2:
        return float("nan")
    grouped = work.groupby("day")[value]
    mean = work[value].mean()
    resid = grouped.sum() - mean * grouped.size()
    k = grouped.ngroups
    variance = float((resid**2).sum()) / len(work) ** 2 * k / (k - 1)
    return float(mean / np.sqrt(variance)) if variance > 0 else float("nan")


def measure(events: pd.DataFrame, bars: dict[str, pd.DataFrame], side: int, horizon: int) -> pd.DataFrame:
    """One row per filled event, for one side and horizon."""
    rows: list[dict[str, object]] = []
    for symbol, group in events.groupby("symbol"):
        frame = bars.get(symbol)
        if frame is None or frame.empty:
            continue
        stamps = _ns(frame["ts_open"])
        low = frame["low"].to_numpy(dtype="float64")
        high = frame["high"].to_numpy(dtype="float64")
        close = frame["close"].to_numpy(dtype="float64")
        for event in group.itertuples():
            pub = int(event.published_at.value)
            ref_i = int(np.searchsorted(stamps, pub, side="right")) - 1
            if ref_i < 0:
                continue
            p_ref = close[ref_i]
            start = int(np.searchsorted(stamps, pub + LAG_MINUTES * MINUTE_NS, side="left"))
            stop = min(start + FILL_WINDOW, len(stamps))
            if start >= len(stamps):
                continue
            window_low = low[start:stop]
            window_high = high[start:stop]
            if side > 0:
                hits = np.nonzero(window_low <= p_ref)[0]
            else:
                hits = np.nonzero(window_high >= p_ref)[0]
            if not len(hits):
                continue
            fill_i = start + int(hits[0])
            exit_target = stamps[fill_i] + (horizon - 1) * MINUTE_NS
            exit_i = int(np.searchsorted(stamps, exit_target, side="right")) - 1
            if exit_i <= fill_i:
                continue
            if exit_target <= stamps[-1]:
                exit_i = int(np.searchsorted(stamps, stamps[fill_i] + horizon * MINUTE_NS, side="right")) - 1
            exit_i = min(max(exit_i, fill_i + 1), len(stamps) - 1)
            raw = side * (close[exit_i] / p_ref - 1.0)
            rows.append(
                {
                    "symbol": symbol,
                    "published_at": event.published_at,
                    "entry_ts": frame["ts_open"].iloc[fill_i],
                    "day": pd.Timestamp(frame["ts_open"].iloc[fill_i]).date(),
                    "filled": 1.0,
                    "ret_bps": raw * 1e4,
                    "net_bps": (raw - COST_BPS / 1e4) * 1e4,
                    "adverse_bps": side * (p_ref / close[ref_i] - 1.0) * 1e4,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default=str(DEFAULT_EVENTS))
    parser.add_argument("--out", default=str(RESULTS / "crypto_maker.json"))
    args = parser.parse_args()

    events = pd.read_parquet(args.events)
    events = events.dropna(subset=["published_at"]).copy()
    events["published_at"] = pd.to_datetime(events["published_at"], utc=True)
    events = events.sort_values("published_at").reset_index(drop=True)
    symbols = sorted(events["symbol"].unique())
    print(f"events={len(events)} assets={len(symbols)} cost={COST_BPS:.0f} bps fill_window={FILL_WINDOW} min")

    attempts = len(events) * 2
    records: list[dict[str, object]] = []
    tables: list[str] = []
    for horizon in HORIZONS:
        block = []
        for side, label in ((1, "long"), (-1, "short")):
            # LazyBars projects high/low away, and a maker fill is defined by the bar's
            # low (long) or high (short), so the bars are read one symbol at a time.
            pieces = []
            for symbol in symbols:
                frame = pd.read_parquet(
                    BARS_DIR / f"{symbol}.parquet",
                    columns=["ts_open", "open", "high", "low", "close"],
                )
                sub = events[events["symbol"] == symbol]
                if len(sub) and len(frame):
                    pieces.append(measure(sub, {symbol: frame}, side, horizon))
                del frame
            filled = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
                columns=["ret_bps", "net_bps", "adverse_bps", "day"]
            )
            filled = filled.assign(horizon=horizon, side=label)
            records.append(
                {
                    "horizon": horizon,
                    "side": label,
                    "attempts": attempts // 2,
                    "filled": int(len(filled)),
                    "fill_rate": float(len(filled) / max(1, attempts // 2)),
                    "mean_adverse_bps": float(filled["adverse_bps"].mean()) if len(filled) else float("nan"),
                    "mean_ret_bps": float(filled["ret_bps"].mean()) if len(filled) else float("nan"),
                    "mean_net_bps": float(filled["net_bps"].mean()) if len(filled) else float("nan"),
                    "t_net": cluster_t(filled, "net_bps") if len(filled) else float("nan"),
                }
            )
            block.append(records[-1])
        tables.append(
            f"### {horizon} min\n\n| сторона | попыток | исполнено | fill rate | adverse, бп | ret, бп | net (9 бп) | t |\n|---|---|---|---|---|---|---|---|\n"
            + "\n".join(
                f"| {row['side']} | {row['attempts']} | {row['filled']} | {row['fill_rate']:.3f} | "
                f"{row['mean_adverse_bps']:+.2f} | {row['mean_ret_bps']:+.2f} | {row['mean_net_bps']:+.2f} | "
                f"{row['t_net']:+.2f} |"
                for row in block
            )
            + "\n"
        )
        for row in block:
            print(
                f"{horizon:3d}m {row['side']:5s} filled {row['filled']:5d}/{row['attempts']}"
                f" ({row['fill_rate']:.3f})  adverse {row['mean_adverse_bps']:+.2f}"
                f"  ret {row['mean_ret_bps']:+.2f}  net {row['mean_net_bps']:+.2f}  t {row['t_net']:+.2f}"
            )

    Path(args.out).write_text(json.dumps(records, indent=2, default=str) + "\n", encoding="utf-8")
    report = ["# Maker-исполнение: измерено\n\n"]
    report.append(
        f"Правило: лимит по цене закрытия бара, содержащего `published_at`; окно {FILL_WINDOW} мин; "
        f"выход по рынку через горизонт; издержки {COST_BPS:.0f} бп на круг "
        f"(maker {MAKER_BPS:.0f} + taker {TAKER_BPS:.0f} + проскальзывание {SLIPPAGE_BPS:.0f}).\n\n"
    )
    report.extend(tables)
    report.append(
        "`adverse` — среднее движение цены между опорной ценой и исполнением: для покупки оно "
        "отрицательно по построению, потому что лимит исполняется только если цена вернулась. "
        "Если adverse съедает сэкономленные на комиссии 19 бп, рычаг пуст.\n"
    )
    Path(RESULTS / "crypto_maker.md").write_text("".join(report), encoding="utf-8")
    print(f"wrote {args.out} and results/crypto_maker.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
