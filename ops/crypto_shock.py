# ruff: noqa: E501  - these scripts print reports
"""How much of a crypto shock is left one minute later?

The question the equity branch could not answer: if a headline reaches us three to five minutes
after publication, is there anything left to trade? This measures it from price alone, without any
news data, over the largest moves in the minute bars.

For every shock bar (top ``--quantile`` of absolute one-minute returns) the script computes the
forward return from *the open of the next bar* - the first price a human or a bot could actually
get after noticing the move - over 1, 5, 15 and 60 minutes, signed by the shock direction.
Positive means continuation (the crowd has not finished), negative means reversal (you are late).

Usage:
    python ops/crypto_shock.py --days 45 --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "crypto" / "1m"
API = "https://api.binance.com/api/v3/klines"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; stock-price-prediction)"}
COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_base",
    "taker_quote",
    "ignore",
]


def _get(params: dict[str, object], attempts: int = 5) -> list[list[str]]:
    """GET with backoff: the public endpoint throttles hard when pushed."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(API, params=params, headers=HEADERS, timeout=30)
            if response.status_code in (418, 429):
                time.sleep(2 ** attempt)
                last = RuntimeError(f"HTTP {response.status_code}")
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - retried, then raised
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"klines request failed after {attempts} attempts: {last}")


def fetch_klines(symbol: str, days: int, pause: float = 0.35) -> pd.DataFrame:
    """One-minute bars from the public REST endpoint, cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{symbol}.parquet"
    need_from = int((time.time() - days * 86400) * 1000)
    if path.exists():
        cached = pd.read_parquet(path)
        if int(cached["open_time"].iloc[0]) <= need_from:
            return cached.loc[cached["open_time"] >= need_from].reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    cursor = need_from
    while cursor < int(time.time() * 1000):
        params = {"symbol": symbol, "interval": "1m", "limit": 1000, "startTime": cursor}
        rows = _get(params)
        if not rows:
            break
        frames.append(pd.DataFrame(rows, columns=COLUMNS)[["open_time", "open", "close", "volume"]])
        cursor = int(rows[-1][0]) + 60_000
        time.sleep(pause)
    if not frames:
        raise SystemExit(f"no bars for {symbol}")
    frame = pd.concat(frames, ignore_index=True).drop_duplicates("open_time")
    frame["open_time"] = frame["open_time"].astype("int64")
    for column in ("open", "close", "volume"):
        frame[column] = frame[column].astype(float)
    frame = frame.sort_values("open_time").reset_index(drop=True)
    frame.to_parquet(path, index=False)
    return frame.loc[frame["open_time"] >= need_from].reset_index(drop=True)


def forward_returns(bars: pd.DataFrame, horizon: int) -> pd.Series:
    """Return from the open of the next bar to the close ``horizon`` bars after the shock bar."""
    entry = bars["open"].shift(-1)
    exit_ = bars["close"].shift(-horizon)
    return exit_ / entry - 1.0


def analyse(symbol: str, bars: pd.DataFrame, quantile: float) -> dict[str, object]:
    close = bars["close"]
    r1 = np.log(close).diff()
    sigma_1m = float(r1.std() * 1e4)
    threshold = float(r1.abs().quantile(quantile))
    shock = r1.abs() >= threshold
    n = int(shock.sum())
    direction = np.sign(r1[shock])

    out: dict[str, object] = {
        "symbol": symbol,
        "bars": int(len(bars)),
        "days": round(len(bars) / 1440, 1),
        "sigma_1m_bps": round(sigma_1m, 2),
        "sigma_5m_bps": round(float(r1.rolling(5).sum().std() * 1e4), 2),
        "shock_threshold_bps": round(threshold * 1e4, 1),
        "n_shocks": n,
        "shock_share": round(n / len(bars), 4),
        "mean_abs_shock_bps": round(float(r1[shock].abs().mean() * 1e4), 1),
    }
    for horizon in (1, 5, 15, 60):
        fwd = forward_returns(bars, horizon)
        signed = (fwd[shock] * direction).dropna()
        out[f"cont_{horizon}m_bps"] = round(float(signed.mean() * 1e4), 2)
        out[f"cont_{horizon}m_median_bps"] = round(float(signed.median() * 1e4), 2)
        out[f"cont_{horizon}m_t"] = round(
            float(signed.mean() / (signed.std(ddof=1) / np.sqrt(len(signed)))), 2
        )
        out[f"cont_{horizon}m_win"] = round(float((signed > 0).mean()), 3)
        out[f"cont_{horizon}m_n"] = int(len(signed))
    # Unconditional drift is the control: a shock sample must beat this to mean anything.
    fwd5 = forward_returns(bars, 5).dropna()
    out["all_5m_mean_bps"] = round(float(fwd5.mean() * 1e4), 2)
    out["all_5m_sd_bps"] = round(float(fwd5.std() * 1e4), 2)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"])
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--quantile", type=float, default=0.998)
    parser.add_argument("--out", default="results/crypto_shock.json")
    args = parser.parse_args()

    rows = []
    for symbol in args.symbols:
        bars = fetch_klines(symbol, args.days)
        row = analyse(symbol, bars, args.quantile)
        rows.append(row)
        print(
            f"{symbol:9s} bars={row['bars']:5d} sigma1m={row['sigma_1m_bps']:6.2f} bps "
            f"shock>{row['shock_threshold_bps']:6.1f} bps n={row['n_shocks']:4d} "
            f"cont5m={row['cont_5m_bps']:+7.2f} bps (t {row['cont_5m_t']:+5.2f}) "
            f"cont60m={row['cont_60m_bps']:+7.2f} bps (t {row['cont_60m_t']:+5.2f})"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
