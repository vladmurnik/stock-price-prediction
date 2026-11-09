# ruff: noqa: E501, E741  - a measurement report, printed as a table
"""Reproduce every number quoted in docs/audits/S1, G7 and docs/data-intraday-supply.md.

Read-only: opens parquet/json artifacts and prints a plain-text report to stdout.
Run:  .venv\\Scripts\\python.exe ops\\audit_measure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
I = ROOT / "data" / "interim"
R = ROOT / "results"


def hdr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def q(s: pd.Series, name: str) -> None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        print(f"  {name}: EMPTY")
        return
    print(
        f"  {name}: n={len(s)} mean={s.mean():+.4f} sd={s.std():.4f} "
        f"min={s.min():+.4f} p05={s.quantile(.05):+.4f} p25={s.quantile(.25):+.4f} "
        f"p50={s.median():+.4f} p75={s.quantile(.75):+.4f} p95={s.quantile(.95):+.4f} "
        f"max={s.max():+.4f}"
    )


# ---------------------------------------------------------------- S1
def s1() -> None:
    hdr("S1. SENTIMENT CLASSIFIER ON HEADLINES")
    for backend in ("finbert", "openjev", "lexicon"):
        f = P / f"article_sentiment_{backend}.parquet"
        if not f.exists():
            print(f"{backend}: MISSING {f}")
            continue
        df = pd.read_parquet(f)
        print(f"\n--- {backend}: {len(df):,} rows, cols={list(df.columns)}")
        if "published_at" in df.columns:
            ts = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
            print(f"  date range: {ts.min()} .. {ts.max()}")
            df["_year"] = ts.dt.year
        for col in ("label_5", "score_5", "p_pos", "p_neg", "sentiment"):
            if col in df.columns:
                q(df[col], col)
        if "label_5" in df.columns:
            vc = df["label_5"].value_counts(dropna=False).sort_index()
            print("  label_5 counts: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
            print(f"  neutral share (label_5==0): {(df['label_5'] == 0).mean():.4f}")
        if {"p_pos", "p_neg"} <= set(df.columns):
            d = (df["p_pos"] - df["p_neg"]).abs()
            print(f"  |p_pos-p_neg| < 0.10: {(d < 0.10).mean():.4f}")
            print(f"  |p_pos-p_neg| < 0.05: {(d < 0.05).mean():.4f}")
            print(f"  |p_pos-p_neg| < 0.02: {(d < 0.02).mean():.4f}")
            if "_year" in df.columns:
                g = df.assign(_d=d).groupby("_year")["_d"]
                print("  by year: " + ", ".join(
                    f"{y}: n={int(v.count())} mean|d|={v.mean():.4f} share<0.1={(v < .1).mean():.3f}"
                    for y, v in g))
        if "ticker" in df.columns and {"p_pos", "p_neg"} <= set(df.columns):
            df["_d"] = df["p_pos"] - df["p_neg"]
            per = df.groupby("ticker")["_d"].agg(["mean", "count"]).sort_values("mean")
            print(f"  per-ticker mean(p_pos-p_neg): {len(per)} tickers, "
                  f"min={per['mean'].min():+.4f} ({per['mean'].idxmin()}), "
                  f"max={per['mean'].max():+.4f} ({per['mean'].idxmax()}), "
                  f"sd={per['mean'].std():.4f}")
            if "_year" in df.columns:
                piv = df.pivot_table(index="ticker", columns="_year", values="_d", aggfunc="mean")
                drift = (piv.max(axis=1) - piv.min(axis=1)).dropna()
                print(f"  per-ticker year drift (max-min of yearly mean): "
                      f"median={drift.median():.4f} p90={drift.quantile(.9):.4f} max={drift.max():.4f}")
                print("  tickers with |yearly mean| > 0.25 in every year: "
                      f"{int(((piv.abs() > 0.25).all(axis=1)).sum())}")

    hdr("S1b. CALIBRATION FILES")
    for name in ("finbert", "openjev", "lexicon"):
        f = ROOT / "reports" / "calibration" / f"{name}.json"
        if not f.exists():
            print(f"{name}: MISSING")
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        print(f"\n--- {name}.json keys: {list(d)}")
        for k, v in d.items():
            if isinstance(v, (int, float, str)):
                print(f"  {k} = {v}")
            elif isinstance(v, dict):
                print(f"  {k} = " + ", ".join(f"{kk}:{vv}" for kk, vv in list(v.items())[:12]))
            elif isinstance(v, list):
                print(f"  {k} = list[{len(v)}] {v[:12]}")


# ---------------------------------------------------------------- G7
def g7() -> None:
    hdr("G7. INTRADAY VALIDATION")
    f = P / "intraday_gdelt_events.parquet"
    if not f.exists():
        print("events parquet MISSING")
        return
    ev = pd.read_parquet(f)
    print(f"events: {len(ev):,} rows, cols={list(ev.columns)}")
    print(ev.head(3).to_string())

    m = json.loads((R / "intraday_gdelt_metrics.json").read_text(encoding="utf-8"))
    print("\nreported metrics json:")
    print(json.dumps(m, indent=2, ensure_ascii=False)[:3000])

    # identify columns
    ycol = next((c for c in ("y", "ret", "target", "excess_ret") if c in ev.columns), None)
    hcol = next((c for c in ("horizon", "h") if c in ev.columns), None)
    scol = next((c for c in ("session", "date") if c in ev.columns), None)
    print(f"\nusing y={ycol} horizon={hcol} session={scol}")
    if not (ycol and hcol and scol):
        return

    for h, g in ev.groupby(hcol):
        y = g[ycol].astype(float)
        n = len(y)
        n_sess = g[scol].nunique()
        sd = y.std(ddof=1)
        naive = y.mean() / (sd / np.sqrt(n))
        # cluster by session
        gm = g.groupby(scol)[ycol].agg(["mean", "count"])
        k = len(gm)
        if k > 1:
            v_cl = (gm["count"] * (gm["mean"] - y.mean()) ** 2).sum() / (k - 1)
            se_cl = np.sqrt(v_cl) / np.sqrt(n)
            t_cl = y.mean() / se_cl if se_cl > 0 else np.nan
            infl = (se_cl / (sd / np.sqrt(n))) ** 2 if sd > 0 else np.nan
        else:
            se_cl = t_cl = infl = np.nan
        mde = (1.96 + 0.84) * np.sqrt(max(infl, 1.0)) * sd / np.sqrt(n)
        print(f"\n  h={h}: n={n} sessions={n_sess} mean={y.mean()*1e4:+.2f}bps sd={sd*1e4:.2f}bps")
        print(f"    naive t={naive:+.2f}  cluster t={t_cl:+.2f}  inflation={infl:.3f}")
        print(f"    MDE(1.96+0.84)*infl^.5*sd/sqrt(n) = {mde*1e4:.2f} bps")
        print(f"    hit_rate(y>0)={float((y > 0).mean()):.4f}  median={y.median()*1e4:+.2f}bps")
        print(f"    events per session: min={gm['count'].min()} max={gm['count'].max()} "
              f"mean={gm['count'].mean():.2f}")
        # bootstrap over sessions
        rng = np.random.default_rng(0)
        means = []
        arr = gm["mean"].to_numpy()
        cnt = gm["count"].to_numpy()
        for _ in range(5000):
            idx = rng.integers(0, k, k)
            w = cnt[idx]
            means.append(float((arr[idx] * w).sum() / w.sum()))
        means = np.array(means)
        print(f"    session bootstrap: sd={means.std()*1e4:.2f}bps "
              f"95%CI=[{np.quantile(means,.025)*1e4:+.2f},{np.quantile(means,.975)*1e4:+.2f}]bps "
              f"implied MDE={(1.96+0.84)*means.std()*1e4:.2f}bps")

    # sign/magnitude table
    print("\n  sign x magnitude (all horizons pooled):")
    y = ev[ycol].astype(float)
    print(f"    y>0: n={int((y > 0).sum())} mean={y[y > 0].mean()*1e4:+.2f}bps")
    print(f"    y<0: n={int((y < 0).sum())} mean={y[y < 0].mean()*1e4:+.2f}bps")
    print(f"    y==0: n={int((y == 0).sum())}")

    # multiplicity
    print("\n  multiplicity: 3 horizons x 2 trend regimes = 6 tests, plus quintile spread and event IC")
    print("    Bonferroni alpha=0.05/6 -> 0.00833 (|t| > 2.64); 0.05/8 -> 0.00625 (|t| > 2.73)")


# ---------------------------------------------------------------- supply
def supply() -> None:
    hdr("SUPPLY. NEWS COVERAGE VS MINUTE BARS")
    f = I / "news_raw.parquet"
    if f.exists():
        df = pd.read_parquet(f)
        ts = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        print(f"news_raw: {len(df):,} rows, published_at {ts.min()} .. {ts.max()}")
        print(f"  rows with published_at >= 2024-11-11: {int((ts >= pd.Timestamp('2024-11-11', tz='UTC')).sum())}")
        print(f"  rows with published_at >= 2024-01-01: {int((ts >= pd.Timestamp('2024-01-01', tz='UTC')).sum())}")
        print(f"  max published_at: {ts.max()}")
    for name in ("news_gdelt_pilot", "news_gdelt"):
        f = I / f"{name}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        ts = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        print(f"\n{name}: {len(df):,} rows, {df['ticker'].nunique() if 'ticker' in df else '?'} tickers, "
              f"{ts.min()} .. {ts.max()}")
        if "ticker" in df.columns:
            print("  per-ticker: " + ", ".join(f"{k}={v}" for k, v in df["ticker"].value_counts().items()))
    # minute cache
    cache = ROOT / "data" / "cache" / "minutes" / "60m"
    if cache.exists():
        files = sorted(cache.glob("*.parquet"))
        print(f"\nminute cache 60m: {len(files)} tickers")
        for p in files:
            b = pd.read_parquet(p)
            idx = pd.to_datetime(b.index, utc=True) if not isinstance(b.index, pd.DatetimeIndex) else b.index
            print(f"  {p.stem}: {len(b)} bars, {idx.min()} .. {idx.max()}, {idx.normalize().nunique()} sessions")


if __name__ == "__main__":
    s1()
    g7()
    supply()
