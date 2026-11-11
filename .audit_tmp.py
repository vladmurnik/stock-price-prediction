import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
def daily_ic(df, ycol, pcol):
    out = df.groupby("date").apply(lambda g: spearmanr(g[ycol], g[pcol]).statistic if len(g) > 2 else np.nan)
    return out.dropna()
for run in ["a4-lexicon", "a5-all", "a6-strict-dates"]:
    m = json.load(open(f"reports/runs/{run}/metrics.json"))
    per = [f["ic"] for f in m["folds"]]
    seedmeans = [float(np.mean(list(f["seeds_ic"].values()))) for f in m["folds"]]
    allseeds = [v for f in m["folds"] for v in f["seeds_ic"].values()]
    print("==", run)
    print("  aggregate.ic.mean        =", m["aggregate"]["ic"]["mean"])
    print("  mean(fold ic)            =", float(np.mean(per)))
    print("  mean(per-fold mean seeds)=", float(np.mean(seedmeans)))
    print("  mean(18 seed-folds)      =", float(np.mean(allseeds)), " sd=", float(np.std(allseeds, ddof=1)), " n=", len(allseeds))
    print("  fold ic sd of:", float(np.std(per, ddof=0)), "| aggregate.std", m["aggregate"]["ic"]["std"], "| np.std ddof1", float(np.std(per, ddof=1)))
    print("  pooled ic", m["pooled"]["ic"], "n_rows", m["pooled"]["n_rows"], "n_dates", m["pooled"]["n_dates"])
    d = pd.read_parquet(f"reports/runs/{run}/predictions.parquet")
    d = d.dropna(subset=["y_eval"])
    dd = daily_ic(d, "y_eval", "y_pred_seed_mean")
    print("  RECOMPUTED pooled IC:", round(float(dd.mean()), 6), "n_dates", len(dd), "n_rows", len(d), "dates_match", len(dd) == m["pooled"]["n_dates"])
    pf = []
    for f in range(6):
        sub = d[d.fold == f]
        pf.append(float(daily_ic(sub, "y_eval", "y_pred_seed_mean").mean()))
    print("  RECOMPUTED per-fold IC:", [round(x, 6) for x in pf])
    print("  RECOMPUTED mean of fold ICs:", round(float(np.mean(pf)), 6))
    print("  metrics per-fold IC     :", [round(x, 6) for x in per])
    # seed-vs-seed_mean check
    print("  y_pred_seed_mean == y_pred everywhere:", bool(np.allclose(d.y_pred, d.y_pred_seed_mean, equal_nan=True)))
