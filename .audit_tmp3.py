import json, numpy as np
from scipy import stats
for run in ["a1-market-only","a2-sentiment-only","a5-all","a4-lexicon","a6-strict-dates"]:
    m = json.load(open(f"reports/runs/{run}/metrics.json"))
    b = m["aggregate"]["ic"]; mean, std, n = b["mean"], b["std"], b["n"]
    lo, hi = mean - 1.96*std/np.sqrt(n), mean + 1.96*std/np.sqrt(n)
    tstat = mean / (std/np.sqrt(n)); p = 2*stats.t.sf(abs(tstat), n-1)
    fstd = np.std(b["per_fold"], ddof=1)
    print(f"{run:20s} mean={mean:+.4f} std={std:.4f} CI=({lo:+.4f},{hi:+.4f}) t={tstat:.3f} p={p:.4f} | pooled={m['pooled']['ic']:+.4f} CI=({m['pooled']['ic_ci95_lo']:+.4f},{m['pooled']['ic_ci95_hi']:+.4f}) dm_vs_zero p={m['dm_vs_zero']['p']:.5f}")
    print(f"{'':20s} per_fold sd(ddof1)={fstd:.4f}  std matches per_fold ddof1: {abs(std-fstd)<1e-12}   aggregate keys={sorted(b.keys())}")
