import json, numpy as np, pandas as pd
from scipy import stats
from scipy.stats import spearmanr

def load(run):
    return json.load(open(f"reports/runs/{run}/metrics.json"))

A5, A6 = load("a5-all"), load("a6-strict-dates")
p5 = np.array([f["ic"] for f in A5["folds"]]); p6 = np.array([f["ic"] for f in A6["folds"]])
d = p6 - p5
print("per-fold diff A6-A5:", [round(x, 6) for x in d])
print("mean diff", round(d.mean(), 6), "sd", round(d.std(ddof=1), 6))
t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
print("paired t (6 folds, df=5):", round(t, 4), "p =", round(2 * stats.t.sf(abs(t), 5), 6))
w = stats.wilcoxon(p5, p6)
print("Wilcoxon:", round(w.statistic, 4), "p =", round(w.pvalue, 6))
tun = stats.ttest_ind(p5, p6)
print("unpaired t:", round(tun.statistic, 4), "p =", round(tun.pvalue, 6))

# two-way decomposition of seed-level variation
rows = []
for run, m in (("A5", A5), ("A6", A6)):
    for f in m["folds"]:
        for s, v in f["seeds_ic"].items():
            rows.append((run, f["fold"], int(s), v))
df = pd.DataFrame(rows, columns=["run", "fold", "seed", "ic"])
print("\nseed-level mean IC per run:", df.groupby("run").ic.mean().round(6).to_dict())
print("seed-level sd per run:", df.groupby("run").ic.std().round(6).to_dict())
# paired across seed/fold cells (same fold & seed => same data & seed)
piv = df.pivot_table(index=["fold", "seed"], columns="run", values="ic")
dd = piv.A6 - piv.A5
print("cell-paired diff: mean", round(dd.mean(), 6), "sd", round(dd.std(), 6), "n", len(dd))
tt = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
print("  t(df=%d)=%.3f p=%.4f" % (len(dd) - 1, tt, 2 * stats.t.sf(abs(tt), len(dd) - 1)))

# independent, daily-IC level test on predictions
def daily_ic(df_, col):
    g = df_.groupby("date").apply(lambda x: spearmanr(x["y_eval"], x[col]).statistic if len(x) > 2 else np.nan)
    return g.dropna()
pr5 = pd.read_parquet("reports/runs/a5-all/predictions.parquet").dropna(subset=["y_eval"])
pr6 = pd.read_parquet("reports/runs/a6-strict-dates/predictions.parquet").dropna(subset=["y_eval"])
i5, i6 = daily_ic(pr5, "y_pred_seed_mean"), daily_ic(pr6, "y_pred_seed_mean")
align = pd.concat([i5.rename("a5"), i6.rename("a6")], axis=1).dropna()
dif = align.a6 - align.a5
print("\ndaily IC: mean a5 %.6f a6 %.6f  mean diff %.6f  sd %.6f  n_dates %d" % (align.a5.mean(), align.a6.mean(), dif.mean(), dif.std(), len(dif)))
nw = dif.mean() / (dif.std(ddof=1) / np.sqrt(len(dif)))
print("  naive t = %.3f p=%.4f" % (nw, 2 * stats.t.sf(abs(nw), len(dif) - 1)))
def nw_t(x, lag=5):
    x = np.asarray(x, float); n = len(x); mu = x.mean(); e = x - mu
    s = (e @ e) / n
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    return mu / np.sqrt(s / n)
print("  Newey-West(lag=5) t = %.3f p=%.4f" % (nw_t(dif), 2 * stats.t.sf(abs(nw_t(dif)), len(dif) - 1)))
print("  corr(a5_daily, a6_daily) = %.4f" % align.corr().iloc[0, 1])
