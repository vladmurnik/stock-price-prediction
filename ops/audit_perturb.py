# ruff: noqa: E501, UP031  - prints a report
"""Negative control for the sentiment classifier: replace the company name with "the company".

If the direction of p_pos - p_neg came from the brand rather than the rest of the headline, the
sign would flip on most headlines. Measured: 12/200 flips, corr 0.975 - see docs/audits/S1.
"""

import re

import numpy as np
import pandas as pd

from spp.config import load_config
from spp.data.news_gdelt_hist import _first_aliases_from_universe
from spp.sentiment.score_corpus import build_backend

rng = np.random.default_rng(42)
news = pd.read_parquet("data/interim/news_raw.parquet")
names = _first_aliases_from_universe()
print("alias map:", len(names), "tickers")

rows = []
for ticker, name in names.items():
    if not name or len(name) < 3:
        continue
    sub = news.loc[news["ticker"] == ticker, ["title"]]
    if sub.empty:
        continue
    hit = sub.loc[sub["title"].astype(str).str.contains(re.escape(name), case=False, regex=True)]
    for title in hit["title"].astype(str).tolist():
        rows.append((ticker, name, title))
print("titles that name the company:", len(rows))
if len(rows) < 200:
    print("WARNING: fewer than 200 candidates")
       
    
cand = pd.DataFrame(rows, columns=["ticker", "name", "title"])
if len(cand) > 200:
    cand = cand.sample(200, random_state=42).reset_index(drop=True)

def scrub(row):
    return re.sub(re.escape(row["name"]), "the company", row["title"], flags=re.I)

cand["scrubbed"] = cand.apply(scrub, axis=1)
print("example:")
print("  orig:", cand["title"].iloc[0])
print("  pert:", cand["scrubbed"].iloc[0])

backend = build_backend("finbert", config=load_config())
orig = backend.score(cand["title"].tolist())
pert = backend.score(cand["scrubbed"].tolist())
d_orig = np.array([s.p_pos - s.p_neg for s in orig])
d_pert = np.array([s.p_pos - s.p_neg for s in pert])
s_orig = np.sign(d_orig)
s_pert = np.sign(d_pert)
flip = (s_orig != s_pert)
print("\n=== perturbation: company name -> 'the company' (n=%d) ===" % len(cand))
print("sign flips: %d / %d = %.4f" % (flip.sum(), len(cand), flip.mean()))
print("zero in original: %d, zero after: %d" % ((s_orig == 0).sum(), (s_pert == 0).sum()))
print("mean |d| orig %.4f  pert %.4f" % (np.abs(d_orig).mean(), np.abs(d_pert).mean()))
print("share |d_orig| < 0.05: %.4f ; < 0.10: %.4f" % ((np.abs(d_orig) < .05).mean(), (np.abs(d_orig) < .10).mean()))
print("corr(d_orig, d_pert) = %.4f" % np.corrcoef(d_orig, d_pert)[0, 1])
big = np.abs(d_orig) >= 0.2
print("among confidently scored (|d|>=0.2, n=%d): flip share %.4f" % (big.sum(), flip[big].mean() if big.any() else np.nan))
print("\nby ticker (top 8 by n):")
tab = pd.DataFrame({"ticker": cand["ticker"], "flip": flip, "d": d_orig})
agg = tab.groupby("ticker").agg(n=("flip", "size"), flips=("flip", "sum"), mean_d=("d", "mean"))
print(agg.sort_values("n", ascending=False).head(8).to_string())

ev = pd.read_parquet("data/processed/intraday_gdelt_events.parquet")
ev = ev[ev.horizon == 60]
a = ev["score"].abs()
print("\n=== pilot event scores (n=%d) ===" % len(ev))
print("mean %.4f median %.4f min %.4f max %.4f" % (ev["score"].mean(), ev["score"].median(), ev["score"].min(), ev["score"].max()))
print("share |score| < 0.05: %.4f ; < 0.10: %.4f" % ((a < .05).mean(), (a < .10).mean()))
print("mean |score| by direction:", ev.groupby("direction")["score"].mean().to_dict())
