"""Check batched open-jev equals per-item scoring and measure throughput."""

from __future__ import annotations

import time

import pandas as pd

from spp.sentiment.openjev import OpenJevBackend

news = pd.read_parquet("data/interim/news_raw.parquet", columns=["title"])
texts = [str(t) for t in news["title"].dropna().drop_duplicates().head(12)]
print("texts:", len(texts))

single = OpenJevBackend(batch_size=1)
start = time.perf_counter()
one = single.score(texts[:4])
single_s = time.perf_counter() - start
print(f"per-item: 4 texts in {single_s:.1f}s -> {4 / single_s:.2f} items/s")

batched = OpenJevBackend(batch_size=8)
start = time.perf_counter()
many = batched.score(texts)
batch_s = time.perf_counter() - start
print(f"batched:  {len(texts)} texts in {batch_s:.1f}s -> {len(texts) / batch_s:.2f} items/s")
print("model_version match:", single.model_version == batched.model_version)

diffs = []
for a, b in zip(one, many[:4], strict=False):
    diffs.append(
        max(
            abs(a.p_pos - b.p_pos),
            abs(a.p_neg - b.p_neg),
            abs(a.p_neu - b.p_neu),
            abs(a.score_5 - b.score_5),
        )
    )
print("max |diff| over first 4:", max(diffs))
print("labels:", [a.label_5 for a in one], [b.label_5 for b in many[:4]])
