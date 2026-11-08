"""Steady-state open-jev throughput for a few batch sizes (load excluded)."""

from __future__ import annotations

import time

import pandas as pd

from spp.sentiment.openjev import OpenJevBackend

news = pd.read_parquet("data/interim/news_raw.parquet", columns=["title"])
texts = [str(t) for t in news["title"].dropna().drop_duplicates().head(64)]

for size in (8, 16, 32):
    backend = OpenJevBackend(batch_size=size)
    backend.score(texts[:size])  # warm-up: loads the model, allocates the arena
    start = time.perf_counter()
    out = backend.score(texts)
    elapsed = time.perf_counter() - start
    print(
        f"batch={size:3d}  n={len(out):3d}  {elapsed:6.2f}s  "
        f"{len(out) / elapsed:5.2f} items/s  "
        f"eta_92k={92927 / (len(out) / elapsed) / 3600:5.2f}h"
    )
