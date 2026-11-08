import time

import pandas as pd

from spp.data.extract import extract_from_url

news = pd.read_parquet("data/interim/news_raw.parquet")
urls = [u for u in news["url"].dropna().unique() if u.startswith("http")]
news["domain"] = news["url"].str.extract(r"https?://(?:www\.)?([^/]+)")[0]
sample = news.dropna(subset=["domain"]).groupby("domain").head(1).head(40)
candidates = [u for u in sample["url"].tolist() if isinstance(u, str)][:6]

for url in candidates:
    t0 = time.perf_counter()
    art = extract_from_url(url)
    dt = time.perf_counter() - t0
    if art is None:
        print(f"FAIL  {dt:5.2f}s  {url}")
        continue
    print(f"OK    {dt:5.2f}s  words={art.word_count:5d} date={art.published_at} ({art.date_precision})"
          f" title={(art.title or '')[:60]!r}")
    print(f"      {url}")
