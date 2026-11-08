import time

import duckdb
import requests

con = duckdb.connect()
con.execute("INSTALL httpfs")
con.execute("LOAD httpfs")
con.execute("SET enable_progress_bar=false")


def parquet_urls(dataset: str) -> list[str]:
    r = requests.get(
        "https://datasets-server.huggingface.co/parquet", params={"dataset": dataset}, timeout=60
    )
    r.raise_for_status()
    files = r.json().get("parquet_files", [])
    return [f["url"] for f in files]


for ds in ("benstaf/FNSPID-filtered-nasdaq-100",):
    urls = parquet_urls(ds)
    print(ds, "files:", len(urls))
    if not urls:
        continue
    t0 = time.time()
    src = urls[0] if len(urls) == 1 else urls
    q = f"""
        SELECT Stock_symbol, count(*) AS n, min(Date) AS d0, max(Date) AS d1
        FROM read_parquet({src!r}, union_by_name=true)
        GROUP BY 1 ORDER BY n DESC
    """
    rows = con.execute(q).fetchall()
    dt = time.time() - t0
    print(f"scanned in {dt:.1f}s, symbols={len(rows)} rows_total={sum(r[1] for r in rows):,}")
    print("top:", rows[:12])
    print("date range:", min(r[2] for r in rows), "->", max(r[3] for r in rows))
