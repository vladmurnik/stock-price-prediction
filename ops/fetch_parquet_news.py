"""Fetch only the relevant news rows from the parquet mirror of FNSPID.

Ready-made path: Hugging Face already hosts FNSPID as parquet
(`sabareesh88/FNSPID_nasdaq`, 15.5M rows). Parquet is columnar, so DuckDB reads
only the five needed columns over HTTP instead of streaming the 23.2 GB CSV,
and the row filter is applied while scanning. Nothing large lands on disk.

Output: `data/interim/news_candidates.parquet` with raw rows, still unfiltered
by the alias matcher (that stays in `spp.data.news_fnspid`).
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import requests
import yaml

DATASETS = ("sabareesh88/FNSPID_nasdaq", "sabareesh88/FNSPID_external")
OUT = "data/interim/news_candidates.parquet"
COLUMNS = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher"]
START = "2019-01-01"
END = "2024-01-01"


def parquet_urls(dataset: str) -> list[str]:
    r = requests.get(
        "https://datasets-server.huggingface.co/parquet", params={"dataset": dataset}, timeout=60
    )
    r.raise_for_status()
    return [f["url"] for f in r.json().get("parquet_files", [])]


def main() -> int:
    universe = yaml.safe_load(open("configs/universe_us_largecap.yaml", encoding="utf-8"))
    tickers = sorted(universe["tickers"])
    quoted = ", ".join(f"'{t}'" for t in tickers)

    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute("SET enable_progress_bar=false")
    con.execute("SET threads=8")

    parts: list[str] = []
    staged: list[str] = []
    for i, ds in enumerate(DATASETS):
        urls = parquet_urls(ds)
        if not urls:
            print(f"{ds}: no parquet files, skipped", flush=True)
            continue
        print(f"{ds}: {len(urls)} parquet file(s)", flush=True)
        t0 = time.time()
        sql = f"""
            SELECT {', '.join(COLUMNS)}
            FROM read_parquet({urls!r}, union_by_name=true)
            WHERE Stock_symbol IN ({quoted})
              AND Date >= '{START}'
              AND Date < '{END}'
        """
        stage = f"data/interim/_part_{i}.parquet"
        con.execute(f"COPY ({sql}) TO '{stage}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        n = con.execute(f"SELECT count(*) FROM '{stage}'").fetchone()[0]
        print(f"{ds}: wrote {n:,} rows in {time.time() - t0:.1f}s", flush=True)
        staged.append(stage)
        parts.append(f"{ds}:{n}")

    if not staged:
        raise SystemExit("no rows staged")

    merged = f"""
        SELECT DISTINCT ON (Stock_symbol, Date, Article_title)
            {', '.join(COLUMNS)}
        FROM read_parquet({staged!r}, union_by_name=true)
        ORDER BY Stock_symbol, Date, Article_title, Publisher
    """
    con.execute(f"COPY ({merged}) TO '{OUT}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    total = con.execute(f"SELECT count(*) FROM '{OUT}'").fetchone()[0]
    for stage in staged:
        Path(stage).unlink(missing_ok=True)
    print(f"merged: {total:,} unique rows", flush=True)
    print("done", parts, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
