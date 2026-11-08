import time

import duckdb

BASE = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/"
con = duckdb.connect()
con.execute("INSTALL httpfs")
con.execute("LOAD httpfs")
con.execute("SET enable_progress_bar=false")

for name in ("All_external.csv",):
    url = BASE + name
    t0 = time.time()
    rows = con.execute(
        f"SELECT count(*) FROM (SELECT * FROM read_csv_auto('{url}', header=true) LIMIT 1000000)"
    ).fetchone()[0]
    dt = time.time() - t0
    print(f"{name}: {rows} rows in {dt:.1f}s -> {rows / dt:,.0f} rows/s")

    t0 = time.time()
    cols = con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{url}', header=true)").fetchall()
    print("schema:", [(c[0], c[1]) for c in cols])
    print(f"describe in {time.time() - t0:.1f}s")
