import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; spp-research/0.1)"}
out = Path("tests/fixtures/live")
now = datetime.now(timezone.utc)
start = (now - timedelta(days=20)).strftime("%Y%m%d%H%M%S")
end = now.strftime("%Y%m%d%H%M%S")
payload = None
for attempt in range(1, 6):
    time.sleep(10)
    r = requests.get(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params={
            "query": '"Apple"',
            "mode": "artlist",
            "format": "json",
            "maxrecords": 50,
            "startdatetime": start,
            "enddatetime": end,
        },
        timeout=90,
        headers=UA,
    )
    print(attempt, r.status_code, len(r.content), r.text[:200].replace("\n", " "))
    if r.status_code == 200 and r.text.strip().startswith("{"):
        payload = r.text
        break
if payload:
    (out / "gdelt_artlist.json").write_text(payload, encoding="utf-8")
    print("saved", len(payload))
