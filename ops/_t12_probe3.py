import time

import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; spp-research/0.1)"}
for attempt in range(1, 4):
    time.sleep(6)
    r = requests.get(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params={
            "query": '"Apple"',
            "mode": "artlist",
            "format": "json",
            "maxrecords": 5,
            "startdatetime": "20261001",
            "enddatetime": "20261108",
        },
        timeout=40,
        headers=UA,
    )
    print(attempt, "GDELT", r.status_code, r.text[:800].replace("\n", " "))
    if r.status_code == 200:
        break
