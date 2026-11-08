import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; spp-research/0.1)"}
out = Path("tests/fixtures/live")
out.mkdir(parents=True, exist_ok=True)

now = datetime.now(timezone.utc)
start = (now - timedelta(days=20)).strftime("%Y%m%d%H%M%S")
end = now.strftime("%Y%m%d%H%M%S")
time.sleep(6)
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
    timeout=60,
    headers=UA,
)
print("GDELT", start, end, r.status_code, len(r.content), r.text[:300].replace("\n", " "))
if r.status_code == 200 and r.content.strip().startswith(b"{"):
    (out / "gdelt_artlist.json").write_bytes(r.content)
    print("saved gdelt_artlist.json")

feeds = {
    "rss_cnbc.xml": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "rss_marketwatch.xml": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "rss_yahoo_aapl.xml": "https://finance.yahoo.com/rss/headline?s=AAPL",
    "rss_investing.xml": "https://www.investing.com/rss/news_25.rss",
}
for name, url in feeds.items():
    rr = requests.get(url, timeout=30, headers=UA)
    print(name, rr.status_code, len(rr.content))
    if rr.status_code == 200:
        (out / name).write_bytes(rr.content)
