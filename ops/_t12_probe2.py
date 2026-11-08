import time

import feedparser
import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; spp-research/0.1)"}
yahoo = [
    "https://finance.yahoo.com/news/rssindex",
    "https://finance.yahoo.com/rss/topstories",
    "https://finance.yahoo.com/rss/headline?s=AAPL",
    "https://finance.yahoo.com/news/rss",
    "https://finance.yahoo.com/rss/",
]
for u in yahoo:
    try:
        r = requests.get(u, timeout=20, headers=UA)
        d = feedparser.parse(r.content)
        print("YAHOO", u, r.status_code, len(r.content), "entries", len(d.entries), "bozo", d.bozo)
        if d.entries:
            e = d.entries[0]
            print("   first:", (e.get("title") or "")[:70], "|", e.get("published") or e.get("pubDate"))
    except Exception as e:
        print("YAHOO ERR", u, type(e).__name__, str(e)[:120])

time.sleep(6)
try:
    r = requests.get(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params={
            "query": '"Apple"',
            "mode": "artlist",
            "format": "json",
            "maxrecords": 5,
            "startdatetime": "20261001",
            "enddatetime": "20261108",
            "sort": "datedesc",
        },
        timeout=40,
        headers=UA,
    )
    print("GDELT", r.status_code, r.text[:600])
except Exception as e:
    print("GDELT ERR", type(e).__name__, str(e)[:200])
