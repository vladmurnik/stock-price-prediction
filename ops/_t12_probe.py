import feedparser
import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; spp-research/0.1)"}
urls = [
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "https://finance.yahoo.com/news/rssindex",
    "https://www.investing.com/rss/news_25.rss",
    "https://www.investing.com/rss/news_25.rss",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
]
for u in urls:
    try:
        r = requests.get(u, timeout=20, headers=UA)
        print("RSS", u, r.status_code, len(r.content), r.headers.get("content-type"))
        d = feedparser.parse(r.content)
        print("   entries:", len(d.entries), "bozo", d.bozo)
        if d.entries:
            e = d.entries[0]
            print("   first:", e.get("title", "")[:70], "|", e.get("published") or e.get("updated") or e.get("pubDate"))
    except Exception as e:
        print("RSS ERR", u, type(e).__name__, str(e)[:150])

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
        },
        timeout=30,
        headers=UA,
    )
    print("GDELT", r.status_code, r.text[:300])
except Exception as e:
    print("GDELT ERR", type(e).__name__, str(e)[:200])

r = requests.get(
    "https://finnhub.io/api/v1/company-news",
    params={"symbol": "AAPL", "from": "2026-10-01", "to": "2026-11-08"},
    timeout=20,
)
print("FINNHUB", r.status_code, r.text[:200])
