# Crypto data: what is committed and what is not

Large raw data is intentionally not in git. Everything committed here can be regenerated,
everything regenerable is documented.

## Committed (derived datasets)

| file | rows | what it is |
|---|---|---|
| `dataset/events.parquet` | 21 588 | cross-sectional events: one row per (headline, symbol), with entry, five horizons, benchmark, cost, split |
| `dataset/events_btc.parquet` | 24 037 | the same for `symbol == BTCUSDT`, absolute return labels, one leg |
| `dataset/events_structural.parquet` | 45 625 | both frames merged and classified into 11 structural classes |
| `listings/events.parquet` | 714 | first trading day of every spot symbol, entry at the second bar |
| `listings/events_perp.parquet` | 923 | first trading day of every UM perpetual, short returns |

Column meanings follow `docs/data-dictionary.md` and the module docstrings in
`src/spp/crypto/dataset.py`.

## Not committed (raw, ~1 GB)

| path | what | how to rebuild |
|---|---|---|
| `crypto/bars/1m/{SYMBOL}.parquet` | 12 majors, one-minute bars, 2023-01…2026-10 | `python ops/crypto_build_bars.py` |
| `crypto/bars/1m_macro/{SYMBOL}.parquet` | BTC/ETH since 2017-08 for the macro branch | `python ops/crypto_macro_bars.py` |
| `crypto/news/wayback.parquet` | 86 171 archived headlines | `python ops/crypto_wayback.py --start 2023-01-01 --end 2026-11-09` |
| `crypto/news/recorder.parquet` | live RSS recorder output | `python ops/crypto_news.py --once` |
| `crypto/funding/` | Binance funding archives, cached | any command that needs them |

## Sources

- Bars: `data.binance.vision` monthly archives (public, no key).
- News: Internet Archive CDX API plus the publishers' own RSS feeds.
- Macro calendar: `bls.gov/bls/news-release/{cpi,empsit}.htm` and the Federal Reserve FOMC pages.
- Listings: the first month present in the Binance Vision S3 listing for each symbol; no
  announcement feed is needed, and the first bar is the listing moment.

## Warning

The first bar of a newly listed symbol is an **auction**. Its `open` is a nominal print
(1INCH opened at 0.200 with a high of 2.900). Entry must be at the open of the second bar;
this single mistake once produced a fake +337% mean return.
