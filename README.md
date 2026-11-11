**🇬🇧 English** · [🇷🇺 Русская версия](README.ru.md)

---

# News → price: two studies, one answer

This repository is a complete, pre-registered attempt to answer a single question: **can public news
be traded profitably?** It was asked twice — once on US equities, once on crypto — and both times the
answer was measured rather than assumed.

**Answer: no, not at these costs.** Of fourteen tested ideas, twelve closed with a negative result
after costs, one (the intraday equity branch) was *not measurable* on the available data, and one —
the CPI macro print at a 60-minute horizon — produced the first positive cell in the project and
remains a **candidate, not a strategy**.

The negative result is the deliverable. Every number in this file is read from an artifact in
`results/`, and two independent checkers recompute them with their own arithmetic:
`ops/crypto_verify.py` (78/78 checks pass) and `ops/crypto_verify2.py` (overlap, bootstrap, funding,
slippage).

| | |
|---|---|
| branches | equities (daily, intraday) and crypto (event study, listings, macro) |
| measured events | 47,790 — 45,625 (headline, symbol) pairs, 714 spot listings, 923 perpetual listings, 528 macro releases |
| data | 1-minute bars 2017–2026, 86,171 archived headlines from 11 sources, 528 release dates |
| tests | 422 passed, 2 skipped; `ruff check src tests ops` clean |
| history | 27 commits, 2026-11-08 … 2026-11-11 — chronology in [docs/history.md](docs/history.md) |

---

## Every idea, and how it ended

| # | idea | what was tested | result | verdict |
|---|---|---|---|---|
| 1 | cross-sectional news, 5/15/60 min | news moves an asset more than the market | +0.09 / +0.29 / +0.87 bps against 28 bps round trip | **failed** |
| 2 | BTC on absolute return | BTC news moves BTC, and BTC has amplitude | mean drift +0.21 / +0.08 / −0.41 bps; model gross +5.11, net −0.55 (t −1.85) | **failed**, narrowly |
| 3 | capacity ladder | the model is the bottleneck | 15 configurations, best top decile +4.84 bps gross → net −23.16 | **failed** |
| 4 | classic market models + rank combination | off-the-shelf models know something LightGBM does not | best cell +4.84 bps gross; the combination never beats the best single model | **failed** |
| 5 | maker execution | passive entry cuts costs from 28 to 9 bps | fill rate 0.86, filled trades lose 3.6–5.5 bps before costs (t −11.98…−45.11) | **failed**: adverse selection |
| 6 | structural events (11 classes) | listings, hacks, bankruptcies move harder | amplitude 30–62 bps against 46.2 for an ordinary comment | **failed** |
| 7 | spot listings | the first minute is the entry into the move | auction median +18.14% is unreachable; after it, −2.65% per hour (t −3.56) | **failed** |
| 8 | perpetual listings | a short catches the post-listing reversal | mean −0.61%; median +1.50%; p99 adverse move +51.33%; 5x liquidated in 9.3% | **failed** |
| 9 | macro prints CPI / NFP / FOMC | the time is known in advance and amplitude is larger | CPI 60 min +28.65 bps (t +2.66), net +14.65; NFP and FOMC negative | **candidate** |
| 10 | source trust | some publishers carry a stronger signal | best-to-worst spread 5.5 bps; ranking across halves Spearman −0.03…+0.15 | **failed**: not reproducible |
| 11 | post-shock continuation | a violent move continues | sign flips across instruments, \|t\| ≤ 1.93, best cell +9.37 bps < 14 | **failed** |
| 12 | daily equity sentiment | yesterday's sentiment predicts tomorrow's return | 2023 holdout IC 0.0127 [−0.0102, 0.0356]; net Sharpe −6.75 | **failed** |
| 13 | ablations A1–A8, three sentiment backends | the signal is lost in the model or the lexicon | every CI contains zero; open-jev 0.0112 < FinBERT 0.0134; lexicon −0.0079 | **failed** |
| 14 | intraday equities, one article = one trade | news decays over hours, not days | 59 events, +2.67 bps at 60 min, MDE 18.3 bps (bootstrap 22–30) | **not measurable** |

Full write-up with designs, tables, reasons and the list of defects found on the way:
**[docs/research/crypto-news-alpha.md](docs/research/crypto-news-alpha.md)**
(Russian: [crypto-news-alpha.ru.md](docs/research/crypto-news-alpha.ru.md)).

## The one candidate, and why it is not a strategy

CPI releases are the only family whose time is published months in advance, so no headline race is
involved. Entering in the direction of the release-minute candle, one minute later, and exiting after
60 minutes pays **+28.65 bps on BTC (t +2.66, net +14.65 after 14 bps)** and **+35.32 bps on ETH
(t +2.48, net +21.32)**.

It stays a candidate for five measured reasons:

1. **36 cells were tested**; Bonferroni needs \|t\| ≈ 3.2 and the best is 2.66. A block bootstrap over
   release days gives p 0.0068 / 0.0114 → family-wise error ≈ 0.24.
2. **One era.** 2017–2020: −0.47 (t −0.03); **2021–2023: +72.15 (t +3.14)**; 2024–2026: +16.49
   (t +1.31). The effect lives where CPI was the market's main event.
3. **Latency.** +28.65 / +20.81 / +11.02 bps at a 1 / 5 / 15-minute entry. Fifteen minutes is already
   below cost, and the live recorder's median detection latency is 15.9 minutes.
4. **Not two instruments.** BTC and ETH correlate at r ≈ 0.59 and agree in sign 68.8% of the time —
   consistency, not independent replication. Effective n ≈ 108.
5. **Never tested forward.** Everything else here is historical; this is the only hypothesis that can
   still be falsified with new data. The pre-registered test: 12 releases, same rule, 14 bps, and the
   effect must show net > 0 with t > 1.5, otherwise the idea is closed.

At 12 trades a year, 14.65 bps each is ≈ 176 bps ≈ 1.8% annually before one bad print — and the
60-minute absolute move on CPI days is 74.5 bps, so a single surprise erases the year. Leverage is
ruled out by the tail, not by taste.

## How to read this repository

| start here | what it is |
|---|---|
| [docs/research/crypto-news-alpha.md](docs/research/crypto-news-alpha.md) | the full study: prologue, method, all 14 ideas, four barriers, defects, verification, appendices |
| [docs/history.md](docs/history.md) | the project's chronology, stage by stage, with artifacts |
| [docs/crypto-results.md](docs/crypto-results.md) | the crypto branch report, 15 sections of tables |
| [docs/audits/C1-crypto-pipeline.md](docs/audits/C1-crypto-pipeline.md) | 8 audits of the crypto pipeline in 14 sections |
| [results/README.md](results/README.md) | index of all 47 artifacts and the script that produced each |
| [data/crypto/README.md](data/crypto/README.md) | what is committed, what is regenerated, and the auction trap |
| [CITATION.cff](CITATION.cff) | how to cite this work |

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"

# equities
python -m spp.cli fetch prices          # 62,900 rows, 50 tickers
python -m spp.cli fetch news            # FNSPID parquet mirror → 124,571 articles
python -m spp.cli score corpus --backend finbert
python -m spp.cli features build --check-leakage
python -m spp.cli train gbdt            # purged CV → reports/runs/<id>/
python -m spp.cli --config configs/final.yaml train test2023   # refuses to run twice
python -m spp.cli backtest --predictions reports/test_2023_predictions.parquet --include-test

# crypto
python ops/crypto_build_bars.py         # 12 majors × 47 months of 1-minute bars
python ops/crypto_macro_bars.py         # BTC/ETH since 2017-08
python ops/crypto_wayback.py --start 2023-01-01 --end 2026-11-09
python ops/crypto_dataset.py            # events + leakage checks
python ops/crypto_train.py && python ops/crypto_sources.py && python ops/crypto_audit.py
python ops/crypto_btc.py && python ops/crypto_ladder.py && python ops/crypto_classic.py
python ops/crypto_maker.py && python ops/crypto_structural.py
python ops/crypto_listings.py --workers 12 && python ops/crypto_listing_perp.py --workers 10
python ops/crypto_macro.py

# independent checks (they import no pipeline module)
python ops/crypto_verify.py
python ops/crypto_verify2.py
```

Tests: `pytest -q` → **422 passed, 2 skipped**; `ruff check src tests ops` → clean.

## Layout

```
src/spp/data/        ingestion, normalisation, dedupe, live feeds
src/spp/sentiment/   FinBERT, lexicon and open-jev backends + calibration
src/spp/features/    market and sentiment assembly, leakage checks
src/spp/models/      baselines, purged CV, GBDT, the 2023 holdout
src/spp/backtest/    portfolio construction, costs, report
src/spp/crypto/      crypto bars, dataset, taxonomy, macro calendar, models
ops/                 experiment scripts, dispatcher, ablation builder, verifiers
results/             every artifact quoted above (json/md only) + README index
data/crypto/         committed derived datasets (events, listings) + README
docs/                design, results, audits, history, research papers
```

## Data committed and data not committed

Raw minute bars (~1 GB) are **not** in git and are rebuilt by `ops/crypto_build_bars.py` and
`ops/crypto_macro_bars.py`. The derived datasets **are** committed — `data/crypto/dataset/` (21,588
cross-sectional events, 24,037 BTC events, 45,625 structural), `data/crypto/listings/` (714 spot, 923
perps) — so every published number can be recomputed without downloading a gigabyte of bars. Details:
[data/crypto/README.md](data/crypto/README.md).

## What went wrong on the way (and was fixed)

Nine defects were found by audit, every one of them biased the result **towards optimism**:

| # | defect | effect of the fix |
|---|---|---|
| 1 | fee charged for one leg instead of two | net overstated by exactly 14 bps per trade |
| 2 | `legs=2` never reached the policy table | the same 14 bps, silently |
| 3 | entry into the listing auction | a fake +337% mean became −2.65% per hour |
| 4 | BTC features empty in the ladder scripts | 4 of 15 features were dead |
| 5 | "insolvency" 48% liquidation recaps | +12.40 (t 2.14) → +13.48 (t 1.63): significance gone |
| 6 | ETF approval tested before rejection | 13 headlines with a flipped sign |
| 7 | a hardcoded "only positive cell" claim | computed from data instead |
| 8 | a source-table horizon labelled 5 min but computed at 15 | label fixed |
| 9 | MDE published without clustering | 0.41/1.29 → 0.47/1.76 with the caveat |

In equities the two defects were similar and just as consequential: sentiment shares were NaN instead
of 0 on days with news but no article of that level, and `sector_ret_1` was entirely empty. Fixing
them took the CV IC from 0.0202 to 0.0156 and the confidence interval covered zero. About a third of
the original "edge" was an artifact of missing values.

## Limitations

* **Market impact** is not modelled: a trade is assumed not to move the price.
* **Position overlap** is not modelled: events may need capital simultaneously; there is no portfolio
  accounting.
* **Funding** is measured only for perpetual listings, on the first day.
* **News sources** are 6 archived publishers; GDELT and several feeds were unavailable, so the corpus
  is CoinDesk-heavy.
* **The structural classifier** is keyword-based, without a manually labelled precision/recall.
* **The CPI effect** lives in one era and has never been tested forward.
* **The 465-event crypto holdout** is untouched but underpowered: its MDE is tens of basis points.
* **Latency**: the live RSS recorder measured p50 15.9 minutes, while the backtest assumes 1.45 — an
  optimistic assumption that works against the strategy, not in its favour.

## Citation

Metadata is in [CITATION.cff](CITATION.cff). If you reuse the datasets, cite the repository and the
artifact you used; every table in the papers names its source file.
