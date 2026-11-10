**🇬🇧 English** · [🇷🇺 Русская версия](README.ru.md)

---

> **Кратко по-русски.** Можно ли по сентименту вчерашних новостей предсказать доходность акции на следующий день?
> **Ответ: нет.** Сигнал есть, но он статистически неотличим от нуля вне выборки (holdout 2023: IC 0.0127, 95% CI [−0.0102, 0.0356], p 0.25),
> а при дневном ребалансе издержки превышают эдж примерно в 30 раз (net Sharpe −6.75).
> Внутридневная ветка (сделка на 1–3 часа после новости) запущена отдельно: 59 событий, MDE 22–30 бп —
> «не измеримо», а не «эффекта нет» (раздел *Intraday branch*).
> Отрицательный результат и есть результат. Полный русский текст — [README.ru.md](README.ru.md).

---

# News Sentiment → Next-Day Equity Returns

Can the sentiment of yesterday's news predict tomorrow's stock return? This repository contains the
full, pre-registered attempt: data ingestion, a leakage-gated feature pipeline, purged
cross-validation, ablations, a single locked 2023 holdout, and an economic backtest.

**Result: no.** The signal is in the expected direction, is not statistically distinguishable from
zero out of sample, and at daily rebalancing transaction costs dominate by roughly 30×. The
negative result is the deliverable — every number below is read from an artifact in `results/`.

Русская версия: [README.ru.md](README.ru.md)

---

## Question and design

* **Target** — next-day log return `y = log(adj_close[t+1] / adj_close[t])`, cross-sectionally
  demeaned. Not the price level.
* **Universe** — 40 US large caps across 8 sectors, 2019-01-01 … 2023-12-31 (50 tickers in the
  price cache including SPY, ^VIX and the 8 sector ETFs).
* **News** — 124,571 articles (38 of the 40 tickers), FNSPID corpus, matched by company-name aliases.
* **Sentiment** — FinBERT (`ProsusAI/finbert`), calibrated on Financial PhraseBank; macro-F1 0.8862
  on the calibration split (`results/calibration/finbert.json`).
* **Model** — LightGBM on 49 features (19 market + 30 sentiment aggregates; `MARKET_FEATURES` and
  `SENTIMENT_FEATURES` in `src/spp/features/`, `n_features: 49` in
  `data/processed/features.parquet.manifest.json`), 3 seeds, Huber loss, early stopping on a
  60-session inner validation block.
* **Validation** — 6 purged, embargoed folds over 2019–2022. 2023 is touched exactly once, behind a
  lock file written before training.

## Feature coverage on the 2019–2023 panel (50,320 rows)

| group | examples | coverage |
|---|---|---|
| market | `ret_1`, `vol_20`, `rsi_14`, `spy_ret_1`, `vix_level`, `sector_ret_1` | 0.977 – 0.997 |
| sentiment same-day | `sent_mean`, `pos_share`, `neg_share`, `neu_share` | 0.523 |
| sentiment decayed | `ewma_24h` | 0.910 |
| open-jev only | `novelty_mean`, `company_specific_share` | 0.523 (open-jev matrices only) |

`novelty_mean` and `company_specific_share` are 0.000 in the FinBERT and lexicon matrices because
only open-jev emits `is_novel` / `is_company_specific`. In the open-jev matrices (A3, A8) both
columns are filled on 52.3% of ticker-days — exactly the share of rows that have any news that day.
Getting there required fixing two defects: the aggregate emitted `is_company_specific_n/_d` while the
reader expected `cs_n/cs_d` (always NaN), and the flags are probabilities, not booleans, so the
`astype("boolean")` cast raised as soon as they were populated. A third defect would have hidden all
of this: the open-jev sample cap of 8,000 degenerated to "2023 only" because the budget was
`max(0, cap − n_rows_2023)` against 29,479 articles in 2023, so A3 and A8 would again have measured
the absence of a feature. The full corpus is now scored (92,927 texts on GPU).

## Ablations (purged CV, 6 folds × 3 seeds)

| id | features | IC | 95% CI | DM p vs 0 |
|---|---|---|---|---|
| A1 | market only (19) | 0.0007 | [−0.0151, 0.0165] | 0.0039 |
| A2 | sentiment only (28) | 0.0134 | [−0.0052, 0.0320] | 0.0074 |
| **A5** | market + sentiment (49) | **0.0156** | [−0.0069, 0.0381] | 0.0054 |
| A4 | market + sentiment, lexicon backend (49) | −0.0079 | [−0.0228, 0.0069] | 0.0000 |
| A6 | market + sentiment, strict_dates = true (49) | 0.0167 | [−0.0021, 0.0356] | 0.0042 |
| A3 | sentiment only, open-jev backend (30) | 0.0112 | [−0.0185, 0.0408] | 0.0020 |
| A7 | rank average of A2 and A3 (same rows) | 0.0137 | [−0.0160, 0.0435] | 0.0000 |
| A8 | market + sentiment + typed flags, open-jev (49) | 0.0076 | [−0.0068, 0.0219] | 0.0004 |

`features` is the column count of the ablation matrix in `reports/runs/<id>/metrics.json`
(`config.features`). The sentiment block is 30 columns; A2, A4 and A5 show 28 and 49 because
`company_specific_share` and `novelty_mean` are all-NaN for FinBERT and the lexicon and are dropped
before fitting. A7 is evaluated on the 15,160 rows where both backends have predictions, and A2 and
A3 are re-scored on that same intersection so the comparison is like-for-like.

Market features alone carry no signal. Sentiment adds a stable +0.013…+0.015 IC, but every
confidence interval contains zero. The lexicon backend is *worse than nothing* (−0.0079): a
dictionary is not a substitute for a contextual model. A6 differs from A5 only by the date
policy, which touched 10 of 124,571 articles, so its 0.0011 edge over A5 is noise, not an effect —
the strict-date rule is not where the signal is lost.

Open-jev does not help. It is a general typed-decision model, not a finance model, and it trails
FinBERT on the same task (0.0112 vs 0.0134). Averaging the two ranks recovers 0.0137, i.e. it lands
between its two inputs and still loses to the A5 control measured on the identical rows (0.0156).
The two columns only open-jev can supply — `novelty_mean` and `company_specific_share` — make things
*worse*: A8 at 0.0076 is less than half of A5 at 0.0156 despite the richer feature set. The premise
behind the open-jev branch, that a general model knows which news is genuinely new and
company-specific, is not supported by this data.

## The single 2023 holdout

```
config_hash   3a1c947f4f51b8c58d9a0117b358de9dd0dd3bd78f83111e36cf3e52a5969acd
matrix_sha256 fab4f76c5957f75645842c28d8375a5d6e151dd86142e6382b73abfb1f3b9d6b
train 37,791 rows | inner valid 2,392 | test rows locked 10,000, scored 9,772 over 245 dates
window 2023-01-03 … 2023-12-29
```

Sources: `reports/test_2023.lock` (hashes, `test_rows: 10000`) and `results/test_2023.json`
(`rows.test: 9772`, `metrics.n_dates: 245`). The difference is the tail without a forward price;
no prediction exists for those rows and they are not in any metric below.

| metric | value | pre-registered gate |
|---|---|---|
| IC | **0.0127** | > 0.02 → FAIL |
| IC 95% CI | [−0.0102, 0.0356] | lower > 0 → FAIL |
| direction accuracy | 0.5089 | — |
| DM vs zero p | 0.2507 | < 0.05 → FAIL |
| decile monotonicity | below 0.5 | > 0.5 → FAIL |

**Verdict: `not_confirmed`** (`results/test_2023.json`).

## Economic test on the same year

| execution | Sharpe gross | Sharpe net | annual | max DD | IC |
|---|---|---|---|---|---|
| next open → close | 0.12 | **−6.75** | −30.2% | −29.3% | 0.0048 |
| close → close | 1.27 | −4.08 | −24.4% | −23.4% | 0.0141 |

Turnover is ~166× per year; at 5+5 bps per side that is ~33%/yr against a gross edge of ~1%.

## Two defects that changed the answer

The first CV run reported IC 0.0202 with a CI excluding zero. Auditing feature coverage found:

1. **Sentiment shares were NaN, not 0,** on days with news but no article of that level
   (`reindex` without `fill_value`). Coverage 0.088–0.423 → 0.523.
2. **`sector_ret_1` was entirely NaN** — sector ETFs are helper tickers and were dropped from the
   price matrix before the sector return was computed. Coverage 0.000 → 0.995.

After the fix the CV IC fell to 0.0156 and the interval covered zero. Roughly a third of the
original "edge" was an artifact of missing values. This is why the leakage gates and ablations run
before the holdout.

## Leakage gates (all PASS, `results/leakage_checks.json`)

1. feature provenance — every column traced to an input file;
2. article signal dates — no article dated after its session;
3. target alignment — `y` uses only `t+1`;
4. disguised target — no feature is a monotone transform of `y`;
5. shuffled-label control — IC ≈ 0 on permuted labels;
6. future shift — shifting features forward does not improve IC beyond threshold.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
python -m spp.cli fetch prices          # 62,900 rows, 50 tickers
python -m spp.cli fetch news            # FNSPID parquet mirror → 124,571 articles
python -m spp.cli score corpus --backend finbert
python -m spp.cli features build --check-leakage
python -m spp.cli train gbdt            # purged CV, writes reports/runs/<id>/
python -m spp.cli --config configs/final.yaml train test2023   # refuses to run twice
python -m spp.cli backtest --predictions reports/test_2023_predictions.parquet --include-test
python ops/ablate.py --only a4 a6       # ablation matrices
python ops/intraday_gdelt.py --news data/interim/news_gdelt_pilot.parquet \
    --start 2026-10-01 --end 2026-10-23   # intraday branch, cached bars only
```

Tests: `pytest -q` (265 passing, 2 skipped), `ruff check src tests ops` (clean).

## Intraday branch: one article, one trade

The daily horizon may simply be the wrong one: a news shock decays in hours, not days, and a
once-a-day rebalance averages it away. `docs/intraday-design.md` pre-registers the alternative
before any number was computed: one article = one trade, entry at the open of the first bar at or
after `published_at + 5 min`, exit at the close of the bar **containing** `entry + horizon`
(60/120/180 min), target = ticker return minus SPY over identical timestamps, 15 bps per round
trip. `configs/intraday.yaml` freezes those choices so the experiment cannot be re-tuned after
seeing the result.

The two free windows do not overlap: FNSPID ends 2023-12-16, yfinance 60-minute bars begin
2024-11-11. The branch therefore runs on GDELT headlines for October 2026, and the sample is a
pilot, not a test.

| horizon | events | mean excess | net | hit rate | cluster t | MDE |
|---|---|---|---|---|---|---|
| 60 min | 59 | +2.67 bps | −12.33 bps | 0.644 | +0.41 | 18.3 bps |
| 120 min | 59 | −1.13 bps | −16.13 bps | 0.525 | −0.15 | 20.6 bps |
| 180 min | 59 | −11.37 bps | −26.37 bps | 0.509 | −1.22 | 26.0 bps |

**Verdict: not measurable.** The sample cannot resolve anything smaller than 18–22 bps, and the
plausible size of a news effect is 8–10 bps (the daily study's gross quintile spread). Three
measured facts forbid stronger wording:

* the five-minute lag is **not** realised: on 60-minute bars the median entry delay is **75
  minutes**, the 75th percentile 247 minutes, and only 45.8 % of events enter within one bar. The
  branch measures the first tradeable bar after publication, not the five-minute reaction;
* **35 of 59 events fall on a single session** (2026-10-01), so the effective sample is six days;
* a bootstrap over sessions puts the MDE at 22–30 bps, 13–21 % above the analytic formula.

The high hit rate is not an edge: the average loss is 1.47× the average win (payoff ratio 0.68) and
the break-even hit rate is 0.654 before costs, against 0.644 realised. Splitting by the 50-session
daily trend — the "trade with the trend" variant — leaves 17 events on **two sessions**, which is
why the report prints a dash instead of the |t| = 13 that arithmetic alone produces.

Why the sample is small is a data problem, not a coding one: GDELT returns 10.5 usable articles per
request, drops **84.5 %** of raw hits because attribution is headline-only, and answers 429 on 20 %
of requests at one request per two minutes. Reaching 2,000 events (MDE ≈ 8 bps) needs ≈ 340
requests and 7–20 hours of crawling, with non-random truncation of the busiest news days. Measured
numbers and the plan: `docs/data-intraday-supply.md`. Audits:
`docs/audits/R16-intraday-lookahead.md`, `docs/audits/G7-intraday-validation.md`,
`docs/audits/S1-sentiment-classification.md`.

## Crypto branch: where latency, not data, is the constraint

Equities failed on supply: no overlap between news and minute bars. Crypto removes that blocker —
Binance serves 1-minute bars back to 2017 without a key, and sixteen public RSS feeds publish within
minutes. The branch is pre-registered in `configs/crypto.yaml` before any event was scored.

The first measurement is not encouraging for "react before the crowd". Over 90 days, entering at the
open of the next bar after a top-0.2 % one-minute move leaves **no continuation**: +0.6 bps on BTC
(t +0.24), −4.0 bps on SOL (t −0.78), and at one hour SOL and DOGE *revert* by 19–24 bps (t ≈ −1.9)
instead of continuing. A round trip costs 12 bps on perpetuals and 20 bps on spot, so the measured
effect does not pay for the fees. Leverage does not change the sign: fees are charged on notional, so
edge and cost scale together.

Being early is physically possible only where the time is known in advance — exchange listings, token
unlocks, ETF decisions, macro prints. Everything else reaches a public feed 1–3 minutes after
publication, which is after the first bar.

Full design, the measured channel table and the frozen kill criteria: `docs/crypto-design.md`.
Chronology of the whole project: `docs/history.md`. Raw numbers: `results/crypto_shock.json`,
`results/crypto_probe.json`.

**Result.** 21,588 events over 11 assets, eleven news sources and 86,171 archived headlines: the
average excess is +0.09 / +0.29 / +0.87 bps at 5 / 15 / 60 minutes against 28 bps round trip (the
label is a spread, so both legs pay), the quantile model's out-of-sample IC is +0.001…+0.030 (sd
0.016–0.031), and the trades it does take lose money after costs (60 min: net −1.17 bps, t −3.01).
Full numbers, the three defects found on the way and the limits of the sample:
[docs/crypto-results.md](docs/crypto-results.md).

**Where the amplitude is.** Structural events on the majors do not have it: mean |return| over an hour
is 30–62 bps against 46 bps for an ordinary commentary piece, and the only directional effect that
reaches the cost line is shorting insolvency (+12.40 bps, t +2.14, net −1.60 at 14 bps). The amplitude
lives in exchange listings, and it is measured there: across 714 spot listings, buying from the second
minute loses **2.65 % in an hour** (t −3.56) and **5.44 % in a day** (t −4.63), because the opening
auction (median +18.14 %) has already absorbed the move. The mirror trade — shorting spot — is
unavailable, and shorting the perpetual across 923 listings gives a +1.50 % median against a −0.65 %
mean: p99 adverse move +51.3 %, and 9.3 % of events would liquidate a 5x short within the first hour.
Full tables: §11–12 of `docs/crypto-results.md`.

## Limitations

* A3 (open-jev) does not beat FinBERT: 0.0112 against 0.0134, and both intervals contain zero. What
  the ablation does establish is that a general typed-decision model brings no extra edge on a
  finance task, and that its two unique columns (`novelty_mean`, `company_specific_share`) do not
  rescue it — A8 with them scores 0.0076 against A5's 0.0156.
* Final models are shallow (6–18 trees): early-stopping IC peaks almost immediately.
* The corpus has a median of 1 article per ticker-day (max 19), so same-day sentiment blocks are
  sparse and most of the information lives in multi-day aggregates.
* Live Finnhub ingestion is implemented but unverified — no `FINNHUB_API_KEY` in the environment.
* Daily rebalancing is the wrong horizon for this effect; see `docs/idea-intraday-t5.md`.

## Layout

```
src/spp/data/        ingestion, normalisation, dedupe, live feeds
src/spp/sentiment/   FinBERT, lexicon, open-jev backends + calibration
src/spp/features/    market + sentiment assembly, leakage checks
src/spp/models/      baselines, purged CV, GBDT, 2023 holdout
src/spp/backtest/    portfolio construction, costs, report
results/             every artifact quoted above (json/md only)
ops/                 experiment scripts, dispatcher, ablation builder
```

Data and trained artifacts are not committed (see `.gitignore`); they are reproducible from the
commands above.
