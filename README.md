**🇬🇧 English** · [🇷🇺 Русская версия](README.ru.md)

---

> **Кратко по-русски.** Можно ли по сентименту вчерашних новостей предсказать доходность акции на следующий день?
> **Ответ: нет.** Сигнал есть, но он статистически неотличим от нуля вне выборки (holdout 2023: IC 0.0127, 95% CI [−0.0102, 0.0356], p 0.25),
> а при дневном ребалансе издержки превышают эдж примерно в 30 раз (net Sharpe −6.75).
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
```

Tests: `pytest -q` (221 passing), `ruff check src tests` (clean).

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
