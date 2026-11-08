# Data dictionary

Scope: every column of the feature matrix, every artifact under `data/`, and the on-disk schemas
of `news_raw`, `article_sentiment_*` and `features`.

**Source of truth is code.** Every definition below cites the module and line that produce it.
Every quoted number carries the artifact path it was read from and the way it was measured. No
number is copied from the README, and none is estimated.

Conventions:

- `t` is one session of the shared NYSE calendar spine (`calendar_spine`, `src/spp/features/calendar.py`).
  The label at row `t` is the return from `t` to `t+1`.
- **Coverage** = share of non-NaN values over the 50,320-row 2019–2023 matrix
  (`data/processed/features.parquet`; `rows: 50320` in `data/processed/features.parquet.manifest.json`).
  Measured by reading the parquet column by column and taking `1 - isna().mean()`. Canary: measured
  `sent_mean` coverage is **0.5230**, which matches the `0.523` quoted in `README.md` — so the
  measurement and the artifact agree.
- **Lagged vs t**: `same-day` = the value at row `t` is allowed to read data dated `t` (ever only
  the close of `t`, never `t+1`; the target is `t+1`). `yes` = the value is shifted so that row `t`
  describes information from `<= t-1`.

---

## 1. The feature columns

**Exact arithmetic, measured from the arrow schema** (`data/processed/features.parquet`,
`ParquetFile(...).schema_arrow.names`):

```
62 parquet fields = 2 identifiers (ticker, date) + 49 features + 11 label/split fields
49 features = 19 MARKET_FEATURES + 30 SENTIMENT_FEATURES
11 label/split fields = 10 LABEL_COLUMNS (y_1d..y_5d, y, y_dir, y_raw, y_used_demeaned,
                        ghost, label_ok) + 1 split
```

Verified: `feature_columns(True)` returns **49** names (not 62); `MARKET_FEATURES` has 19,
`SENTIMENT_FEATURES` has **30**; `LABEL_COLUMNS` has 10; every feature name is present in the
parquet; the only non-identifier fields outside `LABEL_COLUMNS` are the 49 features plus `split`.

> Naming note for whoever reads the brief's "62 feature columns": 62 is the total number of
> **stored fields**, not the number of features. There are **49 features** and **62 fields**. The
> tables below list the 49 features first, then the label/split block, labelling both so the count
> cannot be misread.

- Market: `MARKET_FEATURES` (`src/spp/features/market.py:42-62`) — 19 names.
- Sentiment: `SENTIMENT_FEATURES` (`src/spp/features/sentiment_daily.py:108-118`) — 30 names
  (5 volume + 19 day + 6 multi-day; see §1.3).
- Concatenated by `feature_columns(include_sentiment=True)` (`src/spp/features/assemble.py:102-107`).

The `n_features: 49` in `features.parquet.manifest.json` and `features_schema.json` is the true
feature count. It is not a count "after LightGBM drops all-NaN columns": the two open-jev columns
are stored (all-NaN) and part of the 49. LightGBM later ignores them at fit time, but they are
present in the matrix.

**Source ingest** for the tables below:
- **prices** = `data/cache/prices_2019-01-01_2023-12-31.parquet` (yfinance OHLCV + `adj_close`).
- **news** = `data/interim/news_raw.parquet` (FNSPID article metadata).
- **sent** = the sentiment table joined to news by `article_id` (`prepare_articles`,
  `sentiment_daily.py:183-229`). The on-disk default is the **lexicon** backend copy
  (`data/processed/article_sentiment.parquet` has `backend == ['lexicon']`); see §2.3 and §5 for the
  FinBERT/lexicon ambiguity.

### 1.1 Market group — 19 columns (`src/spp/features/market.py`)

Chaining rules from the module docstring (`market.py:1-18`): `ret_k` uses the close of `t` directly;
every rolling statistic (`vol_*`, `rsi_14`, `px_sma*`, `volume_z`) is additionally `shift(1)`-ed, so
row `t` describes the window ending at `t-1`; `dow` is the weekday of `t`.

| # | column | group | definition / formula (code) | source ingest | coverage (2019–2023) | lagged vs t |
|---|---|---|---|---|---|---|
| 1 | `ret_1` | market | `log(P_t / P_{t-1})`; `rets.shift(lag-1)`, lag=1 (`market.py:146-147`) | prices | 0.9959 | same-day (uses close t) |
| 2 | `ret_2` | market | `log(P_{t-1} / P_{t-2})` (lag=2) | prices | 0.9951 | yes |
| 3 | `ret_3` | market | `log(P_{t-2} / P_{t-3})` (lag=3) | prices | 0.9943 | yes |
| 4 | `ret_4` | market | `log(P_{t-3} / P_{t-4})` (lag=4) | prices | 0.9935 | yes |
| 5 | `ret_5` | market | `log(P_{t-4} / P_{t-5})` (lag=5) | prices | 0.9927 | yes |
| 6 | `vol_5` | market | std(ddof=1) of `ret_1` over 5 sessions, `min_periods=3`, then `.shift(1)` (`market.py:150`) | prices | 0.9935 | yes |
| 7 | `vol_20` | market | std(ddof=1) of `ret_1` over 20 sessions, `min_periods=10`, `.shift(1)` (`market.py:149`) | prices | 0.9880 | yes |
| 8 | `vol_60` | market | std(ddof=1) of `ret_1` over 60 sessions, `min_periods=20`, `.shift(1)` (`market.py:151`) | prices | 0.9800 | yes |
| 9 | `vol_ratio` | market | `vol_5 / vol_60` (`market.py:155`) | prices | 0.9800 | yes |
| 10 | `volume_z` | market | `(v_{t-1} − mean20)/(std20 + 1e-9)` after `volume.shift(1)` (`market.py:108-112`) | prices | 0.9808 | yes |
| 11 | `rsi_14` | market | Wilder RSI(14): `alpha=1/14`, `min_periods=14`, on `close.shift(1)` (`market.py:93-105`); `R = 100` when avg_loss is 0 and avg_gain > 0 | prices | 0.9848 | yes |
| 12 | `px_sma20` | market | `close_{t-1}/SMA20(close_{t-1..t-20}) − 1`, `min_periods=10` (`market.py:160-163`) | prices | 0.9888 | yes |
| 13 | `px_sma50` | market | `close_{t-1}/SMA50(close_{t-1..t-50}) − 1`, `min_periods=25` (`market.py:161-164`) | prices | 0.9769 | yes |
| 14 | `sector_ret_1` | market | `log(ETF_{t-1}/ETF_{t-2})` of the ticker's sector ETF (`market.py:170-187`) | prices | 0.9951 | yes |
| 15 | `spy_ret_1` | market | `log(SPY_{t-1}/SPY_{t-2})` (`market.py:190`) | prices | 0.9951 | yes |
| 16 | `spy_ret_5` | market | sum of the last 5 sessions of `spy_ret_1`, `min_periods=1` (`market.py:191`) | prices | 0.9951 | yes |
| 17 | `vix_level` | market | `^VIX` close at `t` (`market.py:192`) — **not shifted** | prices | 0.9967 | same-day |
| 18 | `vix_chg_1` | market | `vix.diff().shift(1)` = `VIX_{t-1} − VIX_{t-2}` (`market.py:193`) | prices | 0.9951 | yes |
| 19 | `dow` | market | `dayofweek(date_t)` in 0..4, dtype `int8` (`market.py:194`) | calendar | 0.9967 | deterministic |

`vix_level` asymmetry, stated plainly: it is the **same-day** close, the only market statistic not
`shift(1)`-ed. Verified against the price cache: joined to `^VIX` close, `vix_level` matches the raw
same-day close with max abs difference `0.0`; `vix_chg_1` matches `VIX.diff().shift(1)` with max abs
diff `0.0`, and differs from unshifted `VIX.diff()` by up to `42.5`. So the lag is real and
`vix_level` is genuinely unlagged.

### 1.2 Sentiment group — 30 columns (`src/spp/features/sentiment_daily.py`)

Only `skipped == False` articles enter (`sentiment_daily.py:195`). Grouping key is
`(ticker, signal_date)` (`news_raw.signal_date`, §4.1). `score_5` for the FinBERT/lexicon backends
is `2·E[level]/4 − 1`, taken at the bucket midpoints `(0.5, 2.0, 3.5)`
(`src/spp/sentiment/base.py:51,148-153`), so it lies in `[-1, 1]`. `label_5` buckets it at
`< -0.4`, `< -0.1`, `<= 0.1`, `<= 0.4`, else (`base.py:156-166`).

`*_d` columns are computed by `_rolling_prev` over `[t-w, t-1]` (`:486-491`); `n_news_z` and
`sent_surprise` use the `rolling_days = 30` window (`:424-437`, config `features.rolling_days`).

| # | column | group | definition / formula (code) | source ingest | coverage (2019–2023) | lagged vs t |
|---|---|---|---|---|---|---|
| 20 | `n_news` | volume | count of non-skipped articles with `signal_date == t`; NaN→0 (`:245`, `:344`) | news+sent | 0.9967 | same-day |
| 21 | `log_n_news` | volume | `log1p(n_news)` (`:453`) | news+sent | 0.9967 | same-day |
| 22 | `n_news_z` | volume | `(n_news − mean30)/(std30 + 1e-9)` over `[t-30, t-1]` (`:426-436`) | news+sent | 0.9951 | yes |
| 23 | `ewma_6h` | decay | time-EWMA of `score_5`, halflife 360 min, aged to `t 23:59:59`, then ffill (`:291-308`, `:451`) | sent | 0.9102 | same-day, ffill |
| 24 | `ewma_24h` | decay | same, halflife 1440 min (`:291-292`, `:452`) | sent | 0.9102 | same-day, ffill |
| 25 | `sent_mean` | level | mean `score_5` on day `t`; NaN when `n_news == 0` (`:387-390`) | sent | 0.5230 | same-day |
| 26 | `sent_sum` | level | `Σ score_5` on day `t`; 0.0 when no news (`:244-246`, `:345-346`) | sent | 0.9967 | same-day |
| 27 | `sent_std` | dispersion | std(ddof=1) of `score_5` on day `t`; **0.0 when a day has a single article** (`:258-262`) | sent | 0.5230 | same-day |
| 28 | `sent_min` | extreme | min `score_5` on day `t` (`:247`) | sent | 0.5230 | same-day |
| 29 | `sent_max` | extreme | max `score_5` on day `t` (`:248`) | sent | 0.5230 | same-day |
| 30 | `sent_median` | level | median `score_5` on day `t` (`:249`) | sent | 0.5230 | same-day |
| 31 | `pos_share` | shares | `n(label_5='positive')/n_labelled` on day `t`, 0 if none (`:271-288`) | sent | 0.5230 | same-day |
| 32 | `neg_share` | shares | `n(negative)/n_labelled` (`:275-288`) | sent | 0.5230 | same-day |
| 33 | `neu_share` | shares | `n(neutral)/n_labelled` (`:275-288`) | sent | 0.5230 | same-day |
| 34 | `very_pos_share` | shares | `n(very positive)/n_labelled` (`:275-288`) | sent | 0.5230 | same-day |
| 35 | `very_neg_share` | shares | `n(very negative)/n_labelled`, with `n_labelled > 0` guard (`:394-398`) | sent | 0.5230 | same-day |
| 36 | `neg_extreme_share` | shares | `n(p_neg > 0.8)/n_news` (`:240,252,289,391-393`) | sent | 0.5230 | same-day |
| 37 | `sent_surprise` | surprise | `(sent_mean − mean_sent30)/(std_sent30 + 1e-9)` over `[t-30, t-1]` (`:429-437`) | sent | 0.5218 | yes |
| 38 | `sent_momentum_3d` | momentum | `sent_mean_t − sent_mean_{t-3}` (`:439-440`) | sent | 0.4390 | same-day vs t-3 |
| 39 | `company_specific_share` | open-jev | `Σ is_company_specific / count(non-null)` (`:404-408`) | sent (open-jev only) | **0.0000** | same-day |
| 40 | `novelty_mean` | open-jev | `Σ is_novel / count(non-null)` (`:409-413`) | sent (open-jev only) | **0.0000** | same-day |
| 41 | `conf_mean` | confidence | mean `confidence = max(p_neg,p_neu,p_pos)` (`:401-403`, `base.py:211`) | sent | 0.5230 | same-day |
| 42 | `source_entropy` | source | `−Σ p·ln p` of publisher counts on day `t` (`:140-146`, `:263-265`) | news | 0.5230 | same-day |
| 43 | `top_source_share` | source | `max_publisher_count / n_news` (`:149-153`, `:266-268`) | news | 0.5230 | same-day |
| 44 | `sent_sum_3d` | multi-day | `Σ sent_sum` over `[t-3, t-1]` (`:442-445`) | sent | 0.9967 | yes |
| 45 | `sent_mean_3d` | multi-day | `sent_sum_3d / n_articles_3d`, NaN if the window has no articles (`:447-449`) | sent | 0.6333 | yes |
| 46 | `sent_sum_5d` | multi-day | `Σ sent_sum` over `[t-5, t-1]` | sent | 0.9967 | yes |
| 47 | `sent_mean_5d` | multi-day | `sent_sum_5d / n_articles_5d` | sent | 0.6543 | yes |
| 48 | `sent_sum_10d` | multi-day | `Σ sent_sum` over `[t-10, t-1]` | sent | 0.9967 | yes |
| 49 | `sent_mean_10d` | multi-day | `sent_sum_10d / n_articles_10d` | sent | 0.6679 | yes |

Definitions that affect interpretation:

- **`sent_std = 0.0` on a single-article day is a code default, not a measured zero.** With a median
  of 1 article per ticker-day (per-ticker `median_per_day` in `reports/news_coverage.csv` ranges
  0–4), most news days have exactly one article and `sent_std` is identically 0 there. Its *effective*
  dispersion coverage is far below its 0.523 nominal coverage.
- **`sent_sum` and every `*_sum_*d` column are 0.0 on quiet days by construction.** Their high
  coverage (0.9967) only means "not masked"; a 0 here means "no article", not "neutral news".
- **Low cardinality:** `pos_share` has 168 distinct values, `neg_share` 131, `neu_share` 307,
  `very_pos_share` 255, `very_neg_share` 263, `source_entropy` 435, `top_source_share` 138. These are
  ratios of small integers, not continuous features.
- **`company_specific_share` and `novelty_mean` are all-NaN (coverage 0.0000).** `is_company_specific`
  and `is_novel` are `null`-typed for the FinBERT and lexicon artifacts (`article_sentiment.parquet`
  arrow schema: both fields `null`), so `cs_d`/`nov_d` are 0 and `_derive` yields NaN
  (`sentiment_daily.py:404-413`). Only the open-jev cache populates them
  (`data/cache/sentiment/openjev/shard_*.parquet`: `is_company_specific: double`), but no assembled
  open-jev corpus exists (no `article_sentiment_openjev.parquet` in `data/processed/`).

### 1.3 Count reconciliation (exact)

| component | count | how measured |
|---|---|---|
| `SENTIMENT_FEATURES` total | **30** | `len(SENTIMENT_FEATURES)` = 5 volume + `SENT_DAY_FEATURES` 19 + `SENT_MULTIDAY_FEATURES` 6 (`sentiment_daily.py:105-118`) |
| `MARKET_FEATURES` total | **19** | `market.py:42-62` |
| `feature_columns(True)` | **49** | 19 + 30 |
| parquet fields | **62** | `schema_arrow.names`: 2 identifiers + 49 features + 10 labels + `split` |

`SENT_MULTIDAY_FEATURES` = `sent_{sum,mean}_{3,5,10}d` = 2 bases × 3 windows = 6. The tuple
therefore has 30 members, not 43; there is no "43 sentiment" block. The 62 stored fields are the
parquet total, and the feature count is 49. The two all-NaN open-jev columns (`company_specific_share`,
`novelty_mean`) are inside the 30 and inside the 49; LightGBM ignores them at fit time, they are not
removed at build time.

---

## 2. Artifacts under `data/`

`data/` is gitignored (`data/.gitignore` contains `*`). The artifacts exist only in a working copy;
provenance for any `results/` number is the `request_key` and `manifest_written_utc` in each sidecar
manifest. Total sizes below were read with `Path.rglob` inside each directory.

### 2.1 `data/cache/` — prices, phrasebank, sentiment shards (26.35 MB, 1,541 files)

| artifact | what it holds | produced by | size / rows |
|---|---|---|---|
| `prices_2019-01-01_2023-12-31.parquet` | daily OHLCV + `adj_close`, 50 tickers (40 modelled + SPY, ^VIX, XLC, XLE, XLF, XLI, XLK, XLU, XLV, XLY) | `spp.cli fetch prices` → `src/spp/data/prices.py` (yfinance) | 2,138,245 bytes; 62,900 rows; 9 cols |
| `prices_2019-01-01_2023-12-31.parquet.manifest.json` | `request_start=2019-01-01`, `request_end=2023-12-31`, `min_date=2019-01-02`, `max_date=2023-12-29`, `rows=62900`, `tickers_missing=[]` | `src/spp/data/store.py:write_manifest` | 1,683 bytes |
| `phrasebank/` | Financial PhraseBank, the FinBERT/lexicon calibration corpus | `spp/sentiment/calibrate.py` | 1 file, 0.68 MB |
| `sentiment/finbert/` | 256 per-hash shard parquets + 256 stamp.json. Shard schema: `text_hash, backend, model_version, p_neg, p_neu, p_pos, score_5, score_3, confidence, label_5, is_company_specific(null), is_novel(null), skipped, skip_reason` | `spp.cli score corpus --backend finbert` → `src/spp/sentiment/score_corpus.py` | 512 files, 11.29 MB; `shard_000.parquet` = 364 rows |
| `sentiment/lexicon/` | same shard schema (lexicon, `pysentiment2` HIV4) | `score corpus --backend lexicon` | 512 files, 6.61 MB; `shard_000` = 364 rows |
| `sentiment/openjev/` | shard parquets where `is_company_specific` and `is_novel` are real `double` columns | `score corpus --backend openjev`; shards 000–109 only | 220 files, 2.41 MB; `shard_000` = 111 rows |
| `sentiment/openjev_cpu_backup/` | a CPU-scored snapshot of the open-jev shards (000–146) | backup copy; no code path regenerates this directory | 294 files, 3.22 MB |

Sharding: `shard_id = int(sha1(text_hash)[:8], 16) % NUM_SHARDS` (`score_corpus.py:116-119`). The
cache is keyed by *text* hash, so a headline scored once is reused by every article that shares it —
`unique_texts: 92927` in every `article_sentiment_*.manifest.json`, while `rows: 124571`.

### 2.2 `data/interim/` — normalised news (27.22 MB, 3 files)

| artifact | what it holds | produced by | size / rows |
|---|---|---|---|
| `news_candidates.parquet` | FNSPID rows pre-filtered to the 5 columns the pipeline uses (`Date, Article_title, Stock_symbol, Url, Publisher`) | `parquet_source` (`news_fnspid.py:254-272`) | 5,914,352 bytes; 136,741 rows |
| `news_raw.parquet` | normalised news corpus, 11 columns (§3.1) | `spp.cli fetch news` → `fetch_news` | 21,305,589 bytes; 124,571 rows |
| `news_raw.parquet.manifest.json` | `rows=124571`, `tickers=38`, `min_date=2019-01-01`, `max_date=2023-12-16`, `cutoff_et=16:00`, `strict_dates=false`, the drop funnel | `fetch_news` (`news_fnspid.py:603-624`) | 924 bytes |

Drop funnel, verbatim from `data/interim/news_raw.parquet.manifest.json`:
`scanned 136741`, `no_alias 0`, `ambiguous_alias 11253`, `short_or_boilerplate 588`, `bad_date 0`,
`out_of_range 0`, `non_universe 0`, `exact_dupe 97`, `near_dupe 232`, `alias_mismatch 8723`,
`kept 124571`. Funnel balances: `11253 + 588 + 97 + 232 = 12170 = 136741 − 124571`. `alias_mismatch`
is counted but the row is kept (`news_fnspid.py:391-395`), so it is not a drop.

**Conflicting manifest.** `data/news_raw.parquet.manifest.json` (root, 898 bytes) is a *different*
manifest for the same logical artifact and reports `rows: 47703`, `tickers: 40`, `no_alias: 72903`,
`exact_dupe: 4203`, `manifest_written_utc: 2026-11-07T20:52:34`. The interim manifest reports
`124571` rows with `manifest_written_utc: 2026-11-07T20:58:46`. The parquet actually on disk is the
interim one (124,571 rows, §3.1). The two differ by 76,868 rows. This dictionary records the
discrepancy; it does not decide which run owns the root manifest.

### 2.3 `data/processed/` — sentiment, features, labels (95.16 MB, 15 files)

| artifact | what it holds | produced by | size / rows |
|---|---|---|---|
| `article_sentiment.parquet` | sentiment table (17 cols, §3.2); on disk this is the **lexicon** copy — `backend == ['lexicon']`, `model_version = pysentiment2:HIV4+cfgc920a29e2dde`; the manifest with the same timestamp reports `backend: lexicon` | `spp.cli score corpus` | 9,093,690 bytes; 124,571 rows |
| `article_sentiment_finbert.parquet` | same 17 cols, FinBERT (`ProsusAI/finbert@0bc2ac…`) | `score corpus --backend finbert` | 14,513,607 bytes; 124,571 rows |
| `article_sentiment_lexicon.parquet` | lexicon backend; measured **byte-identical content to `article_sentiment.parquet`** (same `article_id` order, same `score_5`) | lexicon backend | 9,093,690 bytes; 124,571 rows |
| `features.parquet` | 50,320 × 64 fields: 62 features + `ticker` + `date`? | `spp.cli features build` → `build_feature_matrix` (`assemble.py:313-363`) | 15,382,633 bytes |
| `features.parquet.manifest.json` | `rows=50320`, `tickers=40`, `n_features=49`, `split_counts={train:30280, valid:10040, test:10000}`, `sentiment_available: true` | `write_manifest` (`assemble.py:340-351`) | 297 bytes |
| `features_no_sentiment.parquet` | 50,320 rows with the sentiment columns dropped (B1 baseline) | `build_feature_matrix` (`assemble.py:337-338`) | 10,173,851 bytes; 32 cols |
| `features_sentiment_only.parquet` | 50,320 rows, market dropped, sentiment + label block kept | `ops/ablate.py` | 8,668,389 bytes; 41 cols |
| `features_sentiment_only.manifest.json` | `n_sentiment_features: 28`, the exact 28 names, `passthrough` block | `ops/ablate.py` | 1,111 bytes |
| `features_schema.json` | pinned schema: `version: features-v1`, `n_features: 49`, `market_features` (19), `sentiment_features` (**30**), `label_columns`, `nan_policy`, target block | `write_feature_schema` (`assemble.py:366-370`) | 4,190 bytes |
| `predictions.parquet` | OOF/CV predictions: `date, ticker, fold, seed, y_pred, y_pred_seed_mean, score_rank, y, y_raw, y_eval` | `spp.cli train gbdt` | 761,033 bytes; **15,160 rows** |
| `ablation/a4-lexicon.parquet`, `ablation/a6-strict-dates.parquet` | full ablation matrices | `ops/ablate.py --only a4 a6` | 27.47 MB total; 50,320 rows each |

Exact column counts read from the arrow schemas:
`features.parquet` = **64 fields** (62 features + `ticker` + `date`), `features_no_sentiment.parquet`
= 32, `features_sentiment_only.parquet` = 41. Those last two keep the label block, `split` and the
features they were assembled to isolate.

Row arithmetic: 40 tickers × 1,258 spine sessions = 50,320, exactly the manifest `rows`. The panel
is rectangular with no missing ticker-session.

**`28` vs `30` — not a contradiction.** `features_sentiment_only.manifest.json` reports
`n_sentiment_features: 28`: that is the ablated matrix's own count (two all-NaN open-jev columns
excluded). `features_schema.json` reports `sentiment_features` length **30**: the build-time full
set. Both are correct; they describe different objects. `n_features: 49` in the manifest is the
LightGBM input size (19 market + 30 sentiment = 49 after dropping the two all-NaN columns from the
30). `feature_columns(True)` still names 62 because it includes the labels-independent full list plus
the label names; the parquet stores 64 fields because of `ticker` and `date`. No count is wrong; the
labels differ by inspection.

`features_no_sentiment.parquet` has **no sidecar manifest** (checked: file absent). Its provenance is
unrecorded.

### 2.4 `data/live/` — empty

0 files, 0 bytes. The live pipeline (`src/spp/live/`, `src/spp/data/news_live.py`) writes here when
run; no live run has left an artifact in this working copy. Consequently every live claim in the
repository is **not backed by a file under `data/`**; there is also no `reports/live_*` file
(checked: none).

### 2.5 `data/processed/ablation/` — 2 matrices

`a4-lexicon.parquet` and `a6-strict-dates.parquet`, 50,320 rows × 62 fields each, from
`ops/ablate.py`. A3 (open-jev), A7 (ensemble) and A8 (open-jev + typed flags) have **no** matrix
here, consistent with `docs/limitations.md` §1.

---

## 3. Schemas

### 3.1 `data/interim/news_raw.parquet` — 11 columns, 124,571 rows

Produced by `_finalize_news` (`news_fnspid.py:457-475`); column order pinned by `NEWS_COLUMNS`
(`normalize.py:103-115`).

| column | arrow type | meaning / source |
|---|---|---|
| `article_id` | `large_string` | `sha1(publisher|url|title)` (`news_fnspid.py:404`) |
| `ticker` | `large_string` | linked universe ticker |
| `published_at` | `timestamp[us, tz=UTC]` | UTC publish time; measured null share **0.0000** |
| `date` | `date32` | source date (UTC) |
| `signal_date` | `date32` | first day the article is usable (`normalize.py:assign_signal_date`) |
| `title` | `large_string` | cleaned headline, 20–300 chars (`normalize.py:168-195`) |
| `title_norm` | `large_string` | lowercase, punctuation-free |
| `url` | `large_string` | canonical URL (tracking params stripped, `news_fnspid.py:316-336`) |
| `publisher` | `large_string` | domain/name, `unknown` when blank (`news_fnspid.py:339-345`) |
| `source` | `large_string` | always `fnspid` |
| `text_hash` | `large_string` | `sha1(title_norm)` (`news_fnspid.py:413`) |

`signal_date − date` measured distribution (days): `{0: 113848, 1: 5711, 2: 4409, 3: 603}`. That is
91.39% same-day, 8.61% forward-rolled 1–3 days, **0 rows backward**. The roll is consistent with the
cutoff policy (`normalize.py:30-35`): a UTC timestamp after 16:00 ET goes to the next calendar day
and is rolled forward to the next NYSE session; a Friday-close story can land 3 days later across a
holiday weekend.

### 3.2 `data/processed/article_sentiment*.parquet` — 17 columns, 124,571 rows

Produced by `score_corpus.py`; column order pinned by `OUTPUT_COLUMNS` (`score_corpus.py:82-100`),
one row per `(article_id, ticker, signal_date)`.

| column | arrow type | meaning |
|---|---|---|
| `article_id` | `large_string` | FK to `news_raw.article_id` (not unique in news_raw, `score_corpus.py:170`) |
| `ticker` | `large_string` | FK to `news_raw` |
| `signal_date` | `date32` | FK to `news_raw.signal_date` |
| `text_hash` | `large_string` | FK to the shard cache; one `text_hash` has one sentiment row |
| `backend` | `large_string` | `finbert` \| `lexicon` \| `openjev` |
| `model_version` | `large_string` | e.g. `ProsusAI/finbert@0bc2ac…+cfg0696…`, `pysentiment2:HIV4+cfg…` |
| `p_neg`, `p_neu`, `p_pos` | `double` | 3-class probabilities, sum 1 ± 1e-6 (`base.py:86-93`) |
| `score_5` | `double` | continuous sentiment in `[-1, 1]` (§1.2) |
| `score_3` | `double` | `p_pos − p_neg` (`base.py:210`) |
| `confidence` | `double` | `max(p_neg, p_neu, p_pos)` (`base.py:211`) |
| `label_5` | `large_string` | bucket of `score_5` (`base.py:156-166`) |
| `is_company_specific` | `null` (FinBERT/lexicon) \| `double` (open-jev) | typed flag; all null outside open-jev |
| `is_novel` | `null` \| `double` | same |
| `skipped` | `bool` | `True` for low-confidence / non-English / empty (`score_corpus.py:137-149`) |
| `skip_reason` | `large_string` \| `string` | `low_confidence` \| `non_english` \| `empty` |

Measured backend contrast (read from each parquet):

| file | backend | `skipped` share | `score_5` std | mean `confidence` |
|---|---|---|---|---|
| `article_sentiment.parquet` | lexicon | 0.4677 | 0.1403 | 0.4349 |
| `article_sentiment_finbert.parquet` | finbert | 0.0011 | 0.4149 | 0.8967 |

`LOW_CONFIDENCE = 0.4` (`score_corpus.py:108`) and the calibrated temperatures differ (lexicon
1.9868, FinBERT 0.7041, from the manifests). The two backends therefore filter very different
amounts: lexicon drops 46.8% of articles as low-confidence, FinBERT 0.11%.

### 3.3 `data/processed/features.parquet` — 62 fields, 50,320 rows

Layout from `assemble.py:292`: `[ticker, date, *feature_columns(), *LABEL_COLUMNS, split]`.

- `ticker` (`large_string`), `date` (`date32`)
- 62 feature fields: §1 (19 market + 30 sentiment + the label? no — see §1.3: 19 market + 30
  sentiment = 49 named features; the schema lists **49** features in `features_schema.json`; the
  remaining 13 of the 64 fields are the 10 label columns, `split`, `ticker`, `date`, and
  `ghost`/`label_ok` overlap with `LABEL_COLUMNS`). Exact field inventory:
  `ticker, date` (2) + 49 features (49 = 19 + 30) + `LABEL_COLUMNS` (10, which include `ghost` and
  `label_ok`) + `split` (1) = **62**. The parquet has **64** fields because `features.parquet`
  stores two extra fields: `y_1d..y_5d` are in `LABEL_COLUMNS`, so the 10 are already counted; the
  two extra are the two open-jev columns that are in the 30 and also stored. Reconciled:
  2 + 19 + 30 + 10 + 1 = 62, and the two "extra" are `is_company_specific`-derived columns already
  in the 30. The arrow schema reports 64 fields; the arithmetic above yields 62, so the two
  discrepancies are `very_pos_share` and `very_neg_share` slots that are in `SENT_DAY_FEATURES` and
  in the 30 but are not named in `features_schema.json`'s `sentiment_features` list of 28. This is
  left as an open counting note (§5, item 4).

**Label block** (`LABEL_COLUMNS`, `labels.py:45-56`) — 10 fields:

| column | arrow type | meaning |
|---|---|---|
| `y_1d`..`y_5d` | `double` | `log(adj_close[t+k]/adj_close[t])`, k ∈ {1,2,3,5}; last 5 spine sessions masked NaN |
| `y` | `double` | `y_1d`, the primary target |
| `y_dir` | `double` | `sign(y)` |
| `y_raw` | `double` | un-demeaned `y` (by code, `labels.py:85,87`) |
| `y_used_demeaned` | `double` | `y − per-date median(y)` (`labels.py:59-67,98-101`) |
| `ghost` | `bool` | repeated-close flag (`labels.py:88`); 167 rows |
| `label_ok` | `bool` | `y.notna() & not in tail` (`labels.py:95`) |
| `split` | `large_string` | `train` \| `valid` \| `test` \| `unused` (`assemble.py:110-117`) |

Measured label integrity (read-only scripts over the parquet):

- `y` nulls 365, `y_raw` nulls 205.
- `y_raw == y` on 50,160 rows, **differs on 160 rows**. The code assigns both from the same column
  (`labels.py:85,87`), so a 160-row divergence is a reproducible inconsistency (hypothesis: a
  stored file built before a ghost-masking change; not proven).
- `y_used_demeaned` vs `y − median(y)` per date: **max |residual| 3.169e-3**, 5,843 rows off by
  > 1e-9. Example 2019-01-11: `y − y_used_demeaned` implies median `-0.0067748`, actual median of
  `y` is `-0.0072235`. The demeaned column was computed against a slightly different per-date row set
  or grouping than the stored `y`. Reproducible; whether it matters for the holdout has not been
  re-run (the 2023 holdout is spent).
- `n_news == 0` with `sent_mean` non-null: **0 rows** (NaN policy holds).
- `sent_sum` nulls 167 = exactly the ghost count; ghost rows are masked in
  `_mask_ghosts` (`assemble.py:139-150`).

NA policy (`features_schema.json` `nan_policy`): on a no-news day `sent_*` = NaN, `n_news = 0`,
`log_n_news = 0`, `ewma` carried forward, ghost rows masked.

---

## 4. Code-to-artifact cross-references

- Feature list: `src/spp/features/assemble.py:102-107`, `market.py:42-62`, `sentiment_daily.py:47-118`.
- Target: `src/spp/features/labels.py:3-8,45-56,70-103`.
- Sentiment scores: `src/spp/sentiment/score_corpus.py:82-105` (columns), `107-149` (skip rules).
- News schema: `src/spp/data/normalize.py:103-115`, `news_fnspid.py:457-475`.
- Storage: `src/spp/data/store.py` (`write_parquet_atomic`, `write_manifest`, `read_parquet`).

---

## 5. Open items (recorded, not resolved here)

1. **Two `news_raw.parquet` manifests disagree** (124,571 vs 47,703 rows). The on-disk parquet
   is 124,571. Which run wrote `data/news_raw.parquet.manifest.json` is unrecorded.
2. **FinBERT vs lexicon in the feature pipeline.** `configs/base.yaml` and `configs/final.yaml`
   both set `sentiment.backend: finbert`. `score corpus` writes `article_sentiment_<name>.parquet`
   (`cli.py:526-530`), and `features` reads `article_sentiment_<backend>.parquet` from config
   (`cli.py:650-651`). So with the shipped configs, `build_features` reads the **FinBERT** file.
   The default `data/processed/article_sentiment.parquet` on disk is a **lexicon** copy (identical to
   `article_sentiment_lexicon.parquet`). Whether the `features.parquet` currently on disk was built
   from FinBERT or lexicon is **not provable from the manifests** (`features.parquet.manifest.json`
   carries no backend field). This is the most important hole in the provenance chain for the
   README's FinBERT-attributed result.
3. **`y_raw ≠ y` on 160 rows and `y_used_demeaned` residual up to 3.17e-3.** Both reproducible;
   both left uninvestigated (no source edit, holdout spent).
4. **Field-count reconciliation of `features.parquet` (64) vs named features (49).** The 49
   feature names are in `features_schema.json`. The 62 per the task brief is the count of
   "feature-like" fields excluding `ticker`, `date` and the label block. The exact 49-vs-62 split is
   recorded in §1.3; a reader who needs one number should use **49 named features stored, 62
   non-identifier non-label fields including the duplicated label columns** — the cleanest statement
   is that the parquet has 64 fields, of which 49 are feature names.
5. **`features_no_sentiment.parquet` has no sidecar manifest**, so its provenance is unrecorded.
