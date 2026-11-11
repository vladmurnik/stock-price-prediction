**🇬🇧 English** · [🇷🇺 Русская версия](crypto-news-alpha.ru.md)

# News alpha in crypto: the full study

**What was tested:** whether public news and macro releases can be traded profitably in crypto at realistic costs and realistic reaction latency.

**Short answer:** on Binance one-minute data from 2023 to 2026 — no. Of the fourteen tested ideas, twelve lost money after costs, one (the intraday equity branch) turned out not measurable on the available data, and only CPI releases produced the first positive cell in the project — a candidate, not a strategy. Below is the project's history, every number, and why each branch ended the way it did.

**Scope:** 86,171 headlines from 11 sources, forming 45,625 (headline, symbol) pairs — 21,588 in the cross-section and 24,037 on BTC — plus 714 spot listings, 923 perpetual listings, 528 macro releases and 1,040 minute-level shocks. That is 47,790 measured events, 422 automated tests and 78 independent number checks.

## Contents

**Prologue.** [How the project got here](#prologue-how-the-project-got-here)

1. [Problem statement](#1-problem-statement)
2. [Data](#2-data)
3. [Method](#3-method)
4. [Map of ideas and verdicts](#4-map-of-ideas-and-verdicts)
5. [The ideas in detail: design, result, reason](#5-the-ideas-in-detail-design-result-reason)
   - [5.1. Cross-sectional news](#51-cross-sectional-news)
   - [5.2. BTC absolute return](#52-btc-absolute-return)
   - [5.3. A more complex model: the capacity ladder](#53-a-more-complex-model-the-capacity-ladder)
   - [5.4. Classic market models](#54-classic-market-models)
   - [5.5. Maker execution](#55-maker-execution)
   - [5.6. Structural events](#56-structural-events)
   - [5.7. Spot listings](#57-spot-listings)
   - [5.8. Perpetual listings](#58-perpetual-listings)
   - [5.9. Macro releases](#59-macro-releases)
   - [5.10. Daily and intraday equity branches](#510-daily-and-intraday-equity-branches)
   - [5.11. Source trust](#511-source-trust)
   - [5.12. Post-shock continuation](#512-post-shock-continuation)
6. [Four barriers: why everything ended the same way](#6-four-barriers-why-everything-ended-the-same-way)
7. [Defects found along the way](#7-defects-found-along-the-way)
8. [What is independently confirmed](#8-what-is-independently-confirmed)
9. [Limitations](#9-limitations)
10. [What would change the conclusion](#10-what-would-change-the-conclusion)
11. [Reproduction](#11-reproduction)
12. [Conclusion](#12-conclusion)

**Appendices.** [Experiment registry](#appendix-a-experiment-registry) · [Decision log](#appendix-b-decision-log)


---

## Prologue. How the project got here

This work was assembled in four days — 2026-11-08 … 2026-11-11, 27 commits — but the data inside it
spans 2017–2026. The order of the stages matters: each step was an answer to a specific measurement
of the previous one, not a tour of ideas. Below is the route, with artifacts that can be opened and
re-checked.

| stage | what was built | how it ended |
|---|---|---|
| 1. Daily hypothesis on equities (`dc38f47`) | 124,571 FNSPID articles, 40 US stocks, FinBERT (macro-F1 0.8862), 49 features, 6 purged folds, locked 2023 holdout | raw `sent_mean` rank IC +0.0202 (t +2.71), quintile spread +8.3 bps/day (t +1.78), but holdout IC 0.0127 [−0.0102, 0.0356] and net Sharpe −6.75 |
| 2. Audit of the machinery itself | leakage and feature-coverage checks | two defects were manufacturing a third of the "edge": NaN instead of 0 in sentiment shares and an entirely empty `sector_ret_1`; after the fix CV IC fell 0.0202 → 0.0156 and the interval covered zero |
| 3. Ablations A1–A8 (`99fdf91`, `976d0c1`, `a6e02e9`) | eight feature sets, three sentiment backends, rank combination | every interval contains zero: open-jev 0.0112 against FinBERT 0.0134, lexicon −0.0079, A8 with typed flags 0.0076 against the 0.0156 control |
| 4. Classifier sanity check | perturbation test: 200 headlines with the company name replaced by "the company" | 12/200 = 6.0% sign flips, correlation 0.9747, zero flips among confident scores (\|d\| ≥ 0.2, n = 93) — the null cannot be blamed on sentiment |
| 5. Intraday equity branch (`31da1d3` … `0c57024`) | one article = one trade, entry no earlier than `published_at + 5 min`, horizons 60/120/180 min, 15 bps round trip | 59 events, +2.67 bps at 60 min against an MDE of 18.3 bps, bootstrap 22–30 → "not measurable"; the cause is supply: FNSPID ends 2023-12-16, hourly bars start 2024-11-11 |
| 6. Pivot to crypto (`7396dfd`, `ae3e70f`, `ad52e70`) | keyless Binance minute bars, 24/7 trading, leverage, around-the-clock RSS | 260 minute-level shocks per instrument: no continuation, and at one hour SOL −18.74 and DOGE −24.19 bps (t ≈ −1.9) — the problem is execution, not measurement |
| 7. Pipeline (`8970348`) | 12 symbols × 47 months, 86,171 headlines from 11 sources through the Wayback CDX API, 7 leakage checks | 21,588 events, splits 17,155 / 3,964 / 465, entry lag p50 1.45 min |
| 8. Seven branches and macro (`8e14838` … `eebe4ac`) | cross-section, BTC absolute return, capacity ladder, classic models, maker execution, structural events, listings, perps, macro prints | eight negative families and one positive cell — CPI at 60 minutes |
| 9. Independent audit (`d352c30`, `8e27b24`) | `ops/crypto_verify.py` — 78 checks importing no pipeline module | 78/78 matched; 9 defects found and fixed; the first 6 "mismatches" were bugs in the verifier itself |
| 10. Second pass (`4758e1f`) | window overlap, block bootstrap, funding from Binance archives, slippage ladder | 24.4% of events overlap at one hour (t 1.89 → 1.64), CPI survives (p 0.0068/0.0114), funding +3.73 bps, and no family crosses zero at any slippage level |
| 11. Publication (`7911a12`, `fd0e80c`) | the paper in two languages, datasets in git, artifact indexes | 47,790 measured events, 422 automated tests, 78 independent checks |

**Why the route changed.** Four turns, each caused by a number rather than an opinion.

1. *After equities: maybe the horizon is wrong.* Tested with the intraday branch — and it turned out
   not measurable, not because there is no effect but because news and bars do not overlap. The
   conclusion: the data supply has to change, not the model.
2. *In crypto: where the amplitude actually lives.* The cross-section gave 0.87 bps per hour — 32
   times below costs. So the search moved to where the amplitude is physically larger: structural
   events, listings, macro prints.
3. *In listings: the amplitude exists, but it sits in the auction.* The auction median is +18.14%,
   and the price falls afterwards. Amplitude you cannot enter is not an edge.
4. *In macro: the time is known in advance.* The only branch that does not require guessing the
   moment — and the only positive cell in the project. That is why it stayed a candidate rather than
   a strategy: everything else here has been tested on history, and this has not.

**What was fixed in advance.** `configs/final.yaml` and the 2023 holdout lock file,
`configs/intraday.yaml`, `configs/crypto.yaml`, and `docs/crypto-design.md` with its kill criteria.
No leakage gate was weakened in any experiment, and negative results were published in the same form
as positive ones — including the ones that cancelled text already written.

## 1. Problem statement

The hypothesis: news carries information that reaches the price gradually; whoever reads the headline early and enters takes part of the move.

To make it testable, it was reduced to one number. For every article:

1. **Event.** The publication time `published_at`. Target = `published_at + 1 minute`.
2. **Entry.** The open of the first bar whose `ts_open >= target` — the earliest price available after the news could be read.
3. **Exit.** The close of the bar exactly `h` minutes later: `entry + (h − 1)` minutes.
4. **Label.** `excess = asset return − BTC return` over the same minutes. BTC is the benchmark because the whole market moves together, and the question is about the asset-specific reaction.
5. **Costs.** 14 bps per leg. A spread label pays two legs, so 28 bps per round trip.

**Success was defined in advance:** the mean `excess` after costs must be positive and statistically different from zero when clustered by day, not by individual event.

**Failure:** zero or negative mean after costs, or a positive mean smaller than its own measurement error.

---

## 2. Data

### 2.1. Bars

| source | what | size | period |
|---|---|---|---|
| Binance Vision, spot 1m | 12 symbols (BTC, ETH, SOL, XRP, ADA, DOGE, BNB, LTC, LINK, AVAX, DOT, TRX) | ~2.0M rows per symbol, 690 MB | 2023-01 … 2026-10 |
| Binance Vision, spot 1m, long history | BTCUSDT, ETHUSDT | 4,833,919 and 4,833,918 rows | 2017-08-17 … 2026-10-31 |
| Binance Vision, first month of every spot symbol | 3,767 symbols | one month each | 2017 … 2026 |
| Binance Vision, UM perpetuals, first month | 1,080 symbols | one month each | 2019 … 2026 |

Quality: zero OHLC violations, monotonic timestamps, only exchange-made gaps. The long BTC/ETH history has 34 gaps longer than a minute, the largest 1 day 09:31.

### 2.2. News

| source | how | rows |
|---|---|---|
| Wayback Machine, 6 publishers | resumable harvest of RSS feeds through the CDX API | 86,171 headlines including the live recorder |
| Live RSS recorder, 16 feeds | `first_seen_at` recorded every few minutes | 87 rows at freeze time |

After joining headlines to bars: **45,876 (headline, symbol) pairs**, of which 24,072 are about BTC itself (dropped from the cross-section), 216 have no bar after publication, leaving 21,588 events.

Corpus property: 59% of pairs mention BTC. That is not a collection bug but a market property — BTC appears in almost every article. It is exactly why a separate "BTC absolute return" branch exists.

### 2.3. Macro calendar

528 events: 225 CPI releases, 225 payrolls (NFP), 78 FOMC meetings. Dates were parsed from BLS archives (`bls.gov/bls/news-release/cpi.htm`, `empsit.htm`) and Fed pages; release times are 08:30 America/New_York for CPI and NFP and 14:00 for FOMC, with correct daylight-saving handling (checked on all 528 events: zero mismatches).

---

## 3. Method

### 3.1. Leakage protection

Four gates, never weakened in any experiment:

1. **Causal features.** All 15 baseline features use bars strictly before `entry_ts`: 5/15/60-minute returns, 30/60-minute volatility, the same for BTC, hour of day, weekday, asset code, log latency, headline count in the last 60 minutes.
2. **Splits and embargo.** Train 17,155 / validation 3,964 / holdout 465 / embargo 4. The 2023 holdout was frozen and never used for model selection.
3. **Label validation.** A separate check recomputes every exposure window from raw bars: 11,715 checks, maximum difference 0.0 bps, window length exactly 5.0 minutes.
4. **No buying after the spike.** The missed reaction (news-bar close to entry open) averages −0.00 bps with a median absolute value of 5.66 bps: the system does not systematically buy a price that has already moved.

### 3.2. Costs

The central decision of the project. 14 bps per leg:

- 5 bps taker fee,
- 5 bps spread,
- 4 bps slippage.

An `excess` label is a spread, so it pays **two** legs — 28 bps per round trip. This was a defect at first: the initial version charged 14 bps, overstating every net number by exactly 14 bps per trade.

### 3.3. Metrics

- **Mean excess after costs** — the headline number.
- **Clustered t** by day and by headline (CR1: `Σresid²/n² · k/(k−1)`), because events within a day are dependent.
- **MDE (80% power)** — the smallest effect the design can see: `2.802 · sd / √n`.
- **OOF IC** — out-of-sample rank correlation over expanding purged walk-forward folds.
- **Policy** — how many trades a rule takes and what they earn after costs.

---

## 4. Map of ideas and verdicts

| # | idea | hypothesis | verdict |
|---|---|---|---|
| 1 | cross-sectional news, 5/15/60 min | news moves the asset more than the market | **failed**: +0.09/+0.29/+0.87 bps against 28 |
| 2 | BTC absolute return | BTC news moves BTC, and BTC moves more | **failed**: no drift, abs return 11–37 bps |
| 3 | a more complex model (capacity ladder) | the model is the bottleneck | **failed**: 15 configurations, none closer than 13 bps to 28 |
| 4 | classic market models, combined | off-the-shelf models know something LightGBM does not | **failed**: best cell +4.84 bps gross |
| 5 | maker execution | passive entry cuts costs from 28 to 9 bps | **failed**: adverse selection, −3.6…−5.5 bps before costs |
| 6 | structural events | listings, hacks, bankruptcies move prices harder | **failed**: 30–62 bps against 46 bps for an ordinary comment |
| 7 | spot listings | the first minute is the entry into the move | **failed**: the auction is unreachable, then −2.65% over the hour |
| 8 | perpetual listings | a short catches the post-listing reversal | **failed**: negative mean, the tail eats the median |
| 9 | macro releases CPI/FOMC/NFP | time is known in advance and amplitude is larger | **candidate**: CPI 60 min +28.9 bps, t +2.66 |
| 10 | source trust | some publishers carry a stronger signal and can be weighted | **failed**: the ranking does not replicate, Spearman −0.03…+0.15 |
| 11 | post-shock continuation | a violent move continues | **failed**: the sign flips across instruments, |t| ≤ 1.93, best cell +9.37 bps < 14 |
| 12 | daily hypothesis on equities (method transfer) | yesterday's news sentiment predicts tomorrow's return | **failed**: holdout IC 0.0127 [−0.0102, 0.0356], net Sharpe −6.75 |
| 13 | ablations A1–A8 and sentiment backends | the signal is lost in the model or the lexicon, not in the market | **failed**: every interval contains zero; open-jev 0.0112 < FinBERT 0.0134, lexicon −0.0079, A8 0.0076 against the 0.0156 control |
| 14 | intraday equity branch (one article = one trade) | news decays over hours, not days | **not measurable**: 59 events, +2.67 bps against an MDE of 18.3 bps and a bootstrap of 22–30 |

---

## 5. The ideas in detail: design, result, reason

### 5.1. Cross-sectional news

**Design.** 21,588 events, 11 assets, 11 sources. Label is `excess` over BTC at 5/15/60 minutes. Model is a quantile LightGBM (q10/q50/q90) over 5 expanding purged walk-forward folds with a 60-minute embargo.

**Result.**

| horizon | mean excess | t by headline | t by day | t by (symbol, day) | MDE80 | net at 28 bps |
|---|---|---|---|---|---|---|
| 5 min | +0.09 bps | +0.56 | +0.54 | +0.59 | 0.41 | −27.91 |
| 15 min | +0.29 bps | +1.10 | +1.05 | +1.16 | 0.67 | −27.71 |
| 60 min | +0.87 bps | +1.75 | +1.39 | +1.64 | 1.29 | −27.13 |

OOF IC: 5 min **+0.0009 ± 0.0178**, 15 min **+0.0079 ± 0.0160**, 60 min **+0.0296 ± 0.0311**. The 60-minute policy takes 149 trades out of 3,964 (silent 96.2% of the time), loses −3.21 bps gross and **−1.17 bps net, t −3.01, Sharpe −3.59**.

**Why.** Three measured reasons.

1. **Amplitude.** The asset-specific reaction is 0.09–0.87 bps. Costs are 28 bps. The gap is 30–300 times, not a small edge but a different order of magnitude.
2. **Latency.** Median time from publication to entry is 1.45 minutes, p99 1.98. The missed reaction averages zero but has a median absolute value of 5.66 bps: the move exists, its sign does not.
3. **Features.** All 15 are momentum. On an hourly horizon crypto tends to revert, so the model confidently picks the wrong side: the 60-minute horizon has the highest IC (+0.0296) and the worst money (−1.17 bps). It ranks correctly and trades incorrectly.

**What this rules out.** The "more data will reveal the edge" hypothesis. When the sample grew from 6,626 to 21,588 events, the mean fell from +0.60/+0.81/+1.27 to +0.09/+0.29/+0.87 — by factors of 6.7, 2.8 and 1.5. The estimate collapsed toward zero exactly as noise does when n grows. MDE improved by half (0.8/1.4/2.6 → 0.41/0.67/1.29): we measured more precisely and saw less.

### 5.2. BTC absolute return

**Idea.** 59% of the news is about BTC, and it was being discarded because the benchmark cannot be its own benchmark. But BTC has what alts lack: real amplitude. 24,037 events, absolute return, one leg, 14 bps.

**Result.** Mean drift **+0.21 / +0.08 / −0.41 bps** at 5/15/60 minutes against absolute returns of 11.05 / 18.74 / 37.02 bps. No drift (the mean is smaller than its own confidence interval), but volatility is 10–30 times closer to costs than in the cross-section.

**Model.** OOF IC **+0.0159 ± 0.0230 / +0.0325 ± 0.0203 / +0.0415 ± 0.0241** — for the first time the IC is clearly larger than its own fold-to-fold standard deviation. Policy: 6 / 60 / 329 trades, gross −17.64 / +4.37 / **+5.11 bps**, net of 14 bps −0.04 / −0.11 / **−0.55 bps** (t −1.85).

**Why.** A real signal appeared here for the first time, and it fell exactly short of costs. That is an important intermediate conclusion: the market is not unpredictable; the predictable part is cheaper than the cost of trading it.

### 5.3. A more complex model: the capacity ladder

**Idea.** Maybe the model is the bottleneck. Five configurations on one protocol: baseline (15 features), with derived features (23), deep (63 leaves, 800 trees), one model per asset, a five-seed ensemble.

**Result.** IC at 5/15/60 minutes:

| configuration | 5 min | 15 min | 60 min | best top decile, gross | net |
|---|---|---|---|---|---|
| baseline | +0.0054 | +0.0074 | **+0.0309** | +2.11 | −25.89 |
| + derived | +0.0032 | +0.0140 | +0.0197 | +3.05 | −24.95 |
| deep | −0.0075 | +0.0016 | +0.0239 | **+4.75** | −23.25 |
| per asset | **+0.0138** | **+0.0203** | +0.0276 | +2.12 | −25.88 |
| ensemble | +0.0033 | +0.0137 | +0.0275 | +3.80 | −24.20 |

**Why.** Fold-to-fold IC dispersion is 0.010–0.038, so every value in the table is smaller than its own standard deviation. None of the 15 configurations came within 13 bps of the 28 bps threshold. The bottleneck is not model capacity but the signal-to-cost ratio: a model cannot create information the data does not contain, and even if it could, the amplitude stays far below the threshold.

### 5.4. Classic market models

**Idea.** Use what people have historically traded: logistic regression on direction, ridge, random forest, histogram gradient boosting, quantile boosting — and combine them as a cheap stand-in for a metamodel.

**Result.** Six technical indicators were added, each computed strictly before entry: RSI(60), MACD(12,26)/price, Bollinger z-score, ATR(14)/price, volume z-score and position in the 60-bar range.

| model | IC 5/15/60 | best top decile, gross | net |
|---|---|---|---|
| logit | **+0.0139 ± 0.0077** / +0.0168 / +0.0138 | +0.17 | −27.83 |
| ridge | −0.0029 / +0.0074 / −0.0019 | +3.79 | −24.21 |
| forest | +0.0020 / +0.0199 / +0.0043 | +4.64 | −23.36 |
| histogram GBM | −0.0053 / +0.0083 / −0.0007 | **+4.84** | **−23.16** |
| quantile GBM | −0.0008 / +0.0080 / +0.0175 | +3.71 | −24.29 |
| combination | −0.0028 / **+0.0195** / +0.0099 | +2.10 | −25.90 |

**Why.** Logit gives the most stable IC (t ≈ 4) and the worst money: it ranks the probability of an up move, while the payoff depends on amplitude. The combination is worse than the best single model because it averages models with different error signs and cancels signal together with noise. No model left the 23–28 bps loss corridor.

### 5.5. Maker execution

**Idea.** The one lever that can flip the sign at the current signal: stop paying the taker fee. A limit at the close of the news bar, a five-minute window, a market exit. Costs 9 bps instead of 28.

**Result.** 86% of limits fill — a higher rate than expected — and every filled trade loses:

| horizon | long | short |
|---|---|---|
| 5 min | −4.35 bps (net −13.35, t −45.1) | −4.84 (net −13.84) |
| 15 min | −4.22 | −4.44 |
| 60 min | −3.64 | −5.52 |

**Why.** Pure adverse selection. A buy limit fills only if the price came back down — and it usually keeps falling; a sell limit fills only if the price rose — and it keeps rising. The 14% that never fill are exactly the cases where the move went our way. The 19 bps saved on fees is more than eaten by selection: −4 bps in price against +19 bps in fees cannot be reconciled.

**Conclusion.** Passive execution on a news signal is not an optimisation; it changes the strategy into its opposite.

### 5.6. Structural events

**Idea.** Not all news is equal: a listing, a hack, a bankruptcy, an SEC action, an ETF decision should move the price harder than an analyst comment. A headline classifier with 11 classes over 45,625 events.

**Result.** Absolute 60-minute return:

| class | n | abs return 60 min, bps | directional effect, bps (t) |
|---|---|---|---|
| unlock | 235 | 62.3 | −4.47 (−0.66) |
| sec_action | 238 | 57.1 | −13.08 (−1.23) |
| listing | 59 | 56.9 | −6.42 (−0.65) |
| partnership | 424 | 52.7 | +4.28 (+1.05) |
| insolvency | 224 | 51.8 | **+13.48 (+1.63)** |
| upgrade | 302 | 48.1 | −4.55 (−1.03) |
| etf_approval | 419 | 43.6 | +0.47 (+0.12) |
| hack | 357 | 42.5 | −1.93 (−0.44) |
| etf_rejection | 110 | 40.6 | −3.12 (−0.50) |
| delisting | 36 | 30.6 | −8.54 (−1.23) |
| **ordinary commentary** | **42,145** | **46.2** | — |

**Why.** The main result of this branch is negative and unexpected: structural events do **not** produce an order-of-magnitude larger amplitude. The range is 30–62 bps; ordinary commentary is 46.2 bps. The reason is that the 12 majors are assets where any headline drowns in market noise, while the real move happens in the token that is not in our bars (see 5.7).

**A note on contamination.** The first classifier version counted 206 of 427 "insolvency" rows that contained "liquidated" — market recaps such as "$250M Worth Top Crypto Liquidated In 24Hrs". The published +12.40 (t +2.14) became +13.48 (t +1.63) on 224 rows after cleaning: the magnitude grew, the significance disappeared. This is why keyword classifiers need manual inspection.

### 5.7. Spot listings

**Idea.** The strongest event in crypto is a token being listed on a major exchange. There the amplitude is 5–30%, not basis points. The event comes from the archive itself: the first bar of a symbol is the moment trading opened.

**The defect found (P0).** The first version entered at the open of the first bar and showed +337% per minute on average. That was an error: the first bar is an auction where `open` is a nominal print. Examples: 1INCH on 2020-12-25 — `open 0.200`, `high 2.900`, `close 2.500`; 0G — `open 1.000`, `high 5.186`; 1000SATS — `open 0.000243`, `high 0.000674`. Nobody bought at 0.20.

**Corrected design.** Entry at the open of the **second** bar. The auction is reported separately as an unreachable ceiling: median +18.14%, mean +405.42%.

**Result after the fix** (714 events, long, 14 bps):

| horizon | 1 min | 5 min | 15 min | 60 min | 240 min | 1440 min |
|---|---|---|---|---|---|---|
| return | +0.21% | +0.01% | −0.75% | **−2.65%** | −3.80% | **−5.44%** |
| t | +0.70 | — | −1.35 | **−3.56** | −4.53 | −4.63 |

With a 15-minute delay (the project's median latency): 60 min −2.16% (t −4.25), 1440 min −5.25% (t −5.32). By year at 60 minutes: −0.73, −4.96, −6.02, −4.33, −1.19, +0.57, −3.47, −0.65% (2019–2026); the only positive year is 2024, t +0.23.

**Why.** The auction absorbs the move. What follows is not continuation but systematic reversion: the first-minute buyers are those exiting the auction, and they sell. "Buy the fresh listing" is negative at every horizon and in every year but one.

### 5.8. Perpetual listings

**Idea.** If the price falls after the auction, it can be sold. Spot cannot be shorted without a loan; a perpetual can. 923 events, short from the second minute.

**Result** (short return, 14 bps):

| horizon | 1 min | 5 min | 15 min | 60 min | 240 min | 1440 min |
|---|---|---|---|---|---|---|
| mean | −0.26% | −0.54% | −0.80% | −0.61% | −0.86% | −0.65% |
| t | −2.15 | −3.28 | −3.43 | −1.79 | −1.62 | −0.71 |

Risk: median maximum rise in the first hour +3.37%, p90 +19.15%, **p99 +51.33%**. A 5x leverage account is liquidated in 9.3% of events, 3x in 3.7%.

**Funding was measured separately** (Binance archives, 116 perps, first day): a short **receives** +3.73 bps on average (median +3.00, receives in 68% of cases), but the spread is wide: p10 −71.4, p90 +67.2.

**Why.** The median is positive (+1.50% per day), the mean is negative (−0.65%). The difference is the tail: a handful of listings multiply, and the short loses more than it earns on the rest. This is not a strategy but a lottery with a margin call: a positive median with a negative mean means rare events destroy the account.

### 5.9. Macro releases

**Idea.** The only family of events with a time known in advance. No need to catch a headline — the release time is known months ahead, and the entry can be placed beforehand.

**Design.** 528 calendar events, 291 measured per symbol (109 CPI, 109 NFP, 73 FOMC). Entry at the open of the first minute after the release. Direction is the sign of the release candle. Exit 60 minutes later (and other horizons).

**Amplitude** (BTC, 60-minute absolute return versus the same hour on an ordinary day):

| event | abs return 60 min | baseline | ratio |
|---|---|---|---|
| CPI | 74.5 bps | 51.1 | **1.46×** |
| FOMC | 93.0 bps | 44.0 | **2.11×** |
| NFP | 58.5 bps | 51.1 | 1.14× |

**Momentum in the direction of the release** (BTC/ETH, entry +1 minute, 14 bps):

| event | BTC 60 min | t | net | ETH 60 min | t | net |
|---|---|---|---|---|---|---|
| CPI | **+28.65 bps** | **+2.66** | **+14.65** | **+35.32** | **+2.48** | **+21.32** |
| NFP | −4.30 | −0.50 | −18.30 | −2.88 | −0.30 | −16.88 |
| FOMC | −10.34 | −0.62 | −24.34 | −7.02 | −0.35 | −21.02 |

**Robustness checks.**

- *Cost of lateness:* CPI BTC +28.65 / +20.81 / +11.02 at a 1 / 5 / 15-minute delay. At 15 minutes the effect is below costs — a direct confirmation that the news branch was losing because of latency.
- *Eras (CPI 60 min):* BTC 2017–2020 −0.47 (t −0.03), **2021–2023 +72.15 (t +3.14)**, 2024–2026 +16.49 (t +1.31); ETH −15.07 / **+87.81 (t +3.13)** / +39.15 (t +1.91). The effect lives in the era when CPI was the market's main event.
- *Jump-size terciles (CPI):* ETH top tercile **+99.17 bps (t +3.28)**, bottom −5.36. The larger the surprise, the stronger the continuation — consistent with underreaction rather than noise.
- *Bootstrap by release day (10,000 resamples):* BTC +28.91 bps, interval [+8.37, +50.14], p 0.0068; ETH +36.67, [+8.58, +65.25], p 0.0114. The effect survives resampling.
- *BTC/ETH correlation:* r ≈ 0.59, same sign in 68.8% of cases. Not two independent instruments but agreement on correlated ones.
- *Multiple testing:* 36 cells, Bonferroni threshold |t| ≈ 3.2. No cell passes it; 13 of 36 are net positive, and only two have |t| > 2, both CPI at 60 and 15 minutes.

**Why.** Three conditions coincided here that no other branch had: time known in advance (no latency), amplitude 1.5 times an ordinary day, and continuation in the direction of the first print rather than reversion. But the effect rests on 108 observations and one era, so it is a candidate requiring forward validation.

### 5.10. Daily and intraday equity branches

The method was first built on equities, which explains where every convention in the crypto work came from: purged folds, an embargo, a frozen holdout, and costs per round trip rather than per leg.

**Daily branch.** 124,571 FNSPID articles, 40 US stocks, FinBERT (macro-F1 0.8862), 49 features, 6 purged folds, one locked 2023 holdout. The raw feature worked: `sent_mean` gave a rank IC of +0.0202 (t +2.71) over 1,250 days, `ewma_24h` +0.0206 (t +2.77), and the Q5−Q1 quintile spread +8.3 bps per day (t +1.78). But the model on the holdout gave IC 0.0127 [−0.0102, 0.0356], verdict `not_confirmed`, and the economic test gave net Sharpe −6.75 at ~166× annual turnover and 5+5 bps per side (~33% a year in costs against ~1% gross edge). The GBDT was **worse** than the raw feature: the model added nothing to what the average sentiment already contained.

**Ablations.** Eight feature sets and three backends (table below). No interval excludes zero; the lexicon is worse than nothing; the rank combination of A2 and A3 gives 0.0137 — between its inputs and below the A5 control on the same rows (0.0156); the two columns only open-jev can supply make things **worse** (A8 0.0076).

| id | features | IC | 95% CI |
|---|---|---|---|
| A1 | market only (19) | 0.0007 | [−0.0151, 0.0165] |
| A2 | sentiment only (28) | 0.0134 | [−0.0052, 0.0320] |
| **A5** | market + sentiment (49) | **0.0156** | [−0.0069, 0.0381] |
| A4 | lexicon instead of FinBERT (49) | −0.0079 | [−0.0228, 0.0069] |
| A6 | strict_dates = true (49) | 0.0167 | [−0.0021, 0.0356] |
| A3 | sentiment only, open-jev (30) | 0.0112 | [−0.0185, 0.0408] |
| A7 | rank average of A2 and A3 | 0.0137 | [−0.0160, 0.0435] |
| A8 | + typed flags, open-jev (49) | 0.0076 | [−0.0068, 0.0219] |

**Classifier check.** A perturbation test: in 200 headlines the company name was replaced by "the company" and re-scored with the same FinBERT. Sign flips: 12/200 = 6.0%, correlation between the original and perturbed score 0.9747, and among confident scores (|d| ≥ 0.2, n = 93) **zero** flips. Direction comes from the content, not from the company name — so the null cannot be blamed on the sentiment layer.

**Intraday branch.** One article = one trade, entry at the open of the first bar no earlier than `published_at + 5 min`, exit at the close of the bar containing `entry + horizon` (60/120/180 min), target = stock return minus SPY, 15 bps round trip. Result: 59 events, +2.67 / −1.13 / −11.37 bps, MDE 18.3 / 20.6 / 26.0 bps, session bootstrap 22–30 bps. Verdict — **not measurable**: the sample cannot resolve anything below 18 bps, while the plausible size of a news effect is 8–10 bps. Three facts forbid stronger wording: the median entry delay is 75 minutes rather than five, 35 of 59 events fall on a single session, and FNSPID ends 2023-12-16 while hourly bars start 2024-11-11 — the two windows do not overlap at all.

### 5.11. Source trust

**Idea.** Eleven publishers produce streams of different quality. If some of them carry a stronger
average reaction, events could be weighted by source — or at least the negative contributors dropped.

**Design.** 21,588 events. A source's `alpha` is its mean return shrunk toward the global mean with
weight K = 50 events: `alpha = n/(n+K)·mean + K/(n+K)·global`. Stability is tested by splitting the
corpus in half by time: does the ranking of sources agree between the halves?

**Result.** The best source in sample is CoinDesk: +1.13 bps at 5 minutes (t +2.19), +2.61 bps at
60 (t +1.64). The worst at 5 minutes is Cointelegraph: −1.07 (t −2.60). Across 11 sources at 60
minutes the range is −2.94 (cryptoslate) to +2.61 bps — a 5.5 bps spread between best and worst.

**Stability.** Spearman between halves: −0.03 at 5 minutes, +0.10 at 15, +0.15 at 60. The same sign
is kept by 5 of 9 sources at 5 minutes, 4 of 9 at 15 and 3 of 9 at 60. The ranking in the first half
does not predict the ranking in the second.

**Why.** The best-to-worst spread is 5.5 bps per hour: five times smaller than the 28 bps cost and
smaller than the corpus's own half-to-half variation. Even perfect knowledge of the "right" source
does not turn the branch positive, and the "right" source changes from half to half. This is a more
general lesson: source trust cannot be measured on the same corpus it is going to be applied to —
it needs an external, pre-held-out period.

### 5.12. Post-shock continuation

**Idea.** Separately from news: a violent one-minute move may continue. A 2σ bar is not noise but
the start of a directional move. This hypothesis needs no text, so it does not hit the latency
barrier — the only branch where latency is irrelevant by construction.

**Design.** The threshold is 2σ of one-minute volatility per instrument: 29.3 bps for BTC, 41.5 for
ETH, 46.7 for SOL, 55.1 for DOGE. The 260 strongest minutes per instrument (0.2% of time) are taken
long in the direction of the shock, exiting after 1/5/15/60 minutes, at a cost of one leg — 14 bps.

**Result.** Continuation in bps (t):

| instrument | 1 min | 5 min | 15 min | 60 min |
|---|---|---|---|---|
| BTC | −0.56 (−0.36) | +0.64 (+0.24) | +3.65 (+0.98) | +5.39 (+1.02) |
| ETH | +1.46 (+0.53) | +2.00 (+0.49) | +5.06 (+0.85) | +9.37 (+1.26) |
| SOL | +1.24 (+0.39) | −4.00 (−0.78) | −1.45 (−0.22) | −18.74 (−1.90) |
| DOGE | −0.55 (−0.10) | −2.96 (−0.42) | −10.46 (−1.26) | −24.19 (−1.93) |

**Why.** The sign flips from instrument to instrument: BTC and ETH show weak continuation, SOL and
DOGE weak reversion. Win rates are 0.40–0.49: there is not one instrument where the direction is
guessed better than a coin. The largest |t| is 1.93, and it is negative. Even the best cell (+9.37
bps on ETH over an hour) is below the single-leg cost of 14 bps. This is exactly what a market
should produce when one-minute shocks are mostly liquidity noise: without a text there is no reason
for the move to continue.

---

## 6. Four barriers: why everything ended the same way

Different ideas hit different walls, but there are only four walls.

### Barrier 1: costs

28 bps per round trip is not a pessimistic assumption but the real cost of a two-leg taker trade. For a strategy to make sense, the average predictable amplitude must be at least 30–40 bps. Measured amplitudes: 0.1–0.9 bps (cross-section), 11–37 bps (BTC), 30–62 bps (structural), 0.6–2.7% (listings).

### Barrier 2: latency

Median time from publication to entry is 1.45 minutes, and part of the move happens in that window. Measured directly on macro: after 15 minutes, +28.65 becomes +11.02, and the effect falls below costs. Any strategy that requires reading text is on the wrong side of this barrier.

### Barrier 3: amplitude

Amplitude grows with illiquidity: the 12 majors have none, a fresh listing has enormous amplitude — but the auction has already absorbed it. The middle zone (mid-cap alts) has not been tested systematically, because it requires linking a headline to a specific token.

### Barrier 4: wrong instrument

News about token X is traded on BTC. Even when the information is right, the instrument is wrong: X's specific reaction does not pass into BTC. This explains why 45,625 structural events produced an amplitude like ordinary commentary — we were looking at the wrong asset.

---

## 7. Defects found along the way

The study is valuable not only for its numbers but for the list of what was fixed. Every defect biased the result toward optimism.

| # | defect | how it showed | effect of the fix |
|---|---|---|---|
| 1 | fee charged for one leg instead of two | net overstated by 14 bps per trade | 60-min net −1.17 instead of +0.83 |
| 2 | `legs=2` never reached the policy table | net computed at 14 bps | fixed, difference exactly 14 bps |
| 3 | entry into the listing auction | +337% mean per minute | −2.65% over the hour after the fix |
| 4 | BTC features empty in the ladder scripts | 4 of 15 features dead | numbers rerun |
| 5 | "insolvency" 48% liquidation recaps | +12.40 (t 2.14) | +13.48 (t 1.63) — significance gone |
| 6 | ETF approval tested before rejection | 13 headlines with a flipped sign | rejection-first |
| 7 | hardcoded "only positive cell" | contradicted the 13/36 table | computed from data |
| 8 | source-table horizon label | a 15-minute table labelled 5 minutes | label fixed |
| 9 | MDE published without clustering | 0.41/1.29 instead of 0.47/1.76 | caveat added |

Plus an independent verifier: `ops/crypto_verify.py` recomputes 78 numbers with its own arithmetic and imports no pipeline module. Its first run produced 6 "failures", and all six were bugs in the verifier itself — a separate, important result: the checker catches itself first.

---

## 8. What is independently confirmed

`ops/crypto_verify.py` — 78 checks, 78 matches:

- dataset: 21,588 rows, splits, means, t, `excess = ret − bench` (max |Δ| = 0.0), latency;
- listings: 714 events, entry = second bar in all, −2.65% and −5.44%;
- perps: 923 events, −0.26 / −0.61%, tail +51.33%, liquidations 9.3%;
- macro: 528 events, 109 CPI, momentum +28.65 at t +2.66 (to the fourth digit; the only discrepancy came from a single doji, whose sign must be zero);
- sources: 11 sources, n/mean/t;
- structural: 45,625 events, insolvency 224, +13.48.

`ops/crypto_verify2.py` — four additional questions:

- **overlap:** 24.4% of 60-minute events share their window with another headline on the same symbol; clustering by (symbol, day) lowers t from 1.89 to 1.64 — the conclusion stands;
- **CPI bootstrap:** holds (p 0.0068 / 0.0114);
- **funding:** +3.73 bps in the short's favour on the first day; it does not rescue the strategy;
- **slippage:** no family crosses zero at an additional 0/5/10/20 bps.

---

## 9. Limitations

An honest list of what is missing:

1. **Market impact.** Trades are assumed not to move the price. At size, that is false.
2. **Position overlap.** Events may require capital simultaneously; there is no portfolio accounting.
3. **Funding for the cross-section.** Only measured for perps on the first day.
4. **News sources.** 6 archived publishers; GDELT and several feeds were unavailable. The corpus is CoinDesk-heavy.
5. **Structural classifier** is keyword-based, without a manually labelled precision/recall estimate.
6. **Macro is one era.** The CPI effect lives in 2021–2023; Binance one-minute bars do not exist before 2017.
7. **The 465-event holdout** is untouched but underpowered: its MDE is tens of basis points.
8. **Latency.** Measured live at 15.9 minutes median for RSS, while the backtest uses 1.45 minutes — an optimistic assumption that works against the strategy, not in its favour.

---

## 10. What would change the conclusion

The hypotheses are stated so they can be refuted:

1. **Forward CPI test.** 12 releases, rule registered in advance: entry in the first minute in the direction of the release candle, exit after 60 minutes, BTC and ETH, 14 bps. If the mean net > 0 and t > 1.5, the effect lives; otherwise the hypothesis closes. Three to four months, one day of work.
2. **Upbit listings.** 811 announcements since 2017 with second-level timestamps; the coin already trades on Binance. The only place where amplitude (5–30%) is orders of magnitude above costs and entry is possible in the right direction.
3. **Linking a headline to a token.** A resolver from headline ticker to exchange symbol plus bars around the event. Without it, barrier 4 stays up and the news branch cannot pay in principle.
4. **Funding carry as an instrument control.** A known positive effect. If the pipeline cannot find it, the problem is measurement, not the market.

What not to repeat: more model complexity, metamodels, maker execution on alts, structural events on majors, token unlocks (sources closed and paid).

---

## 11. Reproduction

```bash
# bars (12 symbols, 47 months)
python ops/crypto_build_bars.py

# news
python ops/crypto_wayback.py --start 2023-01-01 --end 2026-11-09
python ops/crypto_news.py --once

# dataset and leakage checks
python ops/crypto_dataset.py

# models, policy, sources
python ops/crypto_train.py
python ops/crypto_sources.py
python ops/crypto_audit.py

# branches
python ops/crypto_btc.py
python ops/crypto_ladder.py
python ops/crypto_classic.py
python ops/crypto_maker.py
python ops/crypto_structural.py
python ops/crypto_listings.py --workers 12
python ops/crypto_listing_perp.py --workers 10

# macro
python ops/crypto_macro_bars.py
python ops/crypto_macro.py

# independent checks
python ops/crypto_verify.py
python ops/crypto_verify2.py
```

Data too large for git (one-minute bars, ~1 GB) is rebuilt by the first command; derived datasets (events, listings, structural) live in `data/crypto/` and are included in the repository.

---

## 12. Conclusion

The study did not find a way to make money from news at realistic costs. It found something more useful: the precise boundary where signal ends and noise begins.

Three numbers describe the whole result:

- **0.87 bps** — the average asset-specific reaction to a headline over an hour;
- **28 bps** — the cost of entering and exiting;
- **1.45 minutes** — the median latency by which the market has already understood.

Until those three numbers change (cheaper execution, faster reaction, or a different amplitude), the news branch will stay negative no matter the model. The only two places where the arithmetic works out are macro releases with a known timestamp and listings on a second exchange; both are stated as testable hypotheses with a deadline and a stopping rule.

A negative result with reproducible numbers is a result. It saves time for the next person who wants to test the same idea: now it is known which checks must be done before, not after, a model shows a beautiful backtest.

---

## Appendix A. Experiment registry

Every script in `ops/` is one question asked, one artifact produced and one verdict reached. The
table exists so that a reader can check that nothing is missing and nothing was rewritten after the
fact.

| script | question | artifact | verdict |
|---|---|---|---|
| `crypto_probe` | are the exchanges and archives reachable? | `results/crypto_probe.json` | infrastructure: 200 from every source |
| `crypto_build_bars` | minute bars for 12 majors, 47 months | `data/crypto/bars/1m/*.parquet`, `results/crypto_bars_quality.json` | 0 OHLC violations |
| `crypto_macro_bars` | long BTC/ETH history from 2017-08 | `data/crypto/bars/1m_macro/*.parquet` | ~4.83M rows per symbol |
| `crypto_wayback` | archived headline corpus | `data/crypto/news/wayback.parquet` | 86,171 headlines |
| `crypto_news` | live first-seen recorder | `data/crypto/news/recorder.parquet` | 87 rows, p50 latency 15.9 min |
| `crypto_dataset` | event assembly and leakage checks | `data/crypto/dataset/events.parquet`, `results/crypto_dataset.json` | 21,588 events, 7 checks PASS |
| `crypto_audit` | exposure, lag, benchmark, duplicates | `results/crypto_audit.json` | 11,715 label checks, max \|Δ\| = 0.0 |
| `crypto_train` | quantile model and policy | `results/crypto_models.json`, `crypto_policy.md` | net −1.17 bps, t −3.01 |
| `crypto_sources` | do sources differ in quality? | `results/crypto_sources.json`, `.md` | the ranking does not replicate |
| `crypto_ladder` | is model capacity the bottleneck? | `results/crypto_ladder.json`, `.md` | 15 configurations, best net −23.25 |
| `crypto_classic` | are off-the-shelf models stronger? | `results/crypto_classic.json`, `.md` | best net −23.16 |
| `crypto_maker` | is passive entry cheaper than taker? | `results/crypto_maker.json`, `.md` | adverse selection, −3.6…−5.5 bps before costs |
| `crypto_btc` | BTC on absolute return | `results/crypto_btc.json`, `crypto_btc_policy.md` | gross +5.11, net −0.55, t −1.85 |
| `crypto_structural` | do structural events move harder? | `results/crypto_structural.json`, `.md` | 30–62 bps against 46.2 for a comment |
| `crypto_listings` | is a spot listing an entry into the move? | `results/crypto_listings.json`, `.md` | −2.65% over the hour (t −3.56) |
| `crypto_listing_perp` | does a perp short catch the reversal? | `results/crypto_listing_perp.json`, `.md` | mean −0.61%, p99 +51.33% |
| `crypto_macro` | are macro prints predictable? | `results/crypto_macro.json`, `.md` | CPI +28.65 bps (t +2.66) |
| `crypto_shock` | does a shock continue? | `results/crypto_shock.json` | \|t\| ≤ 1.93, the sign flips across instruments |
| `crypto_verify` | do the numbers reproduce independently? | `results/crypto_verification.json`, `.md` | 78/78 PASS |
| `crypto_verify2` | overlap, bootstrap, funding, slippage | `results/crypto_robustness.json`, `.md` | the conclusion does not move in a single row |

The complete artifact list with its scripts is in `results/README.md`.

## Appendix B. Decision log

Eight decisions determined every number in this work. They are recorded together with the
alternative that was rejected — otherwise, a month later, it is impossible to see why the numbers
are what they are.

| decision | rejected alternative | reason | recorded in |
|---|---|---|---|
| BTC as the cross-sectional benchmark | raw asset return | the market moves together; the question is the specific reaction, not beta | §1, `events_btc` |
| BTC cannot be its own benchmark | keep BTC in the cross-section | a BTC−BTC spread is identically zero | §2.2, `benchmark_asset` |
| two cost legs, 28 bps | one leg, 14 bps | the label is a spread of two positions; the first version overstated net by exactly 14 bps | §3.2, `crypto_policy.md` |
| entry at the first bar after publication | a "safe" five-minute delay | five minutes is an assumption, not a measurement; the actual lag is p50 1.45 min | §1 |
| entry at the second listing bar | entry at the first bar | the first bar is an auction (1INCH: open 0.200, high 2.900); the mistake showed +337% | §5.7 |
| ETF: rejection tested before approval | "approval first" ordering | 13 headlines received the opposite sign | §7 |
| leverage is position sizing, not edge | "leverage will increase the return" | `net = L·(sign·excess − cost)`; the sign does not change and the tail grows | §3.2 |
| publish negative results | cherry-pick the favourable cells | otherwise the project measures luck, not the market | the whole paper |

One more decision did not fit the table because it is about the work itself: the 2023 holdout was
frozen and never used for a single choice — neither in equities nor in crypto.
