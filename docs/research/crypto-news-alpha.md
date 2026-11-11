**🇬🇧 English** · [🇷🇺 Русская версия](crypto-news-alpha.ru.md)

# News alpha in crypto: the full study

**What was tested:** whether public news and macro releases can be traded profitably in crypto at realistic costs and realistic reaction latency.

**Short answer:** on Binance one-minute data from 2023 to 2026 — no. Five of the six idea families lost money after costs; the sixth (CPI releases) produced the first positive cell in the project and remains a candidate, not a strategy. Below is the whole path, every number, and why each branch ended the way it did.

**Scope:** 86,171 headlines from 11 sources, 21,588 cross-sectional events, 24,037 BTC events, 45,625 structural events, 714 spot listings, 923 perpetual listings, 528 macro releases. About 120,000 measured events, 422 automated tests, 78 independent number checks.

---

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
| 10 | daily and intraday equity branches | transfer the method to stocks | **failed**: IC 0.0127 unconfirmed; intraday not measurable |

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

The method was first built on equities. The daily branch: 2023 holdout, IC 0.0127, confidence interval [−0.0102, 0.0356], verdict `not_confirmed`, net Sharpe −6.75. The intraday branch: 59 events over 6 sessions, effect +2.67 bps against an MDE of 18–22 bps — **not measurable** with the available data, so it was published as "not measured", not as "does not work".

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
python ops/crypto_bars.py

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
