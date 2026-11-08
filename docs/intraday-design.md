# Intraday event-driven design: one article = one trade

Status: **design, not a result.** Nothing in this document has been measured. It supersedes
`docs/idea-intraday-t5.md` (the earlier t+5 note) and describes the experiment that is being built
now: a cross-sectional signal on the **excess intraday return of a single news article over the same
minutes of SPY**, evaluated on events, with stated costs.

The reader this is written for is one who will try to break it. Every quantity below is either
(a) a definition pinned to a named module, (b) a number copied from an artifact on disk with its
path, or (c) marked **planned** with the exact intended file path. There is a fourth state —
**not measurable on this machine** — and it is the honest status of the target itself; §3 says so in
full.

The model does not predict prices. It ranks events; the claim under test is that the sign/rank of
`y_event` (§1) is predictable out of sample after costs.

---

## 1. Exact definitions

All times are ET (`America/New_York`, DST-aware), matching `src/spp/data/normalize.py:ET`. Bars are
left-labelled: `bar.start` is the timestamp of the first trade in the bar; `bar.end = bar.start +
interval`.

### 1.1 Event unit

An event is one `(article_id, ticker, published_at)` triple surviving dedupe (§1.6). The corpus
already carries one row per `(article_id, ticker)`: `src/spp/data/normalize.py:NEWS_COLUMNS`
(`article_id, ticker, published_at, date, signal_date, title, title_norm, url, publisher, source,
text_hash`).

Daily granularity of `published_at`, **measured** (`data/interim/news_raw.parquet`): 0 NaT, range
`2019-01-01 00:00:00+00:00` … `2023-12-16 23:02:00+00:00`; 99.766 % of stamps have `minute == 0`,
99.769 % have `second == 0`. So for all but ~288 rows the stamp is **hour-resolution**, i.e.
`published_at` is the *start of the publishing hour*, and the true publication is somewhere in
`[published_at, published_at + 60 min)`. This is a fact about the data, not a bug in code, and §6.2
gives the direction of bias it creates.

### 1.2 `entry_ts`

```
session(p)      = NYSE session containing p, from pandas_market_calendars
                  (same calendar as src/spp/data/normalize.py:nyse_trading_days)
open(s)         = 09:30 ET on session(s);   close(s) = 16:00 ET on session(s)
eligible(p)     = exists a bar b with b.start >= p + 5 min and b.start + interval <= close(session(p))
entry_cutoff(p) = close(session(p)) - MAX_HOLD,     MAX_HOLD = 180 min
entry_ts        = min{ b.start : b.start >= p + 5 min, b.start <= entry_cutoff(p), b on session(p) }
event_valid     = published_at is known to minute resolution
                  AND eligible(p)
                  AND entry_ts is not None
                  AND a bar for the ticker AND for SPY exists at entry_ts
```

Rules that are not negotiable in the implementation:

* `>=`, not `>`. The original note wrote `>=` and the loader must not turn it into `>`.
* The `+5 min` lag is an **execution** lag (time to read the headline after it prints), not an
  information lag. It is measured from `published_at`, never from a bar boundary, so the result does
  not depend on bar alignment.
* `entry_cutoff` exists so that every event has the same allowed holding window; events published
  too late for a 180-minute hold are dropped rather than silently shortened. This is what makes the
  60/120/180 comparison a comparison and not three different event sets.
* An event whose `p` falls after `close(s)` or on a non-session day is **not** an intraday event;
  it belongs to the daily/carry experiment and is excluded here (it would otherwise be an
  overnight trade, which is a different question and would import the daily experiment's defects).

Intended module: **planned** `src/spp/live/events.py` (functions `build_entry_ts`, `build_events`).
Nothing in `src/spp/live/` exists today except the docstring-only `__init__.py`.

### 1.3 `exit_ts`

Let `E = entry_ts` and `H ∈ {60, 120, 180}` minutes.

```
exit_ts(H) = min( E + H min , last bar with b.start + interval <= close(session(E)) )
```

So the exit is at most `H` minutes after entry and never after the session close of the **same**
session. No overnight hold, no next-session carry. If the bar at `exit_ts` is missing for the ticker
or SPY, the event is invalid for that `H` (not filled forward — the daily backtest already takes the
same stance, `src/spp/backtest/portfolio.py:build_execution_frame` marks a missing exit as `halted`
rather than carrying the last price).

`H` is a **parameter of the test**, fixed before the run. The plan is to evaluate all three and
report all three; picking the best `H` after the fact converts the test into a multiple-comparison
problem and must be declared in the report if it happens (§6.9).

### 1.4 `y` — excess return over SPY, same minutes

Prices are **raw** `open` of the entry bar and **raw** `close` of the exit bar. Raw, not adjusted,
for the same reason as the daily backtest: `adj_close` is retroactively rescaled by dividends/splits
and is not a transaction price (`src/spp/backtest/portfolio.py` docstring, `_price_returns`).

```
r_i      = exit_close_i / entry_open_i - 1
r_spy    = exit_close_SPY / entry_open_SPY - 1
y_event  = r_i - r_spy
r_log    = log((1 + r_i) / (1 + r_spy))          # diagnostic only, not the target
```

Why raw difference, not log difference: log excess `log(1+r_i) - log(1+r_spy)` is a different
number by `O(r²)` (~10⁻⁶–10⁻⁵ at 2-hour horizons) and neither is "wrong"; picking one and sticking
to it is what matters. The simple difference is used because it is the number a hedged book actually
lives with: long $1 of the name, short $1 of SPY.

Why SPY and not a sector ETF: SPY is the cheapest, most liquid hedge and its bar exists in every
`yfinance` interval. A sector hedge would remove more market variance but a sector ETF's 1-minute
bars are thinner than SPY's and the hedge leg would be the noisiest part of the trade. Sector-hedged
`y` is a possible later variant, not the target.

### 1.5 Trend filter (optional, two independent switches)

Both switches read **daily** data only and only data strictly before the event session:

```
T1 (trend, per name)  : close[ticker, d-1] > SMA20(close[ticker], window d-20 … d-1)
                        SMA20 = mean of the last 20 daily closes, min 15 observations
T2 (market, same day) : sign(close(SPY, d) - open(SPY, d)) == sign(score_event)
                        i.e. trade with the market's intraday direction only
```

* `d` = session of `entry_ts`. `T1` uses `d−1`'s close against an SMA that ends at `d−1`; this is
  exactly the construction already in `src/spp/features/market.py:build_market_features` (lines
  160–164 in the current file: `shifted_close = _shift(close, 1); sma20 = shifted_close.rolling(20, min_periods=10)…;
  px_sma20 = shifted_close / sma20 - 1`). The event filter is `px_sma20 > 0` evaluated on `d`, which
  inside that module is "yesterday's close above the 20-day SMA ending yesterday". Reusing the same
  arithmetic means the filter is verified once, not twice.
* **What `T1` does to the sample:** it removes names that are extended above their 20-day average.
  If the news effect is uniform across names, this only cuts `n` and costs power. It helps only if
  the effect is genuinely concentrated in uptrending names — that is a hypothesis to test as a
  split, and any report must show the filtered and unfiltered samples side by side.
* `T2` uses the **same-day** SPY open-to-close direction, which is known only at the exit. It cannot
  be used as an entry filter — it is a **post-hoc condition**. It is listed here because the agreed
  design names it, but it is a diagnostic split ("does the signal work only on up-market days?"),
  never a tradeable rule. Treating it as a filter would be look-ahead and is listed as a failure mode
  (§6.6).
* `T1` is only defined when `published_at + 5 min` is **after** the open: for pre-market news the
  `entry_ts` is the 09:30 bar, and `T1` at `d−1` is still causal (it only needs `d−1`'s close, known
  the previous evening). `T2`'s same-day open is known at 09:30, so for pre-market events it could
  in principle be used — it is not, for consistency.

Intended module: **planned** `src/spp/live/events.py:apply_trend_filter`, reading the daily frame
already produced by `src/spp/features/market.py`. No new feature code.

### 1.6 Event-level dedupe

The daily pipeline dedupes headlines inside `(ticker, date)`:
`src/spp/data/normalize.py:dedupe_news` — exact by `text_hash`, near by trigram Jaccard `> 0.9`
(`TRIGRAM_THRESHOLD`), survivor by publisher priority (Reuters > Bloomberg > WSJ > CNBC > others,
`PUBLISHER_PRIORITY`), then earliest timestamp, then `article_id`.

For events this is **not enough**, for the same reason it is not enough in live
(`src/spp/data/news_live.py:dedupe_live`, `DEDUPE_BUCKET_HOURS = 6`): a reprint an hour later is a
second event in an event framework and a second trade in the P&L. The event-level rule:

```
dedupe_key  = (ticker, title_hash_normalised)  where title_hash_normalised = text_hash(title_norm)
              after stripping the publisher suffix (normalize.clean_title already does this)
window      = published_at of the survivor ± 6 h  (same bucket size as news_live)
survivor    = publisher priority, then earliest published_at, then article_id
second rule = two events of the same ticker whose entry_ts are equal or within 5 min → keep the
              higher-priority one; the discarded event is recorded in the drop table, not deleted
```

The `±6 h` bucket is deliberately the same constant as live so that the offline event table and the
live loop cannot disagree about what one event is. Syndicated copies across *different* tickers are
kept (same as `normalize.dedupe_news`), because two names genuinely move on one story.

Intended module: **planned** `src/spp/live/events.py:dedupe_events`, reusing
`spp.data.normalize.jaccard`, `trigrams`, `publisher_priority`; no second copy of the trigram code.

---

## 2. Daily approach vs event approach

| | daily (published, 2023 holdout spent) | intraday event (this design) |
|---|---|---|
| unit of observation | `(ticker, signal_date)` panel row; 1 row per ticker-session | one `(article_id, ticker, published_at)` |
| target | `y = log(adj_close[t+1]/adj_close[t])`, cross-sectionally demeaned by date (`src/spp/features/labels.py`, `demean_by_date`) | `y_event = r_i − r_spy` over `[entry_ts, exit_ts]`, raw prices, no demeaning |
| signal timing | `published_at ≤ 16:00 ET` → same day, else next day, then roll to next NYSE session (`normalize.signal_day`) | `published_at + 5 min`, no day policy, no rolling |
| entry | next session `open[t+1]` (`src/spp/backtest/portfolio.py:execution_return_columns`, `next_open_hold_close`) | first bar with `bar.start ≥ published_at + 5 min` |
| exit | `close[t+1]` | `entry + 60/120/180 min` or session close, whichever first |
| hedge | dollar-neutral long/short quintile book (`n_quantiles = 5`, `configs/final.yaml:backtest`) | one short SPY leg per name, same minutes |
| aggregation | per-day, per-ticker (median 1 article/ticker-day, `docs/limitations.md` §2) | none; article is the unit |
| validation | purged CV over `(ticker,date)` pairs, expanding window, `embargo_days = 5` sessions (`src/spp/models/cv.py`) | purged CV over **event dates**; embargo = `ceil(max hold / 390 min)` sessions, see §5 |
| cost model | `turnover × 2 × 0.0010` per day at book level (`portfolio.py:COST_PER_SIDE = 0.0010`, `commission_bps = 5`, `slippage_bps = 5`) | per event, per leg, 4 side-charges per event (§4) |
| turnover | 169.78×/year measured (`reports/backtest_metrics.json:costs.turnover_ann`) | set by event count, not by rebalance |
| lever for costs | smoother book, fewer names | fewer, higher-conviction events (top-k per session) |
| what the market has to be wrong about | the *day* ahead | the *next 1–3 hours* ahead, before the news is priced |

The last row is the honest reason to run the new experiment at all, and the honest reason it may
fail: intraday reaction to a public headline is the *most* efficient horizon there is.

---

## 3. The data problem (measured, not estimated)

Availability on this machine, taken as given by the run brief and **re-verified where cheap**:

| source | interval | window on this box | sessions in window | source of the number |
|---|---|---|---|---|
| `yfinance` | 1m | ~5 trading days | ~5 | run brief (given) |
| `yfinance` | 5m | 60 calendar days, first bar `2026-08-14 09:30 ET` | **60** | verified: 4 661 rows, `2026-08-14 09:30-04:00` … `2026-11-06 15:55-05:00`; session count from `pandas_market_calendars` NYSE |
| `yfinance` | 60m | 730 days, first bar `2023-12-11` | **14** (2023-12-11 … 2023-12-29) | session count from NYSE calendar |
| FNSPID corpus (`data/interim/news_raw.parquet`) | hour-resolution stamps | `2019-01-01` … `2023-12-16` (max `date`) | 1 812 sessions with ≥1 article | manifest `data/interim/news_raw.parquet.manifest.json` |

No API keys are present for Alpaca, Polygon, Finnhub, Tiingo, Alpha Vantage, Databento or EODHD
(run brief; confirmed by absence of the corresponding env vars in `docs/limitations.md` §5, which
records the same fact for `FINNHUB_API_KEY`).

### 3.1 The overlap, computed

The three windows above are disjoint from the news window except one sliver. Measured on the corpus:

| measurement | value | how |
|---|---|---|
| articles with `published_at` in `09:30 ≤ t < 16:00` ET | **56** of 124 571 | `published_at.dt.tz_convert('America/New_York')`, hour/minute test |
| … of which on a weekday | **2** | `dayofweek < 5` |
| articles inside the 60m window (`2023-12-11` … `2023-12-29`) on an intraday `published_at` | **0** | corpus ends `2023-12-16`; the 56 intraday rows on `2023-12-16` are a **Saturday**, not a session; the other 2 weekday rows are 2020 |
| articles inside the 5m window (`2026-08-14` …) | **0** | corpus ends 2023 |
| articles inside the 1m window (~last 5 sessions) | **0** | same |
| share of stamps at `minute == 0` | 0.99766 | whole corpus |
| share of articles stamped `19:00–20:59 ET` | 0.9965 | whole corpus |

**The conclusion is blunt: on this machine there is no configuration in which news and minute bars
overlap.** The only weekday intraday articles in five years are two rows at `2020-06-03 09:53 ET`
(`MRK`, `PFE`, same headline) and one at `2020-05-17 11:07 ET` (a Sunday, not a session). The
`2023-12-16` block is a weekend dump and is not tradeable at all.

### 3.2 The three configurations that are possible

One is possible **today**; the other two are configurations of a *future* state of the machine
(a purchased data source or a forward capture that has been running for a month). They are named
because they are the only three shapes the `(news window ∩ bar window)` can take.

**C1 — Retrospective, full resolution: 60m bars `2023-12-11` … `2023-12-29` ∩ news.**
* Bars: 14 NYSE sessions, `yfinance` 60m. News in overlap: **0** usable intraday rows.
* What it would prove if news existed: the whole event pipeline end-to-end on real bars with real
  costs — `entry_ts`, `exit_ts`, `y`, dedupe, the SPY hedge, the placement of trades. It would prove
  *plumbing*, never *edge*: 14 sessions cannot hold more than a few dozen events, and the plan's
  power arithmetic (§4.3) needs hundreds.
* With today's corpus it proves **nothing measurable**. It is a smoke-test fixture, not an
  experiment.

**C2 — Forward, 1m resolution: live capture from now ∩ live news.**
* Bars: `yfinance` 1m, ~5 sessions (rolling) — i.e. bars are re-downloadable only for the recent
  week, so the capture must run continuously and persist bars (`data/live/minute_bars/`,
  **planned**) or they are lost.
* News: the live feeds already implemented (`src/spp/data/news_live.py`: RSS, GDELT, Finnhub behind
  `FINNHUB_API_KEY` which is absent, Alpha Vantage opt-in). Their yield is measured and poor:
  general feeds yield 1/30, 0/10 and 1/10 records (`docs/limitations.md` §5), per-ticker Yahoo
  raised 3 → 16 of 18.
* What it proves: whether the **live path** produces tradeable events at a usable rate, with exact
  publication timestamps (RSS gives minute resolution; GDELT gives crawl time, which is *later*
  than publication and therefore cancels only in the direction that makes the trade look worse than
  reality — see §6.4). It cannot prove a statistically significant edge: 5 sessions of bars is
  maybe a few hundred events at best, and the events-per-session supply on this corpus is
  **0.0445/session** (56 intraday articles / 1 258 sessions), which over 5 sessions is ≈ 0.
* What it proves *now*: the plumbing and the absence of a signal source. Both are worth having.

**C3 — Retrospective, coarse resolution: 60m bars over 730 days ∩ a purchased news/bar store.**
* This is the only configuration that can carry a **statistical** answer, and it is **not available**:
  it requires a paid intraday source (Polygon/EODHD/Databento) or a key (Alpaca IEX, which covers
  ~2 % of volume and produces sparse bars — a liquidity bias of its own).
* What it would prove: the edge at 60-minute resolution, i.e. the **180-minute horizon only**.
  The 60- and 120-minute horizons cannot be studied at 60m resolution at all (a 60-minute hold is
  one bar; the excess return is then `close − open` of that bar, which is a different quantity and
  must be labelled as such).
* It would **not** prove the 5-minute execution lag is achievable: at 60m resolution the entry is
  the first hour bar whose *start* is ≥ `published_at + 5 min`, so the actual entry lag is anywhere
  in `[5 min, 65 min)`. That is a real dilution and it must be in any report that uses C3.

### 3.3 What is not possible, ever, on free data

A configuration that is both **retrospective** and **full-resolution** and **news-aligned** does not
exist on free endpoints. That is the whole of the data problem. Any future number for the intraday
experiment will come from C2 (forward, 1m, short) or C3 (retrospective, 60m, paid). The design below
is written so that the same event table and the same `y` function serve both, with `interval` as a
parameter, and so that the report must say which configuration it came from.

---

## 4. Cost model per event, and the arithmetic

### 4.1 Per event

Costs follow the daily model's parameters exactly (`configs/final.yaml:backtest`:
`commission_bps = 5`, `slippage_bps = 5`, and `portfolio.py:COST_PER_SIDE = 0.0010`). One event is a
**hedged pair**: long the name, short SPY, entered and exited at the same timestamps.

```
cost_per_side        = commission_bps + slippage_bps = 5 + 5 = 10 bp
sides_per_leg        = 2                                (entry + exit)
legs_per_event       = 2                                (name + SPY hedge)
cost_event(bp)       = 2 × 2 × 10 = 40 bp
y_net_event          = y_event − 0.0040
```

Spread and market impact are **not** added on top of `slippage_bps = 10`: the 10 bp is already called
"slippage" in the daily model and it is the only number the project has measured — and it was not
measured either, it was chosen. For a 2-hour hold in large caps the usual intuition is a few bp; for
a headline-driven move the queue is hostile and 10 bp is optimistically small. The design therefore
does **not** claim the cost is right; it claims the break-even threshold it implies, so that the
reader can substitute their own number:

```
break-even gross excess return per event = cost_event = 40 bp = 0.40 %
```

`borrow_annual = 0.05` (`configs/final.yaml:backtest`) applies to the short SPY leg. Charging it per
calendar day as the daily code does (`portfolio.py:summarise_returns`, `borrow_annual / 252`):
`0.05 / 252 = 0.000198 % = 0.0198 bp` per event day. A trading-hours pro-rata charge
(`0.05 × 2/8760`) is three orders smaller. Both are negligible; the larger one is used.

### 4.2 Events per day needed to overcome costs — the arithmetic and why it is a trap

Let `N` = events traded per session and `μ` = **gross** excess return per event (bp). Daily net P&L:

```
net_day(bp) = N × (μ − 40)
```

* If `μ > 40`: every additional event adds profit, and `N` is limited only by supply.
* If `μ = 30`: each event loses 10 bp, and **more events make it worse**: `N` cannot rescue a
  negative per-event edge. `net_day = −10N`.
* If `μ ≤ 40`: no `N` is enough. **Event count is not a cost lever.** It is a *power* lever (§4.3).

That is the first arithmetic trap in the experiment: a reader who asks "how many trades per day do we
need to cover costs" is asking the wrong question. The right question is "is the per-event gross edge
above 40 bp", and only measurement can answer it.

The second arithmetic trap is **selection**: choosing the top-k events by score per session does not
raise `μ` for free. If the score has no edge, top-k raises variance, not expectation; the daily
backtest demonstrates exactly this — `sharpe_gross = 0.1234` collapsed to `sharpe_net = −6.7506` at
`turnover_ann = 169.78` with `cost_drag_ann = 0.339557` (`reports/backtest_metrics.json`).

### 4.3 Power arithmetic — how many events to *detect* an edge

For a one-sample test on the mean event return, `t = μ / (σ_y / √n)`, so

```
n_needed = (t_target × σ_y / μ)²
```

With `σ_y` = standard deviation of 2-hour excess returns. **`σ_y` is not measured** (there are no
bars) — the figure 150 bp is the order of magnitude of a typical large-cap 2-hour move and is stated
only to show the shape:

| σ_y (assumed) | μ (bp) | t_target | n_needed |
|---|---|---|---|
| 150 bp | 40 (break-even) | 2 | 56 |
| 150 bp | 60 | 2 | 25 |
| 150 bp | 80 | 2 | 14 |
| 150 bp | 20 (below cost) | 2 | 225 |

These numbers are **illustrative only** and must be replaced by the measured `σ_y` from the smoke
fixture (§7 C1) before any power statement is published. The supply side is the binding constraint:
`56 / 1258 = 0.0445` intraday articles per session means the corpus can never supply even the
smallest row of the table at 5-minute entry; the supply lives in a forward capture, not in this
corpus.

Related multiple-testing arithmetic (from the daily ablation, `results/ablation.md`): eight
ablation variants were run and **seven of eight CIs cover zero**; the only one that does not is A4,
whose point estimate is negative (−0.0079). The intraday experiment must therefore declare its
number of variants (horizon × trend filter × top-k) **before** the run and report a corrected
threshold (§5.4), or it will manufacture a "finding" out of eight coin flips.

---

## 5. Validation

### 5.1 Event-level purged CV

* Fold over **event dates** (`session(entry_ts)`), not panel dates. Folds by month, expanding
  window, the same shape as `src/spp/models/cv.py:make_folds` with `n_folds = 6`,
  `test_months = 3`, `min_train_months = 12`.
* Embargo: a training event must exit before the test window opens.
  `embargo_sessions = ceil(MAX_HOLD / 390) = ceil(180/390) = 1`. One session is the minimum that
  makes the exit of a training event precede the test window; a multi-day hold would need more.
  This is a *change* from the daily `embargo_days = 5`, justified by the shorter label and stated
  here because the reviewer will notice it.
* No event appears in two folds; an event whose `article_id` is re-printed is already collapsed by
  §1.6.

### 5.2 Metrics, per horizon `H`

* **mean `y_net`** in bp with a **date-clustered** standard error: `se = std(mean_per_session) /
  √n_sessions`, not `std(events)/√n_events` — events on the same session are not independent, and
  news from one publisher is correlated by construction.
* **hit rate** = share of events with `sign(y_event) == sign(score)` and **balanced hit rate** (the
  daily `direction_metrics` already computes balanced accuracy and coverage,
  `src/spp/models/metrics.py:direction_metrics`).
* **IC** over events, computed per session and averaged across sessions when a session has ≥ 10
  events; otherwise pooled. The daily `ic` requires ≥ 10 names per date
  (`metrics.py:IC_MIN_TICKERS_PER_DATE = 10`) and that floor will kill most intraday sessions, so the
  intraday IC must report **how many sessions survived the floor**, not just the mean.
* **Newey–West t** on the per-session mean series (`metrics.py:newey_west_tstat`,
  `NEWEY_WEST_LAG = 5`), since the daily code already has it and it costs nothing to reuse.
* **Block bootstrap CI** on the mean `y_net`, block = sessions
  (`metrics.py:bootstrap_ic` is date-blocked; the intraday variant blocks by session too).

### 5.3 Controls that are mandatory

1. **Entry-lag grid** `{5, 15, 30, 60}` min. If the edge exists only at 5 min and vanishes at 15, the
   result is about *speed of execution*, not about sentiment. If it is flat across the grid, the edge
   is a drift, not a reaction, and it should be checked against a no-news baseline (§5.3.3).
2. **Sign-flipped score**: the same events with `score → −score` must produce the mirrored P&L. If it
   does not, the pipeline has a directional bug, not a signal.
3. **No-news baseline (pseudo-events)**: run the identical `y` function on random `(ticker, timestamp)`
   pairs matched to the real events' session and time-of-day distributions. The "edge" must beat
   this baseline, not zero. The daily backtest already has a null distribution by permutation
   (`portfolio.py:null_distribution`); the intraday analogue is pseudo-events.
4. **Cost grid** `{0, 20, 40, 80}` bp per event, so the reader can see at which cost the result dies.
5. **Holdout discipline**: a contiguous tail of sessions (or, in C2, the last captured week) is
   evaluated **once**; the daily project already has a lock-file precedent
   (`src/spp/models/holdout.py`, `reports/test_2023.lock`) and the intraday run must use the same
   pattern with a different lock name (`reports/intraday.lock`, **planned**). The 2023 daily holdout
   is **not** reused and **not** re-run.

### 5.4 Multiplicity

Declare the grid before the run: `H ∈ {60, 120, 180}` × `trend ∈ {off, T1}` × `top-k ∈ {all, 5, 10}`
= 18 cells. Either report the whole grid with a Bonferroni-style threshold (`α/18`) or report one
pre-registered cell as primary and the rest as exploratory. A grid whose best cell is quoted without
the grid is a manufacturing defect, not a result.

---

## 6. Failure modes that invalidate the experiment

Each of these, if it happens, makes any positive number **not evidence**. They are listed because
each one is plausible in this codebase, not hypothetically.

1. **Timestamp resolution.** 99.77 % of `published_at` are hour-resolution. If the entry rule is
   applied to those stamps as if they were exact, every "5-minute" entry is really 5–65 minutes late
   and the measured edge is a *post-reaction drift*, not a reaction. **Direction of bias:** the
   event cohort is diluted toward "already priced"; a real edge is deflated, a fake one can appear
   if late entries systematically buy into continuation. **Invalidates** any claim about reaction
   speed; does **not** invalidate a claim about drift.
2. **Weekend/holiday articles.** The 2023-12-16 dump shows the corpus is not session-aligned. An
   event whose `published_at` is on a non-session day must be dropped, not rolled forward. Rolling
   forward reintroduces the daily `signal_date` policy that this experiment exists to remove.
3. **Bar alignment / look-ahead.** If the loader labels bars by their *end* (right-labelled) and the
   entry test uses `bar.start`, trades enter up to one interval late; if it uses `bar.end` with a
   left-labelled frame, trades enter up to one interval *early* = look-ahead. **Invalidates** the
   result. Verification: an assertion that `entry_ts >= published_at + 5 min` for **every** event on
   the frame actually used, not on a sample. Daily code already asserts rather than clamps
   (`normalize.assign_signal_date`); the intraday code must do the same.
4. **Gaps and halts.** Missing bars between entry and exit, or a zero-range bar, mean the exit price
   does not exist. Filling forward or interpolating fabricates a trade. **Invalidates** the affected
   events; the daily backtest already declines to fill (`build_execution_frame` → `halted`).
5. **SPY hedge imperfection.** SPY's beta to a single name is not 1; the residual is *name-specific*
   variance, which raises `σ_y` and drops power (§4.3) but does not bias the mean. The bias appears
   only if the SPY leg is timed differently from the name leg (e.g. SPY entered on the minute, the
   name on the next available bar). **Invalidates** if timestamps differ; requires an assertion that
   both legs share `entry_ts` and `exit_ts`.
6. **`T2` as an entry filter.** Using same-day SPY direction to decide whether to trade is
   look-ahead (the session's direction is known only at 16:00). **Invalidates** any result that uses
   `T2` as a filter; `T2` may only be a post-hoc split.
7. **Survivorship in the universe.** The 40 names are today's large caps, chosen ex post. If an
   event is only counted if its ticker is on today's list, so the cohort over-weights names that
   survived and prospered. **Direction of bias:** greyer, larger, more liquid names; effects in
   small movers are invisible. Same limitation as the daily panel (`docs/limitations.md` §2 and the
   universe file `configs/universe_us_largecap.yaml`).
8. **Duplicate economics.** If dedupe (§1.6) lets a reprint through, one story counts many times and
   the effective `n` is inflated, so a t-statistic looks significant on far fewer independent
   stories. **Invalidates** any significance claim; hit rate is unaffected.
9. **Horizon shopping.** Reporting only the best `H`, or the best trend switch, or the best top-k.
   **Invalidates** the significance claim (§5.4). Must be prevented by pre-registration, not by
   honesty in the write-up.
10. **Cost undercount.** 40 bp assumes 10 bp per side per leg and ignores adverse selection and
    queue position on news. If realised costs are 2× the model, the break-even moves to 80 bp and a
    "significant 50 bp edge" is a loss. **Invalidates** the P&L, not the sign of `y`.
11. **Clocked-out events.** News that prints while the market is closed cannot be an intraday event;
    if such rows are pushed to the next open and then to a 60/120/180 exit, the "intraday" experiment
    silently contains the overnight session. **Invalidates** the attribution to intraday reaction.
12. **Selected sample from the corpus.** The corpus is 124 571 articles over 5 years across 40 names
    with median 1 article per ticker-day (`docs/limitations.md` §2). The subset with a session-time
    stamp is not a random subset of news; it is the subset the source happened to stamp that way.
    **Direction of bias:** unknown, and that is the danger.

---

## 7. What the daily experiment already rules out

The point of this section is to stop the new experiment from re-testing what the old one spent its
holdout on. Every number is from `reports/` and `results/`.

1. **A daily cross-sectional sentiment signal on next-session log return.** `verdict =
   "not_confirmed"`: `ic = 0.012716`, 95 % CI `[−0.010167, 0.035598]`, `ic_t_nw = 0.938293`,
   `dm_vs_zero.p = 0.250742`, `decile_rank_corr = 0.042424` (`reports/test_2023.json`). Re-running
   this variant is not a new question.
2. **The daily version of this idea at the book level.** `sharpe_net = −6.750554` vs
   `sharpe_gross = 0.123431`, `turnover_ann = 169.778571`, `cost_drag_ann = 0.339557`,
   `p_value_le_zero = 1.0` on a 1 000-draw bootstrap (`reports/backtest_metrics.json`).
   The daily rebalance is ruled out **independent of the signal**: at that turnover, cost dominates.
3. **Sentiment backends as the explanation.** In purged CV 2019–2022, A2 (FinBERT, sentiment only)
   `ic = 0.013428`, A3 (open-jev) `0.011165`, A4 (lexicon) `−0.007921`, A7 (ensemble) `0.0137`
   (`results/ablation_cv.json`, `results/ablation.md`). All point estimates are ≤ 0.014 and all CIs
   cover zero. Switching sentiment engine is not a new question; a different **horizon** is.
4. **Day-policy variants.** A6 (`strict_dates = true`) `ic = 0.0167` vs A5 `0.0156`
   (`results/ablation.md`) moved 10 rows of 124 571 — within noise. The day policy is not the
   defect to chase, which is part of why the event design removes it rather than retunes it.
5. **Price-only features.** A1 (market only) `ic = 0.000708`, `pooled_ic = 0.000938`
   (`results/ablation_cv.json`). Technical features alone carry nothing at the daily horizon.

What the daily experiment does **not** rule out, and what the intraday experiment is therefore about:
the same information priced within minutes of publication rather than after a full overnight lag,
measured on a per-event basis with a per-event cost. That is the only open question. Everything else
in the daily results should be treated as closed.

---

## 8. Modules: what exists, what is planned

| path | status | role |
|---|---|---|
| `src/spp/data/normalize.py` | exists | `published_at`, dedupe primitives, `ET`, `nyse_trading_days` |
| `src/spp/data/news_live.py` | exists | `dedupe_live`, `DEDUPE_BUCKET_HOURS = 6`, live feeds |
| `src/spp/features/market.py` | exists | `px_sma20` construction reused by `T1` |
| `src/spp/models/cv.py`, `metrics.py` | exists | fold shape, `newey_west_tstat`, `bootstrap_ic`, `direction_metrics` |
| `src/spp/backtest/portfolio.py` | exists | `COST_PER_SIDE = 0.0010`, `TRADING_DAYS`, `null_distribution` |
| `src/spp/live/__init__.py` | exists (docstring only) | — |
| `src/spp/live/minutes.py` | **planned** | bar download (`yfinance`), NYSE alignment, left-label assertion, gap/halt marking |
| `src/spp/live/events.py` | **planned** | `build_entry_ts`, `build_events`, `dedupe_events`, `compute_y`, `apply_trend_filter` |
| `src/spp/live/evaluate.py` | **planned** | per-horizon metrics, date-clustered SE, pseudo-event baseline, cost grid |
| `configs/intraday.yaml` | **planned** | `interval`, `lag_min = 5`, `horizons = [60,120,180]`, `cost_bps_per_side = 10`, `top_k`, trend switches, lock name |
| `tests/test_intraday_events.py` | **planned** | edge cases below |
| `reports/intraday_baseline.md`, `reports/intraday.lock` | **planned** | first report + one-shot holdout lock |

Edge cases the tests must pin (each is a failure mode from §6 made executable):

1. `published_at` at `09:31` → `entry_ts` is the first bar with `start ≥ 09:36`.
2. `published_at` exactly 5 min before a boundary (`09:25`) → `entry_ts = 09:30`.
3. `published_at = 09:30` → `entry_ts` is the first bar after `09:35`, **not** `09:30`.
4. `published_at` at `15:40` → no valid event for any `H` (`entry_cutoff` = `13:00`).
5. `published_at` on a Saturday → no event (not rolled forward).
6. `published_at` at `16:00:00` exactly → no event.
7. Bar frame right-labelled → loader raises, does not guess.
8. One interval of missing bars inside the hold → event invalid, not filled forward.
9. SPY bar exists, name bar missing → event invalid.
10. Duplicate headline 7 h later → **survives** (`±6 h` window), 5 h later → **dropped**.
11. Two events, same ticker, `entry_ts` within 5 min → higher-priority publisher wins.
12. `H = 180` with `entry` at `12:59` → `exit_ts = 15:59` or last bar ≤ 16:00, whichever first.
13. `score = 0` → not counted in hit rate (the daily `direction_metrics` already drops zeros).
14. Cost with `cost_bps_per_side = 0` → `y_net == y_event` bit-for-bit.

---

## 9. What would make this experiment worth running

Honest summary, for the reviewer who has read all of the above:

* It is **not runnable** in C1 (no news in the 60m window) and **barely runnable** in C2 (5 days of
  1m bars, and the corpus supplies 0.0445 intraday articles per session, so a forward capture sees
  almost nothing). The design is written now so that when the data arrives, it is evaluated once.
* The one configuration that can give a statistical answer is C3, and it costs money.
* The break-even is **40 bp per event**, and the only question is whether the gross per-event edge
  clears it. Event count does not change that; it changes only the standard error.
* If the answer is "no edge", that is a result: it means the daily failure is not aggregation but
  the absence of news predictability at the horizons tested, and the next step is a different
  information set, not a finer grid.
