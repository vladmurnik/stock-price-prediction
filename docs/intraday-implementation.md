# Intraday event study: what the code actually does

Status: **implementation description, no result.** Every rule below is read from the module named
next to it. There is no measured number in this document: `reports/intraday_metrics.json` and
`reports/intraday_baseline.md` do not exist yet, so every outcome field is marked **pending** and
must be copied from that file once the first run produces it — never estimated here.

The design intent (entry/exit semantics, why SPY, why 60/120/180) lives in
`docs/intraday-design.md`. This file is the check against the code: what is implemented, under which
parameter, and where the implementation and the design still disagree (§8).

Modules: `src/spp/live/minutes.py` (bars), `src/spp/live/events.py` (one article → one trade),
`src/spp/backtest/intraday_eval.py` (statistics), `src/spp/live/intraday.py` (orchestration),
`ops/intraday_build.py` (one command), `configs/intraday.yaml` (pre-registered parameters),
`src/spp/cli.py:1362` (the `intraday` command group).

---

## 1. Bar loader (`src/spp/live/minutes.py`)

**Cell shape.** `BAR_COLUMNS = ("ticker", "ts", "open", "high", "low", "close", "volume",
"session")` (`minutes.py:57`). `ts` is a UTC timestamp; `session` is the ET calendar date
(`minutes.py:95` `session_of`).

**Time zone.** Naive input is localised to UTC and never treated as ET (`minutes.py:87`
`to_new_york`). The session window is `[09:30:00, 16:00:00)` ET, half-open (`minutes.py:100`
`in_session`: `seconds >= open_s` and `seconds < close_s`).

**Bar labelling.** A bar is labelled by its start; its end is `start + interval`
(`minutes.py:113` `bar_label_end`), so the 60-minute bar at 15:30 ends at 16:00 and covers 30
minutes only.

**Provider window and depth limits.** yfinance is the only loader (`minutes.py:235` `_download`).
It is called with `auto_adjust=False`, `prepost=False`, `actions=False`, `threads=False`
(`minutes.py:243`). The free feed serves roughly **730 calendar days of 60-minute bars** and
**60 calendar days of 5-minute bars** — stated in `configs/intraday.yaml:8` and
`minutes.py:20`; it is *not* re-derived in code, because the limit is a provider fact. The guard
against silently receiving less is `probe_depth` (`minutes.py:260`), which reports first and last
bar, session count and bars per session for whatever the provider actually returned, over a
`lookback_days` window (default 800, `cli.py:1379`).

**Cache key.** One parquet per `(ticker, interval)`:
`{cache_dir}/{interval}/{sanitised_ticker}.parquet` (`minutes.py:129` `_cache_path`). Each file
holds the **union of every window ever fetched**; a fetch merges `cached ∪ downloaded` and
de-duplicates on `ts` keeping the newest (`minutes.py:198`). The reading window is requested from
the cache only if the cached file reaches within **3 days** of both ends of the request
(`minutes.py:191` `covers_end`, `minutes.py:192` `first_ok`); otherwise it is re-downloaded.
`refresh=True` forces a download regardless (`minutes.py:173`).

Two consequences a reviewer must hold on to:

* the depth limit is **not enforced**, only reported — a run over 2019–2023 at `60m` returns bars
  only for the last ~730 days of the requested window (plus whatever is cached), so the number of
  sessions in a report must be read from the funnel, not from `--start/--end`;
* the 3-day coverage check is a freshness test, not a completeness test: 3 consecutive missing
  days in the middle of a file are invisible to it.

Normalisation (`minutes.py:134` `_normalise_frame`): MultiIndex columns are flattened, OHLCV
lower-cased, rows without `close` dropped, out-of-session rows dropped, duplicates on `ts` keep
the last, sorted ascending. Out-of-session bars are **dropped without a counter** — only the event
funnel counts; no funnel line reports bars removed at load time.

---

## 2. Event rule (`src/spp/live/events.py`)

An event is `(article_id, ticker, published_at)` (`events.py:7`). Definitions, with the code line
that implements each:

```
p            = published_at, converted to UTC                       events.py:159
target       = p + lag_minutes (default 5)                          events.py:205
session(s)   = ET calendar date of p                                events.py:199
entry        = first bar of session(p) with bar.start >= target     events.py:206  (searchsorted side="left")
entry_price  = entry.open, else entry.close if open is not finite>0 events.py:215-218
X(H)         = entry.start + H, H in {60,120,180}                   events.py:285 (_exit_bar)
exit(H)      = first bar of session(p) with bar.start >= X(H)       events.py:287  (searchsorted side="left")
             = last bar of session(p)         if none exists       events.py:290  → truncated=True
r_ticker     = exit.close / entry_price - 1                         events.py:240
r_spy        = spy.close(exit_ts) / spy.open(entry_ts) - 1          events.py:241-246
             (SPY legs are taken at the *event's* entry_ts and exit_ts, not at SPY's own bar after the lag)
excess       = r_ticker - r_spy                                     events.py:247
direction    = +1 if score > dead_zone, -1 if score < -dead_zone, 0 otherwise; NaN → 0  events.py:123
y            = excess * direction if direction != 0 else excess     events.py:263
truncated    = exit was forced to the session's last bar            events.py:264, events.py:290
```

Points the formulas make explicit:

* **`>=`, not `>`.** `searchsorted(..., side="left")` returns the first bar at or after the
  target. An article printed exactly 5 minutes before a bar boundary enters on that bar.
* **One lag, measured from `published_at`.** `entry_delay_min = (entry_ts − published_at)` in
  minutes is stored per event (`events.py:265`), so the realised lag is auditable without
  re-deriving it. Its floor is `lag_minutes`; its ceiling is the bar interval plus the lag.
* **The exit cannot leave the session.** `_exit_bar` searches only inside `session_bars`, so a
  horizon that overruns 16:00 ET is truncated to the last bar of that session and flagged, never
  carried overnight (`events.py:280`).
* **Same stamps for both legs.** SPY is read at the event's `entry_ts` and `exit_ts`
  (`spy_lookup.loc[entry_ts, "open"]`, `spy_lookup.loc[exit_ts, "close"]`). An event whose
  stamp has no SPY bar is dropped as `missing_spy`, not filled (`events.py:224`, `events.py:236`).
* **Benchmark is mandatory.** No SPY bars → `ValueError` (`events.py:188`).
* **`y` is the intended trade's return.** A mean `y` above the cost means the rule made money; the
  unsigned measurement stays in `excess`.
* **Directionless events are kept.** With no score (or an all-NaN score column) every event gets
  `direction = 0` and `y = excess`, which measures the size of the reaction without a rule.

**Dedupe** (`events.py:169-183`): key is `title_hash` (article_id if absent), bucket 6 h
(`dedupe_hours`), earliest copy inside the window kept. In the current corpus
(`data/interim/news_raw.parquet`, 16 columns, verified: no `title_hash`) the fallback key is
`article_id`, which can never join two copies from different rows to the same 6 h window pair —
so dedupe is effectively a by-article pass here. See §5 for the measured copy counts.

---

## 3. Funnel (`events.py:67` `DropStats`)

Every rejection has a name and `total = kept + Σ dropped` is enforced by `DropStats.check()`
(`events.py:101`), which raises rather than reporting a subset as the whole corpus
(`events.py:116`). The counters, in the order the code can increment them:

| counter | meaning | line |
|---|---|---|
| `total` | articles fed in | `events.py:153` |
| `no_publication_time` | `published_at` unparseable / NaT | `events.py:160` |
| `duplicate` | same `(ticker, dedupe_key)` inside 6 h | `events.py:180` |
| `no_bars_for_session` | ticker has no bars on the ET session of `p` | `events.py:202` |
| `after_close` | `target` is after the last bar start of the session | `events.py:208` |
| `no_entry_bar` | entry bar is outside 09:30–16:00 ET, or has no usable price | `events.py:213`, `events.py:219` |
| `rejected_lag` | `entry_ts < target` (must stay 0; implies a broken bar sort) | `events.py:222` |
| `missing_spy` | no SPY bar at `entry_ts` or `exit_ts` | `events.py:225`, `events.py:237`, `events.py:244` |
| `no_exit_bar` | no exit bar for this horizon (only when the session slice is empty) | `events.py:233` |
| `zero_direction` | reserved, currently never incremented | declared `events.py:80` |
| `extra` | free-form counters for future paths | `events.py:82` |
| `kept` | articles that produced events | `events.py:270` |

Granularity caveat: `kept` counts **articles**, the event frame counts **article × horizon** rows
(`events.py:230`). Three horizons with one kept article give `kept = 1` and three event rows. The
funnel therefore balances on articles, and `n_events` in the report is not `kept × 3` (a horizon
can fail independently via `missing_spy` / `no_exit_bar`). Read `reports/intraday_metrics.json`
`:funnel` and `:n_events` separately.

---

## 4. Evaluator (`src/spp/backtest/intraday_eval.py`)

The evaluator never sees bars, tickers or databases; it receives a frame with `date` and `y`
(`intraday_eval.py:18`).

**Unit of observation** is the event, not the ticker-day (`intraday_eval.py:8`).

**Clustered standard error by date** (`intraday_eval.py:64` `cluster_tstat`) — the textbook
one-way cluster:

```
mean      = mean(x)
resid_k   = Σ_{i∈k} x_i − mean · n_k
se²       = [ Σ_k resid_k² / n² ] · G/(G−1)
t         = mean / se ;   p = 2 (1 − Φ(|t|))
naive_se  = sd(x) / sqrt(n)
inflation = se / naive_se
```

with `G` = number of distinct dates. Fewer than 2 clusters → `se = t = p = NaN` with `n_clusters`
reported (`intraday_eval.py:97`) — no number rather than a fake precision. Fewer than 3 events →
the whole block is NaN (`intraday_eval.py:91`). `p` uses the normal approximation, not `t(G−1)`, so
it is mildly optimistic for few dates.

**Cost model** (`intraday_eval.py:203`):

```
cost_per_round_trip = (2 · cost_bps + spread_bps) / 1e4       net_mean = mean(y) − cost
```

With `configs/intraday.yaml:15-16` (`cost_bps: 5.0`, `spread_bps: 5.0`) the round trip costs
`2·5 + 5 = 15 bps = 0.0015` in decimal — the same arithmetic as the YAML comment. `cost_bps` is
**per side**, `spread_bps` the half-spread, charged in full to every event.

**Break-even hit rate** (`intraday_eval.py:159`), under the symmetric win/loss assumption
`mean(y) = (2·hit − 1)·mean|y|`:

```
hit* = 0.5 + cost / (2 · mean|y|)
```

Reported next to the realised `hit_rate_up = mean(y > 0)`. Also reported:
`hit_rate_directional = mean(sign(score) == sign(y))` on `score != 0` (`intraday_eval.py:251`).

**Minimum detectable effect** (`intraday_eval.py:274`):

```
MDE = (1.96 + 0.84) · inflation · σ_y / sqrt(n_events)      = 2.802 · inflation · σ / √n
```

`σ_y` is the event-level dispersion (`std_y`, ddof=1) and `inflation` the realised
`cluster.inflation` from §above (`intraday.py:251-258`). Reading rule, as written by the module:
a mean below the MDE means *the sample cannot answer*, not *the effect is zero*
(`intraday.py:306`).

**Ranking metrics.** `event_ic` = mean per-date Spearman between `score` and `y`, only for dates
with at least `min_events_per_date = 5` events and more than one distinct score
(`intraday_eval.py:224`, `configs/intraday.yaml:28`). `score_quintiles` = top-minus-bottom
quintile excess, buckets formed **inside each date** (`intraday_eval.py:126`).

**Horizons.** `evaluate_by_horizon` keys blocks `h60`, `h120`, `h180` (`intraday_eval.py:263`);
one block per horizon, each with its own `n`, `cluster`, `net_mean`, `breakeven_hit_rate`, `mde`.
No horizon is selected post hoc; the module reports all three (`intraday_design.md` §1.3).

Every metric is a plain Python float / int / None and JSON-serialisable — this is what
`reports/intraday_metrics.json` stores.

---

## 5. Trend filter (`src/spp/live/intraday.py:177` `attach_trend`)

```
price      = daily adj_close (falls back to close if adj_close absent)      intraday.py:188
wide[t,d]  = last price of ticker t on date d                               intraday.py:189
mean[t,d]  = mean(wide[t, d−window+1 … d]) with min_periods = window//2    intraday.py:194
                                     then .shift(1)                         intraday.py:194
trend_up[t,d] = wide[t,d] > mean[t,d−1]                                     intraday.py:195
```

* **The one-session shift is the whole point.** `rolling(...).shift(1)` gives session `d` the mean
  of closes up to `d−1`; nothing from the session being traded enters the flag. Flag on `d` is
  therefore knowable at the `d−1` close.
* **The comparison side is *not* shifted.** The flag for `d` compares `close(d)` to
  `mean(d−1)`, while the design (`intraday-design.md` §1.5, `T1`) defines "previous completed
  session closed above its mean": `close(d−1) > mean(d−1)`. The two differ by one session, and the
  implemented version uses the close of the session it is attached to. `evaluate_panel` only
  *splits* on the flag and never enters on it, so this is not look-ahead in the reported means, but
  the split is not the split the design describes. Recorded as an open item in §8.
* Window default 50 (`configs/intraday.yaml:25` `trend_window: 50`), `min_periods = window // 2 = 25`
  (`intraday.py:194`). A ticker-date with fewer than 25 prior closes gets no flag and is excluded
  from both halves of the split (`intraday.py:235-241`, `notna().any()` gate).
* The split runs only if daily prices were loaded; `ops/intraday_build.py:83-89` warns and skips
  on failure rather than stopping.

Split output: `report["by_trend"]["trend_up" | "trend_down"]`, each a per-horizon block with
`n_events`, `net_mean`, `cluster.t` (`intraday.py:236-248`), and a table in the markdown report
(`intraday.py:311`).

---

## 6. Orchestration and artifacts

`build_panel` (`intraday.py:98`) takes the tickers present in the news window, appends the
benchmark, fetches bars for `[start, end]`, and calls `build_events`. `evaluate_panel`
(`intraday.py:204`) evaluates per horizon, attaches the trend split when prices are given, computes
`mde`, and adds:

* `lag_violations` = count of `entry_ts < published_at` (`intraday.py:259`) — a second, independent
  check of the lag invariant, next to the funnel's `rejected_lag`;
* `truncated_share` = share of event rows whose exit was the session close (`intraday.py:263`);
* `config`, `n_events`, `n_dates`, `funnel` (`intraday.py:217`).

Files written by one run:

| path | writer | contents |
|---|---|---|
| `reports/intraday_metrics.json` | `ops/intraday_build.py:96`, `cli.py:1455` | full report: config, funnel, per-horizon blocks, trend split |
| `reports/intraday_baseline.md` | `write_report`, `intraday.py:267` | the same numbers as tables + the reading rules |
| `data/processed/intraday_events.parquet` | `ops/intraday_build.py:101`, `cli.py:1458` | event-level rows, `EVENT_COLUMNS` (`events.py:47`) |
| `data/cache/minutes/{interval}/{ticker}.parquet` | `minutes.py:209` | union bar cache |

---

## 7. How to run

```powershell
# 1. Reproduce the whole experiment for one window (writes the three artifacts of §6)
.\.venv\Scripts\python.exe ops\intraday_build.py --start 2023-12-11 --end 2023-12-29

#    other supported flags
#    --backend {finbert,openjev,lexicon}   directional score source (default finbert)
#    --config configs/intraday.yaml        parameters; defaults are overlaid with this file
#    --refresh-bars                        re-download instead of trusting the cache
#    --no-trend                            skip the daily trend split

# 2. Check how much history the feed really serves, before trusting a window
.\.venv\Scripts\python.exe -m spp.cli intraday probe --tickers SPY,AAPL --interval 60m --lookback 800

# 3. Same experiment through the CLI (writes the same JSON/markdown; --no-write evaluates only)
.\.venv\Scripts\python.exe -m spp.cli intraday build --start 2023-12-11 --end 2023-12-29 --no-write
```

`ops/intraday_build.py` exits `2` when the window produces no events (`intraday_build.py:80`);
`intraday probe` exits non-zero (`DATA_EXIT`) when every ticker returns zero bars
(`cli.py:1394`).

**Runner defects at the time of writing** (reproduced on this machine, not fixed here — code is
outside this document's zone). All three reproduce on numpy 2.5.4 / pandas 3.1.1.

* `spp.cli intraday probe`, any ticker/interval, raises before any network request:
  `TypeError: tzinfo argument must be None or of a tzinfo subclass, not type 'str'` at
  `minutes.py:272` (`pd.Timestamp(datetime.now(tz="UTC").date())`; reproduced on a bare
  `datetime.date` in the same call). Exit code 1. The depth guard of §1 is therefore unusable.
* `spp.live.events.build_events` raises `TypeError: Cannot compare tz-naive and tz-aware
  timestamps` at `events.py:206` and `events.py:287`. Root cause: a column of UTC `Timestamp`
  objects (whatever the dtype shown in the frame — `datetime64[us, UTC]` or `str`) comes out of
  `Series.to_numpy()` as **object**, and numpy cannot binary-search that against
  `target.to_datetime64()`, which is tz-naive. All eight `build_events` tests in
  `tests/test_events.py` fail on this; `tests/test_minutes.py` (7 tests) passes. `ops/intraday_build.py`
  cannot produce §6's artifacts while this stands.
* `ops/intraday_build.py` and `spp.cli intraday build` stop before `build_events` on the same left
  merge: `MergeError: Merge keys are not unique in right dataset; not a many-to-one merge`
  (`ops/intraday_build.py:47`, `intraday.py:167`, both `validate="many_to_one"`). The sentiment
  files contain **10 duplicate `(article_id, ticker)` rows in 9 groups** each
  (`article_sentiment_finbert.parquet`: 124 571 rows, time-shifted `signal_date`), and
  `news_raw.parquet` repeats the same 10 rows, so the join is not one-to-one on either side.
  `read_parquet` + `merge` reproduced; the CLI exits 1.

---

## 8. Implementation vs design — deltas that must be resolved before a number is published

| # | design (`docs/intraday-design.md`) | code | effect |
|---|---|---|---|
| 1 | `entry_cutoff = close − 180 min`; events published too late for a 180-minute hold are dropped so the three horizons share one event set (§1.2) | no cutoff; a late event is kept and its exit is truncated to the session close (`events.py:288-290`) | the 60/120/180 samples are not the same event set; `truncated` rows carry a shorter holding period than the horizon label |
| 2 | bars must not be filled forward; a missing bar invalidates the event (§1.3) | an exit "not found" means the *last bar of the session*, which after a data gap is a stale earlier bar | a gap inside the hold silently shortens the position instead of voiding it |
| 3 | trend `T1` = `close(d−1) > SMA(d−1)` (§1.5) | flag `d` = `close(d) > mean(d−1)` (`intraday.py:194-195`) | split is one session off from the design's definition |
| 4 | dedupe on normalised headline, trigram/Jaccard, publisher priority (§1.6) | `title_hash` if present, else `article_id` (`events.py:169`); corpus has no `title_hash` | reprints are not merged at headline level |
| 5 | 2023 holdout is spent | nothing in `configs/intraday.yaml` or `ops/intraday_build.py` guards the writing window | a run on `--end 2023-12-31` silently touches spent data; the window must be chosen by the caller |
| 6 | bar depth must be proven before a run | `probe_depth` exists but the build path never calls it | a 60m run over years silently returns ~730 days |
| 7 | corpus rows are one per `(article_id, ticker)` (`intraday-design.md` §1.1) | the corpus repeats 10 such keys (`news_raw.parquet`), and both join sites require uniqueness | the build stops at the join until the key is de-duplicated or the join declared one-to-many |

## 9. First run — what is covered

**Pending.** `reports/intraday_metrics.json` does not exist on disk (checked
2026-11-08), so:

| field | value | source to fill it from |
|---|---|---|
| dates covered (`window`) | **pending** | `reports/intraday_metrics.json:window` |
| sessions with events (`n_dates`) | **pending** | `:n_dates` |
| articles in window / kept (`funnel.total`, `funnel.kept`) | **pending** | `:funnel` |
| events per horizon (`horizons.hXX.n_events`) | **pending** | `:horizons` |
| lag violations | **pending** | `:lag_violations` (must be 0) |
| truncated share | **pending** | `:truncated_share` |

Facts that bound any first run come from the corpus, not from a run. All measured with pandas on
`data/interim/news_raw.parquet` and, for the merge, on
`data/processed/article_sentiment_finbert.parquet`:

* 124 571 rows in 16 columns; `published_at` range
  `2019-01-01 00:00:00+00:00` … `2023-12-16 23:02:00+00:00`; 0 NaT. The file stops on
  2023-12-16, so the example window of §7 (`--start 2023-12-11 --end 2023-12-29`) has no articles
  after that date: **pending** how many sessions it actually yields.
* Hour-resolution stamps: 99.77 % of all timestamps have `minute == 0`; inside 2023-12-11 …
  2023-12-29, 99.91 %. The true publication is therefore somewhere inside the stamped hour, which
  is what the 5-minute lag of §2 is measured against.
* That window holds 1 162 rows over 28 tickers, on six `published_at` dates (2023-12-11 …
  2023-12-16); 144 distinct `(ticker, published_at)` groups, the largest with 44 copies (AMZN,
  2023-12-11 00:00 UTC). Headline dedupe (§8#4) would matter here; `article_id` dedupe cannot
  touch these copies.
* 10 duplicate `(article_id, ticker)` keys in the corpus (9 groups), all in 2020 — sufficient to
  stop both join sites (§8#7), and 10 068 rows share a `(text_hash, ticker)`.

Do not write a date range or an event count into any document until it is copied from
`reports/intraday_metrics.json`.
