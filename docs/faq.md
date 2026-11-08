# FAQ — questions a skeptic will ask

Every answer is tied to an artifact in this repository. If a number is not in an artifact, it is not
written here. Paths are relative to the repository root. Numbers are copied verbatim, including
their precision.

Scope of the claim: this repository measures whether a **cross-sectional signal on next-day log
returns** survives out of sample and beats stated costs. It does not predict prices, and nothing
below is evidence that it could.

---

## 1. IC 0.0127 — why is that not a signal?

IC is the Spearman rank correlation between predicted and realised cross-sectional returns, computed
per date and then averaged. It is not a hit rate and not a return; it is the width of a ranking.

| quantity | value | source |
|---|---|---|
| IC (2023 holdout) | 0.0127 | `results/test_2023.json` → `metrics.ic` |
| IC 95% CI | [-0.0102, 0.0356] | `results/test_2023.json` → `metrics.ic_ci95_lo`, `ic_ci95_hi` |
| IC t (NW, HAC) | 0.9383 | `results/test_2023.json` → `metrics.ic_t_nw` |
| DM vs zero, p | 0.2507 | `results/test_2023.json` → `dm_vs_zero.p` |
| sign-hit rate of the daily IC series | 0.5265 | `results/test_2023.json` → `metrics.ic_hit` |
| direction accuracy | 0.5089 | `results/test_2023.json` → `metrics.dir_acc` |
| decile rank correlation | 0.0424 | `results/test_2023.json` → `decile_rank_corr` |
| R² vs zero | 0.00114 | `results/test_2023.json` → `metrics.r2_vs_zero` |

The CI contains zero with 95% confidence, so the point estimate 0.0127 is not distinguishable from
noise at the pre-registered horizon. The pre-registered gate `ic_gt_0.02` fails
(`results/test_2023.json` → `criteria.ic_gt_0.02 = false`), and so does `ic_ci_lo_gt_0`. A point
estimate inside a zero-crossing CI is not a small signal; per the project's own gate it is
**not_confirmed** (`results/test_2023.json` → `verdict`).

The decile curve also fails to order: 245 daily means from thick buckets can still contain noise,
but decile 4 is negative (-0.000136) while decile 10 is +0.002274 — a non-monotone shape
(`results/test_2023.md`, decile table). The rank correlation across 10 deciles is 0.0424.

One number that *does* look significant would mislead if quoted alone: `ic_hit = 0.5265` (129/245
days with positive IC). Its binomial standard error at n=245 is 0.0319, so 0.0265 above 0.5 is
0.83σ. It is not evidence.

**Answer:** because the holdout CI crosses zero, the HAC t is 0.94, the DM p is 0.25, and the
pre-registered 0.02 threshold was not met.

---

## 2. Adding more features — why will that not help?

More features can only help if the current model is under-fitting the *signal*, not under-fitting the
*noise*. Both diagnostics say the opposite.

| diagnostic | value | source |
|---|---|---|
| pooled CV IC, 49 features (A5) | 0.0159 | `results/ablation.md`, column `pooled IC`, row A5 |
| pooled CV IC, 19 market features only (A1) | 0.0009 | `results/ablation.md`, row A1 |
| A5 minus A1 (pooled) | 0.0150 | difference of the two rows above |
| A2 sentiment-only (28) pooled IC | 0.0136 | `results/ablation.md`, row A2 |
| A6 strict-dates (49) pooled IC | 0.0170 | `results/ablation.md`, row A6 |
| holdout IC at 49 features | 0.0127 | `results/test_2023.json` → `metrics.ic` |
| CV IC degradation, pooled → holdout | 0.0159 → 0.0127 | `results/ablation.md` A5 and `results/test_2023.json` |

The sentiment block adds ~0.0150 over market-only in CV and then delivers 0.0127 on holdout. That
is a **shrinkage pattern typical of a weakly informative block**, not headroom. Adding more columns
of the same kind (more lags, more ratios) targets the same 0.52 coverage and the same one-article
median; it enlarges the numerator of variance without adding an independent source of information
about tomorrow's return.

There is a direct test in the repository: A6 changes nothing but the date policy, and its CV IC
rises from 0.0159 to 0.0170 (`results/ablation.md`, A5 vs A6). That +0.0011 is what a
purely-procedural feature-side change buys — an order of magnitude below what would be needed to
move the holdout past the 0.02 gate. This is a hypothesis about the *size* of incremental
feature effects, not an observation about a future feature set: **no expanded feature set has been
trained**, so "will not help" means "there is no measured headroom to justify the cost", not
"proven impossible".

**Answer:** the CV-to-holdout gap (0.0159 → 0.0127) and the +0.0011 from the date-policy variant
(A5→A6) describe a block that is already saturated with respect to this signal, not an under-fitted
one.

---

## 3. Net Sharpe is -6.75 while gross Sharpe is 0.12 — how?

They are the same portfolio before and after friction. Sharpe = annualised mean / annualised
volatility. The portfolio runs a dollar-neutral book of 8 longs and 8 shorts (mean
`avg_n_long = 8.0`, `avg_n_short = 7.8857`, `results/backtest_metrics.json` → `metrics`), rebalanced
every session.

| quantity | value | source |
|---|---|---|
| Sharpe gross | 0.1234 | `results/backtest_metrics.json` → `metrics.sharpe_gross` |
| Sharpe net | -6.7506 | `results/backtest_metrics.json` → `metrics.sharpe_net` |
| annual vol | 0.05311 | `results/backtest_metrics.json` → `metrics.ann_vol` |
| gross arithmetic annual return | 0.006555 | `results/backtest_metrics.json` → `costs.gross_ann` |
| trading cost drag, annual | 0.339557 | `results/backtest_metrics.json` → `costs.cost_ann` |
| borrow drag, annual | 0.025000 | `results/backtest_metrics.json` → `costs.borrow_ann` |
| annual turnover | 169.779 | `results/backtest_metrics.json` → `costs.turnover_ann` |
| net arithmetic annual return | -0.358002 | `results/backtest_metrics.json` → `costs.net_ann_arith` |

The two Sharpes are consistent with one identity. Gross Sharpe × annual vol = 0.1234 × 0.05311 =
0.006555, which equals `costs.gross_ann` exactly. Net arithmetic return = 0.006555 − 0.339557 −
0.025000 = -0.358002, which equals `costs.net_ann_arith`. Dividing by vol gives
-0.358002 / 0.05311 = -6.74, consistent with `sharpe_net = -6.7506` up to the separate
compounded figure `metrics.ann_return = -0.30209` (geometric vs arithmetic, see §11).

The cost term itself closes: 169.779 × 2 × 10 bps = 0.339557 = `costs.cost_ann` (checked by
arithmetic on the artifact; the formula is `turnover × 2 × cost_per_side` with
`cost_per_side = (5+5)/1e4` in `src/spp/backtest/portfolio.py:359`, `:323`).

So the -6.75 is not a sign error. A gross edge of 0.66% a year is being multiplied by a portfolio
that turns the book over 1.7× a week; the edge has to be ~34% a year gross to stay flat after
costs. It is 0.66%.

**Answer:** trade drag (33.96%/yr) plus borrow (2.50%/yr) dwarfs gross edge (0.66%/yr) by a factor of
51.8 (`costs.cost_share_of_gross = 51.7978`).

---

## 4. What would happen at zero costs?

Zero costs would turn a loss into a near-zero, not into a strategy. Both components have to be
removed for the book to break even, and even then the gross number does not clear the
pre-registered gates.

| scenario | annual return | Sharpe | source |
|---|---|---|---|
| net, as traded | -0.358002 (arithmetic) | -6.7506 | `results/backtest_metrics.json` → `costs.net_ann_arith`, `metrics.sharpe_net` |
| zero trading + zero borrow (gross) | 0.006555 | 0.1234 | `results/backtest_metrics.json` → `costs.gross_ann`, `metrics.sharpe_gross` |
| zero borrow only | 0.006555 − 0.339557 = -0.333002 | -6.2700 (derived) | arithmetic on the same artifact; the -6.2700 figure is derived, **not stored in any artifact** |

The gross Sharpe of 0.1234 at 5.31% vol is a portfolio that is flat in expectation. The 2023 gross
return of 0.50% (`results/backtest_bt-2023-next_open_hold_close.md` §6, year 2023, column `gross ret`)
is within the noise band of a long/short book of 16 names. The `null_distribution` in the same JSON
places a score-permuted book's Sharpe at mean -8.55 with q95 -6.81
(`results/backtest_metrics.json` → `null_distribution`), so the observed -6.75 is barely better than
a random book after costs (`headline.null.p_value = 0.06`).

**Answer:** at zero costs the book becomes 0.12 Sharpe gross and ~0.66%/yr — indistinguishable from
flat. Removing costs does not create signal; it only removes the friction that currently dominates it.

---

## 5. Why cannot 2023 be re-run?

Because it is the only untouched year this project has, and a lock file enforces it.

| evidence | value | source |
|---|---|---|
| lock exists | yes, 635 bytes | `reports/test_2023.lock` |
| lock created | 2026-11-08T08:13:10Z | `reports/test_2023.lock` → `created_at` |
| matrix hash recorded at lock time | `fab4f76c5957f756…` | `reports/test_2023.lock` → `matrix_sha256` |
| config hash recorded at lock time | `3a1c947f4f51b8c5…` | `reports/test_2023.lock` → `config_hash` |
| test window | 2023-01-03 .. 2023-12-29 | `reports/test_2023.lock` → `test_window` |
| behaviour on re-run | raises `GBDTError` | `src/spp/models/holdout.py:85-89` |

`holdout.py` checks `if lock_path.exists() and not force: raise`. There is a `--force` flag
(`src/spp/cli.py:975`), and it exists for deliberate reruns only: once used, the year is no longer a
holdout, it is a validation set that informed the design. Any subsequent number computed on 2023
after a feature change is contaminated by the knowledge that the first run returned
`not_confirmed`, because the change was selected *against* that result.

The rule is not cosmetic. The designed number of holdout uses is one (`results/test_2023.md` →
`# 2023 holdout — single run`). Reproducing the *existing* artifacts is allowed and does not consume
the holdout; it regenerates `results/test_2023.json` from the same locked config and matrix.

**Answer:** `reports/test_2023.lock` blocks a second look, and any `--force` run would stop 2023
from being out of sample. Further claims about 2023 must come from the single existing run.

---

## 6. The final model has 11 trees — why?

The premise conflates two different artifacts. The **deployed** model does not have 11 trees.

| artifact | trees | source |
|---|---|---|
| deployed final model, seed 1/2/3 | 50 each | `reports/runs/gbdt-e124617157/final/seed_{1,2,3}/model.txt`, `Tree=` count = 50; `model_meta.json` → `params.n_estimators = 50` |
| median early-stopped length over 18 fold seed-fits | 6.5 (stored as 6) | values 3,9,13,37,2,24,16,2,136,2,4,3,19,3,2,22,3,47 in `reports/runs/gbdt-e124617157/metrics.json` → `folds[].seed_meta[].best_iteration`; `final_model.n_estimators = 6` (the code applies `int()` to 6.5) |
| A1 market-only final model | 11 | `reports/runs/a1-market-only/metrics.json` → `final_model.n_estimators = 11` |
| A4 lexicon final model | 11 | `reports/runs/a4-lexicon/metrics.json` → `final_model.n_estimators = 11` |

So 11 comes from the ablation runs, not the holdout model. The 50 in the deployed model is a **floor
imposed in code**, not a tuned value: `src/spp/models/gbdt.py:1068` takes
`n_final = int(np.median(lengths))`, which truncates 6.5 to 6, and `src/spp/models/gbdt.py:1073`
then does `params["n_estimators"] = max(n_final, 50)`. The persisted `model_meta.json` keeps
`n_estimators_from: "median early-stopped length across folds"` while storing 50, so the metadata
string does not describe the stored number.

Why so few trees at all: early stopping monitors IC on a 60-session inner validation block
(`configs/final.yaml`, `model.early_stopping_rounds: 100`) and triggers between 2 and 136
iterations. The signal is weak enough that almost any added tree capacity degrades validation IC.
This is a real finding about the signal, and also a limitation: it means the result describes a
**linearly-weak** signal, not the ceiling of what news can carry (`docs/limitations.md` §3).

**Answer:** 11 trees is the A1/A4 ablation config. The deployed model stores 50, produced by the
`max(n_final, 50)` floor over a median early-stopped length of 6.5.

---

## 7. Lexicon A4 came out at -0.0079 — how strong is that conclusion?

It is a **negative in CV and silent on the holdout**, and the two facts have to be quoted together.

| quantity | value | source |
|---|---|---|
| A4 aggregate CV IC | -0.0079 | `results/ablation.md`, row A4; `reports/runs/a4-lexicon/metrics.json` → `aggregate.ic.mean = -0.007921` |
| A4 pooled CV IC | -0.0077 | `results/ablation.md`, row A4, column `pooled IC` |
| A4 CV 95% CI | [-0.0228, 0.0069] | `results/ablation.md`, row A4 |
| A4 DM p vs zero | 0.0000193 | `reports/runs/a4-lexicon/metrics.json` → `dm_vs_zero.p = 1.9327e-05` |
| A4 per-fold IC | -0.0008, +0.0258, -0.0200, -0.0257, -0.0110, -0.0158 | `reports/runs/a4-lexicon/metrics.json` → `aggregate.ic.per_fold` |
| lexicon calibration accuracy | 0.4000 | `results/calibration/lexicon.json` → `accuracy` |
| lexicon calibration macro-F1 | 0.3318 | `results/calibration/lexicon.json` → `macro_f1` |
| lexicon ECE after temperature scaling | 0.2566 | `results/calibration/lexicon.json` → `ece` |
| lexicon 5-class accuracy | 0.2286 | `results/calibration/lexicon.json` → `label_5_accuracy` |
| FinBERT macro-F1, same split | 0.8862 | `results/calibration/finbert.json` → `macro_f1` |

What the evidence supports: under this pipeline the lexicon backend produces a CV IC of the wrong
sign, 5 of 6 folds are non-positive, and the DM p of 1.9e-05 says the loss versus a zero forecast is
not a fluke within CV. The calibration file explains the mechanism: a dictionary that
mis-labels 60% of headlines cannot carry a directional feature.

What it does **not** support, and this is where the claim must stop:

1. **No holdout comparison.** A4 was never run on 2023. The holdout verdict `not_confirmed` belongs
   to the FinBERT A5 model (`results/test_2023.json` → `seeds`, `verdict`). "Lexicon is worse than
   FinBERT" is a CV statement, not a holdout statement.
2. **The CI still contains zero**: [-0.0228, 0.0069]. The sign is stable, the magnitude is not
   bounded away from zero in the usual 95% sense.
3. The DM statistic compares loss, not IC. It says the lexicon predictions are worse than a zero
   forecast in squared error, which is consistent with a bad backend but is not a test of IC.

**Answer:** the conclusion "the lexicon backend is not usable here" is well supported in CV (5 of 6
folds negative, DM p 1.9e-05, macro-F1 0.33) and is explanatory for a bad feature; the conclusion
"FinBERT is better than lexicon" is a CV inference only, and the CI on the A4 IC still crosses zero.

---

## 8. What would an intraday approach fix, and what would it not fix?

The proposal is specified in `docs/idea-intraday-t5.md`. It has not been run: the repository
contains no minute bars (`docs/limitations.md` §6) and no `reports/intraday_baseline.md`.

**Would fix — mechanical defects of the daily dataset:**

| defect in the daily approach | evidence | how intraday removes it |
|---|---|---|
| a full-session lag dilutes the news effect | target is `log(adj_close[t+1]/adj_close[t])` | entry at `published_at + 5 min`, exit at session close (`docs/idea-intraday-t5.md:24-26`) |
| the day policy (`published_at ≤ 16:00 ET` + NYSE calendar) is three places to err | `strict_dates` moved 10 rows of 124,571 (`docs/idea-intraday-t5.md:39`) | the article timestamp *is* the event, no day policy (`:37-40`) |
| daily aggregation mixes events | median 1 article per ticker-day (`docs/limitations.md` §2) | one observation per article (`docs/idea-intraday-t5.md:41-43`) |
| duplicates read as strong sentiment | dedupe exists but daily share is diluted | dedupe by `(title_hash, 6h window)` (`:50-52`) |
| `adj_close` is back-adjusted ex post | IC 0.0459 on adj vs 0.0048 on tradable in the 2023 run (`results/backtest_metrics.json` → `ic.adj_close_ret.ic`, `ic.pooled.ic`) | prices come from bars, not adjusted closes (`:44-46`) |
| turnover is a property of the daily book | 169.78×/yr (`results/backtest_metrics.json` → `costs.turnover_ann`) | turnover set by top-k event selection (`:86-87`) |

**Would not fix:**

1. **The need for minute bars.** This is the first blocker and it is purchased, not written
   (`docs/idea-intraday-t5.md:77-79`). Until it exists, no number in the proposal is measurable.
2. **The possibility that the effect is simply zero at 5 minutes.** The proposal states this
   explicitly: a null at 5 min means the daily failure was absence of predictability, not coarse
   aggregation (`docs/idea-intraday-t5.md:100-104`). It also demands a 30/60-minute lag grid as a
   control for speed-of-reaction (`:88-89`).
3. **The admission-is-late problem.** Entry at `published_at + 5 min` with a public timestamp does
   not model quote staleness or queue position; the 5+5 bps + half-spread charge (`:86`) is the only
   guard.
4. **The existence of *any* tradable edge.** Even if intraday removes every mechanical defect, the
   expected gross effect from an article-level event is unmeasured — and the daily gross edge was
   0.66%/yr (`results/backtest_metrics.json` → `costs.gross_ann`).

**Answer:** it fixes the lag, the day policy, the aggregation, the duplicate inflation, the
`adj_close` contamination and the turnover explosion. It does not supply microsecond admission, does
not provide the minute bars, and does not change the prior probability that the effect is zero.

---

## 9. open-jev was never expected to help — why?

Because on the artifacts that exist, it cannot carry the two features it is uniquely able to fill,
and its calibration sample is too small to say anything.

| quantity | value | source |
|---|---|---|
| open-jev calibration macro-F1 | 0.6080 | `results/calibration/openjev.json` → `macro_f1` |
| open-jev calibration accuracy | 0.56 | `results/calibration/openjev.json` → `accuracy` |
| open-jev eval rows | 25 (`n_eval`), 36 (`rows_evaluated`) | `results/calibration/openjev.json` |
| open-jev per-class support | neg 5, neu 17, pos 3 | `results/calibration/openjev.json` → `per_class_support` |
| open-jev ECE after scaling | 0.2198 | `results/calibration/openjev.json` → `ece` |
| open-jev throughput | 1.0674 items/s | `results/calibration/openjev.json` → `throughput` |
| FinBERT throughput, same file shape | 12.5707 items/s | `results/calibration/finbert.json` → `throughput` |
| articles in the corpus | 124,571 | `README.md`, News line; `docs/limitations.md` §2 |
| unique texts to score | 92,927 | `docs/limitations.md` §1 |
| texts covered at publication time | 5.9 thousand | `docs/limitations.md` §1 |
| minimum macro-F1 the config would accept | 0.70 | `configs/final.yaml` → `sentiment.min_macro_f1` |
| novelty/company-specific coverage | 0.000 | `README.md` feature-coverage table |

Four reasons, in order of strength:

1. **It cannot fill the columns it exists for.** `novelty_mean` and `company_specific_share` are
   all-NaN in every matrix built so far (coverage 0.000, `README.md`). Only open-jev fills
   `is_novel` / `is_company_specific` (`README.md`); FinBERT and lexicon do not. With A3/A8 unclosed
   (`results/ablation.md`, A3 and A8 = "—"), open-jev's entire argument collapses to "another
   sentiment classifier", and a second classifier on the same 1-article-per-day aggregate is not a
   new information source.
2. **The corpus is not scored.** 5.9k of 92,927 texts at publication (`docs/limitations.md` §1). At
   the measured 1.0674 items/s a full pass is about 24 hours of wall clock, not the 7.5 hours
   stated in `limitations.md` §1 (that text quotes ~3.4 items/s); the artifact is the source of
   record here.
3. **Calibration is not informative at n=25.** Support of 5 negative and 3 positive makes any
   macro-F1 unstable, and its 0.6080 would have been **rejected by the config's own 0.70
   threshold** (`configs/final.yaml` → `sentiment.min_macro_f1`), even ignoring the sample size.
4. **Prior reason, not just sample reason.** The daily signal saturates around IC 0.0127–0.0170
   (`results/test_2023.json`, `results/ablation.md` A5/A6) while feature coverage is 0.52. Adding a
   feature that describes *whether* a headline is novel still leaves 48% of ticker-days with no
   article. A typed flag on top of a sparse event cannot manufacture events.

**Answer:** a priori, open-jev's contribution is a flag on rows that half the time do not exist, at
1.07 items/s, with a calibration macro-F1 of 0.6080 on 25 rows — below the project's own 0.70 gate.
That is why it was expected to be neutral, and why the neutral outcome was not a surprise.

---

## 10. The 2023 holdout `verdict` is `not_confirmed` — what exactly failed?

Four pre-registered gates, all four false (`results/test_2023.json` → `criteria`; repeated as
`results/test_2023.md`):

| gate | threshold | observed | pass |
|---|---|---|---|
| `ic_gt_0.02` | IC > 0.02 | 0.0127 | no |
| `ic_ci_lo_gt_0` | CI lower bound > 0 | -0.0102 | no |
| `dm_p_lt_0.05` | DM p < 0.05 | 0.2507 | no |
| `decile_monotonicity_gt_0.5` | > 0.5 | 0.0424 (`decile_rank_corr`) | no |

The economic gate is separate and lives in the backtest: `bootstrap_sharpe_ci_lo_gt_0 = false`
(`results/backtest_metrics.json` → `headline.gates`), with the net Sharpe CI
[-9.2218, -4.6639] (`bootstrap_sharpe.ci95`). The verdict is produced by the gates, not by a Sharpe
target — the backtest report says so in §8 of
`results/backtest_bt-2023-next_open_hold_close.md`.

**Answer:** IC level, IC sign, DM significance, decile monotonicity and economic Sharpe. All five.

---

## 11. Which Sharpe is the real one — I see -6.75, -4.08 and -3.08 in different files?

They are different execution models on the same 2023 scores, and one of them is a different year
range. Choosing between them is not optional; the headline below is the one the pre-registered
config names.

| run | execution | window | gross | net | source |
|---|---|---|---|---|---|
| `bt-2023-next_open_hold_close` | next open → close | 2023-01-03..2023-12-21 | 0.1234 | -6.7506 | `results/backtest_bt-2023-next_open_hold_close.md` §1 |
| `bt-2023-close_to_close` | close → close | 2023-01-03..2023-12-21 | 1.2730 | -4.0816 | `results/backtest_bt-2023-close_to_close.md` §1 |
| `bt-cv-next_open_hold_close` | next open → close | 2021-07-01..2022-12-30 | 0.7063 | -4.2096 | `results/backtest_bt-cv-next_open_hold_close.md` §1 |
| `bt-cv-close_to_close` | close → close | 2021-07-01..2022-12-30 | 1.2035 | -3.0783 | `results/backtest_bt-cv-close_to_close.md` §1 |

`configs/final.yaml` → `backtest.execution: next_open_hold_close`, so -6.7506 is the config's
execution. Close-to-close is the optimistic one because it buys at the same close used to compute
the signal's reference return — it is kept as a diagnostic, not the headline. The CV-window runs are
on 2021–2022 and cannot be compared to the holdout.

Caveat inside a single artifact: `results/backtest_metrics.json` holds `metrics.ann_return =
-0.30209` (geometric) alongside `annual["2023"].ret = -0.29508` and `costs.net_ann_arith =
-0.35800` (arithmetic). The three differ because of compounding and because the arithmetic figure is
`gross_ann − cost_ann − borrow_ann`, not a sum of daily compounded returns. Quoting one without the
other is a source of confusion; the Sharpe values are consistent with the arithmetic figure (§3).

**Answer:** -6.7506, because `configs/final.yaml` selects `next_open_hold_close`; -4.0816 and -3.08
are the close-to-close diagnostics and the CV window.

---

## 12. Was the study pre-registered, or was the design chosen after seeing 2023?

Pre-registered for the holdout. The lock is written before the model is fitted.

| evidence | value | source |
|---|---|---|
| lock created before scoring | `created_at` 2026-11-08T08:13:10Z, identical to `results/test_2023.json` → `created_at` | `reports/test_2023.lock`, `results/test_2023.json` |
| lock stores config hash | `3a1c947f4f51b8c5` | `reports/test_2023.lock` → `config_hash` |
| lock stores matrix hash | `fab4f76c5957f756` | `reports/test_2023.lock` → `matrix_sha256` |
| gates are enumerated before the run | 4 criteria, all false, printed in the artifact | `results/test_2023.json` → `criteria` |
| model code refuses without the lock | `holdout.py:85` | `src/spp/models/holdout.py` |
| model selection used CV only | A5 chosen "best purged-CV IC among the ablations run" | `configs/final.yaml`, header comment |
| feature counts | 49 (19 market + 30 sentiment) | `data/processed/features.parquet.manifest.json` → `n_features`; `results/leakage_checks.json` → `meta.n_features` |

What is *not* perfectly pre-registered: `configs/final.yaml`'s header says "62 columns" while the
matrix has 49 features (two sentiment columns are dropped as all-NaN, `README.md`), and the A5 pick
was made by comparing ablations in CV — legitimate, but it means the final feature set was selected
on 2019–2022 data, and the holdout inherits that selection bias. Six leakage checks pass
(`results/leakage_checks.json` → `passed: true`, `failed: []`), including the shuffled-label check
(IC -0.0106, threshold 0.0171) and the future-shift check (0.0392 vs 0.0490 unshifted).

**Answer:** yes for the holdout, with two documented imperfections: a stale header comment (62 vs 49)
and an A5 feature selection made on CV data before the lock.

---

## 13. Are the calibration numbers good enough to support FinBERT?

They measure the **classifier**, not the trading signal, and they are measured on a different corpus.

| quantity | value | source |
|---|---|---|
| calibration sample | `n_eval` 210 of 300 rows; `rows_before_cap` 3453 | `results/calibration/finbert.json` |
| FinBERT accuracy | 0.9238 | `results/calibration/finbert.json` → `accuracy` |
| FinBERT macro-F1 | 0.8862 | `results/calibration/finbert.json` → `macro_f1` |
| per-class support | neg 34, neu 153, pos 23 | `results/calibration/finbert.json` → `per_class_support` |
| ECE before / after temperature | 0.0531 → 0.0401 | `results/calibration/finbert.json` |
| FinBERT 5-class accuracy | 0.5190 | `results/calibration/finbert.json` → `label_5_accuracy` |

The data is Financial PhraseBank, not the FNSPID headlines the model actually sees. The project has
no labelled headline set, so transfer to its own corpus is **not measured**
(`docs/limitations.md` §8). A macro-F1 of 0.8862 on 210 sentences with 153 neutral is dominated by
the neutral class; the ECE of 0.04 is the number that matters for feature construction, and it is
good. What is unknown is whether that calibration survives the headline distribution. The 5-class
accuracy of 0.5190 is the closest available hint that the finer granularity is much weaker.

**Answer:** sufficient to justify using FinBERT as the sentiment backend, insufficient to claim its
outputs are well calibrated on FNSPID headlines — that transfer is untested.

---

## 14. How do I verify any of this myself?

Every number in this file has a path. The fastest checks, in order of cost:

```powershell
# 1. reopen the holdout artifact (no computation)
Get-Content results/test_2023.json -Raw

# 2. reopen the backtest artifact and redo the cost identity
Get-Content results/backtest_metrics.json -Raw
# 169.779 * 2 * 10 / 1e4 = 0.339557 = costs.cost_ann
# 0.006555 - 0.339557 - 0.025000 = -0.358002 = costs.net_ann_arith

# 3. re-read the CV ablations (no training)
Get-Content results/ablation.md -Raw
```

Things that are **forbidden** as verification, because they consume the holdout or the GPU:
`python -m spp.cli train … --holdout`, `spp features build`, any second run of the 2023 scorer.
A lock file (`reports/test_2023.lock`) enforces the first of these; the other two are blocked by
policy, not by code.

---

## 15. What would make the verdict change?

Not a new feature set, and not a re-run of 2023 (see §5). The gate is `ic_ci_lo_gt_0` and
`ic_gt_0.02` together. The CI is `ic ± 1.96 × ic_se`; with `ic_se = 0.011675`
(`results/test_2023.json` → `metrics.ic_se`) the half-width is 0.02288, so the lower bound clears
zero only if the point estimate reaches 0.0229 — **1.80× the observed 0.0127** and 1.14× the 0.02
gate itself. In this repository, the largest CV IC observed under any of the closed ablations is
0.0167 (A6, `results/ablation.md`), and the holdout IC at 49 features is 0.0127
(`results/test_2023.json`). Neither is within a factor of 1.2 of the gate.

Stated honestly: raising IC from 0.0127 to 0.0229 is not absurd for a weak signal, and a larger
study could clear it by sample size alone if the point estimate were stable. What is not
reasonable is expecting that movement from the same daily aggregates — the CV evidence (§2) shows
a +0.0011 movement from a procedural change, an order of magnitude short of the 0.0102 required.
This last sentence is an inference from measured deltas, marked as such, not a measured outcome.

The honest next step is the one written in `docs/idea-intraday-t5.md`: buy minute bars, measure the
event-level response at 5/15/30/60 minutes, and accept a null at 5 minutes as a result. A new
daily feature set would be measuring the same 0.52 coverage and the same one-article median — and
would have no unfiltered year left to test it on.

**Answer:** a different measurement horizon with minute data, not the same one with more columns.
