# Backtest report — bt-2023-close_to_close

Execution: `close_to_close` · window 2023-01-03 .. 2023-12-21

> Costs: 5 bps commission + 5 bps slippage per side, 5% annual borrow on shorts,
> dollar-neutral, gross 1.0. Returns are raw-price (tradable), not `adj_close`.

## 1. Metrics

| metric | value |
|---|---|
| sharpe_gross | 1.2730 |
| sharpe_net | -4.0816 |
| sortino | -3.7566 |
| ann_return | -24.44% |
| ann_vol | 6.82% |
| max_dd | -23.38% |
| calmar | -1.0452 |
| hit_rate_daily | 39.18% |
| turnover_ann | 169.7786 |
| avg_n_long | 8.0000 |
| avg_n_short | 7.8857 |
| cost_drag_ann | 33.96% |
| borrow_drag_ann | 2.50% |
| dates | 245 (245 traded, 0 skipped) |
| IC (pooled, tradable ret) | 0.0141 | t=1.2083 | NW t=1.0372 | CI95 [-0.0136, 0.0392] |

Sharpe net block-bootstrap CI95 [-6.1927, -2.0326], P(Sharpe<=0)=1.0000 (block=20, n=1000).
Null (scores permuted per date, same book and costs, n=50): mean Sharpe -6.9848, 90% range [-9.0987, -5.4815], p(observed)=0.0000.

## 2. Equity curve

![equity](../figures/equity_bt-2023-close_to_close.png)
![drawdown](../figures/drawdown_bt-2023-close_to_close.png)

## 3. Daily return distribution

![hist](../figures/hist_bt-2023-close_to_close.png)

## 4. Returns by predicted quintile

| bucket (1 = lowest score) | mean daily ret | ann. ret | t | n dates |
|---|---|---|---|---|
| 1 | 0.06% | 15.74% | 0.85 | 245 |
| 2 | 0.08% | 20.83% | 1.39 | 245 |
| 3 | 0.03% | 7.03% | 0.51 | 245 |
| 4 | 0.04% | 11.03% | 0.75 | 245 |
| 5 | 0.11% | 27.74% | 1.65 | 245 |

Monotonicity (Spearman bucket→return): 0.200; top−bottom 0.05% per day.

## 5. Cost decomposition (annualised)

| item | value |
|---|---|
| gross P&L | 8.68% |
| trading cost drag | 33.96% |
| borrow drag | 2.50% |
| net (arithmetic) | -27.78% |
| annual turnover | 169.8x |

## 6. Yearly breakdown and DM tests

| year | net ret | gross ret | sharpe_net | n |
|---|---|---|---|---|
| 2023 | -23.85% | 8.56% | -4.08 | 245 |

- DM vs_zero: dm=7.78 p=0.000 (n=245); dm>0 means the strategy has larger squared loss.
- DM vs_spy_long_only: dm=-7.33 p=0.000 (n=245); dm>0 means the strategy has larger squared loss.

## 7. Data quality

- halted / zero-range rows: 26
- rows without a forward price (kept NaN): 0
- dates flagged (not traded): 0 {}
- IC on `adj_close` returns: 0.0127 (t=1.0892) vs tradable IC 0.0141; mean(adj−raw) per row 0.0001. `adj_close` is back-adjusted ex-post, so its within-session return is not tradable.
- IC without the top-1% |return| rows: 0.0129 (t=1.0905, dropped 98 rows).

## 8. Verdict

- gate `ic_gt_0.02`: **FAIL**
- gate `ic_ci_lo_gt_0`: **FAIL**
- gate `quintile_monotonicity_gt_0.5`: **FAIL**
- gate `bootstrap_sharpe_ci_lo_gt_0`: **FAIL**

**Verdict: not_confirmed**

Criteria are the pre-registered T10 gates, not a Sharpe target. A negative verdict is a publishable result: it says this signal does not pay for its own costs on this universe.

## 9. Reproduce

```powershell
python -m spp.cli backtest --run-id bt-2023-close_to_close --execution close_to_close
```
