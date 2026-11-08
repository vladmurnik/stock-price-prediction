# Backtest report — bt-2023-next_open_hold_close

Execution: `next_open_hold_close` · window 2023-01-03 .. 2023-12-21

> Costs: 5 bps commission + 5 bps slippage per side, 5% annual borrow on shorts,
> dollar-neutral, gross 1.0. Returns are raw-price (tradable), not `adj_close`.

## 1. Metrics

| metric | value |
|---|---|
| sharpe_gross | 0.1234 |
| sharpe_net | -6.7506 |
| sortino | -5.7568 |
| ann_return | -30.21% |
| ann_vol | 5.31% |
| max_dd | -29.29% |
| calmar | -1.0313 |
| hit_rate_daily | 31.43% |
| turnover_ann | 169.7786 |
| avg_n_long | 8.0000 |
| avg_n_short | 7.8857 |
| cost_drag_ann | 33.96% |
| borrow_drag_ann | 2.50% |
| dates | 245 (245 traded, 0 skipped) |
| IC (pooled, tradable ret) | 0.0048 | t=0.4148 | NW t=0.4269 | CI95 [-0.0176, 0.0263] |

Sharpe net block-bootstrap CI95 [-9.2218, -4.6639], P(Sharpe<=0)=1.0000 (block=20, n=1000).
Null (scores permuted per date, same book and costs, n=50): mean Sharpe -8.5530, 90% range [-10.2801, -6.8120], p(observed)=0.0600.

## 2. Equity curve

![equity](../figures/equity_bt-2023-next_open_hold_close.png)
![drawdown](../figures/drawdown_bt-2023-next_open_hold_close.png)

## 3. Daily return distribution

![hist](../figures/hist_bt-2023-next_open_hold_close.png)

## 4. Returns by predicted quintile

| bucket (1 = lowest score) | mean daily ret | ann. ret | t | n dates |
|---|---|---|---|---|
| 1 | 0.07% | 17.18% | 1.16 | 245 |
| 2 | 0.07% | 18.53% | 1.48 | 245 |
| 3 | 0.02% | 4.12% | 0.36 | 245 |
| 4 | 0.06% | 13.98% | 1.10 | 245 |
| 5 | 0.07% | 18.18% | 1.28 | 245 |

Monotonicity (Spearman bucket→return): -0.100; top−bottom 0.00% per day.

## 5. Cost decomposition (annualised)

| item | value |
|---|---|
| gross P&L | 0.66% |
| trading cost drag | 33.96% |
| borrow drag | 2.50% |
| net (arithmetic) | -35.80% |
| annual turnover | 169.8x |

## 6. Yearly breakdown and DM tests

| year | net ret | gross ret | sharpe_net | n |
|---|---|---|---|---|
| 2023 | -29.51% | 0.50% | -6.75 | 245 |

- DM vs_zero: dm=8.14 p=0.000 (n=245); dm>0 means the strategy has larger squared loss.
- DM vs_spy_long_only: dm=-6.88 p=0.000 (n=245); dm>0 means the strategy has larger squared loss.

## 7. Data quality

- halted / zero-range rows: 36
- rows without a forward price (kept NaN): 0
- dates flagged (not traded): 0 {}
- IC on `adj_close` returns: 0.0459 (t=3.4749) vs tradable IC 0.0048; mean(adj−raw) per row -0.0769. `adj_close` is back-adjusted ex-post, so its within-session return is not tradable.
- IC without the top-1% |return| rows: 0.0049 (t=0.4104, dropped 98 rows).

## 8. Verdict

- gate `ic_gt_0.02`: **FAIL**
- gate `ic_ci_lo_gt_0`: **FAIL**
- gate `quintile_monotonicity_gt_0.5`: **FAIL**
- gate `bootstrap_sharpe_ci_lo_gt_0`: **FAIL**

**Verdict: not_confirmed**

Criteria are the pre-registered T10 gates, not a Sharpe target. A negative verdict is a publishable result: it says this signal does not pay for its own costs on this universe.

## 9. Reproduce

```powershell
python -m spp.cli backtest --run-id bt-2023-next_open_hold_close --execution next_open_hold_close
```
