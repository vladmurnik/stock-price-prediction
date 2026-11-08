# Backtest report — bt-cv-next_open_hold_close

Execution: `next_open_hold_close` · window 2021-07-01 .. 2022-12-30

> Costs: 5 bps commission + 5 bps slippage per side, 5% annual borrow on shorts,
> dollar-neutral, gross 1.0. Returns are raw-price (tradable), not `adj_close`.

## 1. Metrics

| metric | value |
|---|---|
| sharpe_gross | 0.7063 |
| sharpe_net | -4.2096 |
| sortino | -3.8472 |
| ann_return | -26.58% |
| ann_vol | 7.26% |
| max_dd | -37.23% |
| calmar | -0.7138 |
| hit_rate_daily | 37.20% |
| turnover_ann | 166.2210 |
| avg_n_long | 8.0000 |
| avg_n_short | 7.8839 |
| cost_drag_ann | 33.24% |
| borrow_drag_ann | 2.50% |
| dates | 379 (379 traded, 0 skipped) |
| IC (pooled, tradable ret) | 0.0152 | t=1.5647 | NW t=1.6516 | CI95 [-0.0028, 0.0314] |

Sharpe net block-bootstrap CI95 [-5.3786, -3.0109], P(Sharpe<=0)=1.0000 (block=20, n=1000).
Null (scores permuted per date, same book and costs, n=50): mean Sharpe -7.3514, 90% range [-8.4660, -6.1872], p(observed)=0.0000.

## 2. Equity curve

![equity](../figures/equity_bt-cv-next_open_hold_close.png)
![drawdown](../figures/drawdown_bt-cv-next_open_hold_close.png)

## 3. Daily return distribution

![hist](../figures/hist_bt-cv-next_open_hold_close.png)

## 4. Returns by predicted quintile

| bucket (1 = lowest score) | mean daily ret | ann. ret | t | n dates |
|---|---|---|---|---|
| 1 | -0.01% | -2.36% | -0.14 | 379 |
| 2 | -0.01% | -1.28% | -0.09 | 379 |
| 3 | -0.01% | -2.49% | -0.20 | 379 |
| 4 | -0.01% | -3.78% | -0.27 | 379 |
| 5 | 0.01% | 1.94% | 0.11 | 379 |

Monotonicity (Spearman bucket→return): 0.100; top−bottom 0.02% per day.

## 5. Cost decomposition (annualised)

| item | value |
|---|---|
| gross P&L | 5.13% |
| trading cost drag | 33.24% |
| borrow drag | 2.50% |
| net (arithmetic) | -30.61% |
| annual turnover | 166.2x |

## 6. Yearly breakdown and DM tests

| year | net ret | gross ret | sharpe_net | n |
|---|---|---|---|---|
| 2021 | -13.37% | 3.13% | -3.81 | 128 |
| 2022 | -27.47% | 4.33% | -4.41 | 251 |

- DM vs_zero: dm=6.85 p=0.000 (n=379); dm>0 means the strategy has larger squared loss.
- DM vs_spy_long_only: dm=-8.10 p=0.000 (n=379); dm>0 means the strategy has larger squared loss.

## 7. Data quality

- halted / zero-range rows: 40
- rows without a forward price (kept NaN): 0
- dates flagged (not traded): 0 {}
- IC on `adj_close` returns: 0.0921 (t=8.3260) vs tradable IC 0.0152; mean(adj−raw) per row -0.1053. `adj_close` is back-adjusted ex-post, so its within-session return is not tradable.
- IC without the top-1% |return| rows: 0.0136 (t=1.3961, dropped 152 rows).

## 8. Verdict

- gate `ic_gt_0.02`: **FAIL**
- gate `ic_ci_lo_gt_0`: **FAIL**
- gate `quintile_monotonicity_gt_0.5`: **FAIL**
- gate `bootstrap_sharpe_ci_lo_gt_0`: **FAIL**

**Verdict: not_confirmed**

Criteria are the pre-registered T10 gates, not a Sharpe target. A negative verdict is a publishable result: it says this signal does not pay for its own costs on this universe.

## 9. Reproduce

```powershell
python -m spp.cli backtest --run-id bt-cv-next_open_hold_close --execution next_open_hold_close
```
