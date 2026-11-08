# Backtest report — bt-cv-close_to_close

Execution: `close_to_close` · window 2021-07-01 .. 2022-12-30

> Costs: 5 bps commission + 5 bps slippage per side, 5% annual borrow on shorts,
> dollar-neutral, gross 1.0. Returns are raw-price (tradable), not `adj_close`.

## 1. Metrics

| metric | value |
|---|---|
| sharpe_gross | 1.2035 |
| sharpe_net | -3.0783 |
| sortino | -2.8627 |
| ann_return | -22.95% |
| ann_vol | 8.34% |
| max_dd | -33.43% |
| calmar | -0.6865 |
| hit_rate_daily | 40.90% |
| turnover_ann | 166.2210 |
| avg_n_long | 8.0000 |
| avg_n_short | 7.8839 |
| cost_drag_ann | 33.24% |
| borrow_drag_ann | 2.50% |
| dates | 379 (379 traded, 0 skipped) |
| IC (pooled, tradable ret) | 0.0193 | t=1.9289 | NW t=2.0036 | CI95 [-0.0001, 0.0357] |

Sharpe net block-bootstrap CI95 [-4.2491, -1.9308], P(Sharpe<=0)=1.0000 (block=20, n=1000).
Null (scores permuted per date, same book and costs, n=50): mean Sharpe -5.9480, 90% range [-7.1588, -4.4891], p(observed)=0.0000.

## 2. Equity curve

![equity](../figures/equity_bt-cv-close_to_close.png)
![drawdown](../figures/drawdown_bt-cv-close_to_close.png)

## 3. Daily return distribution

![hist](../figures/hist_bt-cv-close_to_close.png)

## 4. Returns by predicted quintile

| bucket (1 = lowest score) | mean daily ret | ann. ret | t | n dates |
|---|---|---|---|---|
| 1 | -0.04% | -10.95% | -0.54 | 379 |
| 2 | -0.04% | -10.67% | -0.64 | 379 |
| 3 | -0.00% | -0.67% | -0.04 | 379 |
| 4 | -0.02% | -6.24% | -0.35 | 379 |
| 5 | 0.01% | 3.70% | 0.18 | 379 |

Monotonicity (Spearman bucket→return): 0.900; top−bottom 0.06% per day.

## 5. Cost decomposition (annualised)

| item | value |
|---|---|
| gross P&L | 10.04% |
| trading cost drag | 33.24% |
| borrow drag | 2.50% |
| net (arithmetic) | -25.71% |
| annual turnover | 166.2x |

## 6. Yearly breakdown and DM tests

| year | net ret | gross ret | sharpe_net | n |
|---|---|---|---|---|
| 2021 | -11.70% | 5.12% | -2.89 | 128 |
| 2022 | -23.49% | 10.05% | -3.17 | 251 |

- DM vs_zero: dm=7.58 p=0.000 (n=379); dm>0 means the strategy has larger squared loss.
- DM vs_spy_long_only: dm=-8.35 p=0.000 (n=379); dm>0 means the strategy has larger squared loss.

## 7. Data quality

- halted / zero-range rows: 48
- rows without a forward price (kept NaN): 0
- dates flagged (not traded): 0 {}
- IC on `adj_close` returns: 0.0205 (t=2.0454) vs tradable IC 0.0193; mean(adj−raw) per row 0.0001. `adj_close` is back-adjusted ex-post, so its within-session return is not tradable.
- IC without the top-1% |return| rows: 0.0151 (t=1.4870, dropped 152 rows).

## 8. Verdict

- gate `ic_gt_0.02`: **FAIL**
- gate `ic_ci_lo_gt_0`: **FAIL**
- gate `quintile_monotonicity_gt_0.5`: **PASS**
- gate `bootstrap_sharpe_ci_lo_gt_0`: **FAIL**

**Verdict: insufficient_evidence**

Criteria are the pre-registered T10 gates, not a Sharpe target. A negative verdict is a publishable result: it says this signal does not pay for its own costs on this universe.

## 9. Reproduce

```powershell
python -m spp.cli backtest --run-id bt-cv-close_to_close --execution close_to_close
```
