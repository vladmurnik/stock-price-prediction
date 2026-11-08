# Аблации: IC на purged CV (2019–2022)

Все числа сгенерированы `ops/make_ablation_report.py` из `reports/runs/*/metrics.json`.
6 фолдов × 3 сида, цель — demeaned лог-доходность следующего дня.

| id | признаки | бэкенд | strict_dates | IC | 95% CI | pooled IC | DM p |
|---|---|---|---|---|---|---|---|
| **A1** | market only | finbert | false | 0.0007 | [-0.0151, 0.0165] | 0.0009 | 0.0039 |
| **A2** | sentiment only | finbert | false | 0.0134 | [-0.0052, 0.0320] | 0.0136 | 0.0074 |
| **A5** | market + sentiment | finbert | false | 0.0156 | [-0.0069, 0.0381] | 0.0159 | 0.0054 |
| **A4** | market + sentiment | lexicon | false | -0.0079 | [-0.0228, 0.0069] | -0.0077 | 0.0000 |
| **A6** | market + sentiment | finbert | true | 0.0167 | [-0.0021, 0.0356] | 0.0170 | 0.0042 |
| A3 | sentiment only | open-jev | false | — | — | — | — |
| A8 | market + sentiment + typed flags | open-jev | false | — | — | — | — |

## Единственный holdout 2023

- IC **0.0127**, 95% CI [-0.0102, 0.0356]
- direction accuracy 0.5089, DM p 0.2507
- вердикт: **not_confirmed**

Критерии: decile_monotonicity_gt_0.5=False, dm_p_lt_0.05=False, ic_ci_lo_gt_0=False, ic_gt_0.02=False
