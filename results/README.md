# Что лежит в `results/`

Каждый файл — артефакт прогона, а не ручная запись. Числа в `docs/` и README берутся отсюда;
если файл и текст расходятся, прав файл. Формат: `*.json` — машинный дамп, `*.md` — таблица,
сгенерированная тем же скриптом из того же дампа.

## Крипто-ветка

| артефакт | что измеряет | скрипт |
|---|---|---|
| `crypto_dataset.json` | воронка датасета: сколько заголовков, сколько пар, сколько выброшено | `ops/crypto_dataset.py` |
| `crypto_audit.json` | четыре проверки утечек: экспозиция, лаг, пропущенная реакция, бенчмарк | `ops/crypto_audit.py` |
| `crypto_bars_quality.json` | качество баров: OHLC, монотонность меток, разрывы | `ops/crypto_build_bars.py` |
| `crypto_models.json`, `crypto_policy.md` | OOF IC по фолдам и политика: сколько сделок, gross, net, Sharpe | `ops/crypto_train.py` |
| `crypto_ladder.{json,md}` | лестница ёмкости: 5 конфигураций модели на одном протоколе | `ops/crypto_ladder.py` |
| `crypto_classic.{json,md}` | 6 классических моделей и их объединение | `ops/crypto_classic.py` |
| `crypto_maker.{json,md}` | maker-исполнение: доля исполнений и adverse selection | `ops/crypto_maker.py` |
| `crypto_btc.json`, `crypto_btc_models.json`, `crypto_btc_policy.md` | ветка BTC абсолютной доходностью | `ops/crypto_btc.py` |
| `crypto_sources.{json,md}` | доверие к источникам и проверка устойчивости по половинам | `ops/crypto_sources.py` |
| `crypto_structural.{json,md}` | 11 структурных классов: амплитуда и направленный эффект | `ops/crypto_structural.py` |
| `crypto_listings.{json,md}` | листинги на споте: аукцион отдельно, вход со второго бара | `ops/crypto_listings.py` |
| `crypto_listing_perp.{json,md}` | листинги на перпетуалах: шорт, хвост, ликвидации по плечам | `ops/crypto_listing_perp.py` |
| `crypto_macro.{json,md}` + `crypto_macro_calendar.csv` | 528 релизов: амплитуда, моментум, задержки, эпохи, терцили | `ops/crypto_macro.py` |
| `crypto_shock.json` | продолжение после минутного шока на 4 инструментах | `ops/crypto_shock.py` |
| `crypto_probe.json` | проверка доступности источников данных (не результат исследования) | `ops/crypto_probe.py` |
| `crypto_verification.{json,md}` | 78 независимых проверок: таблица PASS/FAIL и пересчитанные значения | `ops/crypto_verify.py` |
| `crypto_robustness.{json,md}` | перекрытия окон, бутстрэп CPI, фандинг, лестница проскальзывания | `ops/crypto_verify2.py` |

## Ветка акций (дневная и внутридневная)

`test_2023.*` — holdout 2023 года с `.lock`-файлом, `cv_folds.json` — purged-фолды,
`leakage_checks.json` — проверки утечек, `baselines.json` — B0…B3, `backtest_bt-*.md` —
бэктесты с двумя соглашениями об исполнении, `backtest_metrics.json` — их метрики,
`ablation.*` — аблации, `intraday_gdelt_pilot.md` + `intraday_gdelt_metrics.json` —
внутридневной пилот, `audit_measure.txt` — текстовый дамп проверки измерений.

Публичная сводка по этой ветке — в README, метод — в `docs/`.

## Правило

Ни один файл здесь не правится руками: артефакт перегенерируется скриптом. Если нужно другое
число — меняется скрипт и прогон повторяется, а не таблица.
