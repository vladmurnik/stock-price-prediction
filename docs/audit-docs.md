# Аудит документации

Дата: 2026-11-08. Объекты аудита: `README.md`, `README.ru.md`, `docs/limitations.md`,
`docs/idea-intraday-t5.md`, структура каталогов против `git ls-files`, внутренние ссылки.

Метод: каждое утверждение сверено с первоисточником — кодом (`src/spp/**`, `configs/*.yaml`),
артефактом (`reports/*`, `results/*`, `data/**/*.manifest.json`) или запущенной командой.
Команды, которые нельзя запускать во время фонового скоринга (`train`, `test2023`, `backtest`,
`features build`, `score`), сверялись по `--help` и по коду, а не исполнением.

Правило правок: исправлялись только фактические ошибки (пути, флаги, имена файлов, счётчики,
структура, формулировки о чём-то неработающем). Числа аблаций, holdout и бэктеста
(`0.0156`, `0.0127`, `−6.75`, `33.96%` и т. п.) не правились нигде: их генерирует
`ops/make_ablation_report.py` в `results/ablation.md`, ручная правка создала бы расхождение.

## Таблица находок

| # | утверждение | где проверено | статус | исправление / комментарий |
|---|---|---|---|---|
| 1 | «Model — LightGBM on 62 features (32 market + 30 sentiment)» (`README.md:35`), «рыночные + сентимент (62)», «market only (32)» (`README.md:53-55`, `README.ru.md:47-49`) | `src/spp/features/market.py:42` `MARKET_FEATURES` — 19 колонок (`python -c "print(len(MARKET_FEATURES))"` → 19); `src/spp/features/sentiment_daily.py:108` `SENTIMENT_FEATURES` — 30; `data/processed/features.parquet.manifest.json` `"n_features": 49`; `reports/leakage_checks.json` `meta.n_features: 49`; `reports/runs/a5-all/metrics.json` `config.features` — 49 элементов | **неверно** | 62/32 → 49/19. Строки ablation-таблицы: «market only (19)», «market + sentiment (49)», «sentiment only (28)» — число признаков в матрице аблации, а не метрика IC, поэтому правится. IC/CI/DM-p не тронуты |
| 2 | «sentiment only (30)» | `reports/runs/a2-sentiment-only/metrics.json` `config.features` — 28; `data/processed/features_sentiment_only.manifest.json` `n_sentiment_features: 28` | **неверно** | 30 → 28 (в матрице аблации `company_specific_share` и `novelty_mean` отброшены как all-NaN, см. #15) |
| 3 | «test 10,000 (2023-01-03 … 2023-12-29)» (`README.md:70`) и «test 10 000» (`README.ru.md:64`) | `reports/test_2023.lock` `test_rows: 10000`; `results/test_2023.json` `rows.test: 9772`, `metrics.n_rows: 9772`, `metrics.n_dates: 245`; `results/test_2023.md` «test 9772 over 245 dates» | **неверно** | Разделено: «test rows locked 10,000, scored 9,772 over 245 dates» — модельный прогноз существует не для всех строк (tail без forward price) |
| 4 | «Tests: `pytest -q` (216 passing)» (`README.md:128`, `README.ru.md:121`) | запущено `python -m pytest -q -p no:cacheprovider` → `216 passed, 2 skipped in 40.62s`, exit 0 | **неточно** | «216 passed, 2 skipped»; добавлен факт `218 tests collected` |
| 5 | `python -m spp.cli score corpus --backend finbert` | `python -m spp.cli score corpus --help` → exit 0, флаг `--backend`; `src/spp/cli.py:427-470` | ok | — |
| 6 | `python -m spp.cli features build --check-leakage` | `features build --help` → exit 0, флаг `--check-leakage` (`cli.py:607`) | ok | — |
| 7 | `python -m spp.cli train gbdt` | `train gbdt --help` → exit 0 (`cli.py:867`) | ok | — |
| 8 | `python -m spp.cli --config configs/final.yaml train test2023` | `train test2023 --help` → exit 0, флаг `--force` есть; глобальный `--config` определён в callback (`cli.py:126-153`) | ok | — |
| 9 | `python -m spp.cli backtest --predictions ... --include-test` | `backtest --help` → exit 0, оба флага есть (`cli.py:1162-1216`) | ok | — |
| 10 | `python -m spp.cli fetch prices` / `fetch news` | `fetch prices --help`, `fetch news --help` → exit 0 | ok | — |
| 11 | `python ops/ablate.py --only a4 a6` | `python ops/ablate.py --help` → exit 0, `--only {a4,a6,a3,a8}`; a4 и a6 в списке | ok | — |
| 12 | «62,900 rows, 50 tickers» для `fetch prices` | `data/cache/prices_2019-01-01_2023-12-31.parquet.manifest.json`: `rows: 62900`, `tickers_present` — 50 записей | ok | — |
| 13 | «124,571 articles (38 of the 40 tickers)» | `data/interim/news_raw.parquet.manifest.json`: `rows: 124571`, `tickers: 38`, `repo: Zihan1004/FNSPID` | ok | — |
| 14 | «macro-F1 0.8862 … `results/calibration/finbert.json`» | `results/calibration/finbert.json` `macro_f1: 0.886233696467379`, report path реально `reports/calibration/finbert.json` (в `results/` копия есть) | ok | — |
| 15 | «open-jev only · `novelty_mean`, `company_specific_share` · 0.000 (not filled by FinBERT)» | `src/spp/sentiment/finbert.py` и `lexicon.py` **не выставляют** `is_company_specific`/`is_novel` (grep: совпадений нет); значение по умолчанию `None` (`src/spp/sentiment/base.py:81-82`); `src/spp/features/sentiment_daily.py:404-413` даёт NaN при `cs_d==0`; `reports/runs/a5-all/metrics.json` — колонки в матрице all-NaN; `ops/inbox/20261108-025414-t07-quant-model.md:996` «dropping all-NaN feature columns: ['company_specific_share', 'novelty_mean']» | **неверно (неполно)** | Формулировка усилена: колонки NaN не только у FinBERT, а у всех бэкендов, кроме open-jev, и до сих пор ни одна сборка не заполнила их. Добавлен пункт в `docs/limitations.md` (#9) |
| 16 | «A3 (open-jev) is not reported … the ablation is pending» — только в разделе Limitations README | `reports/runs/` содержит только `a1,a2,a4,a5,a6,gbdt-*`; `results/ablation.md` A3/A8 = «—» | ok (по сути) | Перенесено в основной раздел как «не закрыто», чтобы не выглядело как законченный результат |
| 17 | «~3.4 items/s on CPU, so the full 92,927-text corpus is a 7.5-hour job» (`README*`, `docs/limitations.md:15-18`) | `results/calibration/openjev.json` `throughput: 1.0674224017270404`; `ops/HANDOFF.md` «open-jev: … 1.1 items/s»; `ops/inbox/20261108-0830-t10-verified.md` «1.1 items/s makes 124k articles a 31-hour job» | **неверно (устарело)** | Оставлено без изменения в этом прогоне: два источника расходятся (1.07 vs 3.4), см. «Требует решения владельца» |
| 18 | «покрыто 5.9 тыс. текстов» на момент публикации (`docs/limitations.md:18`) | `reports/score_progress.json` — последний прогон `backend: lexicon`, `scored_texts: 0`, `reused_texts: 92927`; артефакта прогресса open-jev нет; `data/processed/article_sentiment_openjev.parquet` отсутствует | **не проверяемо** | Убрано число «5.9 тыс.»; заменено на «прогресс не зафиксирован артефактом» |
| 19 | «holdout 2023: IC 0.0127, CI [−0.0102, 0.0356], p 0.25»; «−6.75 net Sharpe» | `results/test_2023.json`, `results/backtest_bt-2023-next_open_hold_close.json` (`sharpe_net: -6.750554242942211`) | ok | Не трогалось |
| 20 | Ablation-таблица (A1 0.0007, A2 0.0134, A5 0.0156, A4 −0.0079, A6 0.0167) | `results/ablation.md` и `results/ablation_cv.json` — совпадают | ok | Не трогалось |
| 21 | «Turnover is ~166× per year» | `results/backtest_bt-2023-next_open_hold_close.json` `metrics.turnover_ann: 169.77857142857144`; `results/backtest_bt-2023-close_to_close.json` то же | **устарело** | Не исправлялось (число бэктеста). См. «Требует решения владельца» |
| 22 | Exit-коды README-команд | `python -m spp.cli --help` и `fetch|score|features|train|live|report --help` → exit 0 каждая; `pytest -q` → 0; `ruff check src tests` → 0 | ok | — |
| 23 | Раздел «Layout»: `src/spp/data`, `sentiment`, `features`, `models`, `backtest`, `results/`, `ops/` | `git ls-files` — все каталоги существуют; `results/` отслеживается git (22 файла), `data/`, `reports/` — нет (`.gitignore`) | ok | — |
| 24 | «Data and trained artifacts are not committed (see `.gitignore`)» | `.gitignore`: `/data/`, `*.parquet`, `/reports/` | ok | — |
| 25 | `src/spp/live/` представлен в коде только `__init__.py`; `spp.cli live run` — заглушка (`cli.py:1151-1157`) | `python -m spp.cli live --help` → «run  Stub for `live run` (implemented in a later step)»; `git ls-files src/spp/live` → только `__init__.py` | **неверно (скрыто)** | В README добавлена строка, что live-контур не реализован; в `docs/limitations.md` §5 уточнено, что проверены сборщики (`fetch live`), а не торговый цикл |
| 26 | Ссылки между README.md / README.ru.md / docs/*.md | перебор всех `](...)` — все относительные цели существуют: `README.ru.md`, `README.md`, `docs/idea-intraday-t5.md` | ok | Добавлена ссылка `docs/audit-docs.md` |
| 27 | Совпадение фактов README.md ↔ README.ru.md | построчное сравнение чисел и путей | **расхождение** было: оба текста содержали 62/32/30 и 10 000 одинаково, то есть версии согласованы, но обе неверны | Исправлено в обеих версиях одинаково |
| 28 | Строка-переключатель языков (первая строка обоих README) | `README.md:1` `**🇬🇧 English** · [🇷🇺 Русская версия](README.ru.md)`; `README.ru.md:1` `[🇬🇧 English](README.md) · **🇷🇺 Русский**` | ok | Сохранена дословно в обеих версиях |
| 29 | `docs/limitations.md` — «список из 8 пунктов» | файл содержит 8 разделов; план требует 9 обязательных ограничений | **неверно (неполно)** | Добавлен §9 о `company_specific_share`/`novelty_mean` |
| 30 | `results/ablation_cv.json` — те же числа, что в `results/ablation.md` | `a5-all.ic = 0.015596155025294588` — совпадает; в том же файле есть служебные ключи `gbdt-e124617157` (ic 0.0202 — прогон **до** исправления признаков) и `gbdt-ebeeba82cb` (n_folds=1) | **риск** | Ключи не используются README. Помечено здесь, чтобы никто не принял 0.0202 за текущий результат |
| 31 | `reports/ablation_cv.json` | файла нет (есть `results/ablation_cv.json`) | нет дефекта документации | — |

## Что проверено запуском

| команда | exit code |
|---|---|
| `python -m spp.cli --help` | 0 |
| `python -m spp.cli fetch --help` | 0 |
| `python -m spp.cli score --help` | 0 |
| `python -m spp.cli features --help` | 0 |
| `python -m spp.cli train --help` | 0 |
| `python -m spp.cli live --help` | 0 |
| `python -m spp.cli report --help` | 0 |
| `python -m spp.cli fetch prices --help` | 0 |
| `python -m spp.cli fetch news --help` | 0 |
| `python -m spp.cli fetch live --help` | 0 |
| `python -m spp.cli score corpus --help` | 0 |
| `python -m spp.cli score calibrate --help` | 0 |
| `python -m spp.cli features build --help` | 0 |
| `python -m spp.cli train gbdt --help` | 0 |
| `python -m spp.cli train test2023 --help` | 0 |
| `python -m spp.cli backtest --help` | 0 |
| `python ops/ablate.py --help` | 0 |
| `python -m pytest -q -p no:cacheprovider` | 0 (216 passed, 2 skipped) |
| `python -m ruff check src tests` | 0 |

Не запускалось намеренно: `train *`, `test2023`, `backtest`, `features build`, `score corpus`,
`fetch *` — фоновый CPU-bound скоринг, запрет на пересчёт holdout и запрет на тяжёлые команды.

## Требует решения владельца

1. **Скорость open-jev.** В документации «~3.4 items/s → 7.5 ч», в артефакте
   `results/calibration/openjev.json` — 1.07 items/s, в `ops/inbox/20261108-0830-t10-verified.md` —
   «1.1 items/s … 31-hour job». Три значения на один вопрос. Правильнее всего пересчитать
   документ из `results/calibration/openjev.json`; я не стал выбирать между 1.07 и 3.4, потому что
   это меняет заявление о сроке A3 примерно в 4 раза.
2. **Turnover 166 vs 169.8.** Артефакт даёт 169.78×/год в обеих схемах исполнения; в README и
   `docs/limitations.md` и `docs/idea-intraday-t5.md` — «166×». Это число бэктеста, правится
   `ops/make_ablation_report.py`-подобным генератором, а не рукой.
3. **`results/ablation_cv.json` содержит дореформенный прогон** `gbdt-e124617157` с IC 0.0202.
   Файл не цитируется в README, но лежит рядом с актуальными числами и легко провоцирует
   неверную цитату. Нужно либо удалить ключ, либо переименовать с пометкой `_pre_fix`.
4. **A3/A7/A8.** `ops/run_a3_a8_a7.ps1` и `reports/openjev_sample.parquet` уже существуют,
   поэтому фраза «ablation is pending» и фраза «скоринг не завершён» описывают разное состояние.
   После завершения фонового `score corpus --backend openjev` README должны перегенерировать
   через `ops/make_ablation_report.py`, а не редактировать.
5. **`docs/agents.md`** (не входит в отслеживаемые файлы, не мой файл) содержит распределение ролей
   и критерии приёмки docs; `README*` на него не ссылаются. Решить, нужна ли ссылка.
