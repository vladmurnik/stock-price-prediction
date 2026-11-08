# Аудит конфигурации (R13)

Агент: `reviewer`, модель `freepi/deepseek-v4.1-flash`, сессия `ses_e45331ebfffe6twyTVpit2b2fM`.
Все находки воспроизведены агентом на коде и хешах; правок в репозиторий агент не вносил.

## Вердикт

`configs/final.yaml` **функционально идентичен** `base.yaml` — разница в 4 строки, и все они
комментарии (заголовок, обоснование решения, запрет на правку после лока). Никакого «скрытого
override», который требовал бы отдельной предрегистрации, нет. Но поверхность конфига содержит
дефекты молчаливого переопределения: два ключа не читаются вообще, `config_hash` holdout не
покрывает сиды, а `run_id` кросс-валидации не покрывает секцию `sentiment`.

На уже потраченный holdout 2023 это не повлияло (значения в конфиге совпадают с хардкодом), но
лок перестаёт быть честным описанием того, что произвело 2023.

## Находки

| P | файл:строка | что не так | как воспроизведено | правка |
|---|---|---|---|---|
| P1 | `data/news_fnspid.py:575`, `data/news_live.py:1087,1187` | `news.dedupe_jaccard` из конфига не читается: `dedupe_news(raw, threshold=0.9)` и `dedupe_live(combined)` используют литерал `0.9` (`normalize.py:86`, `TRIGRAM_THRESHOLD`) | `inspect.getsource(fetch_news)` → ключа нет; `dedupe_news(threshold=0.5)` оставляет 1 из 2 почти-дублей, `0.9` — 2 | прокинуть `cfg.news.dedupe_jaccard` в `fetch_news`/`collect_live` |
| P1 | `data/news_fnspid.py:481-493,521` | `request_key` кэша новостей не содержит `dedupe_jaccard`, а дедуп выполняется **до** записи артефакта → смена порога возвращает старый кэш с `cached=True` | чтение `_cache_key`; в payload нет поля дедупа | добавить эффективный порог (и `publisher_priority`) в `_cache_key` |
| P1 | `data/normalize.py:93-100` | `news.publisher_priority` мёртв: ранжирование идёт по модульной константе `PUBLISHER_PRIORITY`; ключ читает только валидатор `config.py:76-82` (проверяет непустоту) | `Select-String news.publisher_priority` по `src/spp` → одно попадание, `config.py:81` | принимать список параметром в `dedupe_news`/`dedupe_live`/`_priority_series` |
| P1 | `models/holdout.py:139-141` | `config_hash` = sha256 от `config.model_dump()`, а `--seed`/`--seeds` мутируют только `Context.seed` и передаются в `run_holdout` отдельно ⇒ лок пишет **базовый** хеш. Проверено: `model_copy(seed=7)` даёт `28dd34bcb45a`, лок хранит `3a1c947f4f51`; `lock.seeds` = `[1,2,3]` пишется рядом, поэтому расхождение не видно | сниппет с хешированием под `model_copy(seed=7)` | хешировать `{**model_dump(mode="json"), "effective_seeds": seeds}` либо запретить `--seed != config.seed` для holdout |
| P2 | `models/gbdt.py:678-695` | `run_id` CV покрывает `model/cv/features/feature_columns/seeds/bytes/DEFAULT_PARAMS/git-sha`, но **не** `sentiment`, `news`, `dates`, `universe`. Смена `sentiment.max_length` 128→512 даёт тот же `gbdt-a6edb94c37` — старые артефакты перезаписываются под новым манифестом | три вызова `_make_run_id`; `n_folds` 6→5 меняет id, `max_length` — нет | добавить `config_features_hash` (dedupe_jaccard, max_length, backend, model ids, `features.*`) |
| P2 | `models/gbdt.py:1068` | читается `cfg.model.n_estimators`, которого нет в `ModelCfg` (`extra="forbid"`), реальный потолок — `BOOSTING_ROUNDS=3000` (`gbdt.py:99`). Мёртвая ветка, упадёт при попадании | `hasattr(cfg.model,'n_estimators')` → `False` | брать `params["n_estimators"]` из `_params` |
| P2 | `models/holdout.py:193` | порог `ic > 0.02` захардкожен, ключа в конфиге нет — меняется только правкой кода, ровно то, что лок должен предотвращать | чтение `holdout.py:192-199` | вынести в `backtest.holdout_gates`; под текущим локом только зафиксировать как долг |
| P2 | `features/sentiment_daily.py:291` | имена колонок `ewma_6h/ewma_24h` зашиты, а полураспады берутся из `features.ewma_halflives_min` ⇒ при смене конфига имена врут про горизонт | чтение `sentiment_daily.py:291-292` и `:320-331` | генерировать имена из полураспада либо валидировать соответствие |
| P2 | `cli.py:1147`, `models/holdout.py:63-70` | `live.run_at_utc` и `live.state_path` валидируются, но не используются: `spp live run` — заглушка, путь `data/live` захардкожен | `Select-String "\.live\."` по `src/` → пусто; `live/__init__.py` 68 байт | использовать в `fetch live`/планировщике либо убрать из `base.yaml` |

## Проверено чисто

* `base.yaml` vs `final.yaml`: `git diff --no-index --stat` → 4 вставки, все — комментарии.
* `config_hash` воспроизводится как `3a1c947f4f51` из текущего конфига и совпадает с
  `reports/test_2023.lock` и `docs/faq.md:153`.
* Версионирование кэша сентимента корректно: `_stamp_matches` (`score_corpus.py:319-325`) сверяет
  `model_version` (`<repo>@<rev>+cfg<sha1[:12]>`) и `temperature`; FinBERT кладёт `max_length` в
  хешируемый payload (`finbert.py:55-62`), поэтому смена `max_length` инвалидирует шарды.
* `cv.n_folds`, `cv.embargo_days`, `cv.test_months`, `cv.min_train_months` читаются из `CvCfg`
  (`gbdt.py:766-778`), `cv.py:265-268` отвергает `embargo_days < 1`. Хардкода числа фолдов на
  CV-пути нет.
* `features.ewma_halflives_min`, `rolling_days`, `cross_sectional_rank`, `demean_target` —
  используются (`assemble.py:257-266`, `gbdt.py:761`).
* `backtest.*` (n_quantiles, commission, slippage, borrow, execution) читается из конфига
  (`cli.py:1256-1322`); дефолты в `portfolio.py:344-347` совпадают со значениями конфига.
* `cutoff_et` и `strict_dates` доходят до `SignalPolicy.from_config` и на историческом, и на
  live-пути (`cli.py:279`, `cli.py:1070`).
* У матрицы признаков нет cache-hit логики (`assemble.py:340-351`, манифест только пишется) —
  устаревших признаков из-за смены конфига не бывает.

## Не проверено

* Семантика значений holdout: `train test2023` не перезапускался (запрещено); дрейф `config_hash`
  доказан чтением и хешированием, а не вторым локом.
* Размер эффекта хардкода `0.9` на результат 2023: текущее значение конфига **и есть** 0.9, поэтому
  holdout не был дедуплицирован неверно — дефект латентный.
* `sentiment.min_macro_f1` как гейт калибровки (`cli.py:415`): калибровка не запускалась.
* `openjev.max_length`: `OpenJevBackend` хардкодит `max_state_tokens: 256` в payload
  `model_version` (`openjev.py:89`), `sentiment.max_length` действует только на FinBERT. Открытый
  вопрос, а не подтверждённый дефект.
