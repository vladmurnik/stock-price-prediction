# Методология

Документ описывает **фактический** метод, реализованный в `src/spp/**`, а не задуманный.
Каждое число сопровождается источником (файл и символ). Если параметр задаётся не кодом, а
конфигом, указан конфиг. Если параметра нет нигде — написано `not fixed in code`.

Единица наблюдения — `(ticker, date)` на общей календарной оси (spine). Модель предсказывает
**доходность**, а не цену: это кросс-секционный сигнал на лог-доходность следующей сессии,
оцениваемый вне выборки с явно заданными издержками.

---

## 1. Таргет

Определение (`src/spp/features/calendar.py:horizon_returns`, `src/spp/features/labels.py:build_labels`):

```
y_h[t] = log( adj_close[t+h] / adj_close[t] ),   h ∈ HORIZONS = (1, 2, 3, 5)
y      = y_1                                        # основной таргет
y_dir  = sign(y)
y_raw  = y                                          # неминусованное значение для бэктеста
```

- `HORIZONS = (1, 2, 3, 5)` — `src/spp/features/calendar.py:HORIZONS`.
- `t+h` означает «`h` NYSE-сессий позже» на spine, а не календарных дней
  (`calendar.py:horizon_returns` работает по индексу spine).
- Маскирование хвоста: последние `HORIZON_MASK_SESSIONS = max(HORIZONS) = 5` сессий истории
  получают `y = NaN` и `label_ok = False`, чтобы срез, обрезанный на `2022-12-31`, не обучался
  на метке, заглянувшей в 2023 (`labels.py:HORIZON_MASK_SESSIONS`, `build_labels`).

### Кросс-секционный demean

При `config.features.demean_target = true` (в `configs/final.yaml` — `true`) модель обучается на

```
y_used[t] = y[t] − median_{tickers}( y[t] )        # по каждой дате отдельно
```

- Реализация: `src/spp/features/labels.py:demean_by_date` — `groupby("date")["y"].transform("median")`.
  Это **медиана**, не среднее.
- Операция по-датовая, поэтому причинно безопасна: строка даты `t` не видит другие даты.
- `y` и `y_raw` всегда хранят неминусованное значение; минусованное лежит в отдельной колонке
  `y_used_demeaned` (`labels.py:LABEL_COLUMNS`).
- Тренер берёт таргет из `y_used_demeaned` при `demeaning=True`, иначе из `y`
  (`src/spp/models/gbdt.py:_target`).

### Клиппинг таргета

`objective="huber"` не имеет встроенного клиппинга отклика, поэтому `y_raw` обучающего среза
обрезается по квантилям `PI_BAND = (0.005, 0.995)` (`src/spp/features/clip.py:PI_BAND`,
`clip_bounds`, `clip_targets`). Границы считаются **только по обучающему срезу фолда** и
сохраняются в `model_meta.json` (`target_clip_bounds`). Клиппинг — операция в пространстве метки;
предсказания остаются доходностями, а строка со всеми NaN-фичами по-прежнему даёт NaN.

---

## 2. Цепочка тайминга сигнала

Цель — гарантировать, что значение, привязанное к строке `(ticker, t)`, построено только из
новостей, известных к моменту `t`. Реализация: `src/spp/data/normalize.py:signal_day`,
`assign_signal_date`, `SignalPolicy`.

```
published_at (UTC)
   │  astimezone(America/New_York)
   ▼
local_et
   │  если local_et.time() <= cutoff_et (16:00)  →  signal_date = local_et.date()
   │  иначе                                       →  signal_date = local_et.date() + 1 день
   ▼
signal_date (календарный день)
   │  roll-forward на ближайший NYSE-торговый день (next_or_same)
   ▼
signal_date (торговая сессия)  ──►  агрегируется в строку (ticker, date)
```

- `cutoff_et = "16:00"` — `configs/final.yaml:news.cutoff_et`; значение по умолчанию в коде
  `SignalPolicy.cutoff_et = time(16, 0)` (`normalize.py`).
- `strict_dates = false` — `configs/final.yaml:news.strict_dates`. Для строк без `published_at`
  (date-only источник) `signal_date = source_date` при `strict_dates=false`, иначе `+1 день`
  (`normalize.py:signal_day`).
- Инвариант `signal_date >= date` **проверяется assert'ом**, не клампится
  (`normalize.py:assign_signal_date`, `validate_news`).
- GDELT: `seendate` — время краула, поэтому `published_at` пуст, а `date = signal_date = seendate.date()`
  (консервативная ветка date-only) — `src/spp/data/news_live.py:parse_gdelt_payload`.
- RSS без таймстампа: `published_at = now` (UTC); таймстампы из будущего клампятся к `now`
  (`news_live.py:parse_rss_payload`).

### Как сигнал попадает в фичи

Агрегация идёт по `(ticker, signal_date)` (`src/spp/features/sentiment_daily.py:aggregate_sentiment_daily`).
Внутридневные величины (`sent_*`, доли, энтропия) используют только `signal_date == t`.
Междневные агрегаты (`sent_sum_3d/5d/10d`, `sent_mean_3d/5d/10d`, `n_news_z`, `sent_surprise`,
`sent_momentum_3d`) сдвинуты на spine и читают `[t−k, t−1]` (`sentiment_daily.py:_derive`,
`_rolling_prev`, `_shift_sessions`). Сдвиг именно по spine, а не по компактному индексу тикера,
чтобы строка `t` не прочитала саму себя на дне без новостей.

Политика NaN: нет новостей → `n_news = 0`, `log_n_news = 0`, `ewma_*` carry-forward, все `sent_*`
= **NaN, не 0** (`sentiment_daily.py` docstring, `_derive`). LightGBM потребляет NaN нативно;
импутации нет (`src/spp/features/transform.py:FeaturePrep`).

---

## 3. Шесть групп признаков

В коде есть два кортежа: `MARKET_FEATURES` (19 имён, `src/spp/features/market.py`) и
`SENTIMENT_FEATURES` (30 имён, `src/spp/features/sentiment_daily.py`). Итог —
`feature_columns(include_sentiment=True)` = 49 колонок (`src/spp/features/assemble.py:feature_columns`).
Это подтверждается артефактом: `reports/leakage_checks.json` → `meta.n_features = 49`.

Разбиение на шесть групп ниже — **документационная таксономия** над этими двумя кортежами
(в коде константы «шести групп» нет). Состав групп точен и проверяем по именам колонок.

| # | Группа | Колонки | Источник |
|---|---|---|---|
| 1 | Доходности / моментум | `ret_1, ret_2, ret_3, ret_4, ret_5` | `market.py:MARKET_FEATURES`, `RET_LAGS=(1,2,3,4,5)` |
| 2 | Волатильность | `vol_5, vol_20, vol_60, vol_ratio` | `market.py:build_market_features` |
| 3 | Объём / ликвидность | `volume_z` | `market.py:_volume_z` |
| 4 | Технические уровни | `rsi_14, px_sma20, px_sma50` | `market.py:_rsi`, `build_market_features` |
| 5 | Рыночный / секторный контекст | `sector_ret_1, spy_ret_1, spy_ret_5, vix_level, vix_chg_1, dow` | `market.py:build_market_features` |
| 6 | Новостной сентимент | 30 колонок `SENTIMENT_FEATURES` | `sentiment_daily.py:SENTIMENT_FEATURES` |

Группа 6 (`SENTIMENT_FEATURES`) = 5 базовых (`n_news, log_n_news, n_news_z, ewma_6h, ewma_24h`)
+ 19 дневных (`SENT_DAY_FEATURES`: `sent_mean, sent_sum, sent_std, sent_min, sent_max, sent_median,
pos_share, neg_share, neu_share, very_pos_share, very_neg_share, neg_extreme_share, sent_surprise,
sent_momentum_3d, company_specific_share, novelty_mean, conf_mean, source_entropy, top_source_share`)
+ 6 междневных (`sent_sum_3d, sent_mean_3d, sent_sum_5d, sent_mean_5d, sent_sum_10d, sent_mean_10d`).

### Причинность рыночных фич

- `ret_k = log(P_t / P_{t−k})` — новейшее наблюдение в окне это `t`; таргет `t+1`, поэтому `t`
  читать можно (`market.py` docstring).
- Скользящие статистики (`vol_*`, `px_sma*`, `rsi_14`, `volume_z`) сдвинуты на одну сессию
  (`_shift(..., 1)`), т.е. строка `t` описывает окно, заканчивающееся `t−1`.
- `sector_ret_1`, `spy_ret_1`, `spy_ret_5`, `vix_chg_1` — лаг 1; `vix_level` — уровень на `t`.
- `dow` — день недели, категориальная (`market.py:build_market_features`).

### Трансформация перед моделью

`src/spp/features/transform.py:FeaturePrep` (порядок фиксирован):

1. переупорядочивание колонок к сохранённому списку;
2. `dow` → float (LightGBM видит числа);
3. по-датовый `rank_gauss` по всем некатегориальным колонкам на днях с `>= DEFAULT_MIN_RANK_TICKERS = 10`
   тикерами; на «тонких» днях значение остаётся сырым и помечается `rank_status = "raw"`;
4. **без импутации** — NaN остаётся NaN.

`FeaturePrep` фитится **только на обучающем срезе** (`FeaturePrep.fit`), медианы и доли NaN
сохраняются для диагностики, но не используются для заполнения.

---

## 4. Purged CV с embargo

Реализация: `src/spp/models/cv.py:make_folds`, `assert_no_overlap`, `iter_splits`.

- Окно CV: `CV_WINDOW_START = 2019-01-01`, `CV_WINDOW_END = 2022-12-31`; 2023 жёстко исключён
  (`cv.py:CV_WINDOW_START/END`, `CV_HELD_OUT_YEAR = 2023`).
- `n_folds = 6`, `test_months = 3`, `embargo_days = 5`, `min_train_months = 12` —
  `configs/final.yaml:cv`. `MIN_FOLDS = 3` (`cv.py`).
- Расширяющееся окно: `train_start` у всех фолдов = первая сессия окна (`2019-01-02`), проверяется
  инвариантом «expanding» в `assert_no_overlap`.
- Тестовое окно фолда — `test_months` календарных месяцев, выровненных по концу квартала;
  границы привязываются к сессиям spine (`cv.py:_fold_backwards`, `month_end_session`).

### Embargo: 5 торговых сессий

```
train_end = session_before(test_start, embargo_days + 1)   # 5+1 = 6-я сессия до test_start
valid     = все сессии строго между train_end и test_start  # ровно embargo_days = 5 сессий
```

- `embargo_days = 5` — `configs/final.yaml:cv.embargo_days`; значение по умолчанию в `Fold`
  `embargo_days: int = 5` (`cv.py:Fold`).
- **Почему 5:** максимальный горизонт метки — `max(HORIZONS) = 5` сессий
  (`calendar.py:HORIZONS`, `labels.py:HORIZON_MASK_SESSIONS`). Метка обучающей строки в `t`
  достигает `t+5`; зазор в 5 торговых сессий гарантирует, что окно метки обучения не пересекает
  тест. Embargo измеряется в **торговых сессиях**, а не календарных днях: «5 календарных дней»
  через выходные — это 3 сессии (`cv.py` docstring).
- `embargo_days < 1` запрещён (`cv.py:make_folds`).
- `assert_no_overlap` проверяет пять инвариантов, включая попарную непересекаемость множеств
  `(date, ticker)` в train/valid/test на реальных строках (`cv.py:assert_no_overlap`, проверка 5).

Фактические границы фолдов (источник `reports/cv_folds.json`): 6 фолдов, у каждого
`embargo_sessions = 5`, `test_sessions` = 64/64/62/62/64/63, `total_test_sessions = 379`,
`test_session_share = 0.3759920634920635`, `window_sessions = 1008`.

---

## 5. Внутренний валидационный блок на 60 сессий

`src/spp/models/gbdt.py:EARLY_STOP_SESSIONS = 60`.

Зазор embargo (5 сессий) слишком короткий для ранней остановки: IC по пяти датам имеет
стандартную ошибку порядка 0.4, и бустер останавливается на произвольной итерации
(`gbdt.py:run_gbdt_cv`, комментарий перед блоком). Поэтому:

```
train_days = отсортированные уникальные даты обучающего среза
если len(train_days) > 2 * 60:
    inner_days = train_days[-60:]          # последние 60 сессий обучения
    fit_train  = train[~inner_days]
    fit_valid  = train[inner_days]
иначе:
    fit_train, fit_valid = train, valid    # fallback на валидационный срез фолда
```

Внутренний блок лежит строго до тестового окна, поэтому будущая информация в модель не попадает.
Тот же блок используется в holdout (`src/spp/models/holdout.py:run_holdout`, идентичная логика
`EARLY_STOP_SESSIONS`).

---

## 6. Ранняя остановка по IC

Реализация: `src/spp/models/gbdt.py:_fit_one`, вложенный callback `_ic_early_stop`.

- Метрика остановки — **средний по датам Spearman IC** на внутреннем валидационном блоке
  (`gbdt.py:_rank_ic_by_date`), а не встроенная метрика LightGBM.
- Терпение: `config.model.early_stopping_rounds = 100` (`configs/final.yaml:model`).
- Потолок итераций: `BOOSTING_ROUNDS = 3000` (`gbdt.py`), передаётся как `n_estimators`.
- Если остановка не сработала (`best_iteration` не меньше потолка), берётся
  `FALLBACK_BEST_ITERATION = 500` с WARN (`gbdt.py`).
- Callback намеренно не трогает `env.evaluation_result_list`; он сам считает IC и бросает
  `lgb.callback.EarlyStopException` (`gbdt.py:_ic_early_stop`).

Фактические длины (источник `reports/test_2023.json` → `seeds`): seed 1 → `best_iteration = 155`,
seed 2 → `146`, seed 3 → `67`.

---

## 7. Три сида

- `config.model.seeds = [1, 2, 3]` (`configs/final.yaml:model`); CLI по умолчанию `--seeds 1,2,3`
  (`src/spp/cli.py:train_gbdt_command`).
- Каждый сид — отдельный `LGBMRegressor` с `seed = s` и детерминизмом
  `LGBM_DETERMINISM = {deterministic: True, force_row_wise: True, num_threads: 1}` (`gbdt.py`).
- Предсказания усредняются по сидам: `y_pred_seed_mean` (`gbdt.py:run_gbdt_cv`, `groupby(["date","ticker"])["y_pred"].mean()`).
- Финальная модель для live переобучается на всём CV-окне; число деревьев = медиана
  early-stopped длин по фолдам (`gbdt.py:run_gbdt_cv`, блок `final`).

---

## 8. Функция потерь: Huber

- `objective = "huber"`, `alpha = 0.9` — `src/spp/models/gbdt.py:DEFAULT_PARAMS`.
  `alpha` **не** вынесен в `ModelCfg`/`configs/final.yaml`, поэтому действует значение из
  `DEFAULT_PARAMS` (0.9).
- Остальные гиперпараметры (`DEFAULT_PARAMS`, переопределяемые `ModelCfg`):
  `learning_rate = 0.03`, `num_leaves = 31`, `min_child_samples = 200`, `feature_fraction = 0.8`,
  `bagging_fraction = 0.8`, `bagging_freq = 1`, `lambda_l2 = 1.0`, `n_estimators = 3000`.
  `configs/final.yaml:model` задаёт `num_leaves=31, learning_rate=0.03, min_data_in_leaf=200,
  lambda_l2=1.0`; `min_data_in_leaf` переименовывается в `min_child_samples` в `gbdt.py:_params`.
- Huber устойчив к выбросам, но одиночный скачок в 40% всё равно искажает фит, поэтому
  дополнительно применяется клиппинг таргета по `PI_BAND` (см. §1).

---

## 9. Протокол holdout и lock-файл

Реализация: `src/spp/models/holdout.py:run_holdout`, CLI `python -m spp.cli train test2023`
(`src/spp/cli.py:train_test2023`).

- 2023 используется **ровно один раз**. Lock-файл: `reports/test_2023.lock`
  (`holdout.py:LOCK_NAME = "test_2023.lock"`).
- Если lock существует и не передан `--force`, раннер отказывается стартовать
  (`holdout.py:run_holdout`, `GBDTError`).
- Lock пишется **до** касания модели и содержит: `created_at`, `config_hash` (sha256 от
  `config.model_dump`), `matrix_sha256` (хэш `(date, ticker, y[, y_raw, split])`), `file_sha256`
  (sha256 файла матрицы), `git_sha`, `seeds`, `train_rows`, `inner_valid_rows`, `test_rows`,
  `test_window` (`holdout.py:run_holdout`, `_sha256_frame`, `_sha256_file`).
- Обучение: `split ∈ {train, valid}` (2019–2022), тест: `split == "test"` (2023). Проверки:
  тест не раньше `2023-01-01`, обучение не доходит до 2023, тест не выходит за `dates.test_end`
  (`holdout.py:run_holdout`).
- Ранняя остановка — на последних 60 сессиях обучающего среза `split ∈ {train, valid}`
  (тот же `EARLY_STOP_SESSIONS`), а не на отдельном срезе за пределами обучения. Срез 2022
  (`split == "valid"`) входит в обучение и не расходуется как тест; тест — только 2023.

Фактический lock (источник `reports/test_2023.lock`): `created_at = 2026-11-08T08:13:10.043350+00:00`,
`config_hash = 3a1c947f…`, `matrix_sha256 = fab4f76c…`, `file_sha256 = ff0dd4e0…`, `git_sha = null`,
`seeds = [1,2,3]`, `train_rows = 37791`, `inner_valid_rows = 2392`, `test_rows = 10000`,
`test_window = ["2023-01-03","2023-12-29"]`.

---

## 10. Метрики

Реализация: `src/spp/models/metrics.py`.

### IC (Information Coefficient)

- По-датовый Spearman между `y_true` и `y_pred`, затем среднее по датам
  (`metrics.py:ic`, `_spearman`).
- Дата учитывается только при `>= IC_MIN_TICKERS_PER_DATE = 10` тикерах (`metrics.py`).
- Константное предсказание на дате → IC = 0 (не NaN), счётчик `n_constant_dates`.
- `ic_se = std(ddof=1)/sqrt(n)`, `ic_t = mean/se`, 95% CI = `mean ± 1.96·se`.
- `ic_t_nw` — Newey-West t с `NEWEY_WEST_LAG = 5` (`metrics.py:newey_west_tstat`).
- `ic_hit` — доля дат с IC > 0.
- `bootstrap_ic` — круговой блочный бутстрэп по-датовых IC, `block_length = ceil(n_dates^(1/3))`,
  `n = 1000`, `seed = 42` (`metrics.py:bootstrap_ic`).

### DM (Diebold-Mariano)

- `metrics.py:diebold_mariano` на по-датовых средних потерях; `DM > 0` значит A хуже (B лучше).
- `h = 1` → HAC-дисперсия с `h−1` лагами; `h > 1` → поправка Harvey-Leybourne-Newbold.
- В `gbdt.py:_diebold_by_date` сравниваются MAE модели и MAE нулевого прогноза, усреднённые по датам.

### dir_acc (direction accuracy)

- `metrics.py:direction_metrics`, порог `threshold = 0.0`: `+1` при `y_pred > 0`, `−1` при
  `y_pred < 0`, `0` иначе. Нулевые предсказания исключаются из accuracy; `coverage` — доля
  ненулевых. Также считаются `balanced_accuracy`, `auc` (ранговый), `hit_rate_long/short`.

### Монотонность децилей

- `metrics.py:decile_returns` — по-датовые кросс-секционные децили (10 бакетов), среднее по датам.
- Монотонность: `gbdt.py:_monotonicity` → `top_minus_bottom = decile[10] − decile[1]` и
  `decile_rank_corr` (корреляция номера дециля с доходностью). В holdout —
  `holdout.py:_decile_rank_corr` (ранговая корреляция по 10 децилям).

### Критерии вердикта holdout

`holdout.py:run_holdout`, блок `criteria`:

```
ic_gt_0.02                 : ic > 0.02
ic_ci_lo_gt_0              : ic_ci95_lo > 0
dm_p_lt_0.05               : p(DM) < 0.05
decile_monotonicity_gt_0.5 : decile_rank_corr > 0.5
verdict = "confirmed" если все четыре, иначе "not_confirmed"
```

Фактический результат (источник `reports/test_2023.json`): `ic = 0.012715815268538542`,
`ic_ci95 = [-0.01016682642301041, 0.035598456960087496]`, `ic_t_nw = 0.9382925381115094`,
`dir_acc = 0.5089029881293492`, `auc = 0.516127006768755`, `mae = 0.012259227099017887`,
`r2_vs_zero = 0.0011391199450844214`, `dm_vs_zero = {dm: 1.1485488276744968, p: 0.2507420681034076}`,
`decile_rank_corr = 0.04242424242424241`, `verdict = "not_confirmed"` (все четыре критерия `false`).

---

## 11. Бэктест и издержки

Реализация: `src/spp/backtest/portfolio.py`.

### Исполнение

- Основной режим `next_open_hold_close`: вход по `open[t+1]`, выход по `close[t+1]`
  (`portfolio.py:execution_return_columns`, `build_execution_frame`). Это единственный режим,
  реализуемый при сигнале, сформированном после закрытия `t`.
- Сравнительные режимы: `close_to_close` (`close[t] → close[t+1]`), `next_open`
  (`open[t+1] → close[t+2]`).
- Доходность считается по **сырым** ценам (`ret_oc_raw`); `adj_close`-доходность (`ret_oc_adj`)
  хранится только как диагностика, потому что adj_close ретроспективно скорректирован за дивиденды
  и внутрисессионно не торгуется (`portfolio.py` docstring, `_price_returns`).
- Строка помечается `halted`, если `open == close` (нулевой внутридневной диапазон) или нет цены
  выхода; ничего не заполняется вперёд (`build_execution_frame`).

### Построение книги

- `n_quantiles = 5` (`configs/final.yaml:backtest.n_quantiles`): верхний квинтиль — long, нижний —
  short, остальное — mid (`portfolio.py:leg_holdings`).
- `DEFAULT_MIN_NAMES = 5`, `DEFAULT_MIN_LEG_NAMES = 3` (`portfolio.py`): день не торгуется при
  слишком малом числе имён или тонком плече (`build_day_weights`).
- Долларово-нейтральная книга: `gross = 1.0`, каждое плечо получает `gross/2` (`build_day_weights`).

### Издержки

```
cost_per_side = (commission_bps + slippage_bps) / 1e4
              = (5 + 5) / 1e4 = 0.0010
turnover      = 0.5 * Σ|w_t − w_{t−1}|
cost          = turnover * 2 * cost_per_side
borrow        = (borrow_annual / 252) * gross_short
net_ret       = gross_ret − cost − borrow
```

- `commission_bps = 5`, `slippage_bps = 5`, `borrow_annual = 0.05` — `configs/final.yaml:backtest`.
- `COST_PER_SIDE = 0.0010` — константа в `portfolio.py`; `TRADING_DAYS = 252`.
- Первый день заряжается как полное открытие из кэша (`w_{−1} = 0`) — `portfolio.py:compute_turnover`.
- Пропущенный день даёт `gross_ret = NaN`, а не 0 (ноль занизил бы Sharpe) — `run_backtest`.

### Метрики бэктеста

`portfolio.py:summarise_returns`: `sharpe_gross`, `sharpe_net`, `sortino`, `ann_return`, `ann_vol`,
`max_dd`, `calmar`, `hit_rate_daily`, `turnover_ann`, `cost_drag_ann`, `borrow_drag_ann`,
`pnl_by_year`. Нулевая модель — `null_distribution` (перестановка скоров внутри даты, `n_draws`
по умолчанию 200).

Фактический результат 2023 (источник `reports/backtest_metrics.json`): `sharpe_net = -6.750554242942211`,
`sharpe_gross = 0.12343063157938465`, `ann_return = -0.3020872467006628`, `max_dd = -0.29291008427006615`,
`turnover_ann = 169.77857142857144`, `cost_drag_ann = 0.33955714285714284`,
`borrow_drag_ann = 0.025000000000000005`, `hit_rate_daily = 0.3142857142857143`,
`n_dates = 245`, `n_dates_traded = 245`, `n_halted_rows = 36`.

---

## 12. Live-протокол

- Расписание: `configs/final.yaml:live.run_at_utc = "21:15"`; состояние —
  `data/live/state.json` (`live.state_path`).
- Инференс использует `preprocess.joblib` + `model.txt` + `model_meta.json` из
  `reports/runs/{run_id}/final/seed_{s}/` (`gbdt.py:run_gbdt_cv`, блок `final`;
  `gbdt.py:load_model_dir`, `predict_frame`). Каждый прогон также публикует артефакты под
  фиксированными именами в `reports/latest_gbdt/` (`gbdt.py:_publish_latest`).
  `predict_frame` воспроизводит тренировочный путь (порядок колонок не важен, отсутствующие
  колонки → NaN, не 0); паритет закреплён `golden_vectors.parquet` (`gbdt.py:golden_vectors`).
- Живые фиды: RSS (CNBC, MarketWatch, Yahoo per-ticker, Investing), Finnhub (нужен
  `FINNHUB_API_KEY`), GDELT DOC 2.0, Alpha Vantage (opt-in) — `src/spp/data/news_live.py`.
  Дедуп: точный по `text_hash`, near-dupe по триграммному Jaccard `> 0.9` внутри
  `(ticker, 6-часовой бакет)` (`news_live.py:dedupe_live`, `DEDUPE_BUCKET_HOURS = 6`).
- `--dry-run` выполняет fetch/dedupe/валидацию, но ничего не пишет (`cli.py:fetch_live_command`).

---

## 13. Что не зафиксировано в коде

- `huber alpha = 0.9` живёт в `DEFAULT_PARAMS`, а не в `ModelCfg`/`configs/final.yaml`; менять его
  через конфиг нельзя — только правкой кода.
- Размер вселенной (число тикеров) задаётся `configs/universe_us_largecap.yaml`, не кодом.
- Комментарий в `configs/final.yaml` утверждает «62 columns», тогда как `feature_columns()` даёт
  **49** колонок, и `reports/leakage_checks.json` фиксирует `n_features = 49`. Расхождение —
  дефект комментария конфига, а не кода; фактическое число признаков — 49.
- Точное число тикеров и статей в корпусе — `not fixed in code`; см. `docs/limitations.md`
  (40 тикеров, 124 571 статья — из артефактов, не из кода).
