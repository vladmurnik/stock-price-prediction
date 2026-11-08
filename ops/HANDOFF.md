# Handoff — where the project stopped

Дата остановки: 2026-11-08.

## Состояние

| шаг | статус | артефакт |
|---|---|---|
| T1 scaffold | done | `src/spp/`, `configs/`, 126 тестов |
| T2 prices | done | `data/cache/prices_2019-01-01_2023-12-31.parquet` |
| T3 news | done | `data/interim/news_raw.parquet` — 124 571 статья, 38 тикеров |
| T4 sentiment | done | `reports/calibration/{finbert,openjev,lexicon}.json`, `summary.md` |
| T5 corpus scoring | **paused** | код готов (`score_corpus.py`, CLI `score corpus`), полный прогон не запускался |

## Что осталось сделать, чтобы продолжить

1. Поднять MCP обратно: в `~/.config/opencode/opencode.jsonc` у `web2spec` поставить `"enabled": true` (или удалить строку).
2. Прогнать `python -m pytest -q` и `ruff check src tests` (на момент остановки: 126 passed, ruff clean).
3. Smoke: `python -m spp.cli score corpus --backend finbert --limit 200 --no-write`.
4. Полный прогон FinBERT: ~92 927 уникальных текстов, ~2 часа при 12.6 items/s.
5. `openjev` — только выборка 8 000 статей (`openjev_max_articles`), ~2 часа.
6. Далее T6–T11 (фичи/утечки/CV/LightGBM/аблации/бэктест), T12–T15 (live), T16 (docs).

## Факты, которые нельзя забыть

- Плотность FNSPID для этого universe: медиана 1 статья/тикер-день. Acceptance плана (≥300k, медиана ≥3) не выполнен — это ограничение мощности, а не баг. Фичи должны агрегировать окна 3–5 дней.
- `trafilatura` + `newspaper4k` дают корректное извлечение текста, но крупные издатели отдают 403 на прямой HTTP: 5 из 6 проверенных URL. Полный текст — только обогащение, основной источник заголовков — API (GDELT/Finnhub/RSS).
- open-jev: macro-F1 0.608 на PhraseBank, 1.1 items/s — только аблация A3, не основной бэкенд.
- Основной бэкенд: FinBERT, macro-F1 0.8862, ECE 0.040 после калибровки (T=0.704).
