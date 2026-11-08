# AGENTS.md — как работают агенты проекта

Проект: `stock-price-prediction` (пакет `spp`).
План: `C:\Users\adik\.opencode\plan\stock-sentiment-prediction\` (файлы `00-index.md` … `16-docs.md`).

## Роли

| Агент | Владеет | Не трогает |
|---|---|---|
| `quant-data` | `src/spp/data/`, `data/interim/`, `data/cache/prices*` — T2, T3, T12 | features, models, backtest, live |
| `quant-sentiment` | `src/spp/sentiment/`, `data/processed/article_sentiment.parquet` — T4, T5 | data ingestion, features, models |
| `quant-model` | `src/spp/features/`, `src/spp/models/`, `src/spp/backtest/` — T6–T11 | data ingestion, sentiment, live |
| `quant-live` | `src/spp/live/`, `data/live/`, `reports/live_*` — T13–T15 | model artifacts, features, sentiment |
| `quant-guard` | ничего не пишет; только аудит и вердикт | любые правки |
| `implementer` | каркас, docs, точечные задачи вне доменов — T1, T16 | доменные модули |

Границы файлов жёсткие: агент, которому нужен чужой модуль, сообщает об этом в отчёте, а не правит его.

## Протокол работы

1. Один запуск = одна задача из `ops/queue.json` = один файл плана.
2. Агент читает файл плана, реализует, запускает команды из раздела `Acceptance`, фиксирует реальный вывод.
3. Отчёт всегда начинается со строки `## STATUS: DONE | BLOCKED | NEEDS-DECISION`.
4. Никаких вопросов пользователю: при неоднозначности — безопасная трактовка + `Assumptions`.
5. `BLOCKED` только при реальном блокере (ключ, который нельзя заменить заглушкой; переполненный диск; разрушительное решение). В отчёте — точное действие для разблокировки.
6. Факты вместо прилагательных: числа, команды, пути. «Улучшилось» без CI и t-stat — не результат.
7. Отрицательный результат публикуется так же, как положительный.

## Как агенты «выходят на связь»

- `ops/dispatch.ps1` запускает задачу через `opencode run --agent <id>` и складывает полный отчёт в
  `ops/inbox/{timestamp}-{task}-{agent}.md`.
- Строка статуса пишется в `ops/status.md` (таблица: время, задача, агент, статус, файл отчёта).
- `ops/queue.json` хранит состояние: `pending`, `running`, `done`, `blocked`, `needs-decision`.
- Я (оркестратор) читаю `ops/status.md` и `ops/inbox/` и решаю, что запускать дальше.
- Для длинных задач агент вызывается в фоне (`background: true`) или через `dispatch.ps1 -Loop`.

## Непрерывный цикл

```powershell
cd C:\Users\adik\Desktop\progects\stock-price-prediction

# одна следующая готовая задача
.\ops\dispatch.ps1 -MaxTasks 1

# работать, пока есть готовые задачи (до блокера или пустой очереди)
.\ops\dispatch.ps1 -Loop -MaxTasks 50

# только посмотреть, что будет запущено
.\ops\dispatch.ps1 -DryRun

# конкретная задача
.\ops\dispatch.ps1 -Only t06
```

Очередь уважает `depends_on`: задача запускается только когда все зависимости `done`.
Параллельный запуск двух агентов допустим только при непересекающихся файлах — по умолчанию цикл
последовательный.

## Дисциплина данных и статистики

- Таргет — доходность `log(adj_close[t+1]/adj_close[t])`, не цена.
- 2023 используется один раз, после T10, с `.lock`-файлом.
- Шесть проверок утечек из `06-features-leakage.md` обязательны; провал проверки не «чинится» ослаблением проверки.
- Никакого скрапинга HTML новостных сайтов.
- Ключи только из `.env` / переменных окружения; в отчёты и логи не попадают.
- `quant-guard` вызывается после T10–T11 и после T15; без его `PASS` результат не публикуется.

## Шина сообщений (прямая связь с оркестратором)

Агент не ждёт конца прогона, чтобы сообщить о проблеме или результате.

```
.\venv\Scripts\python.exe ops\bus.py send --from <agent> --type progress --task <id> --text "..."
.\venv\Scripts\python.exe ops\bus.py read --to <agent> --unread
```

Типы: `progress`, `question`, `blocked`, `result`, `handoff`.

Правила:
- минимум три сообщения за запуск: старт, первый значимый шаг, результат;
- вопрос задаётся через `--type question`, ответ читается `read --to <agent> --unread`;
  если ответа нет — выбирай консервативный вариант, фиксируй допущение и продолжай;
- сообщение короче 4000 символов, только факты и числа;
- шина не заменяет финальный отчёт по контракту: `## STATUS: ...` остаётся обязательным;
- `ops/chat.md` — человекочитаемая лента диалога, собирается `python ops/bus.py render`.

## Definition of done задачи

1. Acceptance-команды выполнены, вывод приложен.
2. Артефакты на месте по именам из `00-index.md`.
3. Отчёт по контракту (STATUS, Result, Changed, Verification, Risks, Next).
4. Обновлён `ops/queue.json` (делает dispatcher, не агент).
