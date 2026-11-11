**🇷🇺 Русский** · **🇬🇧 English**

# Исследование: новостная альфа на криптовалютах

## Русский

Это каталог исследования, а не документация к коду. Здесь лежит полный разбор вопроса
«можно ли зарабатывать на публичных новостях и макрособытиях в крипте» — со всеми числами,
всеми ветками и честным объяснением, почему каждая закончилась так, как закончилась.

| документ | что внутри |
|---|---|
| **[crypto-news-alpha.ru.md](crypto-news-alpha.ru.md)** | полное исследование: пролог с историей проекта, 14 идей, 4 барьера, 9 найденных дефектов, независимая проверка, ограничения, план фальсификации, приложения (реестр экспериментов и журнал решений) |
| [crypto-news-alpha.md](crypto-news-alpha.md) | то же на английском |
| [../crypto-results.md](../crypto-results.md) | рабочий отчёт по крипто-ветке: таблицы по каждому эксперименту в порядке выполнения |
| [../audits/C1-crypto-pipeline.md](../audits/C1-crypto-pipeline.md) | 8 аудитов конвейера в 14 разделах, включая независимую перепроверку чисел |
| [../../data/crypto/README.md](../../data/crypto/README.md) | что закоммичено из данных и как пересобрать остальное |

**Короткий ответ.** Двенадцать из четырнадцати идей отрицательны после издержек, одна не измерима, одна осталась кандидатом. Единственная
положительная ячейка — CPI на 60 минутах: +28.65 бп на BTC (t +2.66) и +35.32 на ETH (t +2.48)
против 14 бп одной ноги. Она осталась кандидатом, а не стратегией: 108 наблюдений, одна эпоха,
и поправка на 36 проверенных ячеек не пройдена.

**Как проверить за пять минут.**

```bash
python ops/crypto_verify.py    # 78 независимых проверок опубликованных чисел
python ops/crypto_verify2.py   # перекрытия, бутстрэп, фандинг, проскальзывание
pytest -q                      # 422 теста, включая проверки утечек
```

`crypto_verify.py` не импортирует ни одного модуля пайплайна: он пересчитывает числа своей
арифметикой из тех же parquet-файлов. Именно так и появился список дефектов: проверяющий
сначала поймал себя.

---

## English

This directory holds the study itself, not code documentation. It answers one question — whether
public news and macro releases can be traded profitably in crypto — with every number, every branch
and an honest account of why each one ended where it did.

| document | contents |
|---|---|
| **[crypto-news-alpha.md](crypto-news-alpha.md)** | the full study: a prologue with the project history, 14 ideas, 4 barriers, 9 defects found, independent verification, limitations, falsification plan, appendices (experiment registry and decision log) |
| [crypto-news-alpha.ru.md](crypto-news-alpha.ru.md) | the same in Russian |
| [../crypto-results.md](../crypto-results.md) | the working report, experiment by experiment in execution order (Russian) |
| [../audits/C1-crypto-pipeline.md](../audits/C1-crypto-pipeline.md) | 8 pipeline audits across 14 sections, including the independent number re-check (Russian) |
| [../../data/crypto/README.md](../../data/crypto/README.md) | what data is committed and how to rebuild the rest |

**Short answer.** Twelve of fourteen ideas are negative after costs, one is not measurable, one remained a candidate. The single positive cell is CPI
at 60 minutes: +28.65 bps on BTC (t +2.66) and +35.32 on ETH (t +2.48) against a 14 bps single leg.
It remains a candidate, not a strategy: 108 observations, one era, and the 36-cell correction is not
cleared.

**Verify in five minutes.**

```bash
python ops/crypto_verify.py    # 78 independent checks of the published numbers
python ops/crypto_verify2.py   # overlap, bootstrap, funding, slippage
pytest -q                      # 422 tests, including the leakage gates
```

`crypto_verify.py` imports no pipeline module: it recomputes the numbers with its own arithmetic
from the same parquet files. That is how the defect list came about — the checker caught itself
first.
