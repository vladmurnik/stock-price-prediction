# Disk budget guard (30 GB limit)

Проект должен укладываться в 30 ГБ. Этот скрипт печатает фактическое потребление
и падает с ненулевым кодом, если свободного места меньше порога.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ops\disk_report.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File ops\disk_report.ps1 -MinFreeGB 8
```

Жёсткие правила (нарушение = BLOCKED):

1. Никогда не скачивать `Stock_news/nasdaq_exteral_data.csv` (23.2 ГБ) — только
   `Stock_news/All_external.csv` (5.73 ГБ) через HTTP range-streaming, без записи на диск.
2. Не использовать `datasets.load_dataset` для FNSPID: буферизует гигабайты в
   `~/.cache/huggingface/datasets`.
3. Хранить только отфильтрованный корпус (`data/interim/news_raw.parquet`), не сырой CSV.
4. `pip cache` чистить после установки тяжёлых пакетов (`python -m pip cache purge`).
5. open-jev держать в одном формате: ONNX ИЛИ transformers, не оба (экономия ~1.7 ГБ).
