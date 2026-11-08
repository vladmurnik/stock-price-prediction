# Agent bus

Автоматически собирается из `ops/bus/messages.jsonl` (`python ops/bus.py render`).
Здесь виден прямой диалог агентов с оркестратором: progress, вопросы, результаты.

| # | время (UTC) | от | кому | тип | задача | сообщение |
|---|---|---|---|---|---|---|
| 1 | 00:56:57 | orchestrator | orchestrator | progress | t11 | bus online; backtest CV run finished, verdict not_confirmed |
| 2 | 00:56:57 | quant-model | orchestrator | question | t10 | A3 requires article_sentiment_openjev.parquet; proceed with A1,A2,A4..A8 now and run A3 after openjev scoring? |
| 3 | 00:57:38 | orchestrator | quant-model | progress | t11 | dispatch bus check |
| 4 | 08:19:02 | quant-data | orchestrator | progress | t12 | T12 start: reading 12-live-feeds.md, inspecting normalize.py/store.py/cli.py. FINNHUB_API_KEY is NOT set in env (will implement fully, skip with WARN). Plan: src/spp/data/news_live.py + tests/test_live_feeds.py + etch live CLI, then --dry-run and 30d backfill. |
