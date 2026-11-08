"""T9 acceptance probe: artifacts, test isolation, predict_frame parity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spp.models.gbdt import load_model_dir, predict_frame

runs = sorted(Path("reports/runs").glob("gbdt-*"), key=lambda p: p.stat().st_mtime)
RUN = runs[-1]
print("run dir:", RUN.name)

pred = pd.read_parquet(RUN / "predictions.parquet")
print("predictions rows:", len(pred))
print("folds:", sorted(pred["fold"].unique()), "| seeds per fold:", pred.groupby("fold")["seed"].nunique().unique())

feat = pd.read_parquet("data/processed/features.parquet")
test_dates = set(pd.to_datetime(feat.loc[feat["split"] == "test", "date"]).dt.date)
pred_dates = pd.to_datetime(pred["date"]).dt.date
print("test rows leaked into predictions:", int(pred_dates.isin(test_dates).sum()))
print("prediction window:", pred_dates.min(), "..", pred_dates.max())

final_dir = RUN / "final"
print("final seeds:", sorted(p.name for p in final_dir.glob("seed_*")))
meta = json.loads((final_dir / "seed_1" / "model_meta.json").read_text(encoding="utf-8"))
print("final meta: trees=", meta["best_iteration"], "rows=", meta["train_rows"], "dates=", meta["train_dates"])

metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
print("cv ic mean:", round(metrics["aggregate"]["ic"]["mean"], 4),
      "| ci:", [round(v, 4) for v in (metrics["aggregate"]["ic"].get("ci") or [])] or "n/a",
      "| pooled:", round(metrics["pooled"]["ic"], 4))

series = predict_frame(final_dir / "seed_1", feat.sample(50, random_state=3))
print("predict_frame rows:", len(series))
print("predict_frame std:", float(np.std(series.to_numpy())))
