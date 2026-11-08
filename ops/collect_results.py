"""Collect the evidence a public reader needs into ``results/`` (small text only)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

flat = [
    "test_2023.json",
    "test_2023.md",
    "leakage_checks.json",
    "baselines.json",
    "cv_folds.json",
    "backtest_metrics.json",
]
for name in flat:
    src = REPORTS / name
    if src.is_file():
        shutil.copyfile(src, OUT / name)
        print(f"copied {name} ({src.stat().st_size} B)")

for path in sorted(REPORTS.glob("backtest_*.md")):
    shutil.copyfile(path, OUT / path.name)
    print(f"copied {path.name} ({path.stat().st_size} B)")

runs = OUT / "runs"
runs.mkdir(exist_ok=True)
for run in sorted((REPORTS / "runs").glob("*/metrics.json")):
    target = runs / f"{run.parent.name}.json"
    shutil.copyfile(run, target)
    print(f"copied runs/{target.name} ({run.stat().st_size} B)")

cal = REPORTS / "calibration"
if cal.is_dir():
    dst = OUT / "calibration"
    dst.mkdir(exist_ok=True)
    for path in sorted(cal.glob("*.json")):
        shutil.copyfile(path, dst / path.name)
        print(f"copied calibration/{path.name}")

summary = {}
for cfg in (REPORTS / "runs").glob("*/metrics.json"):
    payload = json.loads(cfg.read_text(encoding="utf-8"))
    block = payload.get("aggregate", {}).get("ic", {})
    summary[cfg.parent.name] = {
        "ic": block.get("mean"),
        "ic_std": block.get("std"),
        "n_folds": block.get("n"),
        "pooled_ic": payload.get("pooled", {}).get("ic"),
        "dm_p": payload.get("dm_vs_zero", {}).get("p"),
    }
(OUT / "ablation_cv.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
