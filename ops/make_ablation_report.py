"""Render ``reports/ablation.md`` from the real metrics files (no hand-typed numbers)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "reports" / "runs"
HOLDOUT = ROOT / "reports" / "test_2023.json"
OUT = ROOT / "reports" / "ablation.md"
OUT_PUBLIC = ROOT / "results" / "ablation.md"

SPEC = [
    ("a1-market-only", "A1", "market only", "finbert", "false"),
    ("a2-sentiment-only", "A2", "sentiment only", "finbert", "false"),
    ("a5-all", "A5", "market + sentiment", "finbert", "false"),
    ("a4-lexicon", "A4", "market + sentiment", "lexicon", "false"),
    ("a6-strict-dates", "A6", "market + sentiment", "finbert", "true"),
    ("a3-openjev", "A3", "sentiment only", "open-jev", "false"),
    ("a7-ensemble", "A7", "ensemble of A2 + A3", "finbert + open-jev", "false"),
    ("a8-openjev", "A8", "market + sentiment + typed flags", "open-jev", "false"),
]


def _ci(mean: float, std: float | None, n: int) -> tuple[float, float]:
    if not std or n <= 1:
        return float("nan"), float("nan")
    half = 1.96 * std / (n**0.5)
    return mean - half, mean + half


def main() -> int:
    rows = []
    for run_id, tag, label, backend, strict in SPEC:
        path = RUNS / run_id / "metrics.json"
        if not path.is_file():
            rows.append((tag, label, backend, strict, None, None, None, None, None))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        block = payload.get("aggregate", {}).get("ic", {})
        mean = block.get("mean")
        std = block.get("std")
        n = int(block.get("n") or 0)
        lo, hi = _ci(mean, std, n) if mean is not None else (None, None)
        rows.append(
            (
                tag,
                label,
                backend,
                strict,
                mean,
                lo,
                hi,
                (payload.get("pooled", {}) or {}).get("ic"),
                ((payload.get("dm_vs_zero", {}) or {}).get("p")),
            )
        )

    lines = [
        "# Аблации: IC на purged CV (2019–2022)",
        "",
        "Все числа сгенерированы `ops/make_ablation_report.py` из `reports/runs/*/metrics.json`.",
        "6 фолдов × 3 сида, цель — demeaned лог-доходность следующего дня.",
        "",
        "| id | признаки | бэкенд | strict_dates | IC | 95% CI | pooled IC | DM p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tag, label, backend, strict, mean, lo, hi, pooled, dm in rows:
        if mean is None:
            lines.append(f"| {tag} | {label} | {backend} | {strict} | — | — | — | — |")
            continue
        ci = "—" if lo is None or lo != lo else f"[{lo:.4f}, {hi:.4f}]"
        lines.append(
            f"| **{tag}** | {label} | {backend} | {strict} | {mean:.4f} | {ci} | "
            f"{pooled:.4f} | {dm:.4f} |"
        )

    if HOLDOUT.is_file():
        payload = json.loads(HOLDOUT.read_text(encoding="utf-8"))
        m = payload["metrics"]
        lines += [
            "",
            "## Единственный holdout 2023",
            "",
            f"- IC **{m['ic']:.4f}**, 95% CI [{m['ic_ci95_lo']:.4f}, {m['ic_ci95_hi']:.4f}]",
            f"- direction accuracy {m['dir_acc']:.4f}, DM p {payload['dm_vs_zero']['p']:.4f}",
            f"- вердикт: **{payload['verdict']}**",
            "",
            "Критерии: " + ", ".join(f"{k}={v}" for k, v in payload["criteria"].items()),
        ]

    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    OUT_PUBLIC.parent.mkdir(exist_ok=True)
    OUT_PUBLIC.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
