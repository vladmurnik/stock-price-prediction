"""Backend selection rule and markdown rendering for the calibration reports.

Selection (straight from the plan): primary backend = the one with the highest macro-F1 among
those that clear ``sentiment.min_macro_f1``.  Ties within ±0.01 go to FinBERT.  If nothing clears
the gate, FinBERT stays primary (it is the most direct financial model) and the failure is
reported as a negative result — never silently swapped for something else.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from spp.sentiment.calibrate import CalibrationResult

__all__ = ["TIE_TOLERANCE", "render_decision", "render_summary", "select_backend"]

TIE_TOLERANCE = 0.01
PREFERRED_ON_TIE = "finbert"


def select_backend(results: Sequence[CalibrationResult], gate: float) -> dict[str, Any]:
    """Apply the plan's selection rule; return the decision plus the supporting numbers."""
    by_name = {r.backend: r for r in results}
    passing = [r for r in results if r.macro_f1 >= gate]
    ranked = sorted(results, key=lambda r: r.macro_f1, reverse=True)

    if not passing:
        chosen = PREFERRED_ON_TIE if PREFERRED_ON_TIE in by_name else ranked[0].backend
        reason = (
            f"no backend cleared the macro-F1 gate {gate:.2f}; {chosen} stays primary as the "
            "financial-domain model and this is recorded as a negative result"
        )
        return {
            "primary": chosen,
            "phase": "below_gate",
            "reason": reason,
            "gate": gate,
            "macro_f1": by_name[chosen].macro_f1 if chosen in by_name else None,
            "ranking": [(r.backend, r.macro_f1) for r in ranked],
        }

    top = max(passing, key=lambda r: r.macro_f1)
    tied = [r for r in passing if abs(r.macro_f1 - top.macro_f1) <= TIE_TOLERANCE]
    if PREFERRED_ON_TIE in {r.backend for r in tied}:
        chosen = PREFERRED_ON_TIE
        reason = (
            f"{len(tied)} backends within ±{TIE_TOLERANCE} of the best macro-F1 "
            f"({top.macro_f1:.4f}); tie-break rule selects {chosen}"
        )
    else:
        chosen = top.backend
        reason = f"{chosen} has the best macro-F1 ({top.macro_f1:.4f}) above the gate {gate:.2f}"

    runner_up = next((r for r in ranked if r.backend != chosen), None)
    return {
        "primary": chosen,
        "phase": "above_gate",
        "reason": reason,
        "gate": gate,
        "macro_f1": by_name[chosen].macro_f1,
        "delta_vs_runner_up": (
            None if runner_up is None else by_name[chosen].macro_f1 - runner_up.macro_f1
        ),
        "runner_up": None if runner_up is None else (runner_up.backend, runner_up.macro_f1),
        "ranking": [(r.backend, r.macro_f1) for r in ranked],
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_summary(
    results: Sequence[CalibrationResult],
    *,
    gate: float,
    seed: int,
    max_rows: int | None,
    phrasebank_repo: str,
    phrasebank_config: str,
    generated_at: str | None = None,
) -> str:
    """``reports/calibration/summary.md``: one table, all three backends, all four metrics."""
    stamp = generated_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# Sentiment backend calibration — financial phrasebank",
        "",
        f"- generated: `{stamp}`",
        f"- data: `{phrasebank_repo}` / `{phrasebank_config}` "
        "(3 classes: negative / neutral / positive)",
        f"- split: stratified 70/30 on `seed={seed}`; evaluation cap "
        f"`max_rows={max_rows if max_rows is not None else 'full'}` "
        "(stratified, applied before the split)",
        "- no backend was fine-tuned on the phrasebank; only one scalar temperature `T` per "
        "backend was fit, on the first 30% of the held-out slice",
        f"- macro-F1 gate: `{gate:.2f}`",
        "",
        "## Headline table",
        "",
        "| backend | model | accuracy | macro-F1 | ECE (T=1) | ECE (T fit) | "
        "Brier | items/s | T | rows |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: x.backend):
        lines.append(
            f"| {r.backend} | `{r.model_version.split('@')[0]}` | {r.accuracy:.4f} | "
            f"{r.macro_f1:.4f} | {r.ece_before:.4f} | {r.ece:.4f} | {r.brier:.4f} | "
            f"{r.throughput:.2f} | {r.temperature:.3f} | {r.rows_evaluated}/{r.rows_before_cap} |"
        )
    lines += [
        "",
        "`rows` = evaluated rows / phrasebank 75%-agreement rows before the cap.",
        "",
        "## Per-class F1",
        "",
        "| backend | F1 negative | F1 neutral | F1 positive | support (neg/neu/pos) |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: x.backend):
        sup = r.per_class_support
        lines.append(
            f"| {r.backend} | {r.per_class_f1['negative']:.4f} | {r.per_class_f1['neutral']:.4f} | "
            f"{r.per_class_f1['positive']:.4f} | "
            f"{sup['negative']}/{sup['neutral']}/{sup['positive']} |"
        )
    lines += [
        "",
        "## 5-level label agreement",
        "",
        "The feature layer consumes `label_5`/`score_5`, so the 3-class gold is projected onto the "
        "5-level axis (negative→`negative`, neutral→`neutral`, positive→`positive`) "
        "and agreement reported.",
        "",
        "| backend | label_5 accuracy |",
        "|---|---|",
    ]
    for r in sorted(results, key=lambda x: x.backend):
        lines.append(f"| {r.backend} | {r.label_5_accuracy:.4f} |")
    lines += [
        "",
        "## Timing and errors",
        "",
        "| backend | elapsed s (scoring) | throughput items/s | per-text errors |",
        "|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: x.backend):
        lines.append(f"| {r.backend} | {r.elapsed_sec:.2f} | {r.throughput:.2f} | {r.n_errors} |")
    lines += [
        "",
        "## ECE reliability tables (temperature applied)",
        "",
    ]
    for r in sorted(results, key=lambda x: x.backend):
        lines += [
            f"### {r.backend} (T={r.temperature:.3f}, ECE={r.ece:.4f})",
            "",
            "| bin | n | mean confidence | accuracy |",
            "|---|---|---|---|",
        ]
        for b in r.ece_bins:
            lines.append(
                f"| {b['lo']:.1f}–{b['hi']:.1f} | {b['n']} | {_fmt(b['conf'])} | {_fmt(b['acc'])} |"
            )
        lines.append("")

    lines += [
        "## How temperature was fit",
        "",
        "Per backend, over the rows of the held-out slice, by minimising NLL on log-probabilities "
        "recovered from the returned `p_*` (`scipy.optimize.minimize_scalar`, bounded 0.5–3.0). "
        "The fit set is the first 30% of the evaluated slice; all metrics above use the "
        "remaining 70%. "
        "Temperature is applied everywhere downstream; it never changes `argmax`, so accuracy and "
        "macro-F1 are identical before and after.",
        "",
    ]
    return "\n".join(lines)


def render_decision(
    results: Sequence[CalibrationResult],
    decision: dict[str, Any],
    *,
    seed: int,
    max_rows: int | None,
    gate: float,
    generated_at: str | None = None,
) -> str:
    """``reports/calibration/decision.md``: the chosen backend, the rule and the numbers."""
    stamp = generated_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    by_name = {r.backend: r for r in results}
    primary = decision["primary"]
    lines = [
        "# Sentiment backend decision (T4)",
        "",
        f"- generated: `{stamp}`",
        f"- rule: primary = max macro-F1 among backends with `macro_f1 >= {gate:.2f}`; "
        f"ties within ±{TIE_TOLERANCE} go to FinBERT",
        f"- seed: `{seed}`, evaluation cap: `{max_rows if max_rows is not None else 'full'}`",
        "",
        "## Decision",
        "",
        f"**Primary backend: `{primary}`**",
        "",
        f"- phase: `{decision['phase']}`",
        f"- reason: {decision['reason']}",
        f"- macro-F1 (`{primary}`): {_fmt(decision.get('macro_f1'))}",
    ]
    if decision.get("runner_up"):
        ru_name, ru_f1 = decision["runner_up"]
        lines.append(f"- runner-up: `{ru_name}` macro-F1 {_fmt(ru_f1)}")
    if decision.get("delta_vs_runner_up") is not None:
        lines.append(f"- margin: {decision['delta_vs_runner_up']:+.4f} macro-F1")

    lines += [
        "",
        "## Ranking (macro-F1)",
        "",
        "| rank | backend | macro-F1 | above gate | accuracy | ECE |",
        "|---|---|---|---|---|---|",
    ]
    for i, (name, f1) in enumerate(decision["ranking"], start=1):
        r = by_name.get(name)
        acc = _fmt(r.accuracy) if r else "—"
        ece = _fmt(r.ece) if r else "—"
        above = "yes" if f1 >= gate else "no"
        lines.append(f"| {i} | {name} | {f1:.4f} | {above} | {acc} | {ece} |")

    lines += [
        "",
        "## Consequences",
        "",
    ]
    if decision["phase"] == "above_gate":
        lines += [
            f"- `configs/base.yaml` → `sentiment.backend: {primary}`. This is written by hand "
            "after review, not auto-applied by the CLI.",
            "- Other backends stay available: open-jev remains the source of `is_novel` and "
            "`is_company_specific` even when it is not primary.",
        ]
    else:
        lines += [
            "- No backend cleared the gate. FinBERT remains primary for the corpus; this is a "
            "**recorded negative result**, not a blocker.",
            "- Suggested next step: tighten the evaluation set (full phrasebank) or accept a lower "
            f"`sentiment.min_macro_f1` with an explicit note, since {gate:.2f} may be optimistic.",
        ]
    lines += [
        "",
        "## Honesty notes",
        "",
        "- open-jev is a *general* typed-decision model (banking77 + SST-5 + BoolQ), not a finance "
        "model. Its financial macro-F1 is reported as measured, including when it trails FinBERT.",
        "- No backend was fine-tuned on the phrasebank. Doing so would make the model's market "
        "metrics dishonest, because the phrasebank is the measurement instrument.",
        "- Accuracy and macro-F1 are invariant under temperature scaling (argmax is preserved); "
        "only ECE, Brier and confidence move.",
        "",
    ]
    return "\n".join(lines)
