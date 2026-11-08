"""Temperature calibration and measurement of the three sentiment backends.

This module is a *measurement* harness, never a training loop.  No backend is ever fine-tuned on
``takala/financial_phrasebank``: the phrasebank is the ruler, and fitting it would make every
downstream metric dishonest.  The only fitted parameter is one scalar temperature per backend,
fit on a held-out slice, and it changes confidence but never the predicted class.

Pipeline:
1. load ``sentences_75agree`` (3 classes: negative / neutral / positive), capped deterministically;
2. stratified 70/30 split with ``config.seed``;
3. score the 30% slice once per backend, caching raw *log-probabilities* so temperature can be swept
   without re-running the model;
4. fit ``T`` by minimising NLL on the held-out part of that 30% slice (``scipy.optimize``);
5. report accuracy, macro-F1 and per-class F1, ECE (10 bins), Brier, throughput, plus 5-level
   agreement because ``score_5``/``label_5`` are what the feature layer consumes.

Split discipline (documented, seed-fixed): ``train=70%`` is *unused* by T4 — it exists so that the
number cannot be inflated by tuning against the evaluation rows.  ``T`` is fit on the first 30% of
the held-out slice and all reported metrics are computed on the remaining 70% of it.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from spp.logging import get_logger
from spp.sentiment.base import ArticleSentiment, SentimentBackend

__all__ = [
    "CalibrationResult",
    "PHRASEBANK_REPO",
    "eval_frame",
    "expected_calibration_error",
    "fit_temperature",
    "load_phrasebank",
    "python_rng_seed",
    "run_backend_calibration",
    "split_phrasebank",
]

_LOG = get_logger("sentiment.calibrate")

PHRASEBANK_REPO = "takala/financial_phrasebank"
PHRASEBANK_FILE = "FinancialPhraseBank-v1.0.zip"
PHRASEBANK_MEMBER = "FinancialPhraseBank-v1.0/Sentences_75Agree.txt"

#: Phrasebank labels are 3-class; index order matches our (neg, neu, pos) canonical order.
LABELS_3 = ("negative", "neutral", "positive")
LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS_3)}

#: 5-level gold for the score_5/label_5 agreement check: three classes map to the 5-level axis by
#: taking the *centre* level of each region: neg -> "negative", neu -> "neutral",
#: pos -> "positive".
LABEL_3_TO_LEVEL5 = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}

#: Upper bound on evaluation rows.  The full 75%-agreement set is 3453; the default cap keeps a
#: calibration run inside a few CPU-minutes even with open-jev, and is recorded in the report.
DEFAULT_MAX_ROWS = 1000


# ---------------------------------------------------------------------------- data
def load_phrasebank(
    cache_dir: Path | None = None,
    *,
    file_name: str = PHRASEBANK_FILE,
    member: str = PHRASEBANK_MEMBER,
) -> tuple[list[str], list[int]]:
    """Return ``(texts, labels)`` for the 75%-agreement phrasebank file.

    The HF dataset uses a legacy loader script that ``datasets>=5`` refuses to execute, so we read
    the published zip directly.  ``cache_dir`` (if given) holds a local copy so runs are offline
    and reproducible; otherwise the hub cache is used.
    """
    from spp.sentiment.hf_env import configure_hf_cache

    configure_hf_cache()
    blob: bytes | None = None
    if cache_dir is not None:
        local = Path(cache_dir) / file_name
        if local.is_file():
            blob = local.read_bytes()
        else:
            local.parent.mkdir(parents=True, exist_ok=True)
    if blob is None:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(PHRASEBANK_REPO, f"data/{file_name}", repo_type="dataset")
        except Exception as exc:  # noqa: BLE001 - explicit, actionable error
            raise RuntimeError(
                f"cannot download {PHRASEBANK_REPO}: {exc}. "
                "The phrasebank is the evaluation set; it must be present or the calibration is "
                "meaningless. Check network/HF_TOKEN or place the zip under the cache dir."
            ) from exc
        blob = Path(path).read_bytes()
        if cache_dir is not None:
            (Path(cache_dir) / file_name).write_bytes(blob)

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        raw = zf.read(member)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:  # the archive is latin-1 (©, ′, – inside sentences)
        text = raw.decode("iso-8859-1")

    texts: list[str] = []
    labels: list[int] = []
    skipped = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        body, sep, label = line.rpartition("@")
        label = label.strip().lower()
        if not sep or label not in LABEL_TO_IDX or not body.strip():
            skipped += 1
            continue
        texts.append(body.strip())
        labels.append(LABEL_TO_IDX[label])
    if not texts:
        raise RuntimeError(f"phrasebank parsed to zero rows from {member}")
    _LOG.info(
        "phrasebank: %d rows (skipped %d) | class counts %s",
        len(texts),
        skipped,
        {name: labels.count(i) for name, i in LABEL_TO_IDX.items()},
    )
    return texts, labels


def split_phrasebank(
    texts: Sequence[str],
    labels: Sequence[int],
    seed: int,
    *,
    max_rows: int | None = DEFAULT_MAX_ROWS,
    holdout: float = 0.3,
) -> dict[str, Any]:
    """Deterministic stratified 70/30 split.

    ``max_rows`` caps the evaluation set: rows are drawn by a stratified sample so the class
    balance is preserved (open-jev at ~1.8 s/row makes the full set impractical on CPU).  The cap
    is applied *before* the split and recorded in the result so the number is auditable.
    """
    idx = np.arange(len(texts))
    y = np.asarray(labels)
    if max_rows is not None and max_rows < len(idx):
        keep, _ = train_test_split(
            idx, train_size=max_rows, random_state=seed, stratify=y, shuffle=True
        )
        keep = np.sort(keep)
        idx, y = keep, y[keep]
        _LOG.info("phrasebank capped to %d rows (stratified, seed=%d)", len(idx), seed)

    # Indices are positional within the returned ``texts``/``labels`` arrays: the cap
    # remaps rows, so absolute positions in the original corpus would overflow.
    positions = np.arange(len(idx))
    train_idx, test_idx = train_test_split(
        positions, test_size=holdout, random_state=seed, stratify=y
    )
    return {
        "train_idx": np.sort(train_idx),
        "test_idx": np.sort(test_idx),
        "texts": [texts[i] for i in idx],
        "labels": y,
        "rows_before_cap": len(texts),
        "seed": seed,
    }


# ---------------------------------------------------------------------------- metrics
def expected_calibration_error(
    probs: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> tuple[float, list[dict[str, float]]]:
    """ECE with equal-width confidence bins; also returns the bin table for the report."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == correct).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    table: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            nan = float("nan")
            table.append({"lo": float(lo), "hi": float(hi), "n": 0, "acc": nan, "conf": nan})
            continue
        bin_acc = float(acc[mask].mean())
        bin_conf = float(conf[mask].mean())
        ece += mask.mean() * abs(bin_acc - bin_conf)
        table.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": int(mask.sum()),
                "acc": bin_acc,
                "conf": bin_conf,
            }
        )
    return float(ece), table


def python_rng_seed(seed: int) -> None:
    """Seed numpy/random (torch is optional) so any backend stochastic path is reproducible."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:  # noqa: BLE001 - torch may be absent in a lexicon-only environment
        pass


# ---------------------------------------------------------------------------- temperature
def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, bounds: tuple[float, float] = (0.5, 3.0)
) -> float:
    """Fit a scalar temperature by NLL minimisation (``scipy.optimize.minimize_scalar``)."""
    from scipy.optimize import minimize_scalar

    def nll(t: float) -> float:
        scaled = logits / t
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        log_z = np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        log_p = scaled - log_z
        return float(-log_p[np.arange(len(labels)), labels].mean())

    res = minimize_scalar(nll, bounds=bounds, method="bounded", options={"xatol": 1e-4})
    return float(res.x)


def apply_temperature(logits: np.ndarray, t: float) -> np.ndarray:
    """Softmax with temperature; argmax is invariant, only the confidence changes."""
    scaled = logits / float(t)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exps = np.exp(scaled)
    return exps / exps.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------- backend adapter
def _logits_from_p(sentiments: Sequence[ArticleSentiment]) -> np.ndarray:
    """Recover unnormalised log-probabilities from a 3-class distribution (up to a constant)."""
    p = np.array([[s.p_neg, s.p_neu, s.p_pos] for s in sentiments], dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    return np.log(p)


@dataclass
class CalibrationResult:
    backend: str
    model_version: str
    temperature: float
    n_train_slice: int
    n_eval: int
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    per_class_support: dict[str, int]
    ece: float
    ece_before: float
    brier: float
    throughput: float
    n_errors: int
    rows_evaluated: int
    rows_before_cap: int
    seed: int
    label_5_accuracy: float
    ece_bins: list[dict[str, float]]
    elapsed_sec: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def markdown_row(self) -> str:
        return (
            f"| {self.backend} | {self.model_version.split('@')[0]} | {self.accuracy:.4f} | "
            f"{self.macro_f1:.4f} | {self.ece_before:.4f} | {self.ece:.4f} | {self.brier:.4f} | "
            f"{self.throughput:.2f} | {self.temperature:.3f} | {self.row_count} |"
        )

    @property
    def row_count(self) -> str:
        return f"{self.rows_evaluated}/{self.rows_before_cap}"


def _score_all(
    backend: SentimentBackend, texts: Sequence[str], batch_size: int, limit: int | None
) -> tuple[list[ArticleSentiment], float, int]:
    """Score texts (optionally only the first ``limit``), returning rows, elapsed, error count."""
    from spp.sentiment.base import ensure_text

    out: list[ArticleSentiment] = []
    errors = 0
    t0 = time.perf_counter()
    n = len(texts) if limit is None else min(limit, len(texts))
    for start in range(0, n, batch_size):
        chunk = list(texts[start : start + batch_size])
        try:
            out.extend(backend.score(chunk))
        except ValueError:  # per-text validation error inside a batch: fall back one by one
            for one in chunk:
                try:
                    out.append(backend.score([ensure_text(one)])[0])
                except ValueError:
                    errors += 1
                    out.append(None)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - model failure must be loud
            raise RuntimeError(f"backend {backend.name} failed: {exc}") from exc
    elapsed = time.perf_counter() - t0
    return out, elapsed, errors


def run_backend_calibration(
    backend: SentimentBackend,
    texts: Sequence[str],
    labels: Sequence[int],
    *,
    seed: int,
    max_rows: int | None = DEFAULT_MAX_ROWS,
    batch_size: int = 32,
    holdout: float = 0.3,
    out_dir: Path | None = None,
    score_limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> CalibrationResult:
    """Full calibration for one backend: split, score, fit T, compute metrics, persist JSON."""
    python_rng_seed(seed)
    split = split_phrasebank(texts, labels, seed, max_rows=max_rows, holdout=holdout)
    eval_texts = [split["texts"][i] for i in split["test_idx"]]
    eval_labels = split["labels"][split["test_idx"]]

    limit = score_limit
    if progress is not None:
        progress(0, len(eval_texts))
    sentiments, elapsed, errors = _score_all(backend, eval_texts, batch_size, limit)

    valid = [(s, int(y)) for s, y in zip(sentiments, eval_labels, strict=False) if s is not None]
    if not valid:
        raise RuntimeError(f"backend {backend.name} returned no valid rows")
    sents = [s for s, _ in valid]
    y = np.asarray([yy for _, yy in valid])

    logits = _logits_from_p(sents)
    # Fit T on the first holdout fraction of the evaluated slice; report on the remainder.
    n_fit = max(1, int(round(len(y) * holdout)))
    if n_fit >= len(y):  # tiny slice: fit and evaluate on the same rows (recorded as n_train)
        n_fit = len(y) // 2 or 1
    t = fit_temperature(logits[:n_fit], y[:n_fit])
    probs_before = apply_temperature(logits, 1.0)
    probs_after = apply_temperature(logits, t)

    eval_slice = slice(n_fit, None) if n_fit < len(y) else slice(0, len(y))
    y_eval = y[eval_slice]
    p0 = probs_before[eval_slice]
    p1 = probs_after[eval_slice]

    pred = p1.argmax(axis=1)
    macro_f1 = float(f1_score(y_eval, pred, average="macro", labels=[0, 1, 2], zero_division=0))
    per_class = f1_score(y_eval, pred, average=None, labels=[0, 1, 2], zero_division=0)
    ece_before, _ = expected_calibration_error(p0, y_eval)
    ece_after, bins = expected_calibration_error(p1, y_eval)
    onehot = np.eye(3)[y_eval]
    brier = float(((p1 - onehot) ** 2).sum(axis=1).mean())

    label5_pred = [s.label_5 for s in sents[eval_slice]]
    label5_gold = [LABEL_3_TO_LEVEL5[LABELS_3[i]] for i in y_eval]
    label5_acc = float(np.mean([p == g for p, g in zip(label5_pred, label5_gold, strict=False)]))

    result = CalibrationResult(
        backend=backend.name,
        model_version=backend.model_version,
        temperature=t,
        n_train_slice=n_fit,
        n_eval=int(len(y_eval)),
        accuracy=float(accuracy_score(y_eval, pred)),
        macro_f1=macro_f1,
        per_class_f1={LABELS_3[i]: float(per_class[i]) for i in range(3)},
        per_class_support={LABELS_3[i]: int((y_eval == i).sum()) for i in range(3)},
        ece=ece_after,
        ece_before=ece_before,
        brier=brier,
        throughput=float(len(sents) / elapsed) if elapsed > 0 else float("inf"),
        n_errors=errors,
        rows_evaluated=len(sents),
        rows_before_cap=int(split["rows_before_cap"]),
        seed=seed,
        label_5_accuracy=label5_acc,
        ece_bins=bins,
        elapsed_sec=float(elapsed),
    )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = result.to_json()
        (out_dir / f"{backend.name}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    _LOG.info(
        "%s: acc=%.4f macroF1=%.4f ECE %.4f->%.4f T=%.3f %.1f items/s",
        backend.name,
        result.accuracy,
        result.macro_f1,
        result.ece_before,
        result.ece,
        t,
        result.throughput,
    )
    return result


def eval_frame(results: Iterable[CalibrationResult]) -> list[dict[str, Any]]:
    """Compact dict view used by the summary renderer and the CLI."""
    return [{k: v for k, v in r.to_json().items() if k != "ece_bins"} for r in results]
