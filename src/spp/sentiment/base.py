"""The shared ``ArticleSentiment`` contract and helpers every backend must satisfy.

A backend turns one headline into one :class:`ArticleSentiment`.  The contract is deliberately
narrow so that FinBERT (3 softmax classes), open-jev (5 ordered levels) and a lexicon (a signed
score) can all be projected onto the same 5-level representation.

Invariants (enforced in :meth:`ArticleSentiment.__post_init__`):

* ``p_neg + p_neu + p_pos == 1`` within ``1e-6``;
* ``0 <= p_* <= 1``;
* ``-1 <= score_5 <= 1`` and ``score_5 == 2 * E[level] / 4 - 1``;
* ``score_3 == p_pos - p_neg``;
* ``confidence == max(p_*)``.

Levels are the five ordered labels ``very negative .. very positive`` mapped to ``0..4``; the
midpoint of each bucket is used to project a coarse 3-class distribution onto the 5-level axis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol

__all__ = [
    "LEVEL_LABELS",
    "LEVEL_MIDPOINTS",
    "PROB_SUM_TOL",
    "ArticleSentiment",
    "BackendError",
    "ModelUnavailableError",
    "SentimentBackend",
    "levels_to_probs",
    "score5_from_levels",
    "score5_from_probs",
]

#: Ordered bucket labels for the 5-level axis (index == level).
LEVEL_LABELS: tuple[str, ...] = (
    "very negative",
    "negative",
    "neutral",
    "positive",
    "very positive",
)
NUM_LEVELS = len(LEVEL_LABELS)

#: Expected level of a sentence known only to be positive/neutral/negative.  Used to lift a
#: 3-class distribution onto the 5-level axis without inventing information about intensity.
LEVEL_MIDPOINTS: tuple[float, float, float] = (0.5, 2.0, 3.5)  # very neg/neg, neutral, pos/very pos

PROB_SUM_TOL = 1e-6


class BackendError(RuntimeError):
    """Raised when a backend cannot honour the contract (never a silent fallback)."""


class ModelUnavailableError(BackendError):
    """A model could not be loaded (missing network, HF token, or local cache).

    Raised instead of silently degrading to another backend, so a downstream metric can never be
    computed from an unintended model.
    """


@dataclass(frozen=True)
class ArticleSentiment:
    """One headline's sentiment, independent of which backend produced it."""

    p_neg: float
    p_neu: float
    p_pos: float
    score_5: float
    score_3: float
    confidence: float
    label_5: str
    backend: str
    model_version: str
    is_company_specific: float | None = None
    is_novel: float | None = None

    NUM_LEVELS: ClassVar[int] = NUM_LEVELS

    def __post_init__(self) -> None:
        probs = (self.p_neg, self.p_neu, self.p_pos)
        for name, value in zip(("p_neg", "p_neu", "p_pos"), probs, strict=False):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite probability in [0, 1], got {value!r}")
        total = sum(probs)
        if abs(total - 1.0) > PROB_SUM_TOL:
            raise ValueError(f"probabilities must sum to 1, got {total!r}")
        if not math.isfinite(self.score_5) or not -1.0 <= self.score_5 <= 1.0:
            raise ValueError(f"score_5 must be in [-1, 1], got {self.score_5!r}")
        expected_3 = self.p_pos - self.p_neg
        if abs(self.score_3 - expected_3) > 1e-9:
            raise ValueError(
                f"score_3 must equal p_pos - p_neg ({expected_3!r}), got {self.score_3!r}"
            )
        if abs(self.confidence - max(probs)) > 1e-9:
            raise ValueError(
                f"confidence must equal max(p) ({max(probs)!r}), got {self.confidence!r}"
            )
        if self.label_5 not in LEVEL_LABELS:
            raise ValueError(f"label_5 must be one of {LEVEL_LABELS}, got {self.label_5!r}")
        optional_fields = (
            ("is_company_specific", self.is_company_specific),
            ("is_novel", self.is_novel),
        )
        for name, value in optional_fields:
            if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be None or a probability in [0, 1], got {value!r}")
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if not self.model_version:
            raise ValueError("model_version must be non-empty")

    def as_row(self) -> dict[str, object]:
        """Flat mapping matching the ``article_sentiment.parquet`` schema (no article metadata)."""
        return {
            "backend": self.backend,
            "model_version": self.model_version,
            "p_neg": self.p_neg,
            "p_neu": self.p_neu,
            "p_pos": self.p_pos,
            "score_5": self.score_5,
            "score_3": self.score_3,
            "confidence": self.confidence,
            "label_5": self.label_5,
            "is_company_specific": self.is_company_specific,
            "is_novel": self.is_novel,
        }


def score5_from_levels(level_probs: Sequence[float]) -> float:
    """``2 * E[level] / (L - 1) - 1`` directly from a full 5-level distribution.

    This is the faithful path for a backend that predicts 5 ordered levels (open-jev).  Using the
    collapsed 3-class probabilities instead would cap the score at ±0.75 and erase intensity.
    """
    if len(level_probs) != NUM_LEVELS:
        raise ValueError(f"expected {NUM_LEVELS} level probabilities, got {len(level_probs)}")
    expected_level = sum(i * p for i, p in enumerate(level_probs))
    return 2.0 * expected_level / (NUM_LEVELS - 1) - 1.0


def score5_from_probs(p_neg: float, p_neu: float, p_pos: float) -> float:
    """``2 * E[level] / (L - 1) - 1`` with the expected level taken at bucket midpoints."""
    expected_level = (
        LEVEL_MIDPOINTS[0] * p_neg + LEVEL_MIDPOINTS[1] * p_neu + LEVEL_MIDPOINTS[2] * p_pos
    )
    return 2.0 * expected_level / (NUM_LEVELS - 1) - 1.0


def label5_from_score(score_5: float) -> str:
    """Bucket a continuous score into the 5 labels using the plan's thresholds."""
    if score_5 < -0.4:
        return LEVEL_LABELS[0]
    if score_5 < -0.1:
        return LEVEL_LABELS[1]
    if score_5 <= 0.1:
        return LEVEL_LABELS[2]
    if score_5 <= 0.4:
        return LEVEL_LABELS[3]
    return LEVEL_LABELS[4]


def levels_to_probs(level_probs: Sequence[float]) -> tuple[float, float, float]:
    """Collapse a 5-level distribution into ``(p_neg, p_neu, p_pos)``.

    ``p_pos = P(level >= 3)``, ``p_neg = P(level <= 1)``, ``p_neu`` is the remainder.
    """
    if len(level_probs) != NUM_LEVELS:
        raise ValueError(f"expected {NUM_LEVELS} level probabilities, got {len(level_probs)}")
    p_neg = float(level_probs[0] + level_probs[1])
    p_neu = float(level_probs[2])
    p_pos = float(level_probs[3] + level_probs[4])
    total = p_neg + p_neu + p_pos
    if abs(total - 1.0) > 1e-9:  # renormalise to absorb float drift
        p_neg, p_neu, p_pos = p_neg / total, p_neu / total, p_pos / total
    return p_neg, p_neu, p_pos


def build_article_sentiment(
    *,
    p_neg: float,
    p_neu: float,
    p_pos: float,
    label_5: str,
    backend: str,
    model_version: str,
    score_5: float | None = None,
    is_company_specific: float | None = None,
    is_novel: float | None = None,
) -> ArticleSentiment:
    """Assemble the derived fields consistently.

    ``score_5`` is derived from the 3-class probabilities by default (FinBERT, lexicon).  A
    backend that predicts full 5-level distributions (open-jev) passes it explicitly so the
    intensity information is not lost in the 5→3 collapse.
    """
    if score_5 is None:
        score_5 = score5_from_probs(p_neg, p_neu, p_pos)
    return ArticleSentiment(
        p_neg=p_neg,
        p_neu=p_neu,
        p_pos=p_pos,
        score_5=score_5,
        score_3=p_pos - p_neg,
        confidence=max(p_neg, p_neu, p_pos),
        label_5=label_5,
        backend=backend,
        model_version=model_version,
        is_company_specific=is_company_specific,
        is_novel=is_novel,
    )


class SentimentBackend(Protocol):
    """Structural protocol implemented by every backend (FinBERT, open-jev, lexicon)."""

    name: str
    model_version: str

    def score(self, texts: Sequence[str]) -> list[ArticleSentiment]:
        """Score a batch of headlines; return one :class:`ArticleSentiment` per input."""
        ...


def ensure_text(text: str) -> str:
    """Reject empty/degenerate input loudly instead of propagating NaN downstream."""
    if not isinstance(text, str):
        raise ValueError(f"headline must be a string, got {type(text).__name__}")
    stripped = text.strip()
    if not stripped:
        raise ValueError("headline is empty")
    return stripped
