"""Lexicon backend: ``pysentiment2.HIV4`` over the headline.

This exists as a cheap *control*: if a transformer cannot beat a dictionary on the same labels,
the transformer is not adding sentiment signal.  It is deliberately simple and deterministic.

Mapping (per plan, with two corrections):
* the raw evidence is the HIV-4 positive/negative term counts ``(pos, neg)``;
* ``p_*`` from a softmax over logits ``[neg, 1, pos]`` at temperature 1, where the neutral logit
  ``1`` is the same one-pseudo-observation of *neutral* evidence that the ``+ 1`` in the plan's
  ``score_3 = (pos - neg) / (pos + neg + 1)`` already implies.  The plan wrote ``[neg, 0, pos]``;
  with a neutral logit of exactly ``0`` the neutral class is unreachable (any matched term has a
  count >= 1 > 0), every headline collapses into a polarity, and a zero-evidence row becomes a
  three-way tie that ``argmax`` resolves to index 0, i.e. "negative".  That is the defect this
  constant fixes;
* ``score_3 = p_pos - p_neg``, ``score_5`` from the bucket midpoints, ``label_5`` a bucket of that
  ``score_5`` — the ``ArticleSentiment`` contract, so the three fields always agree;
* ``confidence = 1 - 1 / (1 + pos + neg)`` (a *lexicon* confidence = how much evidence was found).

Contract note: ``ArticleSentiment.confidence`` is defined as ``max(p_*)``.  The lexicon's own
evidence-based confidence is quantity of matched terms, a different concept, so it is not stored
in that field; it is derivable from the row via ``-1/(confidence-1) - 1`` (inverse of the formula).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from spp.sentiment.base import (
    ArticleSentiment,
    ModelUnavailableError,
    build_article_sentiment,
    ensure_text,
    label5_from_score,
    score5_from_probs,
)
from spp.sentiment.hf_env import stable_hash

__all__ = ["LexiconBackend"]


class LexiconBackend:
    """Dictionary sentiment (Harvard IV-4 via pysentiment2)."""

    name = "lexicon"

    #: Neutral logit = one pseudo-observation of neutral evidence, matching the ``+ 1`` smoothing
    #: in ``score_3 = (pos - neg) / (pos + neg + 1)``.  A headline needs at least two net matched
    #: terms before the dictionary overrules the "nothing found" verdict.
    NEUTRAL_LOGIT = 1.0

    def __init__(self, dictionary: str = "HIV4") -> None:
        self.dictionary = dictionary.upper()
        self.model_version = (
            f"pysentiment2:{self.dictionary}+cfg{stable_hash({'dict': self.dictionary})[:12]}"
        )
        self._lex = None

    def _load(self) -> None:
        if self._lex is not None:
            return
        try:
            import pysentiment2 as ps
        except ImportError as exc:  # pragma: no cover - dependency guarded by pyproject
            raise ModelUnavailableError(
                "pysentiment2 not installed; run `pip install -e .`"
            ) from exc
        if not hasattr(ps, self.dictionary):
            raise ModelUnavailableError(f"pysentiment2 has no dictionary {self.dictionary!r}")
        lex = getattr(ps, self.dictionary)()
        lex.init_dict()
        self._lex = lex

    def score(self, texts: Sequence[str]) -> list[ArticleSentiment]:
        cleaned = [ensure_text(t) for t in texts]
        if not cleaned:
            return []
        self._load()
        assert self._lex is not None
        out: list[ArticleSentiment] = []
        for text in cleaned:
            scored = self._lex.get_score(self._lex.tokenize(text))
            pos = float(scored["Positive"])
            neg = float(scored["Negative"])
            p_neg, p_neu, p_pos = self._softmax(neg, pos)
            # ``label_5`` must bucket the row's own ``score_5`` (base.py contract), not the raw
            # count ratio: ``build_article_sentiment`` derives ``score_5`` from these ``p_*``.
            score_5 = score5_from_probs(p_neg, p_neu, p_pos)
            out.append(
                build_article_sentiment(
                    p_neg=p_neg,
                    p_neu=p_neu,
                    p_pos=p_pos,
                    label_5=label5_from_score(score_5),
                    backend=self.name,
                    model_version=self.model_version,
                )
            )
        return out

    @classmethod
    def _softmax(cls, neg: float, pos: float) -> tuple[float, float, float]:
        """Softmax over ``[neg, NEUTRAL_LOGIT, pos]`` at temperature 1.

        The neutral logit is the no-evidence baseline, so a headline whose matched terms cancel
        out (or that contains no dictionary terms at all) is predicted *neutral* instead of the
        three-way tie that a ``0`` neutral logit produces — a tie ``argmax`` always resolves to the
        first class, i.e. "negative".
        """
        logits = (neg, cls.NEUTRAL_LOGIT, pos)
        m = max(logits)
        exps = [math.exp(z - m) for z in logits]
        total = sum(exps)
        return (exps[0] / total, exps[1] / total, exps[2] / total)

    @staticmethod
    def lexicon_confidence(pos: float, neg: float) -> float:
        """Plan's lexicon confidence: ``1 - 1/(1 + pos + neg)``."""
        return 1.0 - 1.0 / (1.0 + pos + neg)
