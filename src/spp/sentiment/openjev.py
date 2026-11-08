"""open-jev backend: ``com-kotobalabs/open-jev-deberta-v3-large`` typed decisions.

open-jev is a *general* typed-decision model (banking77 + SST-5 + BoolQ), not a finance model.
We submit one state (the headline) with three questions in a single forward pass:

* ``score``  — sentiment on 5 ordered levels ``very negative .. very positive``;
* ``choice`` — what the news concerns (``company-specific`` / ``sector-wide`` / ``market-wide``);
* ``noul``   — "is this a materially new event".

The 5-level distribution gives ``p_*`` directly:
``p_pos = P(level >= 3)``, ``p_neg = P(level <= 1)``, ``p_neu`` is the remainder.

Measured macro-F1 on the financial phrasebank is reported as-is, including when it trails
FinBERT.  The backend is compute-bound on CPU, so ``score`` batches states through the same
collator: one forward pass per ``batch_size`` headlines instead of one per headline.  The
batched path reproduces ``OpenJev.decide`` element-wise; ``tests/test_openjev_batching.py``
pins that equivalence.
"""

from __future__ import annotations

from collections.abc import Sequence

from spp.logging import get_logger
from spp.sentiment.base import (
    ArticleSentiment,
    ModelUnavailableError,
    build_article_sentiment,
    ensure_text,
    label5_from_score,
    levels_to_probs,
    score5_from_levels,
)
from spp.sentiment.hf_env import configure_hf_cache, resolve_model_version

__all__ = ["OPENJEV_BATCH_SIZE", "OPENJEV_QUESTIONS", "OpenJevBackend"]

_LOG = get_logger("sentiment.openjev")

#: Headlines per forward pass.  Eight keeps the activations well inside the ~5 GB of RAM
#: that is actually free on the target box while still cutting the per-item cost by ~3-5x.
OPENJEV_BATCH_SIZE = 8

OPENJEV_QUESTIONS: list[dict] = [
    {
        "type": "score",
        "instructions": (
            "How positive is the sentiment of this headline for the company's stock price?"
        ),
        "options": ["very negative", "negative", "neutral", "positive", "very positive"],
    },
    {
        "type": "choice",
        "instructions": "What does this news concern?",
        "options": ["company-specific", "sector-wide", "market-wide"],
    },
    {
        "type": "noul",
        "instructions": "This headline reports a materially new event for the company.",
    },
]

_INSTALL_HINT = (
    "open-jev is unavailable. To enable it: `pip install git+https://github.com/kotoba-lang/"
    "typed-decisions` (it pins transformers<5) and pre-download the weights with "
    "`huggingface-cli download com-kotobalabs/open-jev-deberta-v3-large`. "
    "No other backend is substituted silently."
)


class OpenJevBackend:
    """Batched forward passes over headlines, three typed questions per state."""

    name = "openjev"

    def __init__(
        self,
        model_id: str = "com-kotobalabs/open-jev-deberta-v3-large",
        device: str | None = None,
        questions: list[dict] | None = None,
        batch_size: int = OPENJEV_BATCH_SIZE,
    ) -> None:
        self.model_id = model_id
        self.questions = questions or OPENJEV_QUESTIONS
        self._device = device
        self.batch_size = max(1, int(batch_size))
        self.model_version = resolve_model_version(
            model_id,
            {"kind": "openjev", "questions": self.questions, "max_state_tokens": 256},
        )
        self._model = None
        self.truncated = 0

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        if self._model is not None:
            return
        configure_hf_cache()
        try:
            from typed_decisions.open_jev import OpenJev
        except ImportError as exc:
            raise ModelUnavailableError(_INSTALL_HINT) from exc
        try:
            self._model = OpenJev.from_pretrained(self.model_id, device=self._device)
        except Exception as exc:  # noqa: BLE001 - explicit error, never a silent fallback
            raise ModelUnavailableError(
                f"cannot load open-jev ({self.model_id}): {exc}. {_INSTALL_HINT}"
            ) from exc
        _LOG.info("open-jev loaded (temperature=%.3f)", float(self._model.model.temperature))

    # ------------------------------------------------------------------ scoring
    def score(self, texts: Sequence[str]) -> list[ArticleSentiment]:
        cleaned = [ensure_text(t) for t in texts]
        if not cleaned:
            return []
        self._load()
        assert self._model is not None
        size = self.batch_size
        if size <= 1:
            return [self._compose(self._model.decide(text, self.questions)) for text in cleaned]
        out: list[ArticleSentiment] = []
        for start in range(0, len(cleaned), size):
            out.extend(self._score_batch(cleaned[start : start + size]))
        return out

    def _score_batch(self, texts: Sequence[str]) -> list[ArticleSentiment]:
        """One forward pass for ``len(texts)`` states, mirroring ``OpenJev.decide``.

        ``OpenJev.decide_batch`` is a plain Python loop, so the batching has to happen here:
        the library collator already accepts a list of ``(state, questions)`` pairs and pads
        them, and the head emits ``[batch, n_questions, n_options]`` logits.
        """
        import torch
        from typed_decisions.schema import readout

        model = self._model
        assert model is not None
        questions = [model._question(i, q) for i, q in enumerate(self.questions)]
        collated = model.collator(
            [(text, questions) for text in texts], model.device
        )
        with torch.no_grad():
            logits = model.model(
                collated["input_ids"],
                collated["attention_mask"],
                collated["opt_pos"],
                collated["opt_mask"],
                collated["q_pos"],
                collated["seg"],
            ).float()
            probs = (logits / model.model.temperature).softmax(-1)

        out: list[ArticleSentiment] = []
        for row in range(len(texts)):
            decisions: list[dict] = []
            for qi, question in enumerate(questions):
                values = probs[row, qi, : len(question.options)].tolist()
                read = readout(question.kind, values)
                keyed = dict(zip(question.options, values))
                if question.kind == "choice":
                    decisions.append(
                        {
                            "choice": question.options[read["choice"]],
                            "probabilities": keyed,
                            "confidence": read["confidence"],
                        }
                    )
                elif question.kind == "score":
                    decisions.append(
                        {
                            "score": read["score"],
                            "probabilities": keyed,
                            "confidence": read["confidence"],
                        }
                    )
                else:
                    decisions.append({"noul": read["noul"]})
            out.append(self._compose(decisions))
        return out

    def _compose(self, decisions: list[dict]) -> ArticleSentiment:
        score_d, choice_d, noul_d = decisions[0], decisions[1], decisions[2]
        options = self.questions[0]["options"]
        # ``probabilities`` is keyed by option string; re-order to the declared level order.
        level_probs = [float(score_d["probabilities"].get(opt, 0.0)) for opt in options]
        total = sum(level_probs) or 1.0
        level_probs = [p / total for p in level_probs]
        p_neg, p_neu, p_pos = levels_to_probs(level_probs)
        score_5 = score5_from_levels(level_probs)
        company = choice_d["probabilities"].get("company-specific")
        return build_article_sentiment(
            p_neg=p_neg,
            p_neu=p_neu,
            p_pos=p_pos,
            score_5=score_5,
            label_5=label5_from_score(score_5),
            backend=self.name,
            model_version=self.model_version,
            is_company_specific=None if company is None else float(company),
            is_novel=None if noul_d.get("noul") is None else float(noul_d["noul"]),
        )
