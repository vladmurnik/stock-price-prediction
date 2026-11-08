"""FinBERT backend: ``ProsusAI/finbert`` 3-class sentence classifier.

Label order is read from ``config.id2label`` at load time — never hardcoded — because published
checkpoints do not agree on whether index 0 is *positive* or *negative*.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from spp.logging import get_logger
from spp.sentiment.base import (
    ArticleSentiment,
    ModelUnavailableError,
    build_article_sentiment,
    ensure_text,
    label5_from_score,
)
from spp.sentiment.hf_env import configure_hf_cache, resolve_model_version

__all__ = ["FinBertBackend"]

_LOG = get_logger("sentiment.finbert")


def _physical_cores() -> int:
    """Number of physical cores, used to pick a sensible torch thread count."""
    try:
        import psutil  # type: ignore[import-not-found]

        return psutil.cpu_count(logical=False) or max(1, (os.cpu_count() or 2) // 2)
    except Exception:  # noqa: BLE001 - psutil is optional
        return max(1, (os.cpu_count() or 2) // 2)


class FinBertBackend:
    """``score_3 = p_pos - p_neg``; ``score_5`` derived from the same distribution."""

    name = "finbert"

    def __init__(
        self,
        model_id: str = "ProsusAI/finbert",
        max_length: int = 128,
        batch_size: int = 32,
        num_threads: int | None = None,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.num_threads = num_threads if num_threads is not None else max(1, _physical_cores() - 1)
        self._device = device
        self.model_version = resolve_model_version(
            model_id,
            {
                "kind": "finbert",
                "max_length": self.max_length,
                "id2label": "from-config",
            },
        )
        self._tok = None
        self._model = None
        self._index: dict[str, int] = {}
        self._torch = None
        self._truncated = 0

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        if self._model is not None:
            return
        configure_hf_cache()
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency guarded by pyproject
            raise ModelUnavailableError(
                "transformers/torch not installed; run `pip install -e .` in the project venv"
            ) from exc

        torch.set_num_threads(self.num_threads)
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            tok = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        except Exception as exc:  # noqa: BLE001 - report, never fall back
            raise ModelUnavailableError(
                f"cannot load FinBERT ({self.model_id}): {exc}. "
                "Check network access to huggingface.co, set HF_TOKEN if the repo is gated, "
                "or pre-download with `huggingface-cli download ProsusAI/finbert`."
            ) from exc

        model.to(device).eval()
        # Map model label names -> our canonical index order (neg/neu/pos).
        canonical = {"negative": "neg", "neutral": "neu", "positive": "pos"}
        index: dict[str, int] = {}
        for idx, label in model.config.id2label.items():
            key = canonical.get(str(label).strip().lower())
            if key is None:
                raise ModelUnavailableError(
                    f"unexpected FinBERT label {label!r} at index {idx}; "
                    f"id2label={model.config.id2label}"
                )
            index[key] = int(idx)
        if set(index) != {"neg", "neu", "pos"}:
            raise ModelUnavailableError(f"FinBERT id2label is incomplete: {model.config.id2label}")

        self._tok, self._model, self._index, self._torch = tok, model, index, torch
        self._device = device
        _LOG.info(
            "FinBERT loaded on %s (threads=%d, id2label=%s)",
            device,
            self.num_threads,
            model.config.id2label,
        )

    @property
    def truncated_count(self) -> int:
        """How many headlines were cut to ``max_length`` tokens (logged, never dropped)."""
        return self._truncated

    # ------------------------------------------------------------------ scoring
    def score(self, texts: Sequence[str]) -> list[ArticleSentiment]:
        cleaned = [ensure_text(t) for t in texts]
        if not cleaned:
            return []
        self._load()
        assert self._tok is not None and self._model is not None and self._torch is not None
        torch = self._torch
        out: list[ArticleSentiment] = []
        for start in range(0, len(cleaned), self.batch_size):
            chunk = cleaned[start : start + self.batch_size]
            enc = self._tok(
                chunk,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            n_tokens = enc["input_ids"].shape[1]
            if n_tokens >= self.max_length:
                self._truncated += int((enc["attention_mask"].sum(dim=1) >= self.max_length).sum())
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.inference_mode():
                logits = self._model(**enc).logits
                probs = logits.softmax(dim=-1).cpu()
            for row in probs:
                p_neg = float(row[self._index["neg"]])
                p_neu = float(row[self._index["neu"]])
                p_pos = float(row[self._index["pos"]])
                out.append(self._compose(p_neg, p_neu, p_pos))
        return out

    def _compose(self, p_neg: float, p_neu: float, p_pos: float) -> ArticleSentiment:
        score_3 = p_pos - p_neg  # plan: FinBERT's score_5 == score_3 on this axis
        return build_article_sentiment(
            p_neg=p_neg,
            p_neu=p_neu,
            p_pos=p_pos,
            label_5=label5_from_score(score_3),
            backend=self.name,
            model_version=self.model_version,
        )
