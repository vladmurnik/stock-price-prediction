"""Sharded, resumable corpus scoring (T5).

Design (see ``05-corpus-scoring.md``):

* **Deduplicate first.**  One ``text_hash`` can appear under several tickers (syndication).
  Each *unique* hash is scored exactly once and fanned out to every source row on assembly,
  so 124k rows become ~93k model calls.
* **Sharding.**  ``shard_id = int(sha1(text_hash)[:8], 16) % 256`` into
  ``data/cache/sentiment/{backend}/shard_{id:03d}.parquet``.  A shard is computed whole and
  written atomically (``.tmp`` -> ``os.replace``), so an interrupted run never leaves a
  half-written shard that a later resume would trust.
* **Cache stamp.**  Each shard carries a sidecar JSON with ``model_version`` and ``temperature``.
  A shard is reused only when both match; otherwise it is recomputed and *left on disk* under
  the same name (never deleted).  Changing an open-jev question therefore invalidates shards
  instead of silently mixing model generations.
* **Filters after scoring.**  Empty text, non-English and low-confidence rows are marked
  ``skipped=True`` with a reason and kept.  Dropping them is the feature layer's decision.
* **Determinism.**  Rows are sorted by ``article_id`` and all arithmetic is deterministic, so a
  rerun with the same config produces a byte-identical artifact.

open-jev is compute-bound (~1.8 s/state on CPU), so it only runs on a seeded stratified sample
(all 2023 articles plus a proportional draw per (ticker, year)); the sampled ids are persisted
to ``reports/openjev_sample.parquet`` so the selection is reproducible.
"""

from __future__ import annotations

import hashlib
import json
import math
import signal
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spp.data.store import read_parquet, write_manifest, write_parquet_atomic
from spp.logging import get_logger
from spp.sentiment.base import (
    LEVEL_LABELS,
    ArticleSentiment,
    BackendError,
    label5_from_score,
    score5_from_probs,
)

__all__ = [
    "BACKEND_CHOICES",
    "LOW_CONFIDENCE",
    "MIN_ASCII_SHARE",
    "NUM_SHARDS",
    "OUTPUT_COLUMNS",
    "ScoreSummary",
    "apply_temperature_to_probs",
    "assemble_article_sentiment",
    "build_backend",
    "build_openjev_sample",
    "ascii_share",
    "load_news",
    "load_temperature",
    "score_corpus",
    "shard_id_for",
    "shard_path",
    "skip_reason_for",
    "validate_output",
    "write_article_sentiment",
]

_LOG = get_logger("sentiment.score_corpus")

#: Shard files per backend: ``shard_000.parquet`` .. ``shard_255.parquet``.
NUM_SHARDS = 256

BACKEND_CHOICES: tuple[str, ...] = ("finbert", "openjev", "lexicon")

#: ``article_sentiment.parquet`` columns, in plan order.
OUTPUT_COLUMNS: tuple[str, ...] = (
    "article_id",
    "ticker",
    "signal_date",
    "text_hash",
    "backend",
    "model_version",
    "p_neg",
    "p_neu",
    "p_pos",
    "score_5",
    "score_3",
    "confidence",
    "label_5",
    "is_company_specific",
    "is_novel",
    "skipped",
    "skip_reason",
)

#: Sentiment columns carried by a shard; the rest come from ``news_raw`` at assembly time.
SENTIMENT_COLUMNS: tuple[str, ...] = tuple(
    c for c in OUTPUT_COLUMNS if c not in ("article_id", "ticker", "signal_date")
)

#: Rows scoring below this confidence are kept but flagged ``low_confidence``.
LOW_CONFIDENCE = 0.4
#: Headlines with a smaller ASCII share are treated as non-English (Cyrillic/CJK regimes).
MIN_ASCII_SHARE = 0.85
#: A progress line is emitted every this many texts scored in the current run.
PROGRESS_EVERY = 10_000


# --------------------------------------------------------------------------- pure helpers
def shard_id_for(text_hash: str) -> int:
    """Stable shard assignment: ``int(sha1(text_hash)[:8], 16) % NUM_SHARDS``."""
    digest = hashlib.sha1(text_hash.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % NUM_SHARDS


def shard_path(cache_dir: Path, backend: str, shard_id: int) -> Path:
    return Path(cache_dir) / backend / f"shard_{shard_id:03d}.parquet"


def shard_stamp_path(path: Path) -> Path:
    return path.with_name(path.stem + ".stamp.json")


def ascii_share(text: str) -> float:
    """Share of ASCII characters in ``text``; empty strings count as 0."""
    if not text:
        return 0.0
    return sum(1 for ch in text if ch.isascii()) / len(text)


def skip_reason_for(text: str, confidence: float) -> str | None:
    """First failing post-scoring filter, or ``None`` when the row is usable.

    Order matters: an unencodable headline is ``empty``, a foreign-language headline is
    ``non_english``, and only then do we consider the model's own uncertainty.
    """
    if not isinstance(text, str) or not text.strip():
        return "empty"
    if ascii_share(text) < MIN_ASCII_SHARE:
        return "non_english"
    if math.isfinite(confidence) and confidence < LOW_CONFIDENCE:
        return "low_confidence"
    return None


def apply_temperature_to_probs(
    p_neg: float, p_neu: float, p_pos: float, temperature: float
) -> tuple[float, float, float]:
    """Temperature-scale a 3-class distribution (``T`` > 1 softens, ``T`` < 1 sharpens)."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    probs = np.clip(np.asarray([p_neg, p_neu, p_pos], dtype=np.float64), 1e-12, 1.0)
    logits = np.log(probs) / float(temperature)
    logits -= logits.max()
    scaled = np.exp(logits)
    scaled /= scaled.sum()
    return float(scaled[0]), float(scaled[1]), float(scaled[2])


# --------------------------------------------------------------------------- input loading
def load_news(news_path: Path) -> pd.DataFrame:
    """Load ``news_raw.parquet`` and keep only what scoring needs.

    ``article_id`` is **not** unique in the raw corpus (a story can be republished under
    several ids), so uniqueness is never asserted here; the row granularity of the raw file is
    preserved end to end.
    """
    frame = read_parquet(news_path)
    required = ("article_id", "ticker", "signal_date", "title", "text_hash")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{news_path} is missing columns {missing}")
    frame = frame.loc[:, list(required)].copy()
    frame["title"] = frame["title"].astype("string").fillna("").astype(str)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.normalize()
    frame["article_id"] = frame["article_id"].astype(str)
    frame["text_hash"] = frame["text_hash"].astype(str)
    return frame.sort_values("article_id", kind="stable").reset_index(drop=True)


def load_temperature(reports_dir: Path, backend: str) -> tuple[float, str | None]:
    """Read ``T`` (and the calibration model_version) from ``reports/calibration/<backend>.json``.

    A missing artifact is fatal for a neural backend: scoring the corpus with an uncalibrated
    temperature would quietly change the confidence the feature layer consumes.  The lexicon
    backend has no reliability to calibrate, so ``T = 1.0``.
    """
    path = Path(reports_dir) / "calibration" / f"{backend}.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return float(payload["temperature"]), payload.get("model_version")
    if backend == "lexicon":
        return 1.0, None
    raise BackendError(
        f"no calibration artifact for {backend!r} at {path}. Run "
        f"`python -m spp.cli score calibrate --backend {backend}` first; corpus scores are "
        "required to carry the calibrated temperature."
    )


# --------------------------------------------------------------------------- backend factory
def build_backend(
    name: str,
    *,
    config: Any,
    device: str | None = None,
) -> Any:
    """Construct the named backend from ``config.sentiment``.  No silent substitution."""
    sent = config.sentiment
    if name == "finbert":
        from spp.sentiment.finbert import FinBertBackend

        return FinBertBackend(
            model_id=sent.finbert_model,
            max_length=sent.max_length,
            batch_size=sent.batch_size,
            device=device,
        )
    if name == "openjev":
        from spp.sentiment.openjev import OpenJevBackend

        return OpenJevBackend(model_id=sent.openjev_model, device=device)
    if name == "lexicon":
        from spp.sentiment.lexicon import LexiconBackend

        return LexiconBackend()
    raise ValueError(f"unknown backend {name!r}; choose from {BACKEND_CHOICES}")


# --------------------------------------------------------------------------- open-jev sample
def build_openjev_sample(
    news: pd.DataFrame,
    *,
    max_articles: int,
    seed: int,
    all_years: Sequence[int] = (2023,),
) -> pd.DataFrame:
    """Seeded (ticker, year) stratified sample; every row from ``all_years`` is included.

    The remaining budget is spread proportionally over the other (ticker, year) strata, so a
    thin stratum still contributes when the overall rate is small.  The output is a pure
    function of ``(news, max_articles, seed)``.
    """
    if max_articles <= 0:
        raise ValueError(f"max_articles must be positive, got {max_articles}")
    frame = news.assign(year=news["signal_date"].dt.year.astype(int))
    forced_mask = frame["year"].isin({int(y) for y in all_years})
    n_forced = int(forced_mask.sum())
    budget = max(0, int(max_articles) - n_forced)

    strata = [(k, g) for k, g in frame.loc[~forced_mask].groupby(["ticker", "year"], sort=True)]
    n_rest = sum(len(g) for _k, g in strata)
    if n_rest == 0 or budget == 0:
        quotas = {k: 0 for k, _g in strata}
    elif budget >= n_rest:
        quotas = {k: len(g) for k, g in strata}
    else:
        # Largest-remainder allocation: floor share, then hand out leftovers by remainder.
        exact = {k: budget * len(g) / n_rest for k, g in strata}
        quotas = {k: int(math.floor(v)) for k, v in exact.items()}
        leftover = budget - sum(quotas.values())
        order = sorted(exact, key=lambda k: (-(exact[k] - quotas[k]), str(k)))
        for key in order[:leftover]:
            quotas[key] += 1

    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for key, group in strata:
        n_take = quotas[key]
        if n_take <= 0:
            continue
        chosen = np.sort(rng.choice(len(group), size=n_take, replace=False))
        picked.extend(group.index[chosen].tolist())

    keep = sorted(set(picked) | set(frame.index[forced_mask]))
    sample = (
        frame.loc[keep, ["article_id", "ticker", "signal_date", "text_hash", "year"]]
        .rename(columns={"signal_date": "date"})
        .sort_values(["ticker", "year", "article_id"], kind="stable")
        .reset_index(drop=True)
    )
    return sample


def write_openjev_sample(sample: pd.DataFrame, path: Path, *, backend: str, seed: int) -> Path:
    """Persist the sample ids plus a manifest describing how they were drawn."""
    write_parquet_atomic(sample, path, date32=("date",))
    write_manifest(
        path.with_name(path.name + ".manifest.json"),
        {
            "kind": "openjev_sample",
            "backend": backend,
            "seed": seed,
            "rows": int(len(sample)),
            "per_year": {str(k): int(v) for k, v in sample["year"].value_counts().items()},
            "tickers": int(sample["ticker"].nunique()),
        },
    )
    return path


# --------------------------------------------------------------------------- shard io
def _read_stamp(path: Path) -> dict[str, Any]:
    stamp = shard_stamp_path(path)
    if not stamp.is_file():
        return {}
    try:
        return json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _stamp_matches(path: Path, model_version: str, temperature: float) -> bool:
    stamp = _read_stamp(path)
    try:
        stored_t = float(stamp["temperature"])
    except (KeyError, TypeError, ValueError):
        return False
    return stamp.get("model_version") == model_version and abs(stored_t - temperature) < 1e-9


def _shard_hashes(path: Path) -> set[str]:
    """Hashes already scored in ``path``.  A corrupt shard yields ``set()`` (never trusted)."""
    if not path.is_file():
        return set()
    try:
        frame = pd.read_parquet(path, columns=["text_hash"])
    except Exception:  # noqa: BLE001 - unreadable shard must be recomputed
        _LOG.warning("shard %s is unreadable; it will be recomputed", path)
        return set()
    return set(frame["text_hash"].astype(str).tolist())


def _write_shard(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    backend: str,
    model_version: str,
    temperature: float,
) -> None:
    """Persist ``rows`` into ``path``, merging with what is already there.

    A resumed run only rescores the *missing* hashes of a reused shard, so the existing rows
    must survive: overwriting the file here silently dropped every already-scored text, which
    is exactly how 1,800 hashes went missing.  New rows win on a hash collision.
    """
    frame = pd.DataFrame(rows)
    missing = [c for c in SENTIMENT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"shard rows are missing columns {missing}")
    frame = frame.loc[:, list(SENTIMENT_COLUMNS)]

    if path.is_file():
        try:
            previous = pd.read_parquet(path)
            if set(SENTIMENT_COLUMNS).issubset(previous.columns):
                frame = pd.concat(
                    [previous.loc[:, list(SENTIMENT_COLUMNS)], frame], ignore_index=True
                )
        except Exception:  # noqa: BLE001 - unreadable shard is rebuilt from the new rows
            pass

    frame = frame.drop_duplicates("text_hash", keep="last")
    frame = frame.sort_values("text_hash", kind="stable").reset_index(drop=True)
    write_parquet_atomic(frame, path)
    write_manifest(
        shard_stamp_path(path),
        {
            "backend": backend,
            "model_version": model_version,
            "temperature": float(temperature),
            "rows": int(len(frame)),
        },
    )


# --------------------------------------------------------------------------- summary
@dataclass
class ScoreSummary:
    """Counts produced by one :func:`score_corpus` call (also the progress artifact)."""

    backend: str
    model_version: str
    temperature: float
    unique_texts: int = 0
    scored_texts: int = 0
    reused_texts: int = 0
    skipped_by_reason: dict[str, int] = field(default_factory=dict)
    articles: int = 0
    shards_written: int = 0
    shards_reused: int = 0
    elapsed_sec: float = 0.0
    items_per_sec: float = 0.0
    interrupted: bool = False
    out_path: str | None = None
    sample_path: str | None = None

    def as_json(self) -> dict[str, Any]:
        skipped = {k: int(v) for k, v in sorted(self.skipped_by_reason.items())}
        return {
            "backend": self.backend,
            "model_version": self.model_version,
            "temperature": self.temperature,
            "unique_texts": self.unique_texts,
            "scored_texts": self.scored_texts,
            "reused_texts": self.reused_texts,
            "skipped_by_reason": skipped,
            "skipped_total": int(sum(skipped.values())),
            "articles": self.articles,
            "shards_written": self.shards_written,
            "shards_reused": self.shards_reused,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "items_per_sec": round(self.items_per_sec, 3),
            "interrupted": self.interrupted,
            "out_path": self.out_path,
            "sample_path": self.sample_path,
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }


class _InterruptGuard:
    """Turn the first Ctrl+C into a clean stop at a shard boundary."""

    def __init__(self) -> None:
        self.requested = False
        self._previous: Any = None

    def __enter__(self) -> _InterruptGuard:
        try:
            self._previous = signal.signal(signal.SIGINT, self._handle)
        except (ValueError, OSError):  # not on the main thread
            self._previous = None
        return self

    def __exit__(self, *exc: object) -> None:
        if self._previous is not None:
            try:
                signal.signal(signal.SIGINT, self._previous)
            except (ValueError, OSError):  # pragma: no cover
                pass

    def _handle(self, signum: int, frame: Any) -> None:  # noqa: ARG002
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        _LOG.warning("interrupt received: finishing the current shard, then stopping")


# --------------------------------------------------------------------------- scoring loop
def _score_texts(
    backend: Any,
    texts: Sequence[str],
    *,
    batch_size: int,
    on_progress: Callable[[int], None] | None = None,
) -> list[ArticleSentiment | None]:
    """Score ``texts`` in batches; blank headlines become ``None`` instead of being encoded."""
    results: list[ArticleSentiment | None] = [None] * len(texts)
    batch = max(1, int(batch_size))
    i = 0
    while i < len(texts):
        chunk = list(texts[i : i + batch])
        live = [j for j, t in enumerate(chunk) if isinstance(t, str) and t.strip()]
        if live:
            try:
                scored = backend.score([chunk[j] for j in live])
            except (MemoryError, RuntimeError) as exc:
                message = str(exc).lower()
                if "memory" in message and batch > 1:
                    batch = max(1, batch // 2)
                    _LOG.warning("out of memory at batch %d; halving to %d", batch * 2, batch)
                    continue
                if isinstance(exc, BackendError):
                    raise
                _LOG.warning("batch failed (%s); retrying items individually", exc)
                scored = []
                for j in live:
                    try:
                        scored.append(backend.score([chunk[j]])[0])
                    except (ValueError, RuntimeError):
                        scored.append(None)  # type: ignore[arg-type]
            if len(scored) != len(live):
                raise BackendError(
                    f"backend {backend.name} returned {len(scored)} results for {len(live)} inputs"
                )
            for pos, j in enumerate(live):
                results[i + j] = scored[pos]
        i += len(chunk)
        if on_progress is not None:
            on_progress(len(chunk))
    return results


def _calibrated_row(
    sent: ArticleSentiment | None,
    *,
    backend: str,
    model_version: str,
    temperature: float,
    text_hash: str,
) -> dict[str, Any]:
    """Build a shard row, applying the corpus temperature and the skip filters.

    ``T`` rescales ``p_*`` and confidence for every backend.  ``score_5``/``label_5`` are
    recomputed from the rescaled 3-class view only for backends whose ``score_5`` is defined
    that way (FinBERT, lexicon); open-jev's five-level ``score_5`` cannot be recovered from the
    collapsed distribution and is preserved verbatim (argmax is invariant to ``T``).
    """
    if sent is None:
        third = 1.0 / 3.0
        row: dict[str, Any] = {
            "text_hash": text_hash,
            "backend": backend,
            "model_version": model_version,
            "p_neg": third,
            "p_neu": third,
            "p_pos": third,
            "score_5": 0.0,
            "score_3": 0.0,
            "confidence": third,
            "label_5": LEVEL_LABELS[2],
            "is_company_specific": None,
            "is_novel": None,
        }
    else:
        row = sent.as_row()
        row["text_hash"] = text_hash
        if abs(temperature - 1.0) > 1e-12:
            p_neg, p_neu, p_pos = apply_temperature_to_probs(
                sent.p_neg, sent.p_neu, sent.p_pos, temperature
            )
            row.update(
                p_neg=p_neg,
                p_neu=p_neu,
                p_pos=p_pos,
                score_3=p_pos - p_neg,
                confidence=max(p_neg, p_neu, p_pos),
            )
            if backend != "openjev":
                score_5 = score5_from_probs(p_neg, p_neu, p_pos)
                row["score_5"] = score_5
                row["label_5"] = label5_from_score(score_5)
        row["backend"] = backend
        row["model_version"] = model_version
    return row


def score_corpus(
    news: pd.DataFrame,
    backend_name: str,
    *,
    cache_dir: Path,
    reports_dir: Path,
    temperature: float,
    batch_size: int = 32,
    shards: Sequence[int] | None = None,
    force: bool = False,
    out_path: Path | None = None,
    progress_every: int = PROGRESS_EVERY,
    sample_ids: set[str] | None = None,
    backend: Any | None = None,
    config: Any | None = None,
) -> ScoreSummary:
    """Score the corpus with per-shard resume.

    ``shards`` restricts work to a subset of shard ids (used for smoke runs and by tests).
    ``sample_ids`` restricts scoring to the given ``text_hash``-bearing row selection — callers
    use it for open-jev, where the whole corpus must never be scored.  ``backend`` lets callers
    inject an already-constructed backend (tests, CLI); otherwise one is built from ``config``.
    """
    started = time.perf_counter()
    if backend is None:
        if config is None:
            raise ValueError("score_corpus needs either backend=... or config=...")
        backend = build_backend(backend_name, config=config)
    model_version = str(backend.model_version)
    temperature = float(temperature)

    if backend_name not in BACKEND_CHOICES and backend_name != getattr(backend, "name", None):
        raise ValueError(f"unknown backend {backend_name!r}; choose from {BACKEND_CHOICES}")
    if backend_name == "openjev" and sample_ids is None:
        raise ValueError(
            "open-jev must never score the full corpus: pass sample_ids= from "
            "build_openjev_sample(...)"
        )

    work = news
    if sample_ids is not None:
        keep = news.loc[news["text_hash"].isin(sample_ids), :]
        if keep.empty:
            raise ValueError("sample_ids matched no corpus row")
        work = keep

    summary = ScoreSummary(
        backend=backend_name, model_version=model_version, temperature=temperature
    )

    # Dedupe: one representative title per hash (first by article_id, the frame is pre-sorted).
    uniques = (
        work.loc[:, ["text_hash", "title"]]
        .drop_duplicates("text_hash", keep="first")
        .sort_values("text_hash", kind="stable")
        .reset_index(drop=True)
    )
    summary.unique_texts = int(len(uniques))
    pairs = list(zip(uniques["text_hash"].tolist(), uniques["title"].tolist(), strict=False))

    wanted = set(range(NUM_SHARDS)) if shards is None else {int(s) for s in shards}
    out_of_range = sorted(wanted - set(range(NUM_SHARDS)))
    if out_of_range:
        raise ValueError(f"shard ids must be in 0..{NUM_SHARDS - 1}, got {out_of_range}")

    by_shard: dict[int, list[tuple[str, str]]] = {}
    for text_hash, title in pairs:
        sid = shard_id_for(text_hash)
        if sid in wanted:
            by_shard.setdefault(sid, []).append((text_hash, title))

    todo: dict[int, list[tuple[str, str]]] = {}
    for sid in sorted(by_shard):
        path = shard_path(cache_dir, backend_name, sid)
        if not force and _stamp_matches(path, model_version, temperature):
            have = _shard_hashes(path)
            pending = [(h, t) for h, t in by_shard[sid] if h not in have]
            summary.shards_reused += 1
            summary.reused_texts += len(by_shard[sid]) - len(pending)
            if pending:
                todo[sid] = pending
        else:
            todo[sid] = by_shard[sid]

    pending_total = sum(len(v) for v in todo.values())
    score_started = time.perf_counter()
    since_log = 0

    def _progress(n: int) -> None:
        nonlocal since_log
        since_log += n
        if progress_every <= 0 or since_log < progress_every:
            return
        since_log = 0
        done = summary.scored_texts
        rate = done / max(1e-9, time.perf_counter() - score_started)
        eta = (pending_total - done) / rate if rate > 0 else math.inf
        _LOG.info(
            "%s: %d/%d texts scored, %.1f items/s, ETA %.0f s",
            backend_name,
            done,
            pending_total,
            rate,
            eta,
        )

    with _InterruptGuard() as guard:
        for sid in sorted(todo):
            if guard.requested:
                summary.interrupted = True
                break
            pending = todo[sid]
            path = shard_path(cache_dir, backend_name, sid)
            texts = [title for _h, title in pending]
            scored = _score_texts(backend, texts, batch_size=batch_size, on_progress=_progress)
            rows: list[dict[str, Any]] = []
            for (text_hash, title), sent in zip(pending, scored, strict=False):
                row = _calibrated_row(
                    sent,
                    backend=backend_name,
                    model_version=model_version,
                    temperature=temperature,
                    text_hash=text_hash,
                )
                reason = skip_reason_for(title, row["confidence"])
                row["skipped"] = reason is not None
                row["skip_reason"] = reason
                rows.append(row)
            try:
                _write_shard(
                    path,
                    rows,
                    backend=backend_name,
                    model_version=model_version,
                    temperature=temperature,
                )
            except OSError as exc:
                raise BackendError(
                    f"cannot write shard {sid:03d} to {path}: {exc}. The partial .tmp is "
                    "discarded and not trusted; free space under data/cache/sentiment/ and rerun."
                ) from exc
            summary.shards_written += 1
            summary.scored_texts += len(rows)

    scored_elapsed = max(1e-9, time.perf_counter() - score_started)
    summary.items_per_sec = summary.scored_texts / scored_elapsed if summary.scored_texts else 0.0

    if out_path is not None and not summary.interrupted:
        assembled = assemble_article_sentiment(
            news=work,
            cache_dir=cache_dir,
            backend=backend_name,
            model_version=model_version,
            shards=sorted(by_shard),
        )
        summary.articles = int(len(assembled))
        summary.skipped_by_reason = {
            str(k): int(v)
            for k, v in assembled["skip_reason"].dropna().value_counts().sort_index().items()
        }
        write_article_sentiment(assembled, out_path, summary, reports_dir=reports_dir)
        summary.out_path = str(out_path)

    summary.elapsed_sec = time.perf_counter() - started
    _LOG.info(
        "score %s: %d texts scored, %d shard(s) written, %d reused, %.1f items/s, %.1f s total",
        backend_name,
        summary.scored_texts,
        summary.shards_written,
        summary.shards_reused,
        summary.items_per_sec,
        summary.elapsed_sec,
    )
    return summary


# --------------------------------------------------------------------------- assembly
def assemble_article_sentiment(
    *,
    news: pd.DataFrame,
    cache_dir: Path,
    backend: str,
    model_version: str,
    shards: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Concat shards for ``model_version`` and left-join them onto ``news`` row for row."""
    ids = range(NUM_SHARDS) if shards is None else [int(s) for s in shards]
    frames: list[pd.DataFrame] = []
    for sid in ids:
        path = shard_path(cache_dir, backend, sid)
        if not path.is_file():
            continue
        shard = pd.read_parquet(path)
        if "model_version" in shard and (shard["model_version"] != model_version).any():
            shard = shard.loc[shard["model_version"] == model_version]
        if not shard.empty:
            frames.append(shard)
    if not frames:
        raise BackendError(
            f"no shards for {backend!r} @ {model_version} under {Path(cache_dir) / backend}; "
            "score the corpus first"
        )
    per_hash = pd.concat(frames, ignore_index=True)
    per_hash = per_hash.drop_duplicates("text_hash", keep="first")

    joined = news.merge(per_hash, on="text_hash", how="left", validate="many_to_one")
    missing = joined["backend"].isna()
    if missing.any():
        sample = joined.loc[missing, "article_id"].head(5).tolist()
        raise BackendError(
            f"{int(missing.sum())} article row(s) have no shard entry for {backend!r} "
            f"(e.g. {sample}); rerun without restricting shards or with --force"
        )
    joined["skipped"] = joined["skipped"].astype(bool)
    joined = joined.loc[:, list(OUTPUT_COLUMNS)]
    return joined.sort_values("article_id", kind="stable").reset_index(drop=True)


def write_article_sentiment(
    frame: pd.DataFrame,
    out_path: Path,
    summary: ScoreSummary,
    *,
    reports_dir: Path,
) -> None:
    """Atomically write ``article_sentiment.parquet`` and ``reports/score_progress.json``."""
    write_parquet_atomic(frame, out_path, date32=("signal_date",))
    write_manifest(
        out_path.with_name(out_path.name + ".manifest.json"),
        {
            "kind": "article_sentiment",
            "backend": summary.backend,
            "model_version": summary.model_version,
            "temperature": summary.temperature,
            "rows": int(len(frame)),
            "unique_texts": int(frame["text_hash"].nunique()),
            "skipped_by_reason": summary.skipped_by_reason,
        },
    )
    progress_path = Path(reports_dir) / "score_progress.json"
    write_manifest(progress_path, summary.as_json())


def validate_output(frame: pd.DataFrame, news: pd.DataFrame) -> list[str]:
    """Structural violations of the T5 contract; empty list means the artifact is acceptable."""
    problems: list[str] = []
    missing_cols = [c for c in OUTPUT_COLUMNS if c not in frame.columns]
    if missing_cols:
        return [f"missing columns: {missing_cols}"]
    if len(frame) != len(news):
        problems.append(f"row count {len(frame)} != news rows {len(news)}")
    key = ["article_id", "ticker", "signal_date"]
    if frame.duplicated(key).any():
        problems.append("duplicate (article_id, ticker, signal_date) rows")
    if frame.groupby("article_id")["text_hash"].nunique().max() > 1:
        problems.append("one article_id maps to several text_hash values")

    def _keyset(df: pd.DataFrame) -> set[tuple]:
        # signal_date comes back from parquet as ``datetime.date`` objects, so compare strings.
        dates = pd.to_datetime(df["signal_date"]).dt.strftime("%Y-%m-%d")
        return set(
            zip(df["article_id"].astype(str), df["ticker"].astype(str), dates, strict=False)
        )

    if _keyset(frame) != _keyset(news):
        problems.append("(article_id, ticker, signal_date) keys do not match news_raw")
    if not frame["article_id"].is_monotonic_increasing:
        problems.append("rows are not sorted by article_id")
    probs = frame[["p_neg", "p_neu", "p_pos"]].to_numpy(dtype=float)
    if not np.isfinite(probs).all():
        problems.append("non-finite probabilities")
    else:
        dev = float(np.abs(probs.sum(axis=1) - 1.0).max())
        if dev > 1e-6:
            problems.append(f"probabilities do not sum to 1 (max dev {dev:.2e})")
    if float(frame["score_5"].abs().max()) > 1.0 + 1e-9:
        problems.append("score_5 outside [-1, 1]")
    dev3 = float((frame["score_3"] - (frame["p_pos"] - frame["p_neg"])).abs().max())
    if dev3 > 1e-9:
        problems.append(f"score_3 != p_pos - p_neg (max dev {dev3:.2e})")
    skipped = frame["skipped"].astype(bool)
    if bool((skipped & frame["skip_reason"].isna()).any()):
        problems.append("skipped rows without a skip_reason")
    if bool((frame["skip_reason"].notna() & ~skipped).any()):
        problems.append("skip_reason present on a row that is not skipped")
    return problems


def sha1_file(path: Path) -> str:
    """SHA1 of a file on disk — used to prove a rerun is byte-identical."""
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
