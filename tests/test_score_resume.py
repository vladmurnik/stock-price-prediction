"""T5 acceptance tests: shard resume, determinism, dedup fan-out and skip accounting.

No network and no GPU: the backends here are deterministic fakes that satisfy the
``SentimentBackend`` protocol, so the tests exercise the scoring machinery itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from spp.sentiment.base import BackendError, build_article_sentiment
from spp.sentiment.score_corpus import (
    NUM_SHARDS,
    OUTPUT_COLUMNS,
    ScoreSummary,
    assemble_article_sentiment,
    build_openjev_sample,
    load_news,
    score_corpus,
    sha1_file,
    shard_id_for,
    shard_path,
    skip_reason_for,
    validate_output,
)


class FakeBackend:
    """Deterministic stand-in: negative when the headline contains 'fall', else neutral."""

    def __init__(self, name: str = "finbert", model_version: str = "fake@1+cfg000") -> None:
        self.name = name
        self.model_version = model_version
        self.calls: list[str] = []

    def score(self, texts):
        self.calls.extend(texts)
        out = []
        for text in texts:
            if "fall" in text.lower():
                row = build_article_sentiment(
                    p_neg=0.8,
                    p_neu=0.15,
                    p_pos=0.05,
                    label_5="negative",
                    backend=self.name,
                    model_version=self.model_version,
                )
            elif "low" in text.lower():
                # Hand-built low-confidence distribution (0.38 < LOW_CONFIDENCE).
                row = build_article_sentiment(
                    p_neg=0.38,
                    p_neu=0.34,
                    p_pos=0.28,
                    label_5="negative",
                    backend=self.name,
                    model_version=self.model_version,
                )
            else:
                row = build_article_sentiment(
                    p_neg=0.1,
                    p_neu=0.8,
                    p_pos=0.1,
                    label_5="neutral",
                    backend=self.name,
                    model_version=self.model_version,
                )
            out.append(row)
        return out


def make_news(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """rows = (article_id, ticker, date, title)."""
    frame = pd.DataFrame(rows, columns=["article_id", "ticker", "signal_date", "title"])
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    # text_hash mirrors the real pipeline: a digest of the headline.  Duplicate titles
    # therefore share one hash and are scored once, exactly as in the corpus.
    frame["text_hash"] = frame["title"].map(lambda t: f"h{abs(hash(t)) % 10**10:010d}")
    return frame.sort_values("article_id", kind="stable").reset_index(drop=True)


ROWS = [
    ("a1", "AAPL", "2021-03-01", "Shares rise on strong results"),
    ("a2", "AAPL", "2021-03-02", "Stocks fall sharply"),
    ("a3", "MSFT", "2022-04-01", "Stocks fall sharply"),  # duplicated title, other ticker
    ("a4", "MSFT", "2023-05-01", "Mild outlook is low risk"),
    ("a5", "AAPL", "2023-06-01", ""),
    ("a6", "MSFT", "2023-07-01", "Акции растут"),
]


@pytest.fixture()
def env(tmp_path: Path):
    news = make_news(ROWS)
    cache = tmp_path / "cache"
    reports = tmp_path / "reports"
    reports.mkdir()
    return news, cache, reports


def test_load_news_tolerates_duplicate_article_ids(tmp_path: Path) -> None:
    frame = make_news(ROWS)
    frame["article_id"] = ["dup", "dup"] + [f"x{i}" for i in range(len(ROWS) - 2)]
    path = tmp_path / "news_raw.parquet"
    frame.to_parquet(path, index=False)
    loaded = load_news(path)
    assert len(loaded) == len(frame)
    assert list(loaded.columns) == ["article_id", "ticker", "signal_date", "title", "text_hash"]


def test_shard_ids_are_stable_and_spread() -> None:
    ids = {shard_id_for(f"h{i:08d}") for i in range(5000)}
    assert min(ids) >= 0 and max(ids) < NUM_SHARDS
    assert len(ids) > 200
    assert shard_id_for("h00000001") == shard_id_for("h00000001")


def test_skip_reasons_are_ordered_and_bounded() -> None:
    assert skip_reason_for("", 0.9) == "empty"
    assert skip_reason_for("   ", 0.9) == "empty"
    assert skip_reason_for("Акции растут", 0.9) == "non_english"
    assert skip_reason_for("Stocks fall", 0.35) == "low_confidence"
    assert skip_reason_for("Stocks fall", 0.95) is None


def test_score_dedupes_and_writes_every_article(tmp_path: Path) -> None:
    news = make_news(ROWS)
    cache, reports = tmp_path / "cache", tmp_path / "reports"
    reports.mkdir()
    backend = FakeBackend()
    out = tmp_path / "article_sentiment.parquet"
    summary = score_corpus(
        news,
        "finbert",
        cache_dir=cache,
        reports_dir=reports,
        temperature=1.0,
        out_path=out,
        backend=backend,
        progress_every=0,
    )
    # 6 rows but only 5 unique hashes: the duplicate is scored once.  The blank headline
    # (a5) never reaches the model, so 4 texts are actually encoded.
    assert summary.unique_texts == 5
    assert len(backend.calls) == 4
    frame = pd.read_parquet(out)
    assert list(frame.columns) == list(OUTPUT_COLUMNS)
    assert len(frame) == len(news)
    assert frame["article_id"].is_monotonic_increasing
    assert validate_output(frame, news) == []
    # Duplicate hash fans out to both tickers with identical scores.
    fall_hash = frame.loc[frame["article_id"] == "a2", "text_hash"].iloc[0]
    fall = frame.loc[frame["text_hash"] == fall_hash]
    assert len(fall) == 2
    assert fall["score_5"].nunique() == 1
    assert set(fall["ticker"]) == {"AAPL", "MSFT"}
    # Skip reasons are recorded, never silently dropped.
    by_id = frame.set_index("article_id")
    assert by_id.loc["a5", "skip_reason"] == "empty"
    assert by_id.loc["a6", "skip_reason"] == "non_english"
    assert by_id.loc["a4", "skip_reason"] == "low_confidence"
    assert bool(by_id.loc["a1", "skipped"]) is False
    progress = json.loads((reports / "score_progress.json").read_text(encoding="utf-8"))
    assert sum(progress["skipped_by_reason"].values()) == int(frame["skipped"].sum())
    assert progress["scored_texts"] == 5


def test_resume_reuses_shards_and_is_byte_identical(tmp_path: Path) -> None:
    news = make_news(ROWS)
    cache, reports = tmp_path / "cache", tmp_path / "reports"
    reports.mkdir()
    out = tmp_path / "article_sentiment.parquet"
    first_backend = FakeBackend()
    first = score_corpus(
        news,
        "finbert",
        cache_dir=cache,
        reports_dir=reports,
        temperature=1.0,
        out_path=out,
        backend=first_backend,
        progress_every=0,
    )
    assert first.scored_texts == 5
    digest = sha1_file(out)

    # Second run: a backend that would fail if called, proving nothing is recomputed.
    class ExplodingBackend(FakeBackend):
        def score(self, texts):
            raise AssertionError("resume must not rescore cached shards")

    second = score_corpus(
        news,
        "finbert",
        cache_dir=cache,
        reports_dir=reports,
        temperature=1.0,
        out_path=out,
        backend=ExplodingBackend(),
        progress_every=0,
    )
    assert second.scored_texts == 0
    assert second.reused_texts == 5
    assert second.shards_reused >= 1
    assert sha1_file(out) == digest


def test_partial_shard_run_then_resume_covers_everything(tmp_path: Path) -> None:
    news = make_news(ROWS)
    cache, reports = tmp_path / "cache", tmp_path / "reports"
    reports.mkdir()
    target = shard_id_for(news["text_hash"].iloc[0])
    first = score_corpus(
        news,
        "finbert",
        cache_dir=cache,
        reports_dir=reports,
        temperature=1.0,
        shards=[target],
        backend=FakeBackend(),
        progress_every=0,
        out_path=None,
    )
    assert first.shards_written == 1
    # Full run must now only score the shards that were not done yet.
    second = score_corpus(
        news,
        "finbert",
        cache_dir=cache,
        reports_dir=reports,
        temperature=1.0,
        backend=FakeBackend(),
        progress_every=0,
        out_path=None,
    )
    assert second.reused_texts >= 1
    covered = sum(len(pd.read_parquet(p)) for p in (cache / "finbert").glob("*.parquet"))
    assert covered == 5


def test_stale_temperature_invalidates_shard(tmp_path: Path) -> None:
    news = make_news(ROWS)
    cache, reports = tmp_path / "cache", tmp_path / "reports"
    reports.mkdir()
    score_corpus(
        news,
        "finbert",
        cache_dir=cache,
        reports_dir=reports,
        temperature=1.0,
        backend=FakeBackend(),
        progress_every=0,
    )
    backend = FakeBackend()
    summary = score_corpus(
        news,
        "finbert",
        cache_dir=cache,
        reports_dir=reports,
        temperature=0.7,
        backend=backend,
        progress_every=0,
    )
    assert summary.scored_texts == 5  # recomputed under the new T
    assert summary.reused_texts == 0
    # Old shard files are overwritten in place, never deleted into a hole.
    written = [s for s in range(NUM_SHARDS) if _exists(cache, s)]
    assert written and all(shard_path(cache, "finbert", s).is_file() for s in written)


def _exists(cache: Path, shard_id: int) -> bool:
    return shard_path(cache, "finbert", shard_id).is_file()


def test_force_recomputes(tmp_path: Path) -> None:
    news = make_news(ROWS)
    cache, reports = tmp_path / "cache", tmp_path / "reports"
    reports.mkdir()
    score_corpus(
        news,
        "finbert",
        cache_dir=cache,
        reports_dir=reports,
        temperature=1.0,
        backend=FakeBackend(),
        progress_every=0,
    )
    forced = score_corpus(
        news,
        "finbert",
        cache_dir=cache,
        reports_dir=reports,
        temperature=1.0,
        backend=FakeBackend(),
        progress_every=0,
        force=True,
    )
    assert forced.scored_texts == 5
    assert forced.reused_texts == 0


def test_openjev_requires_sample(tmp_path: Path) -> None:
    news = make_news(ROWS)
    with pytest.raises(ValueError, match="never score the full corpus"):
        score_corpus(
            news,
            "openjev",
            cache_dir=tmp_path / "c",
            reports_dir=tmp_path,
            temperature=1.0,
            backend=FakeBackend(name="openjev"),
            progress_every=0,
        )


def test_openjev_sample_is_stratified_includes_2023_and_is_reproducible() -> None:
    rows = []
    for year in (2019, 2020, 2021, 2022, 2023):
        for ticker in ("AAPL", "MSFT", "CVX"):
            for i in range(20):
                rows.append(
                    (f"{ticker}{year}{i:02d}", ticker, f"{year}-06-01", f"{year} {ticker} {i}")
                )
    news = make_news(rows)
    sample = build_openjev_sample(news, max_articles=70, seed=42)
    assert sample["article_id"].is_unique
    assert set(sample.loc[sample["year"] == 2023, "article_id"]) == set(
        news.loc[news["signal_date"].dt.year == 2023, "article_id"]
    )
    assert len(sample) <= 70
    again = build_openjev_sample(news, max_articles=70, seed=42)
    pd.testing.assert_frame_equal(sample, again)


def test_temperature_rescales_confidence_but_keeps_argmax(tmp_path: Path) -> None:
    news = make_news(ROWS)
    cache, reports = tmp_path / "cache", tmp_path / "reports"
    reports.mkdir()
    out = tmp_path / "article_sentiment.parquet"
    score_corpus(
        news,
        "finbert",
        cache_dir=cache,
        reports_dir=reports,
        temperature=0.7,
        out_path=out,
        backend=FakeBackend(),
        progress_every=0,
    )
    frame = pd.read_parquet(out)
    assert validate_output(frame, news) == []
    sharp = frame.loc[frame["article_id"] == "a2"].iloc[0]
    assert sharp["p_neg"] > 0.8  # T < 1 sharpens
    # p_neg=0.8 sits at bucket midpoint 0.5 => score_5 = -0.5625 => "very negative".
    assert sharp["label_5"] == "very negative"


def test_validate_output_flags_missing_rows(tmp_path: Path) -> None:
    news = make_news(ROWS)
    cache, reports = tmp_path / "cache", tmp_path / "reports"
    reports.mkdir()
    out = tmp_path / "article_sentiment.parquet"
    score_corpus(
        news,
        "finbert",
        cache_dir=cache,
        reports_dir=reports,
        temperature=1.0,
        out_path=out,
        backend=FakeBackend(),
        progress_every=0,
    )
    frame = pd.read_parquet(out).iloc[:-1]
    problems = validate_output(frame, news)
    assert any("row count" in p for p in problems)


def test_assemble_refuses_incomplete_coverage(tmp_path: Path) -> None:
    news = make_news(ROWS)
    cache, reports = tmp_path / "cache", tmp_path / "reports"
    reports.mkdir()
    score_corpus(
        news,
        "finbert",
        cache_dir=cache,
        reports_dir=reports,
        temperature=1.0,
        backend=FakeBackend(),
        progress_every=0,
    )
    with pytest.raises(BackendError, match="no shard entry|no shards for"):
        assemble_article_sentiment(
            news=news.head(1),
            cache_dir=cache,
            backend="finbert",
            model_version="fake@1+cfg000",
            shards=[shard_id_for("h_missing")],
        )


def test_summary_json_counts_are_consistent(tmp_path: Path) -> None:
    news = make_news(ROWS)
    cache, reports = tmp_path / "cache", tmp_path / "reports"
    reports.mkdir()
    score_corpus(
        news,
        "finbert",
        cache_dir=cache,
        reports_dir=reports,
        temperature=1.0,
        out_path=tmp_path / "a.parquet",
        backend=FakeBackend(),
        progress_every=0,
    )
    payload = json.loads((reports / "score_progress.json").read_text(encoding="utf-8"))
    for key in ("backend", "model_version", "temperature", "skipped_by_reason", "articles"):
        assert key in payload
    assert payload["articles"] == len(news)
    assert isinstance(ScoreSummary("finbert", "v", 1.0).as_json()["skipped_total"], int)
