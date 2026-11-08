"""Merging a shard must never drop rows that were scored by an earlier run."""

from __future__ import annotations

import pandas as pd
import pytest

from spp.sentiment.score_corpus import SENTIMENT_COLUMNS, _write_shard


def _row(text_hash: str, p_pos: float) -> dict:
    return {
        "text_hash": text_hash,
        "backend": "finbert",
        "model_version": "model@sha",
        "p_neg": 1.0 - p_pos,
        "p_neu": 0.0,
        "p_pos": p_pos,
        "score_5": 2.0 * p_pos - 1.0,
        "score_3": 2.0 * p_pos - 1.0,
        "confidence": max(p_pos, 1.0 - p_pos),
        "label_5": "neutral",
        "is_company_specific": None,
        "is_novel": None,
        "skipped": False,
        "skip_reason": None,
    }


def test_write_shard_merges_with_existing_rows(tmp_path):
    path = tmp_path / "shard_000.parquet"
    _write_shard(path, [_row("aaa", 0.9)], backend="finbert", model_version="m", temperature=1.0)
    _write_shard(path, [_row("bbb", 0.1)], backend="finbert", model_version="m", temperature=1.0)

    frame = pd.read_parquet(path)
    assert list(frame.columns) == list(SENTIMENT_COLUMNS)
    assert set(frame["text_hash"]) == {"aaa", "bbb"}


def test_write_shard_keeps_the_new_value_on_collision(tmp_path):
    path = tmp_path / "shard_001.parquet"
    _write_shard(path, [_row("ccc", 0.2)], backend="finbert", model_version="m", temperature=1.0)
    _write_shard(path, [_row("ccc", 0.8)], backend="finbert", model_version="m", temperature=1.0)

    frame = pd.read_parquet(path)
    assert len(frame) == 1
    assert frame.loc[0, "p_pos"] == pytest.approx(0.8)


def test_write_shard_rebuilds_a_corrupt_file(tmp_path):
    path = tmp_path / "shard_002.parquet"
    path.write_bytes(b"not parquet")
    _write_shard(path, [_row("ddd", 0.5)], backend="finbert", model_version="m", temperature=1.0)
    assert set(pd.read_parquet(path)["text_hash"]) == {"ddd"}
