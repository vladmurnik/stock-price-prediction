"""Batched open-jev must reproduce the per-item decision exactly.

The 400M-parameter model makes this a slow test: it is skipped unless ``SPP_SLOW_TESTS=1``.
Set it once after installing the backend to re-verify the batched forward pass.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SPP_SLOW_TESTS") != "1",
    reason="loads the open-jev weights; set SPP_SLOW_TESTS=1 to run",
)

HEADLINES = [
    "Apple beats quarterly earnings expectations on strong iPhone demand",
    "Regulator opens antitrust probe into the company's app store",
    "The company announced a new chief financial officer",
]


def test_batched_matches_single_item() -> None:
    from spp.sentiment.openjev import OpenJevBackend

    single = OpenJevBackend(batch_size=1)
    batched = OpenJevBackend(batch_size=8)

    one = single.score(HEADLINES)
    many = batched.score(HEADLINES)

    assert single.model_version == batched.model_version
    assert len(one) == len(many) == len(HEADLINES)
    for expected, actual in zip(one, many, strict=True):
        assert expected.label_5 == actual.label_5
        assert expected.p_pos == pytest.approx(actual.p_pos, abs=1e-6)
        assert expected.p_neg == pytest.approx(actual.p_neg, abs=1e-6)
        assert expected.p_neu == pytest.approx(actual.p_neu, abs=1e-6)
        assert expected.score_5 == pytest.approx(actual.score_5, abs=1e-6)


def test_batch_size_is_floored_at_one() -> None:
    from spp.sentiment.openjev import OpenJevBackend

    assert OpenJevBackend(batch_size=0).batch_size == 1
    assert OpenJevBackend(batch_size=16).batch_size == 16
