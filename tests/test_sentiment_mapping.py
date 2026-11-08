"""T4 tests: the ``ArticleSentiment`` contract, backend mapping and temperature behaviour.

No network and no model downloads here — the tests use the lexicon backend (local dictionary) and
plain numeric fixtures, so they run in seconds and stay deterministic in CI.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from spp.sentiment.base import (
    LEVEL_LABELS,
    LEVEL_MIDPOINTS,
    ArticleSentiment,
    build_article_sentiment,
    label5_from_score,
    levels_to_probs,
    score5_from_levels,
    score5_from_probs,
)
from spp.sentiment.calibrate import apply_temperature, expected_calibration_error, fit_temperature
from spp.sentiment.lexicon import LexiconBackend

HEADLINES = [
    "The company reported a sharp increase in quarterly profit and revenue.",
    "Shares fell after the firm cut its full-year outlook.",
    "The annual meeting will be held in Helsinki in May.",
    "Record earnings per share beat every analyst estimate.",
    "The company announced a restructuring and layoffs of 2,000 staff.",
    "No plans to move production to Russia were disclosed.",
    "The board approved a share buyback programme worth 10 billion euros.",
    "Operating profit fell 14 percent amid rising raw material costs.",
    "The company appointed a new chief executive officer.",
    "Total assets rose slightly year on year.",
]


# ---------------------------------------------------------------- contract & 5-level mapping
def test_score5_from_levels_matches_expected_level_formula() -> None:
    # Point masses on the 5 levels must map to -1, -0.5, 0, 0.5, 1 (the open-jev path).
    for level, expected in zip(range(5), (-1.0, -0.5, 0.0, 0.5, 1.0), strict=False):
        probs = [1.0 if i == level else 0.0 for i in range(5)]
        assert score5_from_levels(probs) == pytest.approx(expected, abs=1e-9)


def test_score5_from_probs_uses_bucket_midpoints() -> None:
    # The 3-class path cannot express intensity beyond the bucket midpoint: a point mass on
    # "negative" sits at level 0.5, i.e. -0.75.  This is why open-jev bypasses it.
    p_neg, p_neu, p_pos = levels_to_probs([1.0, 0.0, 0.0, 0.0, 0.0])
    assert score5_from_probs(p_neg, p_neu, p_pos) == pytest.approx(-0.75)
    assert levels_to_probs([0.0, 0.0, 1.0, 0.0, 0.0]) == pytest.approx((0.0, 1.0, 0.0))


def test_article_sentiment_rejects_probabilities_not_summing_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        ArticleSentiment(
            p_neg=0.5,
            p_neu=0.5,
            p_pos=0.5,
            score_5=0.0,
            score_3=0.0,
            confidence=0.5,
            label_5="neutral",
            backend="x",
            model_version="v",
        )


def test_article_sentiment_rejects_score5_out_of_range() -> None:
    with pytest.raises(ValueError, match="score_5"):
        ArticleSentiment(
            p_neg=0.0,
            p_neu=1.0,
            p_pos=0.0,
            score_5=1.5,
            score_3=0.0,
            confidence=1.0,
            label_5="neutral",
            backend="x",
            model_version="v",
        )


def test_score3_is_p_pos_minus_p_neg() -> None:
    row = build_article_sentiment(
        p_neg=0.2, p_neu=0.3, p_pos=0.5, label_5="positive", backend="b", model_version="v"
    )
    assert row.score_3 == pytest.approx(row.p_pos - row.p_neg)
    assert row.confidence == pytest.approx(max(row.p_neg, row.p_neu, row.p_pos))


def test_levels_to_probs_maps_buckets() -> None:
    # level 0 + 1 -> negative, level 2 -> neutral, level 3 + 4 -> positive
    p_neg, p_neu, p_pos = levels_to_probs([0.1, 0.2, 0.3, 0.25, 0.15])
    assert p_neg == pytest.approx(0.3)
    assert p_neu == pytest.approx(0.3)
    assert p_pos == pytest.approx(0.4)
    assert p_neg + p_neu + p_pos == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (-0.9, "very negative"),
        (-0.41, "very negative"),
        (-0.4, "negative"),
        (-0.11, "negative"),
        (-0.1, "neutral"),
        (0.1, "neutral"),
        (0.11, "positive"),
        (0.4, "positive"),
        (0.41, "very positive"),
    ],
)
def test_label5_thresholds(score: float, expected: str) -> None:
    assert label5_from_score(score) == expected


def test_level_labels_are_ordered() -> None:
    assert LEVEL_LABELS == ("very negative", "negative", "neutral", "positive", "very positive")
    assert LEVEL_MIDPOINTS == (0.5, 2.0, 3.5)


# ---------------------------------------------------------------- temperature
def test_temperature_does_not_change_argmax() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 3))
    base = apply_temperature(logits, 1.0).argmax(axis=1)
    for t in (0.5, 0.8, 1.5, 3.0):
        assert np.array_equal(apply_temperature(logits, t).argmax(axis=1), base)


def test_temperature_is_applied_to_probabilities() -> None:
    logits = np.array([[2.0, 0.0, -2.0]])
    sharp = apply_temperature(logits, 0.5)
    soft = apply_temperature(logits, 2.0)
    assert sharp[0].max() > soft[0].max()
    assert sharp.sum() == pytest.approx(1.0)
    assert soft.sum() == pytest.approx(1.0)


def test_fit_temperature_sharpens_confident_correct_predictions() -> None:
    """Overconfident *and correct* logits carry real skill: NLL is minimised by sharpening (T<1).

    The direction is a property of the data, not of the optimiser — so we assert it explicitly
    instead of assuming T must exceed 1.
    """
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 3, size=500)
    logits = np.full((500, 3), -6.0)
    logits[np.arange(500), labels] = 6.0
    t = fit_temperature(logits, labels)
    assert 0.5 <= t < 1.0


def test_fit_temperature_bounds_are_respected() -> None:
    rng = np.random.default_rng(11)
    logits = rng.normal(scale=50.0, size=(300, 3))
    labels = rng.integers(0, 3, size=300)
    t = fit_temperature(logits, labels)
    assert 0.5 <= t <= 3.0


# ---------------------------------------------------------------- ECE
def test_ece_is_zero_for_perfect_calibration() -> None:
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    labels = np.array([0, 1, 2])
    ece, bins = expected_calibration_error(probs, labels)
    assert ece == pytest.approx(0.0, abs=1e-9)
    assert sum(b["n"] for b in bins) == 3


def test_ece_penalises_confident_errors() -> None:
    probs = np.array([[0.95, 0.03, 0.02], [0.95, 0.03, 0.02]])
    labels = np.array([1, 1])  # both wrong
    ece, _ = expected_calibration_error(probs, labels)
    assert ece > 0.9


# ---------------------------------------------------------------- lexicon backend
def test_lexicon_returns_valid_contract_rows() -> None:
    rows = LexiconBackend().score(HEADLINES)
    assert len(rows) == len(HEADLINES)
    for row in rows:
        assert abs(row.p_neg + row.p_neu + row.p_pos - 1.0) <= 1e-6
        assert -1.0 <= row.score_5 <= 1.0
        assert row.label_5 in LEVEL_LABELS
        assert row.score_3 == pytest.approx(row.p_pos - row.p_neg)
        assert row.backend == "lexicon"
        assert row.is_novel is None and row.is_company_specific is None


def test_lexicon_separates_positive_and_negative_headlines() -> None:
    rows = LexiconBackend().score([HEADLINES[0], HEADLINES[1]])
    assert rows[0].score_3 > rows[1].score_3


def test_lexicon_is_deterministic() -> None:
    a = LexiconBackend().score(HEADLINES)
    b = LexiconBackend().score(HEADLINES)
    assert [r.score_5 for r in a] == [r.score_5 for r in b]


# ---------------------------------------------------------------- edge cases
@pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
def test_empty_text_raises_value_error_not_nan(bad: str) -> None:
    with pytest.raises(ValueError):
        LexiconBackend().score([bad])


def test_non_string_input_raises() -> None:
    with pytest.raises(ValueError):
        LexiconBackend().score([None])  # type: ignore[list-item]


def test_non_english_text_does_not_crash() -> None:
    rows = LexiconBackend().score(["Компания объявила о рекордной прибыли."])
    assert len(rows) == 1
    assert 0.0 <= rows[0].confidence <= 1.0


def test_lexicon_softmax_predicts_neutral_without_evidence() -> None:
    """Regression: a no-evidence headline must not fall through to a polarity.

    With a neutral logit of ``0`` the distribution was uniform and ``argmax`` picked index 0
    ("negative") for every headline the dictionary did not recognise (2146 of the 3453 phrasebank
    rows), which is what drove 3-class accuracy far below chance.
    """
    p_neg, p_neu, p_pos = LexiconBackend._softmax(0.0, 0.0)
    assert p_neu > p_neg == pytest.approx(p_pos)
    assert p_neg + p_neu + p_pos == pytest.approx(1.0)

    # One matched term ties with the neutral prior (and wins the argmax on its own side), while
    # two net matched terms overrule it.
    one = LexiconBackend._softmax(0.0, 1.0)
    assert one[1] == pytest.approx(one[2])
    assert one[1] > one[0]
    two = LexiconBackend._softmax(0.0, 2.0)
    assert two[2] > two[1]

    # neg evidence moves mass to the negative class; the positive class keeps the least.
    l_neg, l_neu, l_pos = LexiconBackend._softmax(3.0, 0.0)
    assert l_neg > l_neu > l_pos
    # pos evidence mirrors it.
    r_neg, r_neu, r_pos = LexiconBackend._softmax(0.0, 3.0)
    assert r_pos > r_neu > r_neg


def test_lexicon_predictions_cover_all_three_classes() -> None:
    """Regression: the lexicon used to predict only {negative, positive} and never neutral."""
    classes = ("negative", "neutral", "positive")
    rows = LexiconBackend().score(HEADLINES)
    preds = [classes[int(np.argmax((row.p_neg, row.p_neu, row.p_pos)))] for row in rows]
    assert set(preds) == set(classes)


def test_minimize_scalar_is_reachable() -> None:
    """Guard: the plan mandates scipy.optimize.minimize_scalar for the temperature fit."""
    res = minimize_scalar(lambda x: (x - 1.5) ** 2, bounds=(0.5, 3.0), method="bounded")
    assert 0.5 <= res.x <= 3.0
