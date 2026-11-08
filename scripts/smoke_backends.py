"""Smoke script: 10 fixed headlines through each backend, validating the contract.

Run: `.venv\\Scripts\\python.exe scripts\\smoke_backends.py [backend...]`
Not part of the test suite (it downloads models); the acceptance checks live in tests/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HEADLINES = [
    "Apple reports record quarterly revenue, beating analyst estimates.",
    "Tesla cuts delivery targets for the third quarter.",
    "JPMorgan announces the departure of its chief financial officer.",
    "The company declared a quarterly dividend of $0.42 per share.",
    "Boeing faces a new regulatory probe into 737 MAX production.",
    "Nvidia unveils next-generation AI chips at its developer conference.",
    "Walmart warns that household spending remains under pressure.",
    "The board approved a $10 billion share buyback programme.",
    "Losses widened to $2.1 billion as raw material costs surged.",
    "Annual shareholder meeting will be held on May 14 in Atlanta.",
]


def check(backend_name: str) -> None:
    from spp.sentiment.base import LEVEL_LABELS, ArticleSentiment

    if backend_name == "finbert":
        from spp.sentiment.finbert import FinBertBackend

        backend = FinBertBackend()
    elif backend_name == "openjev":
        from spp.sentiment.openjev import OpenJevBackend

        backend = OpenJevBackend()
    elif backend_name == "lexicon":
        from spp.sentiment.lexicon import LexiconBackend

        backend = LexiconBackend()
    else:
        raise SystemExit(f"unknown backend {backend_name}")

    rows: list[ArticleSentiment] = backend.score(HEADLINES)
    assert len(rows) == len(HEADLINES), (len(rows), len(HEADLINES))
    for text, row in zip(HEADLINES, rows):
        total = row.p_neg + row.p_neu + row.p_pos
        assert abs(total - 1.0) <= 1e-6, (text, total)
        assert -1.0 <= row.score_5 <= 1.0, (text, row.score_5)
        assert row.label_5 in LEVEL_LABELS, row.label_5
        assert abs(row.score_3 - (row.p_pos - row.p_neg)) < 1e-9
        print(
            f"{backend_name:8s} {row.label_5:13s} s5={row.score_5:+.3f} s3={row.score_3:+.3f} "
            f"c={row.confidence:.3f}  {text[:58]}"
        )
    print(f"{backend_name}: OK ({len(rows)} rows, model_version={backend.model_version})\n")


if __name__ == "__main__":
    names = sys.argv[1:] or ["lexicon", "finbert", "openjev"]
    for name in names:
        check(name)
