"""Build an ablation matrix that keeps only the sentiment feature block.

The GBDT trainer intersects the schema feature list with the matrix columns and
drops what is missing (warning logged), so a matrix without market columns is a
valid sentiment-only ablation input. Everything that is not a feature
(labels, split masks, calendar columns) must survive untouched.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

SENTIMENT_PREFIXES = (
    "n_news",
    "log_n_news",
    "n_news_z",
    "sent_",
    "pos_share",
    "neg_share",
    "neu_share",
    "neg_extreme_share",
    "ewma_",
    "company_specific_share",
    "novelty_mean",
    "conf_mean",
    "source_entropy",
    "top_source_share",
)


def _schema_features() -> list[str]:
    """Feature list from the build schema; falls back to an empty list."""
    path = PROC / "features_schema.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("features", "feature_columns", "columns"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, list) and value and isinstance(value[0], str):
            return value
    return []


def main() -> int:
    frame = pd.read_parquet(PROC / "features.parquet")
    sentiment = [c for c in frame.columns if c.startswith(SENTIMENT_PREFIXES)]
    features = set(_schema_features())
    # Keep every non-feature column (labels, split, ghost, calendar) plus the sentiment block.
    passthrough = [c for c in frame.columns if c not in features or c in sentiment]
    keep = list(dict.fromkeys(passthrough + sentiment))
    out = PROC / "features_sentiment_only.parquet"
    frame.loc[:, keep].to_parquet(out, index=False)

    manifest = {
        "source": str(PROC / "features.parquet"),
        "output": str(out),
        "rows": int(len(frame)),
        "n_columns": len(keep),
        "n_sentiment_features": len(sentiment),
        "sentiment_features": sentiment,
        "passthrough": [c for c in keep if c not in sentiment],
    }
    (PROC / "features_sentiment_only.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"wrote {out} rows={len(frame)} cols={len(keep)} sentiment={len(sentiment)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
