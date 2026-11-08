"""Diagnose where signal_date becomes NaT in the live GDELT path."""

from __future__ import annotations

import json
import pathlib

import pandas as pd

from spp.config import load_config, load_universe
from spp.data.normalize import SignalPolicy, alias_index_from_universe
from spp.data.news_live import _calendar, parse_gdelt_payload
from spp.data.news_fnspid import _finalize_news

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "live"

payload = json.loads((FIX / "gdelt_artlist.json").read_text(encoding="utf-8"))
config = load_config()
index = alias_index_from_universe(load_universe())
policy = SignalPolicy.from_config(config.news.cutoff_et, config.news.strict_dates)

frame = parse_gdelt_payload(payload, index=index)
print("parsed rows:", len(frame))
print("columns:", list(frame.columns))
print(frame[["date", "published_at"]].head(3).to_string())
print("dtypes:\n", frame[["date", "published_at"]].dtypes)

days = _calendar(pd.Timestamp("2026-10-01").date(), pd.Timestamp("2026-11-10").date())
out = _finalize_news(frame, policy, days)
print("\nafter finalize:")
print(out[["date", "published_at", "signal_date"]].head(3).to_string())
print("signal_date nulls:", int(out["signal_date"].isna().sum()), "of", len(out))
