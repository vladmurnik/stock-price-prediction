from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from spp.data.news_fnspid import parquet_source
from spp.data.normalize import SignalPolicy, alias_index_from_universe


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2020-06-01 21:30:00 UTC", "2020-06-02 14:00:00 UTC"],
            "Article_title": ["Apple unveils new Mac", "Microsoft beats estimates"],
            "Stock_symbol": ["AAPL", "MSFT"],
            "Url": ["https://example.com/a", "https://example.com/m"],
            "Publisher": ["Reuters", "CNBC"],
        }
    )


def test_parquet_source_yields_fnspid_rows(tmp_path):
    path = tmp_path / "candidates.parquet"
    _frame().to_parquet(path, index=False)

    rows = list(parquet_source(path)("shard"))
    assert len(rows) == 2
    line_no, record = rows[0]
    assert line_no == 1
    assert record["Stock_symbol"] == "AAPL"
    assert record["Article_title"] == "Apple unveils new Mac"
    assert record["Publisher"] == "Reuters"


def test_parquet_source_rejects_missing_columns(tmp_path):
    path = tmp_path / "broken.parquet"
    pd.DataFrame({"Date": ["2020-06-01"], "Stock_symbol": ["AAPL"]}).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        list(parquet_source(path)("shard"))


def test_parquet_source_feeds_the_pipeline(tmp_path):
    from spp.config import load_universe
    from spp.data.news_fnspid import fetch_news

    path = tmp_path / "candidates.parquet"
    _frame().to_parquet(path, index=False)
    universe = load_universe()
    index = alias_index_from_universe(universe)
    policy = SignalPolicy.from_config("16:00", strict_dates=False)

    result = fetch_news(
        date(2020, 6, 1),
        date(2020, 6, 30),
        tmp_path / "interim",
        tmp_path / "reports",
        index,
        policy,
        shards=(str(path),),
        source=parquet_source(path),
        refresh=True,
    )
    assert len(result.frame) == 2
    assert set(result.frame["ticker"]) == {"AAPL", "MSFT"}
    assert (result.frame["signal_date"] >= result.frame["date"]).all()
