"""Parquet storage primitives: atomic writes, cache manifests, partition helpers.

Every stage of the pipeline writes through :func:`write_parquet_atomic` so a crash
mid-write can never leave a half-written artifact behind. Reruns must produce
byte-identical parquet: callers sort deterministically and pin the compression scheme.

The cache manifest (:func:`read_manifest` / :func:`write_manifest`) records *what
actually came back from the network* so that a request for a wider range can be
served from an overlapping cached file up to the recorded fetch boundary, and a
request beyond that boundary is retried instead of silently returning NaN.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "PARQUET_COMPRESSION",
    "manifest_path",
    "normalize_date_range",
    "read_manifest",
    "read_parquet",
    "write_manifest",
    "write_parquet_atomic",
]

#: Pinned compression so repeated runs produce byte-identical files.
PARQUET_COMPRESSION = "snappy"


def normalize_date_range(start: date | None, end: date | None) -> tuple[date | None, date | None]:
    """Validate that ``start <= end`` (both inclusive) and return them unchanged."""
    if start is not None and end is not None and start > end:
        raise ValueError(f"start {start} > end {end}")
    return start, end


def write_parquet_atomic(
    df: pd.DataFrame,
    path: str | Path,
    *,
    compression: str = PARQUET_COMPRESSION,
    index: bool = False,
    date32: Iterable[str] = (),
) -> Path:
    """Write ``df`` to ``path`` atomically: ``.tmp`` in the target dir, then ``os.replace``.

    Parquet determinism relies on the caller having a stable row/column order; this
    function only guarantees the file is either the old one or the complete new one.
    Columns named in ``date32`` are stored as arrow ``date32`` (pandas keeps them as
    ``datetime64[ns]`` in memory).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.stem}.", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        table = pa.Table.from_pandas(df, preserve_index=index)
        for name in date32:
            position = table.schema.get_field_index(name)
            if position < 0:
                continue
            column = table.column(position)
            if pa.types.is_date32(column.type) or pa.types.is_date64(column.type):
                continue
            table = table.set_column(position, name, column.cast(pa.date32()))
        pq.write_table(table, tmp, compression=compression)
        os.replace(tmp, target)
    finally:
        if tmp.exists():  # pragma: no cover - only on failure paths
            try:
                tmp.unlink()
            except OSError:
                pass
    return target


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a parquet file into pandas.

    Arrow ``date32`` columns are converted back to ``datetime64[ns]`` so that the
    in-memory contract never depends on the storage encoding (dates carry no time).
    """
    frame = pd.read_parquet(Path(path), engine="pyarrow")
    for name in frame.columns:
        series = frame[name]
        if series.dtype == object and len(series) and isinstance(series.iloc[0], date):
            frame[name] = pd.to_datetime(series).astype("datetime64[ns]")
    return frame


def _dump_primitive(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _dump_primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_dump_primitive(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a sidecar JSON manifest atomically with sorted keys."""

    def _default(obj: Any) -> Any:  # pragma: no cover - defensive
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"not JSON serialisable: {type(obj).__name__}")

    data = {str(k): _dump_primitive(v) for k, v in payload.items()}
    data["manifest_written_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.stem}.", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(
            json.dumps(data, indent=2, sort_keys=True, default=_default, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, target)
    finally:
        if tmp.exists():  # pragma: no cover - only on failure paths
            try:
                tmp.unlink()
            except OSError:
                pass
    return target


def manifest_path(artifact: str | Path) -> Path:
    """Sidecar manifest path for a parquet artifact: ``x.parquet`` -> ``x.manifest.json``."""
    p = Path(artifact)
    return p.with_name(p.name + ".manifest.json")


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Load a sidecar JSON manifest; missing file yields an empty dict."""
    target = Path(path)
    if not target.is_file():
        return {}
    with target.open(encoding="utf-8") as fh:
        loaded = json.load(fh)
    return loaded if isinstance(loaded, dict) else {}


def parquet_columns(path: str | Path) -> tuple[str, ...]:
    """Column names of a parquet file without loading it fully."""
    import pyarrow.parquet as pq

    return tuple(pq.ParquetFile(Path(path)).schema_arrow.names)


def iter_batches(items: Iterable[str], size: int) -> Iterable[list[str]]:
    """Split ``items`` into consecutive lists of at most ``size`` elements."""
    if size <= 0:
        raise ValueError(f"batch size must be positive, got {size}")
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
