"""Logging setup: stdout + daily file under ``reports/logs``."""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

__all__ = ["LOG_FORMAT", "get_logger", "setup_logging"]

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_LOG_DIR = Path("reports") / "logs"
_MARKER = "_spp_configured"


def _log_level() -> int:
    raw = os.environ.get("SPP_LOG_LEVEL", "INFO").strip().upper()
    level = logging.getLevelName(raw)
    return level if isinstance(level, int) else logging.INFO


def _log_dir() -> Path:
    override = os.environ.get("SPP_LOG_DIR")
    return Path(override) if override else _DEFAULT_LOG_DIR


def setup_logging(
    level: int | str | None = None, log_dir: str | Path | None = None
) -> logging.Logger:
    """Configure the root ``spp`` logger idempotently."""
    root = logging.getLogger("spp")
    if getattr(root, _MARKER, False) and log_dir is None and level is None:
        return root
    # An explicit level/log_dir means "reconfigure": drop previously installed handlers.
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    resolved = _log_level() if level is None else level
    root.setLevel(resolved)
    root.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    directory = Path(os.fspath(log_dir)) if log_dir is not None else _log_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        log_file = directory / f"{datetime.now(UTC).date().isoformat()}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:  # read-only checkout, missing reports/ mount, ...
        root.warning("file logging disabled (%s): %s", directory, exc)

    setattr(root, _MARKER, True)
    return root


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured logger; ``None`` yields the package logger."""
    setup_logging()
    if not name or name == "spp":
        return logging.getLogger("spp")
    short = name.removeprefix("spp.")
    return logging.getLogger(f"spp.{short}")
