"""Typed project configuration loaded from ``configs/*.yaml``."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "BacktestCfg",
    "Config",
    "CvCfg",
    "DatesCfg",
    "FeaturesCfg",
    "LiveCfg",
    "ModelCfg",
    "NewsCfg",
    "PathsCfg",
    "SentimentCfg",
    "TickerInfo",
    "Universe",
    "load_config",
    "load_universe",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _Section(BaseModel):
    """Base for config sections: unknown keys are a hard error."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DatesCfg(_Section):
    train_start: date
    train_end: date
    valid_end: date
    test_end: date

    @model_validator(mode="after")
    def _check_order(self) -> DatesCfg:
        checks = [
            ("train_start", self.train_start, self.train_end),
            ("train_end", self.train_end, self.valid_end),
            ("valid_end", self.valid_end, self.test_end),
        ]
        for name, left, right in checks:
            if not left < right:
                raise ValueError(f"dates must be increasing: {name}={left} !< {right}")
        return self


class NewsCfg(_Section):
    cutoff_et: str
    strict_dates: bool = False
    dedupe_jaccard: float = Field(ge=0.0, le=1.0)
    publisher_priority: list[str] = Field(
        default_factory=lambda: ["reuters", "bloomberg", "wsj", "cnbc"]
    )

    @field_validator("cutoff_et")
    @classmethod
    def _check_cutoff(cls, value: str) -> str:
        hour, sep, minute = value.partition(":")
        if not sep or not (hour.isdigit() and minute.isdigit()):
            raise ValueError(f"cutoff_et must be HH:MM, got {value!r}")
        if not (0 <= int(hour) < 24 and 0 <= int(minute) < 60):
            raise ValueError(f"cutoff_et out of range: {value!r}")
        return value

    @field_validator("publisher_priority")
    @classmethod
    def _check_priority(cls, value: list[str]) -> list[str]:
        cleaned = [v.strip().casefold() for v in value if v and v.strip()]
        if not cleaned:
            raise ValueError("news.publisher_priority must not be empty")
        return cleaned


class SentimentCfg(_Section):
    backend: Literal["finbert", "openjev", "lexicon"] = "finbert"
    finbert_model: str
    openjev_model: str
    batch_size: int = Field(gt=0)
    max_length: int = Field(gt=0)
    openjev_max_articles: int = Field(gt=0)
    min_macro_f1: float = Field(ge=0.0, le=1.0)


class FeaturesCfg(_Section):
    ewma_halflives_min: list[int]
    rolling_days: int = Field(gt=0)
    cross_sectional_rank: bool = True
    demean_target: bool = True

    @field_validator("ewma_halflives_min")
    @classmethod
    def _check_halflives(cls, value: list[int]) -> list[int]:
        if not value or any(v <= 0 for v in value):
            raise ValueError(f"ewma_halflives_min must be non-empty positive ints, got {value}")
        return value


class ModelCfg(_Section):
    num_leaves: int = Field(ge=2)
    learning_rate: float = Field(gt=0.0, le=1.0)
    min_data_in_leaf: int = Field(ge=1)
    lambda_l2: float = Field(ge=0.0)
    seeds: list[int]
    early_stopping_rounds: int = Field(ge=1)

    @field_validator("seeds")
    @classmethod
    def _check_seeds(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("model.seeds must not be empty")
        return value


class CvCfg(_Section):
    n_folds: int = Field(ge=2)
    test_months: int = Field(ge=1)
    embargo_days: int = Field(ge=0)
    min_train_months: int = Field(ge=1)


class BacktestCfg(_Section):
    n_quantiles: int = Field(ge=2)
    commission_bps: float = Field(ge=0.0)
    slippage_bps: float = Field(ge=0.0)
    borrow_annual: float = Field(ge=0.0)
    execution: Literal["next_open_hold_close", "next_close", "next_open", "close_to_close"]


class LiveCfg(_Section):
    run_at_utc: str
    state_path: Path

    @field_validator("run_at_utc")
    @classmethod
    def _check_run_at(cls, value: str) -> str:
        hour, sep, minute = value.partition(":")
        if not sep or not (hour.isdigit() and minute.isdigit()):
            raise ValueError(f"run_at_utc must be HH:MM, got {value!r}")
        if not (0 <= int(hour) < 24 and 0 <= int(minute) < 60):
            raise ValueError(f"run_at_utc out of range: {value!r}")
        return value


class PathsCfg(_Section):
    data: Path
    reports: Path


class Config(_Section):
    seed: int
    universe: Path
    dates: DatesCfg
    news: NewsCfg
    sentiment: SentimentCfg
    features: FeaturesCfg
    model: ModelCfg
    cv: CvCfg
    backtest: BacktestCfg
    live: LiveCfg
    paths: PathsCfg


class TickerInfo(_Section):
    sector: str
    aliases: tuple[str, ...]

    @field_validator("aliases")
    @classmethod
    def _check_aliases(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("aliases must not be empty")
        return value


class Universe(_Section):
    tickers: dict[str, TickerInfo]
    sector_etf: dict[str, str] = {}
    exclude_phrases: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_sectors(self) -> Universe:
        unknown = {info.sector for info in self.tickers.values()} - set(self.sector_etf)
        if unknown:
            raise ValueError(f"sectors without an ETF mapping: {sorted(unknown)}")
        return self


def _resolve(path: str | Path, base_dir: Path) -> Path:
    """Resolve relative paths against the project root."""
    p = Path(os.fspath(path)).expanduser()
    return p if p.is_absolute() else (base_dir / p)


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return payload


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise NotADirectoryError(f"not a directory: {path}")


def load_config(path: str | Path = "configs/base.yaml", root: str | Path | None = None) -> Config:
    """Load and validate ``configs/base.yaml``; create data/report directories."""
    base_dir = _resolve(root or PROJECT_ROOT, Path.cwd())
    cfg_path = _resolve(path, base_dir)
    cfg = Config.model_validate(_load_yaml(cfg_path))

    for directory in (cfg.paths.data, cfg.paths.reports):
        _ensure_dir(_resolve(directory, base_dir))
    _ensure_dir(_resolve(cfg.paths.reports, base_dir) / "logs")
    # live.state_path is relative to the project root as well.
    _ensure_dir(_resolve(cfg.live.state_path, base_dir).parent)

    universe_path = _resolve(cfg.universe, base_dir)
    if not universe_path.is_file():
        raise FileNotFoundError(f"universe file not found: {universe_path}")
    return cfg


def load_universe_config(path: str | Path = "configs/universe_us_largecap.yaml") -> Universe:
    """Load and validate the universe YAML (tickers, sector ETFs, exclusion phrases)."""
    return Universe.model_validate(_load_yaml(_resolve(path, PROJECT_ROOT)))


def load_universe(path: str | Path = "configs/universe_us_largecap.yaml") -> dict[str, TickerInfo]:
    """Load the ticker universe, validating sector ETF coverage."""
    return dict(load_universe_config(path).tickers)


def load_universe_exclusions(
    path: str | Path = "configs/universe_us_largecap.yaml",
) -> tuple[str, ...]:
    """Phrases that must never be linked to a universe ticker (e.g. ``apple hospitality``)."""
    return tuple(load_universe_config(path).exclude_phrases)


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag (1/true/yes/on)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
