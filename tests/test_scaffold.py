"""Tests for the T1 scaffold: config, logging and CLI stubs."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from spp.config import load_config, load_universe
from spp.logging import get_logger, setup_logging

CONFIG = Path("configs/base.yaml")
UNIVERSE = Path("configs/universe_us_largecap.yaml")
runner = CliRunner()


def test_load_config_returns_typed_sections() -> None:
    cfg = load_config(CONFIG)
    assert cfg.seed == 42
    assert cfg.dates.train_start == date(2019, 1, 1)
    assert cfg.dates.test_end == date(2023, 12, 31)
    assert cfg.backtest.n_quantiles == 5
    assert cfg.paths.data.is_dir()
    assert cfg.paths.reports.is_dir()


def test_load_config_rejects_non_monotonic_dates(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["dates"]["valid_end"] = payload["dates"]["train_end"]
    broken = tmp_path / "base.yaml"
    broken.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="dates must be increasing"):
        load_config(broken)


def test_load_config_rejects_single_quantile(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["backtest"]["n_quantiles"] = 1
    broken = tmp_path / "base.yaml"
    broken.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(broken)


def test_load_universe_is_8_sectors_by_5() -> None:
    tickers = load_universe(UNIVERSE)
    assert len(tickers) == 40
    sectors: dict[str, int] = {}
    for info in tickers.values():
        sectors[info.sector] = sectors.get(info.sector, 0) + 1
    assert len(sectors) == 8
    assert set(sectors.values()) == {5}
    assert all(2 <= len(info.aliases) <= 5 for info in tickers.values())


def test_logging_writes_daily_file(tmp_path: Path) -> None:
    from spp import logging as spp_logging

    root = get_logger("data.prices")
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    if hasattr(root, spp_logging._MARKER):
        delattr(root, spp_logging._MARKER)

    setup_logging(log_dir=tmp_path)
    get_logger("data.prices").info("hello")
    files = list(tmp_path.glob("*.log"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "| INFO | spp.data.prices | hello" in text

    for handler in list(root.handlers):
        handler.close()
        root.removeHandler(handler)
    if hasattr(root, spp_logging._MARKER):
        delattr(root, spp_logging._MARKER)


def test_cli_help_lists_all_groups() -> None:
    from spp.cli import app

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for group in ("fetch", "score", "features", "train", "backtest", "live", "report"):
        assert group in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["fetch", "live"],
        ["train", "all"],
        ["live", "run"],
        ["report", "ablation"],
        ["report", "live"],
        ["report", "final"],
    ],
)
def test_cli_stubs_exit_two(args: list[str]) -> None:
    from spp.cli import app

    result = runner.invoke(app, [*args, "--config", str(CONFIG)])
    assert result.exit_code == 2, result.output
    assert "not implemented yet" in result.output


def test_cli_rejects_unknown_option() -> None:
    from spp.cli import app

    result = runner.invoke(app, ["fetch", "prices", "--bogus"])
    assert result.exit_code == 2


def test_cli_version() -> None:
    from spp.cli import app

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "spp" in result.output
