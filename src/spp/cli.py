"""Unified entrypoint: ``spp <group> <command> [--config ...]``.

Groups call into their own module once the matching step (T2+) lands; only argument
parsing, path resolution and exit codes live here.
"""

from __future__ import annotations

import os
import random
import sys
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any, NoReturn

import pandas as pd
import typer

from spp import __version__
from spp.config import PROJECT_ROOT, Config, load_config
from spp.logging import get_logger

__all__ = ["app", "run"]

NOT_IMPLEMENTED_EXIT = 2
DATA_EXIT = 1

DEFAULT_PRICES_START = date(2019, 1, 1)
DEFAULT_PRICES_END = date(2023, 12, 31)


class Context:
    """State shared by the callback and the commands."""

    def __init__(self, config: Config, seed: int, run_id: str) -> None:
        self.config = config
        self.seed = seed
        self.run_id = run_id
        self.log = get_logger("spp.cli")


def _stub_command(name: str) -> Callable[..., None]:
    """Build a Typer command stub that also accepts the global flags."""

    def command(
        ctx: typer.Context,
        config: Annotated[
            Path | None,
            typer.Option("--config", "-c", help="Config override (same as the global option)."),
        ] = None,
        seed: Annotated[int | None, typer.Option("--seed", help="Seed override.")] = None,
        run_id: Annotated[str | None, typer.Option("--run-id", help="Run id override.")] = None,
    ) -> None:
        parent = ctx.parent.obj if ctx.parent is not None else None
        if isinstance(parent, Context):
            ctx.obj = parent
            if config is not None:
                parent.config = load_config(config)
            if seed is not None:
                parent.seed = seed
                _apply_seed(seed)
            if run_id is not None:
                parent.run_id = run_id
                os.environ["SPP_RUN_ID"] = run_id
        _not_implemented(ctx, name)

    command.__name__ = name.replace(" ", "_")
    command.__doc__ = f"Stub for `{name}` (implemented in a later step)."
    return command


def _not_implemented(ctx: typer.Context, command: str) -> NoReturn:
    if isinstance(ctx.obj, Context):
        ctx.obj.log.info("stub invoked: %s", command)
    typer.secho(f"not implemented yet: {command}", fg=typer.colors.YELLOW, err=True)
    raise typer.Exit(code=NOT_IMPLEMENTED_EXIT)


def _apply_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:  # numpy is core; keep the CLI usable without it
        return
    np.random.seed(seed)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"spp {__version__}")
        raise typer.Exit()


def _parse_date(value: str | None, fallback: date, label: str) -> date:
    if value is None:
        return fallback
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        typer.secho(f"--{label} must be YYYY-MM-DD, got {value!r}", fg=typer.colors.RED, err=True)
        raise typer.BadParameter(str(exc), param_hint=f"--{label}") from exc


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _parent(ctx: typer.Context) -> Context:
    obj = ctx.obj if ctx.obj is not None else (ctx.parent.obj if ctx.parent else None)
    if isinstance(obj, Context):
        return obj
    cfg = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    return Context(cfg, cfg.seed, datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))


app = typer.Typer(
    name="spp",
    help="News-sentiment -> stock return prediction pipeline.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path,
        typer.Option(
            "--config", "-c", help="YAML config loaded for every command.", dir_okay=False
        ),
    ] = Path("configs/base.yaml"),
    seed: Annotated[
        int | None, typer.Option("--seed", help="Override config seed (default: config value).")
    ] = None,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Run id; default is a UTC timestamp.")
    ] = None,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Load config and fix the seed before dispatching to a command."""
    cfg = load_config(config)
    resolved_seed = cfg.seed if seed is None else seed
    resolved_run = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    _apply_seed(resolved_seed)
    os.environ["SPP_RUN_ID"] = resolved_run
    os.environ["SPP_SEED"] = str(resolved_seed)
    ctx.obj = Context(cfg, resolved_seed, resolved_run)


fetch_app = typer.Typer(help="Ingest prices, news and live feeds.", no_args_is_help=True)
score_app = typer.Typer(help="Sentiment scoring and calibration.", no_args_is_help=True)
features_app = typer.Typer(help="Feature matrix construction.", no_args_is_help=True)
train_app = typer.Typer(help="Model training.", no_args_is_help=True)
live_app = typer.Typer(help="Paper-trading loop.", no_args_is_help=True)
report_app = typer.Typer(help="Report rendering.", no_args_is_help=True)


@fetch_app.command("prices")
def fetch_prices_command(
    ctx: typer.Context,
    start: Annotated[
        str | None,
        typer.Option("--start", help="First trading day (inclusive), YYYY-MM-DD."),
    ] = None,
    end: Annotated[
        str | None,
        typer.Option("--end", help="Last trading day (inclusive), YYYY-MM-DD."),
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Ignore the cache and re-download from yfinance."),
    ] = False,
) -> None:
    """Download 40 universe tickers + SPY/^VIX/8 sector ETFs into one parquet cache."""
    from spp.config import load_universe
    from spp.data.prices import (
        default_end,
        fetch_prices,
        helper_tickers,
        write_anomaly_report,
    )

    c = _parent(ctx)
    root = PROJECT_ROOT
    start_d = _parse_date(start, DEFAULT_PRICES_START, "start")
    end_d = _parse_date(end, max(DEFAULT_PRICES_END, default_end()), "end")
    if start_d > end_d:
        typer.secho(f"--start {start_d} is after --end {end_d}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=DATA_EXIT)

    universe = load_universe(_resolve(root, c.config.universe))
    helpers = helper_tickers(_resolve(root, c.config.universe))
    tickers = list(universe) + [h for h in helpers if h not in universe]
    cache_dir = _resolve(root, c.config.paths.data) / "cache"
    reports_dir = _resolve(root, c.config.paths.reports)

    result = fetch_prices(tickers, start_d, end_d, cache_dir, refresh=refresh)
    anomalies_path = write_anomaly_report(result.anomalies, reports_dir / "prices_anomalies.csv")

    typer.echo(f"cache:   {result.path} ({result.frame.shape[0]} rows)")
    missing = list(result.missing)
    typer.echo(f"tickers: {result.coverage.shape[0]} present, {len(missing)} missing {missing}")
    typer.echo(f"range:   {start_d} .. {end_d}")
    if not result.coverage.empty:
        cov = result.coverage["coverage"]
        typer.echo(
            f"coverage: median {cov.median():.3f}, min {cov.min():.3f}, "
            f"tickers >=95%: {int((cov >= 0.95).sum())}/{len(cov)}"
        )
        dmin = min(result.coverage["first_date"])
        dmax = max(result.coverage["last_date"])
        typer.echo(f"data:     {dmin} .. {dmax}")
    typer.echo(f"anomalies: {len(result.anomalies)} row(s) -> {anomalies_path}")
    if result.violations:
        for violation in result.violations[:10]:
            typer.secho(f"VALIDATION: {violation}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=DATA_EXIT)


@fetch_app.command("news")
def fetch_news_command(
    ctx: typer.Context,
    start: Annotated[
        str | None,
        typer.Option("--start", help="First source date (inclusive), YYYY-MM-DD."),
    ] = None,
    end: Annotated[
        str | None,
        typer.Option("--end", help="Last source date (inclusive), YYYY-MM-DD."),
    ] = None,
    shard: Annotated[
        list[str] | None,
        typer.Option("--shard", help="FNSPID CSV shard relative to the repo root (repeatable)."),
    ] = None,
    fixture: Annotated[
        Path | None,
        typer.Option("--fixture", help="Replay a local CSV instead of streaming HF.", exists=True),
    ] = None,
    from_parquet: Annotated[
        Path | None,
        typer.Option(
            "--from-parquet",
            help="Read rows from a local parquet mirror instead of streaming HF.",
            exists=True,
        ),
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Ignore the cached artifact and re-ingest."),
    ] = False,
) -> None:
    """Stream FNSPID, link tickers, dedupe and write ``data/interim/news_raw.parquet``."""
    from spp.config import load_universe
    from spp.data.news_fnspid import (
        FNSPID_SHARDS,
        fetch_news,
        fixture_source,
        parquet_source,
        summarize,
    )
    from spp.data.normalize import SignalPolicy, alias_index_from_universe

    c = _parent(ctx)
    root = PROJECT_ROOT
    start_d = _parse_date(start, c.config.dates.train_start, "start")
    end_d = _parse_date(end, c.config.dates.test_end, "end")
    if start_d > end_d:
        typer.secho(f"--start {start_d} is after --end {end_d}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=DATA_EXIT)

    universe = load_universe(_resolve(root, c.config.universe))
    index = alias_index_from_universe(universe)
    policy = SignalPolicy.from_config(c.config.news.cutoff_et, c.config.news.strict_dates)
    shards = tuple(shard) if shard else FNSPID_SHARDS
    source = None
    if from_parquet is not None:
        source = parquet_source(from_parquet)
        shards = (str(from_parquet),)
    elif fixture is not None:
        source = fixture_source(fixture)

    result = fetch_news(
        start_d,
        end_d,
        _resolve(root, c.config.paths.data) / "interim",
        _resolve(root, c.config.paths.reports),
        index,
        policy,
        shards=shards,
        source=source,
        refresh=refresh,
    )
    summary = summarize(result.frame, result.stats, result.coverage)
    typer.echo(f"artifact: {result.path} ({len(result.frame)} rows, cached={result.cached})")
    typer.echo(
        "funnel:   scanned {scanned} -> bad_date {bad_date} -> out_of_range {out_of_range} "
        "-> non_universe {non_universe} -> short {short_or_boilerplate} / no_alias {no_alias} "
        "/ ambiguous {ambiguous_alias} -> exact {exact_dupe} / near {near_dupe}"
        " -> kept {kept}".format(**summary)
    )
    if not result.coverage.empty:
        typer.echo(
            "coverage: median articles/ticker-day "
            f"{summary.get('median_per_day_weighted', float('nan')):.2f}, "
            f"min median {summary.get('min_median_per_day', float('nan')):.2f}, "
            f"max empty share {summary.get('max_empty_share', float('nan')):.3f}"
        )
    top = list((summary.get("publishers") or {}).items())[:5]
    typer.echo(f"top publishers: {top}")
    if result.violations:
        for violation in result.violations:
            typer.secho(f"VALIDATION: {violation}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=DATA_EXIT)


@score_app.command("calibrate")
def score_calibrate_command(
    ctx: typer.Context,
    backend: Annotated[
        str,
        typer.Option("--backend", help="finbert | openjev | lexicon | all (default: all)."),
    ] = "all",
    max_rows: Annotated[
        int,
        typer.Option("--max-rows", help="Cap on phrasebank rows (stratified); 0 = use all."),
    ] = 1000,
    openjev_rows: Annotated[
        int,
        typer.Option("--openjev-rows", help="Row cap for open-jev specifically (CPU-bound)."),
    ] = 120,
    no_write: Annotated[
        bool,
        typer.Option("--no-write", help="Print metrics but do not write reports/calibration/*."),
    ] = False,
) -> None:
    """Measure the sentiment backends on the financial phrasebank and rank them."""
    from spp.sentiment.base import BackendError
    from spp.sentiment.calibrate import load_phrasebank, run_backend_calibration
    from spp.sentiment.calibrate_report import render_decision, render_summary, select_backend

    c = _parent(ctx)
    root = PROJECT_ROOT
    cfg = c.config.sentiment
    wanted = ["finbert", "openjev", "lexicon"] if backend == "all" else [backend]
    unknown = set(wanted) - {"finbert", "openjev", "lexicon"}
    if unknown:
        typer.secho(f"unknown backend(s): {sorted(unknown)}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=NOT_IMPLEMENTED_EXIT)

    out_dir = _resolve(root, c.config.paths.reports) / "calibration"
    cache_dir = _resolve(root, c.config.paths.data) / "cache" / "phrasebank"
    texts, labels = load_phrasebank(cache_dir)
    common_cap = None if max_rows <= 0 else max_rows

    results = []
    for name in wanted:
        if name == "finbert":
            from spp.sentiment.finbert import FinBertBackend

            be = FinBertBackend(
                model_id=cfg.finbert_model,
                max_length=cfg.max_length,
                batch_size=cfg.batch_size,
            )
            cap = common_cap
        elif name == "openjev":
            from spp.sentiment.openjev import OpenJevBackend

            be = OpenJevBackend(model_id=cfg.openjev_model)
            cap = min(common_cap or openjev_rows, openjev_rows)
        else:
            from spp.sentiment.lexicon import LexiconBackend

            be = LexiconBackend()
            cap = common_cap

        typer.echo(f"calibrating {name} ...")
        try:
            res = run_backend_calibration(
                be,
                texts,
                labels,
                seed=c.seed,
                max_rows=cap,
                batch_size=cfg.batch_size,
                out_dir=None if no_write else out_dir,
            )
        except BackendError as exc:
            typer.secho(f"{name}: FAILED — {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=DATA_EXIT) from exc
        results.append(res)
        typer.echo(
            f"  {name}: acc={res.accuracy:.4f} macroF1={res.macro_f1:.4f} "
            f"ECE={res.ece:.4f} (T={res.temperature:.3f}) {res.throughput:.1f} items/s"
        )

    decision = select_backend(results, cfg.min_macro_f1)
    if not no_write:
        payload = {
            "summary": render_summary(
                results,
                gate=cfg.min_macro_f1,
                seed=c.seed,
                max_rows=common_cap,
                phrasebank_repo="takala/financial_phrasebank",
                phrasebank_config="sentences_75agree",
            ),
            "decision": render_decision(
                results, decision, seed=c.seed, max_rows=common_cap, gate=cfg.min_macro_f1
            ),
        }
        (out_dir / "summary.md").write_text(payload["summary"], encoding="utf-8")
        (out_dir / "decision.md").write_text(payload["decision"], encoding="utf-8")
        typer.echo(f"reports written to {out_dir}")

    typer.echo(f"\nprimary backend (rule, gate {cfg.min_macro_f1:.2f}): {decision['primary']}")
    typer.echo(decision["reason"])
    typer.echo("note: configs/base.yaml is not auto-edited — apply the decision by hand.")


@score_app.command("corpus")
def score_corpus_command(
    ctx: typer.Context,
    backend: Annotated[
        str,
        typer.Option("--backend", help="finbert | openjev | lexicon (default: from config)."),
    ] = "",
    shards: Annotated[
        str,
        typer.Option("--shards", help="Shard subset, e.g. '0-63' or '0,5,9'. Default: all."),
    ] = "",
    force: Annotated[
        bool,
        typer.Option("--force", help="Ignore cached shards and recompute the selected range."),
    ] = False,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Score only the first N unique texts (smoke runs). 0 = all."),
    ] = 0,
    no_write: Annotated[
        bool,
        typer.Option("--no-write", help="Score shards but do not write the joined artifact."),
    ] = False,
    max_articles: Annotated[
        int,
        typer.Option(
            "--max-articles",
            help=(
                "Override sentiment.openjev_max_articles for open-jev. The config cap (8000) "
                "is smaller than the 2023 row count, so the stratified sample degenerates to "
                "'2023 only' and the join fails validation; pass 0 to score the whole corpus."
            ),
        ),
    ] = -1,
) -> None:
    """Score `news_raw.parquet` into sharded caches and `article_sentiment.parquet`."""
    from spp.sentiment.base import BackendError
    from spp.sentiment.score_corpus import (
        NUM_SHARDS,
        assemble_article_sentiment,
        build_backend,
        build_openjev_sample,
        load_news,
        load_temperature,
        score_corpus,
        validate_output,
        write_openjev_sample,
    )

    c = _parent(ctx)
    root = PROJECT_ROOT
    cfg = c.config.sentiment
    name = backend or cfg.backend
    if name not in {"finbert", "openjev", "lexicon"}:
        typer.secho(f"unknown backend: {name!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=NOT_IMPLEMENTED_EXIT)

    data_root = _resolve(root, c.config.paths.data)
    reports_dir = _resolve(root, c.config.paths.reports)
    news_path = data_root / "interim" / "news_raw.parquet"
    if not news_path.is_file():
        typer.secho(f"missing {news_path}; run `fetch news` first", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=DATA_EXIT)

    try:
        shard_ids = _parse_shards(shards, NUM_SHARDS)
        news = load_news(news_path)
        temperature, cal_version = load_temperature(reports_dir, name)
        be = build_backend(name, config=c.config)
        if cal_version is not None:
            # A calibration run under a different HF revision describes a different model.
            base = cal_version.split("+", 1)[0]
            if not str(be.model_version).startswith(base):
                typer.secho(
                    f"WARNING: calibrated model_version {cal_version} does not match "
                    f"{be.model_version}; re-run `score calibrate --backend {name}`.",
                    fg=typer.colors.YELLOW,
                    err=True,
                )
        sample_ids = None
        sample_path = reports_dir / "openjev_sample.parquet"
        if name == "openjev":
            cap = cfg.openjev_max_articles if max_articles < 0 else max_articles
            sample = build_openjev_sample(
                news, max_articles=cap or len(news), seed=c.seed
            )
            write_openjev_sample(sample, sample_path, backend=name, seed=c.seed)
            sample_ids = set(sample["text_hash"].tolist())
            typer.echo(
                f"open-jev sample: {len(sample)} articles over {sample['ticker'].nunique()} "
                f"tickers (cap {cap}) -> {sample_path}"
            )
        if limit > 0:
            keep = news["text_hash"].drop_duplicates().head(limit)
            news = news.loc[news["text_hash"].isin(set(keep))]
    except BackendError as exc:
        typer.secho(f"{name}: FAILED — {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=DATA_EXIT) from exc

    out_path = (
        None
        if no_write
        else data_root / "processed" / f"article_sentiment_{name}.parquet"
    )
    shards_label = len(shard_ids) if shard_ids is not None else "all"
    typer.echo(
        f"scoring {len(news)} rows ({news['text_hash'].nunique()} unique texts) with {name} "
        f"(T={temperature:.4f}, shards={shards_label}) ..."
    )
    try:
        summary = score_corpus(
            news,
            name,
            cache_dir=data_root / "cache" / "sentiment",
            reports_dir=reports_dir,
            temperature=temperature,
            batch_size=cfg.batch_size,
            shards=shard_ids,
            force=force,
            out_path=out_path,
            backend=be,
            config=c.config,
            sample_ids=sample_ids,
        )
    except BackendError as exc:
        typer.secho(f"{name}: FAILED — {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=DATA_EXIT) from exc

    typer.echo(
        f"  scored={summary.scored_texts} reused={summary.reused_texts} "
        f"shards={summary.shards_written} written/{summary.shards_reused} reused "
        f"{summary.items_per_sec:.1f} items/s in {summary.elapsed_sec:.1f}s"
    )
    if summary.interrupted:
        typer.secho("interrupted: shards are durable, rerun to resume", fg=typer.colors.YELLOW)
        raise typer.Exit(code=130)

    if out_path is not None:
        assembled = assemble_article_sentiment(
            news=news,
            cache_dir=data_root / "cache" / "sentiment",
            backend=name,
            model_version=summary.model_version,
        )
        problems = validate_output(assembled, news)
        if problems:
            for problem in problems:
                typer.secho(f"VALIDATION: {problem}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=DATA_EXIT)
        _print_score_summary(summary, assembled, out_path)


def _parse_shards(spec: str, total: int) -> list[int] | None:
    """Parse ``'0-63'`` / ``'0,5,9'`` / ``''`` (all) into sorted shard ids."""
    spec = (spec or "").strip()
    if not spec:
        return None
    ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"bad shard range {part!r}")
            ids.update(range(lo, hi + 1))
        else:
            ids.add(int(part))
    bad = sorted(i for i in ids if i < 0 or i >= total)
    if bad:
        raise ValueError(f"shard ids outside 0..{total - 1}: {bad}")
    return sorted(ids)


def _print_score_summary(summary: Any, frame: Any, out_path: Path) -> None:
    skipped = int(frame["skipped"].sum())
    total = len(frame)
    typer.echo(f"wrote {out_path} ({total} rows, {total - skipped} scored, {skipped} skipped)")
    for reason, count in sorted(summary.skipped_by_reason.items()):
        typer.echo(f"  skipped[{reason}] = {count}")


@features_app.command("build")
def features_build_command(
    ctx: typer.Context,
    start: Annotated[
        str | None,
        typer.Option("--start", help="First session, YYYY-MM-DD (default: train_start)."),
    ] = None,
    end: Annotated[
        str | None, typer.Option("--end", help="Last session, YYYY-MM-DD (default: test_end).")
    ] = None,
    check_leakage: Annotated[
        bool,
        typer.Option("--check-leakage", help="Run all six leakage checks; exit 1 on failure."),
    ] = False,
    no_write: Annotated[
        bool, typer.Option("--no-write", help="Build in memory but do not write artifacts.")
    ] = False,
    sentiment_path: Annotated[
        Path | None,
        typer.Option("--sentiment", help="Override data/processed/article_sentiment.parquet."),
    ] = None,
) -> None:
    """Build ``data/processed/features.parquet`` from the price cache and article sentiment."""
    from spp.data.prices import load_prices
    from spp.data.store import read_parquet
    from spp.features.assemble import build_feature_matrix, feature_columns
    from spp.features.leakage import run_all_checks, write_report

    c = _parent(ctx)
    root = PROJECT_ROOT
    data_root = _resolve(root, c.config.paths.data)
    reports_dir = _resolve(root, c.config.paths.reports)
    start_d = _parse_date(start, c.config.dates.train_start, "start")
    end_d = _parse_date(end, c.config.dates.test_end, "end")

    prices = load_prices(data_root / "cache", start_d, end_d)
    if prices.empty:
        typer.secho("no cached prices; run `spp fetch prices` first", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=DATA_EXIT)
    backend = c.config.sentiment.backend
    sent_file = sentiment_path or (data_root / "processed" / f"article_sentiment_{backend}.parquet")
    sentiment = read_parquet(sent_file) if sent_file.is_file() else None
    if sentiment is None:
        typer.secho(
            f"WARNING: {sent_file} missing; building with sentiment columns all-NaN",
            fg=typer.colors.YELLOW,
            err=True,
        )
    news_file = data_root / "interim" / "news_raw.parquet"
    news = read_parquet(news_file) if news_file.is_file() else None

    try:
        result = build_feature_matrix(
            prices,
            sentiment=sentiment,
            news=news,
            config=c.config,
            data_dir=data_root,
            start=start_d,
            end=end_d,
            write=not no_write,
        )
    except Exception as exc:
        typer.secho(f"feature assembly failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=DATA_EXIT) from exc

    matrix = result.features
    typer.echo(
        f"features: {len(matrix)} rows, {matrix['ticker'].nunique()} tickers, "
        f"{result.schema['n_features']} features, "
        f"range {matrix['date'].min().date()}..{matrix['date'].max().date()}"
    )
    counts = matrix["split"].value_counts()
    typer.echo(
        "splits: " + ", ".join(f"{k}={int(counts.get(k, 0))}" for k in ("train", "valid", "test"))
    )
    missing_y = int(matrix.loc[matrix["split"].isin(["train", "valid"]), "y"].isna().sum())
    typer.echo(f"y NaN in train/valid: {missing_y}")
    for name, path in result.paths.items():
        typer.echo(f"  {name}: {path}")

    if not check_leakage:
        return
    cols = feature_columns(include_sentiment=True)
    train_mask = (matrix["split"] == "train").to_numpy()
    report = run_all_checks(
        features=matrix,
        feature_columns=cols,
        panel=result.panel,
        inventory=result.inventory,
        articles=result.articles,
        labels=result.labels,
        adj_close=result.adj_close,
        train_mask=train_mask,
        seed=c.seed,
    )
    path = write_report(report, reports_dir / "leakage_checks.json")
    for res in report.results:
        flag = "PASS" if res.passed else "FAIL"
        colour = typer.colors.GREEN if res.passed else typer.colors.RED
        typer.secho(f"  check {res.check_id} {res.name}: {flag}", fg=colour)
        if not res.passed:
            for violation in res.violations[:5]:
                typer.secho(f"      {violation}", fg=typer.colors.RED, err=True)
    typer.echo(f"leakage report: {path}")
    if not report.passed:
        raise typer.Exit(code=DATA_EXIT)


@train_app.command("baselines")
def train_baselines_command(
    ctx: typer.Context,
    features: Annotated[
        Path | None,
        typer.Option(
            "--features", help="Matrix parquet (default: data/processed/features.parquet)."
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Report path (default: reports/baselines.json)."),
    ] = None,
    n_boot: Annotated[
        int, typer.Option("--n-boot", help="Bootstrap draws for the IC CI (default 1000).")
    ] = 1000,
) -> None:
    """Fit B0–B3 on train (2019–2021), evaluate on valid (2022) only.

    Never reads 2023: the loader drops ``split == "test"`` rows and asserts the
    valid slice stays inside 2022.
    """
    import time

    from spp.data.store import read_parquet, write_parquet_atomic
    from spp.models.baselines import (
        compute_baselines,
        predict_baselines,
        series_to_frame,
        write_baselines_report,
    )

    c = _parent(ctx)
    root = PROJECT_ROOT
    data_root = _resolve(root, c.config.paths.data)
    reports_dir = _resolve(root, c.config.paths.reports)
    matrix_path = features or (data_root / "processed" / "features.parquet")
    if not matrix_path.is_file():
        typer.secho(
            f"missing {matrix_path}; run `features build` first", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=DATA_EXIT)

    started = time.perf_counter()
    frame = read_parquet(matrix_path)
    n_test = int((frame["split"].astype(str) == "test").sum())
    frames = predict_baselines(frame, train_end=c.config.dates.train_end)
    results = compute_baselines(frames, n_deciles=10, n_boot=n_boot)
    report = write_baselines_report(results, out or (reports_dir / "baselines.json"))
    series_frame = series_to_frame(results)
    if not series_frame.empty:
        series_frame["date"] = pd.to_datetime(series_frame["date"])
        write_parquet_atomic(
            series_frame, reports_dir / "baselines_ic_series.parquet", date32=("date",)
        )
    elapsed = time.perf_counter() - started

    typer.echo(f"matrix: {matrix_path} ({len(frame)} rows, {n_test} test rows dropped)")
    b0 = results["B0"]
    typer.echo(f"split: valid_2022, n_dates={b0['n_dates']} n_rows={b0['n_rows']}")
    typer.echo(
        f"{'code':<7} {'ic':>9} {'t_nw':>7} {'ic95':>17} {'mae':>9} {'rmse':>9}"
        f" {'dir':>7} {'auc':>7}"
    )

    def _fmt(block: dict, key: str, width: int = 9, digits: int = 4) -> str:
        value = block.get(key)
        return f"{value:>{width}.{digits}f}" if value is not None else f"{'nan':>{width}}"

    for code in ("B0", "B1", "B2", "B3_mom", "B3_rev"):
        block = results[code]
        typer.echo(
            f"{code:<7} {_fmt(block, 'ic')} {_fmt(block, 'ic_t_nw', 7, 2)} "
            f"[{_fmt(block, 'ic_ci95_lo', 7, 4)},{_fmt(block, 'ic_ci95_hi', 7, 4)}] "
            f"{_fmt(block, 'mae')} {_fmt(block, 'rmse')} {_fmt(block, 'dir_acc', 7, 3)} "
            f"{_fmt(block, 'auc', 7, 3)}"
        )
    for code, block in results["dm"].items():
        typer.echo(f"DM {code} vs B0: dm={block['dm_vs_b0']:.3f} p={block['p_vs_b0']:.3g}")
    typer.echo(f"report: {report}  ({elapsed:.1f}s)")


@train_app.command("cv")
def train_cv_command(
    ctx: typer.Context,
    features: Annotated[
        Path | None,
        typer.Option(
            "--features", help="Matrix parquet (default: data/processed/features.parquet)."
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Report path (default: reports/cv_folds.json)."),
    ] = None,
) -> None:
    """Build purged, embargoed walk-forward folds over 2019-2022; never touches 2023."""
    from spp.data.store import read_parquet
    from spp.features.calendar import nyse_sessions
    from spp.models.cv import (
        CV_WINDOW_END,
        CV_WINDOW_START,
        CvCfg,
        assert_no_overlap,
        fold_row_counts,
        make_folds,
        summarize_folds,
        write_folds_report,
    )

    c = _parent(ctx)
    root = PROJECT_ROOT
    data_root = _resolve(root, c.config.paths.data)
    reports_dir = _resolve(root, c.config.paths.reports)
    matrix_path = features or (data_root / "processed" / "features.parquet")
    if not matrix_path.is_file():
        typer.secho(
            f"missing {matrix_path}; run `features build` first", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=DATA_EXIT)

    frame = read_parquet(matrix_path)
    n_test = int((frame["split"].astype(str) == "test").sum())
    spine = nyse_sessions(CV_WINDOW_START, CV_WINDOW_END)
    cv = c.config.cv
    cfg = CvCfg(
        n_folds=cv.n_folds,
        test_months=cv.test_months,
        embargo_days=cv.embargo_days,
        min_train_months=getattr(cv, "min_train_months", 12),
    )
    folds = make_folds(spine, cfg, spine)
    assert_no_overlap(folds, spine)

    counts = fold_row_counts(frame, folds)
    payload = summarize_folds(folds, spine)
    payload["row_counts"] = counts
    payload["rows_dropped_test"] = n_test
    path = write_folds_report(payload, out or (reports_dir / "cv_folds.json"))

    typer.echo(f"matrix: {matrix_path} ({len(frame)} rows, {n_test} test rows excluded)")
    typer.echo(
        f"window: {payload['window']['start']} .. {payload['window']['end']} "
        f"({payload['window_sessions']} sessions), folds={len(folds)}"
    )
    typer.echo(
        f"{'id':<3} {'train_end':<12} {'valid':<24} {'test':<24} "
        f"{'emb':>4} {'train':>7} {'valid':>7} {'test':>6}"
    )
    for row, count in zip(payload["folds"], counts, strict=True):
        typer.echo(
            f"{row['fold_id']:<3} {row['train_end']:<12} "
            f"{row['valid_start']}..{row['valid_end']:<11} "
            f"{row['test_start']}..{row['test_end']:<11} "
            f"{row['embargo_sessions']:>4} {count['train_rows']:>7} "
            f"{count['valid_rows']:>7} {count['test_rows']:>6}"
        )
    typer.echo(f"test coverage: {payload['test_session_share']:.1%} of the CV window")
    typer.echo(f"report: {path}")


@train_app.command("gbdt")
def train_gbdt_command(
    ctx: typer.Context,
    features: Annotated[
        Path | None,
        typer.Option(
            "--features", help="Matrix parquet (default: data/processed/features.parquet)."
        ),
    ] = None,
    seeds: Annotated[
        str,
        typer.Option("--seeds", help="Comma separated seeds (default: 1,2,3)."),
    ] = "1,2,3",
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Reuse a run directory instead of hashing the inputs."),
    ] = None,
    max_folds: Annotated[
        int | None,
        typer.Option("--max-folds", help="Smoke test: train only the first N folds."),
    ] = None,
) -> None:
    """Train the primary LightGBM on purged folds; 2023 stays held out."""
    from spp.models.gbdt import GBDTError, run_gbdt_cv

    c = _parent(ctx)
    root = PROJECT_ROOT
    data_root = _resolve(root, c.config.paths.data)
    matrix_path = features or (data_root / "processed" / "features.parquet")
    if not matrix_path.is_file():
        typer.secho(
            f"missing {matrix_path}; run `features build` first", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=DATA_EXIT)
    try:
        seed_list = [int(part) for part in seeds.split(",") if part.strip()]
    except ValueError as exc:
        typer.secho(f"bad --seeds: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=DATA_EXIT) from exc
    if not seed_list:
        typer.secho("--seeds is empty", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=DATA_EXIT)

    typer.echo(f"matrix: {matrix_path}")
    typer.echo(f"seeds: {seed_list}, max_folds: {max_folds or 'all'}")
    try:
        result = run_gbdt_cv(
            c.config,
            features_path=matrix_path,
            seeds=seed_list,
            run_id=run_id,
            project_root=root,
            max_folds=max_folds,
        )
    except GBDTError as exc:
        typer.secho(f"training failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=DATA_EXIT) from exc

    metrics = result.metrics
    aggregate = metrics.get("aggregate", {}) or {}
    pooled = metrics.get("pooled", {}) or {}
    typer.echo(f"run: {result.run_id} -> {result.run_dir}")
    typer.echo(
        f"{'fold':<5} {'valid':<24} {'ic':>8} {'mae':>8} {'dir_acc':>8} {'best_iter':>9}"
    )
    for row in metrics.get("folds", []):
        valid = row.get("valid") or row.get("validation") or ""
        if isinstance(valid, dict):
            valid = f"{valid.get('start', '?')}..{valid.get('end', '?')}"
        typer.echo(
            f"{row.get('fold', row.get('fold_id', '?')):<5} {str(valid):<24} "
            f"{float(row.get('ic', float('nan'))):>8.4f} "
            f"{float(row.get('mae', float('nan'))):>8.5f} "
            f"{float(row.get('dir_acc', float('nan'))):>8.3f} "
            f"{row.get('best_iteration', row.get('best_iter')) or '?':>9}"
        )
    ic_block = aggregate.get("ic", {}) or {}
    ic_mean = ic_block.get("mean", float("nan"))
    ic_std = ic_block.get("std", float("nan"))
    n_folds = int(ic_block.get("n", 0) or 0)
    # 95% CI across folds; with a single fold the interval is undefined, so report NaN.
    half = 1.96 * ic_std / (n_folds**0.5) if n_folds > 1 and ic_std == ic_std else float("nan")
    ic_lo = ic_mean - half if half == half else float("nan")
    ic_hi = ic_mean + half if half == half else float("nan")
    typer.echo(
        f"aggregate: ic={float(ic_mean):.4f} ci=[{float(ic_lo):.4f}, {float(ic_hi):.4f}] "
        f"pooled_ic={float(pooled.get('ic', float('nan'))):.4f} "
        f"elapsed={result.elapsed_s:.1f}s"
    )
    dm = metrics.get("dm_vs_zero", {}) or {}
    typer.echo(f"DM vs zero: dm={dm.get('dm')} p={dm.get('p')}")
    typer.echo(f"importance: {result.run_dir / 'feature_importance.csv'}")
    typer.echo("held out: 2023 untouched")


@train_app.command("test2023")
def train_test2023(
    ctx: typer.Context,
    features: Annotated[
        Path | None,
        typer.Option("--features", help="Feature matrix to score."),
    ] = None,
    seeds: Annotated[
        str | None,
        typer.Option("--seeds", help="Comma separated seeds; defaults to config.model.seeds."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Ignore an existing lock; only for a deliberate rerun."),
    ] = False,
) -> None:
    """Single-shot 2023 holdout. Refuses to run twice unless --force is given."""
    from spp.models.gbdt import GBDTError
    from spp.models.holdout import run_holdout

    c = _parent(ctx)
    seed_list = [int(s) for s in seeds.split(",")] if seeds else None
    try:
        payload = run_holdout(
            c.config,
            features_path=features,
            seeds=seed_list,
            project_root=PROJECT_ROOT,
            force=force,
        )
    except GBDTError as exc:
        typer.secho(f"holdout refused: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=DATA_EXIT) from exc

    m = payload["metrics"]
    typer.echo(
        f"2023 holdout: ic={m['ic']:.4f} ci=[{m['ic_ci95_lo']:.4f}, {m['ic_ci95_hi']:.4f}] "
        f"dir_acc={m['dir_acc']:.4f} dm_p={payload['dm_vs_zero']['p']:.4f}"
    )
    typer.echo(f"verdict: {payload['verdict']}")
    for name, ok in payload["criteria"].items():
        typer.echo(f"  {name}: {'PASS' if ok else 'FAIL'}")
    typer.echo(f"report: {Path(c.config.paths.reports) / 'test_2023.md'}")


@fetch_app.command("live")
def fetch_live_command(
    ctx: typer.Context,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Fetch, dedupe and print; write no artifacts."),
    ] = False,
    since: Annotated[
        str | None,
        typer.Option("--since", help="Lower bound (ISO-8601); default now-24h."),
    ] = None,
    backfill_days: Annotated[
        int | None,
        typer.Option(
            "--backfill-days", help="API backfill window in calendar days (Finnhub+GDELT)."
        ),
    ] = None,
    feed: Annotated[
        list[str] | None,
        typer.Option(
            "--feed",
            help="Feed id or URL (repeatable); default: cnbc, marketwatch, yahoo, investing.",
        ),
    ] = None,
    no_gdelt: Annotated[bool, typer.Option("--no-gdelt", help="Skip GDELT.")] = False,
    no_finnhub: Annotated[bool, typer.Option("--no-finnhub", help="Skip Finnhub.")] = False,
    cross_check: Annotated[
        bool,
        typer.Option("--cross-check", help="Also query Alpha Vantage (25 req/day)."),
    ] = False,
    ticker: Annotated[
        list[str] | None,
        typer.Option("--ticker", help="Restrict to these tickers (repeatable); default: universe."),
    ] = None,
) -> None:
    """Collect live news from RSS/Finnhub/GDELT, dedupe, write the day file + health CSV.

    ``--dry-run`` performs the same fetch/dedupe/validation but only prints the per-source
    counts and the health table; nothing is written.
    """
    from datetime import timedelta

    from spp.config import load_universe
    from spp.data.news_live import (
        ALPHA_VANTAGE_ENV,
        FINNHUB_ENV,
        backfill,
        collect_live,
        health_path,
        write_daily,
        write_health,
    )
    from spp.data.normalize import SignalPolicy, alias_index_from_universe

    c = _parent(ctx)
    root = PROJECT_ROOT
    universe = load_universe(_resolve(root, c.config.universe))
    tickers = [t.upper() for t in (ticker or list(universe))]
    unknown = sorted(set(tickers) - set(universe))
    if unknown:
        typer.secho(f"tickers outside the universe: {unknown}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=DATA_EXIT)
    index = alias_index_from_universe(universe)
    policy = SignalPolicy.from_config(c.config.news.cutoff_et, c.config.news.strict_dates)
    now = datetime.now(UTC)
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError as exc:
            typer.secho(f"--since must be ISO-8601, got {since!r}", fg=typer.colors.RED, err=True)
            raise typer.BadParameter(str(exc), param_hint="--since") from exc
        if since_dt.tzinfo is None:
            since_dt = since_dt.replace(tzinfo=UTC)
    else:
        since_dt = now - timedelta(hours=24)

    feeds_arg: list[str] | None = None
    if feed:
        feeds_arg = [f if "://" in f else f for f in feed]

    if backfill_days is not None:
        result = backfill(
            tickers,
            backfill_days,
            index=index,
            finnhub_key=os.environ.get(FINNHUB_ENV),
            policy=policy,
            until=now,
            include_gdelt=not no_gdelt,
        )
    else:
        result = collect_live(
            tickers,
            index,
            since_dt,
            until=now,
            feeds=feeds_arg if feeds_arg is not None else None,
            policy=policy,
            include_finnhub=not no_finnhub,
            include_gdelt=not no_gdelt,
            include_alpha_vantage=cross_check,
            finnhub_key=os.environ.get(FINNHUB_ENV),
            alpha_vantage_key=os.environ.get(ALPHA_VANTAGE_ENV),
        )

    typer.echo(f"rows after dedup: {len(result.frame)}  dedupe: {result.dedupe}")
    for row in result.health:
        typer.echo(
            f"  {row.source:<18} items={row.n_items:<5} kept={row.n_after_dedup:<5} "
            f"max={row.max_published_at or '-'} {row.status}"
        )
    if result.violations:
        for violation in result.violations:
            typer.secho(f"VALIDATION: {violation}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=DATA_EXIT)
    if dry_run:
        typer.secho("dry-run: nothing written", fg=typer.colors.YELLOW)
        return

    health_file = write_health(result.health, health_path(_resolve(root, c.config.paths.reports)))
    typer.echo(f"health: {health_file}")
    if not result.frame.empty:
        days = sorted({ts.date() for ts in result.frame["date"]})
        live_dir = _resolve(root, c.config.paths.data) / "live"
        paths = [
            write_daily(
                result.frame[result.frame["date"] == pd.Timestamp(day)],
                live_dir,
                day,
                policy=policy,
            )
            for day in days
        ]
        typer.echo(f"written: {[str(p) for p in paths]}")
    else:
        typer.secho("no rows to write", fg=typer.colors.YELLOW)
    if not any(r.status == "OK" for r in result.health):
        # Every source failed or was empty: distinguish "feed broken" from "no news" above.
        errors = [r for r in result.health if r.status == "ERROR"]
        if errors:
            raise typer.Exit(code=DATA_EXIT)


# Remaining stubs: one registration per plan command.
for _app, _group, _names in (
    (train_app, "train", ("all",)),
    (live_app, "live", ("run",)),
    (report_app, "report", ("ablation", "live", "final")),
):
    for _name in _names:
        _app.command(_name)(_stub_command(f"{_group} {_name}"))

@app.command("backtest")
def backtest_command(
    ctx: typer.Context,
    predictions: Annotated[
        Path | None,
        typer.Option(
            "--predictions",
            help="Predictions parquet (default: data/processed/predictions.parquet).",
        ),
    ] = None,
    prices: Annotated[
        Path | None,
        typer.Option(
            "--prices",
            help="Price cache parquet (default: auto-detected in data/cache).",
        ),
    ] = None,
    execution: Annotated[
        str | None,
        typer.Option(
            "--execution",
            help="next_open_hold_close | close_to_close | next_open (default: config).",
        ),
    ] = None,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Artifact suffix (default: config execution).")
    ] = None,
    include_test: Annotated[
        bool,
        typer.Option(
            "--include-test",
            help=(
                "Evaluate the holdout (2023) predictions; otherwise the window stops at valid_end."
            ),
        ),
    ] = False,
    sector_cap: Annotated[
        float | None,
        typer.Option("--sector-cap", help="Max gross share per sector (e.g. 0.3)."),
    ] = None,
    n_quantiles: Annotated[
        int | None, typer.Option("--n-quantiles", help="Default: config backtest.n_quantiles.")
    ] = None,
    cost_bps: Annotated[
        float | None,
        typer.Option("--cost-bps", help="Override per-side cost in bps (sensitivity only)."),
    ] = None,
    compare: Annotated[
        bool,
        typer.Option("--compare", help="Also run close_to_close as a comparison variant."),
    ] = True,
    null_draws: Annotated[
        int,
        typer.Option(
            "--null-draws",
            help="Permutation draws for the null Sharpe (default 50).",
        ),
    ] = 50,
) -> None:
    """Long/short quintile backtest with costs; never reads 2023."""
    from spp.backtest.report import reproducible_stats, run_backtest_suite, write_artifacts
    from spp.data.store import read_parquet

    c = _parent(ctx)
    root = PROJECT_ROOT
    data_root = _resolve(root, c.config.paths.data)
    reports_dir = _resolve(root, c.config.paths.reports)
    pred_path = predictions or (data_root / "processed" / "predictions.parquet")
    if not pred_path.is_file():
        typer.secho(
            f"missing {pred_path}; run `train gbdt` first", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=DATA_EXIT)
    price_path = prices
    if price_path is None:
        candidates = sorted((data_root / "cache").glob("prices_*.parquet"))
        if not candidates:
            typer.secho(
                f"no prices cache under {data_root / 'cache'}; run `fetch prices`",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=DATA_EXIT)
        price_path = candidates[-1]
    if not price_path.is_file():
        typer.secho(f"missing {price_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=DATA_EXIT)

    mode = execution or c.config.backtest.execution
    if mode not in {"next_open_hold_close", "close_to_close", "next_open"}:
        typer.secho(f"unknown --execution {mode!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=DATA_EXIT)

    # Never touch 2023 unless the holdout predictions are being evaluated explicitly.
    cutoff = pd.Timestamp(
        c.config.dates.test_end if include_test else c.config.dates.valid_end
    )
    commission = c.config.backtest.commission_bps
    slippage = c.config.backtest.slippage_bps
    if cost_bps is not None:
        commission, slippage = cost_bps / 2.0, cost_bps / 2.0
    nq = n_quantiles or c.config.backtest.n_quantiles

    predictions_frame = read_parquet(pred_path)
    predictions_frame["date"] = pd.to_datetime(predictions_frame["date"])
    n_2023 = int((predictions_frame["date"] > cutoff).sum())
    if n_2023:
        typer.secho(
            f"WARNING: dropping {n_2023} prediction rows after {cutoff.date()} (2023 is held out)",
            fg=typer.colors.YELLOW,
            err=True,
        )
    price_frame = read_parquet(price_path)
    from spp.config import load_universe

    universe = load_universe(_resolve(root, c.config.universe))
    sector_map = {ticker: info.sector for ticker, info in universe.items()}

    modes = [mode]
    if compare and "close_to_close" not in modes:
        modes.append("close_to_close")

    typer.echo(f"predictions: {pred_path}")
    typer.echo(f"prices:      {price_path}")
    typer.echo(
        f"costs:       {commission:.1f}+{slippage:.1f} bps/side, borrow "
        f"{c.config.backtest.borrow_annual:.0%}/yr, {nq} quantiles"
    )
    reports = {}
    for current in modes:
        suffix = run_id or current
        if len(modes) > 1:
            suffix = f"{run_id or 'bt'}-{current}"
        report = run_backtest_suite(
            predictions_frame,
            price_frame,
            current,
            sector_map=sector_map,
            n_quantiles=nq,
            commission_bps=commission,
            slippage_bps=slippage,
            borrow_annual=c.config.backtest.borrow_annual,
            sector_cap=sector_cap,
            max_date=cutoff,
            run_id=suffix,
            null_draws=null_draws,
            seed=c.seed,
        )
        reports[current] = report
        m = report.result.summary
        k = report.ic_block.get("pooled", {}) or {}
        typer.echo(
            f"{current:<22} sharpe_gross={m['sharpe_gross']:>6.2f} "
            f"sharpe_net={m['sharpe_net']:>6.2f} "
            f"ann={m['ann_return']:>7.2%} dd={m['max_dd']:>7.2%} "
            f"IC={k.get('ic', float('nan')):>7.4f}"
        )

    config_snapshot = {
        "execution": mode,
        "n_quantiles": nq,
        "commission_bps": commission,
        "slippage_bps": slippage,
        "borrow_annual": c.config.backtest.borrow_annual,
        "sector_cap": sector_cap,
        "seed": c.seed,
        "valid_end": str(cutoff.date()),
    }
    written = write_artifacts(
        reports, reports_dir, reports_dir / "figures", config_snapshot=config_snapshot
    )
    primary = reports[mode]
    stats = reproducible_stats(reports_dir / "backtest_daily_returns.csv")
    payload = primary.as_dict()["metrics"]
    checks = {
        "identity_net=gross-cost-borrow": stats["identity_residual_max"] <= 1e-12,
        "dollar_neutral": stats["max_abs_net_weight"] < 1e-9,
        "sharpe_reproducible": abs(stats["sharpe_net"] - payload["sharpe_net"]) < 1e-9,
        "maxdd_reproducible": abs(stats["max_dd"] - payload["max_dd"]) < 1e-9,
    }
    for name, ok in checks.items():
        typer.echo(f"check {name}: {'PASS' if ok else 'FAIL'}")
    typer.echo(f"report: {written['primary:json'].parent / ('backtest_' + primary.run_id + '.md')}")
    typer.echo(f"verdict: {primary.headline['verdict']}  gates={primary.headline['gates']}")
    typer.echo("held out: 2023 untouched")
    if not all(checks.values()):
        raise typer.Exit(code=DATA_EXIT)


app.add_typer(fetch_app, name="fetch")
app.add_typer(score_app, name="score")
app.add_typer(features_app, name="features")
app.add_typer(train_app, name="train")
app.add_typer(live_app, name="live")
app.add_typer(report_app, name="report")


def run() -> None:  # pragma: no cover - console wrapper
    """Console-script entrypoint referenced by ``[project.scripts]``."""
    app()


if __name__ == "__main__":
    sys.exit(app())
