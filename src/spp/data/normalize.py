"""T3 — news normalisation: title cleaning, entity linking, dedupe, ``signal_date`` policy.

Every function here is pure and deterministic; :mod:`spp.data.news_fnspid` handles I/O
and hands batches to these helpers. The contract of ``data/interim/news_raw.parquet``
is defined in :mod:`spp.data.news_fnspid`.

Title cleaning (``03-news-history.md`` §2)
    HTML entities are unescaped, publisher suffixes at the *end* of the headline
    (`` - Reuters``, `` | MarketWatch``) are stripped, typographic quotes are folded to
    ASCII, whitespace is collapsed, the result is truncated to
    :data:`MAX_TITLE_CHARS`, and headlines shorter than :data:`MIN_TITLE_CHARS` or made
    only of boilerplate (``Stock Market News``, ``Top Stories``) are rejected.

Entity linking (§3)
    Aliases come from ``configs/universe_us_largecap.yaml``. Matching is
    case-insensitive on word boundaries; the bare ticker token is matched **only in upper
    case** (``AAPL`` matches, ``aapl`` inside *a word* does not). Every alias that is a
    suffix of a longer matching alias is discarded (``Apple`` vs ``Apple Hospitality``),
    and a headline matching aliases of two *different* tickers is dropped as ambiguous.
    A headline with no alias never enters the corpus.

Dedupe (§4)
    Exact duplicates: same ``(ticker, text_hash)`` → keep the earliest ``published_at``
    (tie-break: earlier ``date``, then publisher priority, then ``article_id``). Near
    duplicates: trigram Jaccard on ``title_norm`` above ``news.dedupe_jaccard`` inside one
    ``(ticker, date)`` group; the survivor is decided by publisher priority
    (Reuters > Bloomberg > WSJ > CNBC > others), then earliest timestamp, then
    ``article_id``. Syndicated copies across different tickers are *not* merged.

``signal_date`` (§5)
    ``published_at`` known: same day when the UTC timestamp converts to ET <= ``cutoff_et``
    (16:00), otherwise +1 calendar day. Date-only source rows: the source date itself
    (``strict_dates: false``) or +1 day (``strict_dates: true``). The resulting day is then
    rolled forward to the next NYSE trading day. The invariant ``signal_date >= date`` is
    asserted, never clamped.
"""

from __future__ import annotations

import hashlib
import html
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from typing import Final
from zoneinfo import ZoneInfo

import pandas as pd

from spp.logging import get_logger

__all__ = [
    "DropStats",
    "ENTITY_AMBIGUOUS",
    "ENTITY_NONE",
    "ENTITY_OK",
    "MAX_TITLE_CHARS",
    "MIN_TITLE_CHARS",
    "NEWS_COLUMNS",
    "PUBLISHER_PRIORITY",
    "SignalPolicy",
    "TRIGRAM_THRESHOLD",
    "assign_signal_date",
    "clean_title",
    "dedupe_news",
    "jaccard",
    "link_ticker",
    "normalize_title",
    "publisher_priority",
    "text_hash",
    "trigrams",
    "validate_news",
]

LOG = get_logger("data.normalize")

ET: Final = ZoneInfo("America/New_York")
UTC: Final = UTC

MIN_TITLE_CHARS: Final[int] = 20
MAX_TITLE_CHARS: Final[int] = 300
TRIGRAM_THRESHOLD: Final[float] = 0.9

ENTITY_OK: Final[str] = "ok"
ENTITY_NONE: Final[str] = "no_alias"
ENTITY_AMBIGUOUS: Final[str] = "ambiguous"

#: Plan §4: Reuters > Bloomberg > WSJ > CNBC > others. Lower rank wins.
PUBLISHER_PRIORITY: Final[tuple[tuple[str, str], ...]] = (
    ("reuters", "Reuters"),
    ("bloomberg", "Bloomberg"),
    ("wsj", "WSJ"),
    ("wall street journal", "WSJ"),
    ("dow jones", "WSJ"),
    ("cnbc", "CNBC"),
)
DEFAULT_PUBLISHER_RANK: Final[int] = len(PUBLISHER_PRIORITY)

NEWS_COLUMNS: Final[tuple[str, ...]] = (
    "article_id",
    "ticker",
    "published_at",
    "date",
    "signal_date",
    "title",
    "title_norm",
    "url",
    "publisher",
    "source",
    "text_hash",
)

# --------------------------------------------------------------------------- #
# Title cleaning
# --------------------------------------------------------------------------- #
_HTML_TAG = re.compile(r"<[^>]{0,40}>")
_WHITESPACE = re.compile(r"\s+")
#: ` - Reuters`, ` | MarketWatch`, ` -- CNBC.com` at the very end of a headline.
_PUBLISHER_SUFFIX = re.compile(
    r"\s+[-|–—]+\s*(?:reuters|bloomberg|wsj|the wall street journal|wall street journal|"
    r"marketwatch|cnbc(?:\s*\.com)?|barron'?s|forbes|associated press|ap|investor'?s business daily|"
    r"ibd|fox business|business insider|motley fool|the motley fool|seeking alpha|zacks|"
    r"benzinga|yahoo finance|nasdaq(?:\.com)?|streetinsider|thestreet|usa today|"
    r"financial times|ft\.com|fortune|time|newsweek|the new york times|new york times)\s*$",
    re.IGNORECASE,
)
_QUOTES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u200b": "",
        "\u200e": "",
        "\u200f": "",
        "\ufeff": "",
    }
)
#: Headlines that carry no information whatsoever (case-insensitive exact match).
BOILERPLATE_TITLES: Final[frozenset[str]] = frozenset(
    {
        "stock market news",
        "top stories",
        "market news",
        "latest news",
        "news",
        "breaking news",
        "stocks",
        "stock quotes",
        "business news",
    }
)


def clean_title(raw: object) -> str | None:
    """Return the cleaned headline, or ``None`` when the row must be dropped.

    Dropped: empty / non-string values, boilerplate-only headlines and anything shorter
    than :data:`MIN_TITLE_CHARS` characters after cleaning.
    """
    if raw is None:
        return None
    text = str(raw)
    if not text or text == "None":
        return None
    text = html.unescape(text)  # &amp; -> &, &#39; -> ', &quot; -> "
    text = html.unescape(text)  # FNSPID double-escapes in places (&amp;amp;)
    text = _HTML_TAG.sub(" ", text)
    text = text.translate(_QUOTES)
    text = unicodedata.normalize("NFKC", text)
    text = _WHITESPACE.sub(" ", text).strip()
    text = _PUBLISHER_SUFFIX.sub("", text).strip()
    text = _WHITESPACE.sub(" ", text).strip(" -|:;,.")
    if len(text) > MAX_TITLE_CHARS:
        cut = text[:MAX_TITLE_CHARS]
        space = cut.rfind(" ")
        text = (cut[:space] if space > MIN_TITLE_CHARS else cut).strip()
    if len(text) < MIN_TITLE_CHARS:
        return None
    if text.casefold() in BOILERPLATE_TITLES:
        return None
    return text


_NON_ALNUM = re.compile(r"[^0-9a-z&+]+")


def normalize_title(title: str) -> str:
    """Lowercase, punctuation-free, whitespace-collapsed form used for dedupe."""
    lowered = unicodedata.normalize("NFKD", title).casefold()
    lowered = lowered.replace("'", "").replace("’", "").replace("&", "")
    if localized := stripped_diacritics(lowered):
        lowered = localized
    return _NON_ALNUM.sub(" ", lowered).strip()


def stripped_diacritics(text: str) -> str:
    """Remove combining marks (``é`` -> ``e``) without touching base letters."""
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.combining(ch))


def text_hash(title_norm: str) -> str:
    return hashlib.sha1(title_norm.encode("utf-8")).hexdigest()


def article_id(publisher: str, url: str, title: str) -> str:
    """``sha1(publisher + '|' + url + '|' + title)`` — plan schema."""
    payload = "\x1f".join((publisher, url, title))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Entity linking
# --------------------------------------------------------------------------- #
_ALIAS_SPLIT = re.compile(r"[\s\-_/]+")


@dataclass(frozen=True)
class _AliasEntry:
    alias: str
    ticker: str
    pattern: re.Pattern[str]

    @property
    def n_tokens(self) -> int:
        return len(_ALIAS_SPLIT.split(self.alias))


class AliasIndex:
    """Compiled alias matcher for the ticker universe.

    ``aliases`` maps ``ticker -> aliases`` (as in the universe YAML). The bare ticker is
    always registered as an alias, matched case-sensitively; named aliases match
    case-insensitively on word boundaries.
    """

    def __init__(
        self,
        aliases: Mapping[str, Sequence[str]],
        exclude_phrases: Sequence[str] = (),
    ) -> None:
        entries: list[_AliasEntry] = []
        for ticker, names in sorted(aliases.items()):
            seen = {ticker.casefold()}
            entries.append(
                _AliasEntry(
                    alias=ticker.casefold(),
                    ticker=ticker,
                    pattern=_token_pattern(ticker, case_sensitive=True),
                )
            )
            for name in names:
                cleaned = name.strip()
                key = cleaned.casefold()
                if not cleaned or key in seen:
                    continue
                seen.add(key)
                entries.append(
                    _AliasEntry(
                        alias=key,
                        ticker=ticker,
                        pattern=_token_pattern(cleaned, case_sensitive=False),
                    )
                )
        # Longest alias first so "apple hospitality" is decided before "apple".
        self._entries = sorted(entries, key=lambda e: (-len(e.alias), e.ticker))
        self._tickers = tuple(sorted(aliases))
        self._exclusions = tuple(
            _token_pattern(phrase, case_sensitive=False)
            for phrase in exclude_phrases
            if phrase.strip()
        )

    @property
    def tickers(self) -> tuple[str, ...]:
        return self._tickers

    def aliases_for(self, ticker: str) -> tuple[str, ...]:
        """Named aliases registered for ``ticker``, longest first (bare ticker excluded)."""
        key = ticker.casefold()
        names = {
            e.alias
            for e in self._entries
            if e.ticker == ticker and e.alias != key
        }
        return tuple(sorted(names, key=lambda a: (-len(a), a)))

    def match(self, title: str) -> tuple[str | None, str]:
        """Return ``(ticker, status)``; status is :data:`ENTITY_OK`/``_NONE``/``_AMBIGUOUS``.

        Longest alias wins. Aliases that are token-substrings of a longer *matching* alias
        (``Apple`` inside ``Apple Hospitality``) are discarded first; if two different
        tickers still match, the headline is ambiguous and dropped. Aliases masked by an
        ``exclude_phrase`` never match: a headline about ``Apple Hospitality`` is reported
        as ambiguous, never as ``AAPL``.
        """
        hits: list[tuple[_AliasEntry, re.Match[str]]] = [
            (entry, match)
            for entry in self._entries
            if (match := entry.pattern.search(title)) is not None
        ]
        if not hits:
            return None, ENTITY_NONE
        blocked = [
            span
            for pattern in self._exclusions
            for span in (m.span() for m in pattern.finditer(title))
        ]
        if blocked:
            kept = [
                (entry, match)
                for entry, match in hits
                if not any(
                    start <= match.start() and match.end() <= end for start, end in blocked
                )
            ]
            if not kept:
                # The phrase names a different company (e.g. Apple Hospitality REIT).
                return None, ENTITY_AMBIGUOUS
            hits = kept
        matched = [entry for entry, _ in hits]
        survivors = [
            e
            for e in matched
            if not any(
                o is not e
                and len(o.alias) > len(e.alias)
                and _is_token_subset(e.alias, o.alias)
                for o in matched
            )
        ]
        tickers = {e.ticker for e in survivors}
        if len(tickers) == 1:
            (ticker,) = tickers
            return ticker, ENTITY_OK
        return None, ENTITY_AMBIGUOUS


def _is_token_subset(short: str, long: str) -> bool:
    """True when every token of ``short`` appears in ``long`` (both already case-folded)."""
    return set(_ALIAS_SPLIT.split(short)) <= set(_ALIAS_SPLIT.split(long))


def _token_pattern(alias: str, *, case_sensitive: bool) -> re.Pattern[str]:
    parts = [re.escape(p) for p in _ALIAS_SPLIT.split(alias.strip()) if p]
    body = r"[\s\-_/]+".join(parts)
    flags = 0 if case_sensitive else re.IGNORECASE
    # Word boundary is custom: \b does not work around punctuation such as "disney+".
    return re.compile(rf"(?<![0-9A-Za-z]){body}(?![0-9A-Za-z])", flags)


#: Phrases naming a *different* company that shares an alias with a universe ticker.
#: Any alias match fully inside such a phrase is discarded (see ``AliasIndex.match``).
DEFAULT_EXCLUDE_PHRASES: Final[tuple[str, ...]] = (
    "apple hospitality",
    "apple reit",
    "apple bank",
)


@lru_cache(maxsize=8)
def build_alias_index(
    aliases_key: tuple[tuple[str, tuple[str, ...]], ...],
    exclude_key: tuple[str, ...] = DEFAULT_EXCLUDE_PHRASES,
) -> AliasIndex:
    """Cached :class:`AliasIndex` from a hashable ``((ticker, (alias, ...)), ...)``."""
    return AliasIndex(dict(aliases_key), exclude_key)


def alias_index_from_universe(
    universe: Mapping[str, object],
    exclude_phrases: Sequence[str] = DEFAULT_EXCLUDE_PHRASES,
) -> AliasIndex:
    """Build the index from ``load_universe()`` (``{ticker: TickerInfo(aliases=...)}}``)."""
    key = tuple(
        sorted(
            (str(ticker), tuple(str(a) for a in getattr(info, "aliases", ())))
            for ticker, info in universe.items()
        )
    )
    return build_alias_index(key, tuple(exclude_phrases))


def load_alias_index(universe_path: str | None = None) -> AliasIndex:
    """Load the alias index straight from the universe YAML."""
    from spp.config import PROJECT_ROOT, load_universe, load_universe_exclusions

    path = universe_path or str(PROJECT_ROOT / "configs" / "universe_us_largecap.yaml")
    exclusions = load_universe_exclusions(path) or DEFAULT_EXCLUDE_PHRASES
    return alias_index_from_universe(load_universe(path), exclusions)


def link_ticker(title: str, index: AliasIndex) -> tuple[str | None, str]:
    """Public wrapper: ``(ticker, status)`` for one headline."""
    return index.match(title)


# --------------------------------------------------------------------------- #
# Trigrams / Jaccard
# --------------------------------------------------------------------------- #
def trigrams(text: str) -> frozenset[str]:
    """Character trigrams of the normalised title (padded so short titles still compare)."""
    padded = f"  {text}  "
    if len(padded) < 3:
        return frozenset()
    return frozenset(padded[i : i + 3] for i in range(len(padded) - 2))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# --------------------------------------------------------------------------- #
# Publisher priority
# --------------------------------------------------------------------------- #
def publisher_priority(publisher: str) -> int:
    """Rank of the publisher domain/name (0 = Reuters, higher = less preferred)."""
    key = publisher.strip().casefold()
    if not key:
        return DEFAULT_PUBLISHER_RANK
    for needle, _canonical in PUBLISHER_PRIORITY:
        if needle in key:
            return PUBLISHER_PRIORITY.index((needle, _canonical))
    return DEFAULT_PUBLISHER_RANK


# --------------------------------------------------------------------------- #
# signal_date
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SignalPolicy:
    """Leak-free ``signal_date`` policy from ``configs/base.yaml:news``."""

    cutoff_et: time = time(16, 0)
    strict_dates: bool = False

    @classmethod
    def from_config(cls, cutoff_et: str, strict_dates: bool) -> SignalPolicy:
        hour, minute = cutoff_et.split(":")
        return cls(time(int(hour), int(minute)), bool(strict_dates))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def signal_day(
    source_date: date,
    published_at: datetime | None,
    policy: SignalPolicy,
) -> date:
    """Raw ``signal_date`` before trading-calendar alignment (unit-testable)."""
    if published_at is not None:
        local = _as_utc(published_at).astimezone(ET)
        base = local.date()
        if local.timetz().replace(tzinfo=None) <= policy.cutoff_et and local.date() == base:
            return max(base, source_date)
        return max(base + timedelta(days=1), source_date)
    if policy.strict_dates:
        return source_date + timedelta(days=1)
    return source_date


def assign_signal_date(
    dates: Sequence[date],
    published: Sequence[datetime | None],
    policy: SignalPolicy,
    trading_days: Sequence[date] | None = None,
) -> list[date]:
    """Vectorised-over-rows ``signal_date`` incl. roll-forward to the next trading day."""
    out: list[date] = []
    cal = _TradingCalendar(trading_days) if trading_days is not None else None
    for source, stamp in zip(dates, published, strict=True):
        day = signal_day(source, stamp, policy)
        if cal is not None:
            day = cal.next_or_same(day)
        if day < source:  # pragma: no cover - guarded by max() above
            raise AssertionError(f"signal_date {day} precedes date {source}")
        out.append(day)
    return out


class _TradingCalendar:
    """Next-trading-day lookup backed by an explicit sorted day list."""

    def __init__(self, days: Sequence[date]) -> None:
        self._days = sorted(set(days))
        if not self._days:
            raise ValueError("trading_days must not be empty")

    def next_or_same(self, day: date) -> date:
        import bisect

        pos = bisect.bisect_left(self._days, day)
        if pos >= len(self._days):
            raise ValueError(
                f"{day} is after the last trading day in the calendar ({self._days[-1]}); "
                "extend the explicit calendar"
            )
        return self._days[pos]


def nyse_trading_days(start: date, end: date) -> list[date]:
    """NYSE session days in ``[start, end]`` (falls back to business days if unavailable)."""
    try:
        import pandas_market_calendars as mcal

        schedule = mcal.get_calendar("NYSE").schedule(
            start_date=start.isoformat(), end_date=(end + timedelta(days=1)).isoformat()
        )
        days = [stamp.date() for stamp in schedule.index]
    except Exception as exc:  # pragma: no cover - calendar package missing
        LOG.warning("NYSE calendar unavailable (%s); using business days", exc)
        days = [ts.date() for ts in pd.bdate_range(start, end)]
    return sorted(set(days))


# --------------------------------------------------------------------------- #
# Dedupe
# --------------------------------------------------------------------------- #
@dataclass
class DropStats:
    """Rows removed at every stage, for ``reports/news_coverage.csv``."""

    scanned: int = 0
    bad_date: int = 0
    out_of_range: int = 0
    non_universe: int = 0
    short_or_boilerplate: int = 0
    no_alias: int = 0
    alias_mismatch: int = 0
    ambiguous_alias: int = 0
    exact_dupe: int = 0
    near_dupe: int = 0
    kept: int = 0
    by_ticker: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, int]:
        return {k: v for k, v in self.__dict__.items() if k != "by_ticker"}

    def merge(self, other: DropStats) -> None:
        for key in self.as_dict():
            setattr(self, key, getattr(self, key) + getattr(other, key))
        for ticker, count in other.by_ticker.items():
            self.by_ticker[ticker] = self.by_ticker.get(ticker, 0) + count


def _sort_key(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic ordering: ``(date, ticker, published_at, article_id)`` with NaT last."""
    work = df.copy()
    work["_pub_missing"] = work["published_at"].isna()
    work = work.sort_values(
        ["date", "ticker", "_pub_missing", "published_at", "article_id"],
        kind="stable",
        na_position="last",
    )
    return work.drop(columns=["_pub_missing"])


def _priority_series(publishers: pd.Series) -> pd.Series:
    return publishers.map(lambda p: publisher_priority(str(p))).astype("int32")


def dedupe_news(df: pd.DataFrame, threshold: float = TRIGRAM_THRESHOLD) -> tuple[pd.DataFrame, DropStats]:
    """Exact + near-duplicate removal; returns the survivors and drop counters.

    Survivor of an exact duplicate inside ``(ticker, date)``: earliest ``published_at``
    (NaT loses), then publisher priority, then ``article_id``. Survivor of a near-duplicate
    group inside ``(ticker, date)``: best publisher priority, then earliest timestamp, then
    ``article_id`` — independent of input row order, so reruns are byte-identical.
    """
    stats = DropStats()
    if df.empty:
        return df.copy(), stats

    work = df.copy()
    work["_priority"] = _priority_series(work["publisher"])
    if "published_at" not in work.columns:
        work["published_at"] = pd.NaT
    work["_pub_missing"] = work["published_at"].isna()

    before = len(work)
    work = work.sort_values(
        ["ticker", "text_hash", "_pub_missing", "published_at", "_priority", "article_id"],
        kind="stable",
    ).drop_duplicates(subset=["ticker", "date", "text_hash"], keep="first")
    stats.exact_dupe = before - len(work)

    # Near-duplicates: greedy clustering inside (ticker, date) in priority order.
    keep_mask = pd.Series(False, index=work.index)
    for (_ticker, _day), group in work.groupby(["ticker", "date"], sort=True, observed=True):
        ordered = group.sort_values(
            ["_priority", "_pub_missing", "published_at", "article_id"], kind="stable"
        )
        survivors: list[frozenset[str]] = []
        survivor_index: list[object] = []
        for idx, title_norm in ordered["title_norm"].items():
            grams = trigrams(str(title_norm))
            if any(jaccard(grams, kept) > threshold for kept in survivors):
                continue
            survivors.append(grams)
            survivor_index.append(idx)
        keep_mask.loc[survivor_index] = True
    stats.near_dupe = int((~keep_mask).sum())
    work = work.loc[keep_mask]

    work = work.drop(columns=["_priority", "_pub_missing"])
    work = _sort_key(work).reset_index(drop=True)
    stats.kept = len(work)
    return work, stats


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_news(df: pd.DataFrame, tickers: Iterable[str] | None = None) -> list[str]:
    """Return invariant violations (empty list means the frame is clean)."""
    violations: list[str] = []
    missing = [c for c in NEWS_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if df.empty:
        return ["frame is empty"]

    frame = df
    dates = pd.to_datetime(frame["date"], errors="coerce")
    signals = pd.to_datetime(frame["signal_date"], errors="coerce")
    if bool(dates.isna().any()) or bool(signals.isna().any()):
        violations.append("null date or signal_date")
    else:
        broken = signals < dates
        if bool(broken.any()):
            sample = frame.loc[broken, ["ticker", "date", "signal_date"]].head(3)
            violations.append(
                f"signal_date < date in {int(broken.sum())} rows, e.g. "
                f"{sample.to_dict(orient='records')}"
            )
    if bool((frame["text_hash"].astype(str) != frame["title_norm"].map(text_hash)).any()):
        violations.append("text_hash does not match sha1(title_norm)")
    dupes = frame.duplicated(subset=["ticker", "date", "text_hash"]).sum()
    if dupes:
        violations.append(f"{int(dupes)} exact duplicates survived dedupe")
    if bool((frame["title"].astype(str).str.len() < MIN_TITLE_CHARS).any()):
        violations.append("title shorter than MIN_TITLE_CHARS survived cleaning")
    if tickers is not None:
        unknown = sorted(set(frame["ticker"].astype(str)) - set(tickers))
        if unknown:
            violations.append(f"tickers outside the universe: {unknown[:10]}")
    return violations


def near_dupe_pairs(df: pd.DataFrame, threshold: float = TRIGRAM_THRESHOLD) -> int:
    """Count surviving pairs with Jaccard > threshold inside ``(ticker, date)`` (must be 0)."""
    count = 0
    if df.empty:
        return 0
    for _key, group in df.groupby(["ticker", "date"], sort=False):
        if len(group) < 2:
            continue
        grams = [trigrams(str(t)) for t in group["title_norm"]]
        for i in range(len(grams)):
            for j in range(i + 1, len(grams)):
                if jaccard(grams[i], grams[j]) > threshold:
                    count += 1
    return count


def is_finite(value: float) -> bool:
    return isinstance(value, float) and math.isfinite(value)
