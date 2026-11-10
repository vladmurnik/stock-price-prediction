"""Deterministic headline taxonomy for structural crypto events.

The corpus is dominated by market commentary ("Bitcoin dips below $21K"), where the
expected move after the headline is a few basis points - the same order as the noise.
A small share of headlines describe a *structural* event instead: a listing, a
delisting, an exploit, an ETF decision, an insolvency. Those move prices by tens or
hundreds of basis points, so the round-trip cost stops being the binding constraint.

Every title gets at most one class (first match by priority) and a sign: the direction
the class is expected to push the named asset. The rules are deliberately narrow -
a false positive pollutes a class whose ``n`` is already in the hundreds, so a pattern
requires both a subject and an action word wherever the generic word is ambiguous
(``list`` alone is not a listing, ``approval`` alone is not an ETF approval).

``classify`` is pure and deterministic: the same title always yields the same class,
on the training split and on the holdout alike. It reads nothing but the headline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

__all__ = ["CLASSES", "DIRECTIONS", "EventClass", "classify", "top_symbol_hint"]


@dataclass(frozen=True)
class EventClass:
    """One structural class: a name, the expected direction, the rule, and a note."""

    name: str
    direction: int
    regex: re.Pattern[str]
    note: str


_EXCHANGE: Final[str] = (
    r"(?:binance|coinbase|kraken|okx|bybit|upbit|bitstamp|gemini|kucoin|bitfinex|huobi|"
    r"bitget|mexc|gate\.io|crypto\.com|bitso|independent reserve|bakkt|robinhood)"
)
_LIST_VERB: Final[str] = (
    r"(?:list(?:s|ed|ing)?|adds? (?:support|trading)|will add|to add support|"
    r"trading debut|starts? trading|goes? live)"
)
_LISTING: Final[re.Pattern[str]] = re.compile(
    rf"\b{_EXCHANGE}\b[^.]{{0,80}}?\b{_LIST_VERB}\b"
    rf"|\b{_LIST_VERB}\b[^.]{{0,60}}?\b(?:on|at|by)\b[^.]{{0,20}}?\b{_EXCHANGE}\b",
    re.IGNORECASE,
)
_DELISTING: Final[re.Pattern[str]] = re.compile(
    r"\b(?:delist(?:s|ed|ing)?|to be delisted|suspend(?:s|ed|ing)? trading|"
    r"halt(?:s|ed|ing)? trading|remov(?:e|es|ed) (?:from|as) )\b",
    re.IGNORECASE,
)
_HACK: Final[re.Pattern[str]] = re.compile(
    r"\b(?:hack(?:s|ed|ing)?|exploit(?:s|ed|ing)?|drained|stolen|breach(?:ed|es)?|"
    r"rug ?pull(?:ed)?|attacked)\b",
    re.IGNORECASE,
)
_INSOLVENCY: Final[re.Pattern[str]] = re.compile(
    r"\b(?:bankrupt(?:cy)?|insolven(?:t|cy)|chapter 11|liquidat(?:ed|ion)|"
    r"collapse[sd]?|wind(?:s|ing)? down|shuts? down)\b",
    re.IGNORECASE,
)
_ETF: Final[re.Pattern[str]] = re.compile(r"\betf\b", re.IGNORECASE)
_APPROVE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:approv(?:e|es|ed|al)|green ?light|authoriz(?:e|ed|ation)|clears?|"
    r"wins? (?:approval|the case)|to begin trading)\b",
    re.IGNORECASE,
)
_REJECT: Final[re.Pattern[str]] = re.compile(
    r"\b(?:reject(?:s|ed|ion)?|deni(?:es|ed|al)|delay(?:s|ed)?|postpon(?:e|es|ed)|"
    r"withdraw(?:s|n)?|turns? down|denied)\b",
    re.IGNORECASE,
)
_SEC_SUBJECT: Final[str] = r"(?:sec|doj|cftc|fca|treasury|irs|ofac|finra|doj)"
_SEC_ACTION: Final[str] = (
    r"(?:sues?|sued|charges?|charged|lawsuit|settlement|fine[sd]?|penalt(?:y|ies)|"
    r"probe[sd]?|investigat(?:e|es|ed|ion)|enforcement|subpoena|indict(?:s|ed)?|"
    r"arrest(?:s|ed)?)"
)
_SEC_ACTION_RE: Final[re.Pattern[str]] = re.compile(
    rf"\b{_SEC_SUBJECT}\b[^.]{{0,90}}?\b{_SEC_ACTION}\b"
    rf"|\b{_SEC_ACTION}\b[^.]{{0,90}}?\b{_SEC_SUBJECT}\b",
    re.IGNORECASE,
)
_UNLOCK: Final[re.Pattern[str]] = re.compile(
    r"\b(?:unlock(?:s|ed|ing)?|token release|token vesting|vesting|cliff|airdrop(?:s|ped)?)\b",
    re.IGNORECASE,
)
_UPGRADE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:mainnet|hard fork|network upgrade|upgrade (?:goes live|activates?)|"
    r"activates? (?:the )?upgrade|testnet|migration to)\b",
    re.IGNORECASE,
)
_PARTNERSHIP: Final[re.Pattern[str]] = re.compile(
    r"\b(?:partner(?:s|ed|ship)?|integrat(?:e|es|ed|ion)|collaborat(?:e|es|ed|ion)|"
    r"team(?:s|ed)? up|joint venture)\b",
    re.IGNORECASE,
)
_WHALE: Final[re.Pattern[str]] = re.compile(r"\bwhales?\b", re.IGNORECASE)

#: Priority order: the first class whose rule matches wins. Delisting comes before
#: listing because "Binance to delist X" names an exchange and a list-like verb.
CLASSES: Final[tuple[EventClass, ...]] = (
    EventClass(
        "delisting",
        -1,
        _DELISTING,
        "exchange removes support: liquidity leaves, the quote usually gaps down",
    ),
    EventClass(
        "hack",
        -1,
        _HACK,
        "exploit or theft: the named asset loses credibility and holders exit",
    ),
    EventClass(
        "insolvency",
        -1,
        _INSOLVENCY,
        "counterparty failure: contagion, forced selling, credit risk repricing",
    ),
    EventClass(
        "listing",
        +1,
        _LISTING,
        "new venue for the asset: a new pool of buyers, historically a pump",
    ),
    EventClass(
        "sec_action",
        -1,
        _SEC_ACTION_RE,
        "enforcement against a named party: legal overhang",
    ),
    EventClass(
        "unlock",
        -1,
        _UNLOCK,
        "supply becomes liquid (unlock, vesting, airdrop): mechanical sell pressure",
    ),
    EventClass(
        "upgrade",
        +1,
        _UPGRADE,
        "protocol milestone: attention and a narrative bid",
    ),
    EventClass(
        "partnership",
        +1,
        _PARTNERSHIP,
        "distribution or integration story: narrative bid",
    ),
    EventClass(
        "whale",
        0,
        _WHALE,
        "large holder action: direction unknown, amplitude possibly large",
    ),
)

#: Class name -> expected direction of the named asset (``0`` means unknown).
#: The two ETF classes are compound (see :func:`classify`) and carry their own signs:
#: an approval is a bid, a rejection is an offer.
DIRECTIONS: Final[dict[str, int]] = {
    **{spec.name: spec.direction for spec in CLASSES},
    "etf_approval": +1,
    "etf_rejection": -1,
}

_ETF_CLASSES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("etf_approval", _APPROVE),
    ("etf_rejection", _REJECT),
)


def classify(title: str | None) -> str | None:
    """The structural class of ``title``, or ``None`` when it is ordinary commentary.

    Compound rules are checked before the generic ones: an ETF headline that also
    carries an approval or rejection word is an ETF decision, not a partnership.
    """
    if not title:
        return None
    text = str(title)
    if _ETF.search(text):
        for name, pattern in _ETF_CLASSES:
            if pattern.search(text):
                return name
    for spec in CLASSES:
        if spec.regex.search(text):
            return spec.name
    return None


def top_symbol_hint() -> None:
    """Placeholder for callers that want type checking; the taxonomy is title-only."""
    return None
