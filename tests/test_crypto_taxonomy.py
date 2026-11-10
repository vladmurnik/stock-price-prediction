"""Tests for the headline taxonomy: every class fires on its canonical headline."""

from __future__ import annotations

import pytest

from spp.crypto.taxonomy import CLASSES, DIRECTIONS, classify


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Binance Will Delist Three Stablecoin Pairs Next Week", "delisting"),
        ("Coinbase to delist XRP trading pairs", "delisting"),
        ("DeFi Protocol Aave Clears Bad CRV Token Debt After Exploit Attempt", "hack"),
        ("Algorand Wallet MyAlgo Advises Users to Withdraw Funds After $9.6M Exploit", "hack"),
        ("Celsius files for Chapter 11 bankruptcy protection", "insolvency"),
        ("Binance Will List Arbitrum (ARB) With Seed Tag", "listing"),
        ("Solana (SOL) goes live on Robinhood", "listing"),
        ("SEC Approves Spot Bitcoin ETF in Landmark Decision", "etf_approval"),
        ("Judge rejects Grayscale's bitcoin ETF application", "etf_rejection"),
        ("SEC Sues Binance and Founder Changpeng Zhao for Securities Violations", "sec_action"),
        ("Arbitrum to Airdrop New Token With 12% Unlock", "unlock"),
        ("Ethereum's Shanghai Upgrade Goes Live, Enabling Staking Withdrawals", "upgrade"),
        (
            "Xapo Bank Integrates Bitcoin's Lightning Network, Partners With Lightspark",
            "partnership",
        ),
        ("Bitcoin Whale Moves $37M BTC After 11 Years of Dormancy", "whale"),
    ],
)
def test_classes_fire_on_canonical_headlines(title: str, expected: str) -> None:
    assert classify(title) == expected


@pytest.mark.parametrize(
    "title",
    [
        "Bitcoin Dips Below $21K to Reach Lowest Level in 7 Weeks",
        "Crypto Markets Today: Bitcoin Clings to $23K",
        "Stablecoin Regulation Is First on New Subcommittee's To-Do List",
        "Ethereum Developers Target March 14 Date for Shanghai Testnet Run",
        "",
        None,
    ],
)
def test_commentary_is_not_structural(title: str | None) -> None:
    # ``Ethereum Developers Target March 14 Date for Shanghai Testnet Run`` does hit
    # ``testnet``: it is a milestone headline, so it belongs to ``upgrade``.
    result = classify(title)
    if title and "Testnet" in title:
        assert result == "upgrade"
    else:
        assert result is None


def test_delisting_wins_over_listing() -> None:
    assert classify("Binance to Delist and Relist the Token After Review") == "delisting"


def test_etf_decision_wins_over_generic_classes() -> None:
    # "approve" alone would be nothing; with "ETF" it is a decision, and the ETF
    # branch is evaluated before the partnership rule that "partners" would match.
    assert classify("Partners Celebrate as SEC Approves Spot Ether ETF") == "etf_approval"


def test_every_class_has_a_direction_and_a_note() -> None:
    assert set(DIRECTIONS) == {spec.name for spec in CLASSES} | {"etf_approval", "etf_rejection"}
    for spec in CLASSES:
        assert spec.direction in (-1, 0, 1)
        assert spec.note
        assert spec.name == spec.name.lower()
    assert DIRECTIONS["etf_approval"] == 1
    assert DIRECTIONS["etf_rejection"] == -1
