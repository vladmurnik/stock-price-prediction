"""Crypto branch: bars, news, dataset, models and the leverage-aware policy.

Everything here is keyless and reproducible from public archives: Binance Vision monthly klines for
prices, GDELT for the historical news sample and the project's own RSS recorder for true
publication-to-detection latency. The package is deliberately separate from the equity code in
``spp.data`` / ``spp.features``: the market is 24/7, the calendar is not a trading calendar, and a
one-day horizon means something entirely different.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
