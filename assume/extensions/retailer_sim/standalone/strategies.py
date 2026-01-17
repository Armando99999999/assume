# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Wrapper standalone per le strategie condivise.

Questo modulo evita duplicazioni re-esportando le classi/metodi definiti in
``decision.py``. Gli script nella cartella ``standalone`` possono quindi
continuare a importare da ``standalone.strategies`` senza modifiche, pur
riutilizzando l'implementazione condivisa.
"""

from __future__ import annotations

from ..decision import (
    BiddingStrategy,
    RandomBiddingStrategy,
    StrategyInputs,
    build_strategy,
    estimate_msd_price,
)

__all__ = [
    "BiddingStrategy",
    "RandomBiddingStrategy",
    "StrategyInputs",
    "build_strategy",
    "estimate_msd_price",
]
