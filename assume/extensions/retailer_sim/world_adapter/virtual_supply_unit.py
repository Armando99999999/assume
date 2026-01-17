# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Alias storico per mantenere compatibilità con `virtual_supply_unit`.

Documentazione rapida:
    - In passato il file esportava un'unità di supply virtuale dedicata.
    - Oggi la logica è stata unificata in ``TernaBalancingUnit``; molte configurazioni
      puntano ancora a ``virtual_supply_unit.VirtualSupplyUnit``.
    - Questo modulo funge quindi da semplice alias per evitare cambiamenti alle config.
"""

from __future__ import annotations

from .terna_unit import TernaBalancingUnit

# Alias pubblico compatibile con le vecchie configurazioni.
VirtualSupplyUnit = TernaBalancingUnit
