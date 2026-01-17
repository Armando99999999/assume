# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Strategie per il retailer e la controparte virtuale nel world adapter.

Walkthrough in stile colloquio tecnico
--------------------------------------
1. ``set_shared_strategy`` registra la ``BiddingStrategy`` condivisa, la lista
   delle sessioni MI e un flag `real_market` usato per arricchire gli ordini.
2. Le funzioni helper `_hour_in_window`, `_mi_reference_for_timestamp`,
   `_resolve_session` gestiscono il mapping fra timestamp e sessioni MI.
3. ``RetailerMGPStrategy`` genera gli ordini MGP interrogando `RetailerUnit`,
   riutilizzando la logica `plan_day_ahead` dello standalone.
4. ``RetailerMIStrategy`` replica le correzioni intraday: controlla l'eligibilità
   degli slot, invoca `plan_intraday` e applica i regolatori dell'unità.
5. ``TernaBalancingStrategy`` simula la controparte Terna-like pubblicando ordini
   di vendita in base alla capacità massima disponibile.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from assume.common.base import SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.strategies import BaseStrategy

from ..decision import BiddingStrategy

SHARED_STRATEGY: Optional[BiddingStrategy] = None
SESSION_DEFINITIONS: List[Dict[str, Any]] = []
REAL_MARKET_FLAG: bool = False


# --------------------------------------------------------------------------- #
# Helper per mapping sessioni / timestamp                                     #
# --------------------------------------------------------------------------- #


def _hour_in_window(hour_value: float, start: float, end: float) -> bool:
    """True se l'ora decimale cade nella finestra [start, end), con gestione rollover 24h."""
    if end <= start:
        end += 24
    value = hour_value
    if value < start:
        value += 24
    return start <= value < end


def _mi_reference_for_timestamp(timestamp: pd.Timestamp) -> str:
    """
    Trova la sessione MI responsabile dello slot, convertendo il timestamp in ora
    decimale e confrontandolo con le finestre configurate. Serve per passare il
    corretto riferimento di prezzo a `plan_day_ahead` (fallback su MI1).
    """
    if not SESSION_DEFINITIONS:
        return "MI1"
    slot_hour = timestamp.hour + timestamp.minute / 60.0
    for session in SESSION_DEFINITIONS:
        start = float(session.get("coverage_start_hour", session.get("open_hour", 0.0)))
        end = float(session.get("coverage_end_hour", session.get("close_hour", 24.0)))
        if _hour_in_window(slot_hour, start, end):
            return session.get("name", "MI1")
    return SESSION_DEFINITIONS[0].get("name", "MI1")


def _resolve_session(market_id: str | None) -> Tuple[int, Dict[str, Any]]:
    """
    Associa un identificativo di mercato a (indice sessione, metadati) con fallback
    sulla prima sessione configurata. La ricerca e' case-insensitive.
    """
    name = (market_id or "MI1").upper()
    for idx, session in enumerate(SESSION_DEFINITIONS, start=1):
        if session.get("name", "").upper() == name:
            return idx, session
    if name in {"MI", "MI1"} and SESSION_DEFINITIONS:
        return 1, SESSION_DEFINITIONS[0]
    if name == "MI2" and len(SESSION_DEFINITIONS) >= 2:
        return 2, SESSION_DEFINITIONS[1]
    if SESSION_DEFINITIONS:
        return 1, SESSION_DEFINITIONS[0]
    return 1, {"name": "MI1"}


# --------------------------------------------------------------------------- #
# Registrazione della strategia condivisa                                     #
# --------------------------------------------------------------------------- #


def set_shared_strategy(
    strategy: BiddingStrategy,
    *,
    session_definitions: Optional[List[Dict[str, Any]]] = None,
    real_market: bool = False,
) -> None:
    """
    Salva la ``BiddingStrategy`` condivisa e i metadati delle sessioni MI.

    Questo consente di riutilizzare la stessa strategia in entrambi gli ambienti
    (standalone e world adapter) senza duplicarne lo stato.
    """
    global SHARED_STRATEGY, SESSION_DEFINITIONS, REAL_MARKET_FLAG
    SHARED_STRATEGY = strategy
    if session_definitions is not None:
        SESSION_DEFINITIONS = list(session_definitions)
    REAL_MARKET_FLAG = bool(real_market)


# --------------------------------------------------------------------------- #
# Strategie concrete                                                          #
# --------------------------------------------------------------------------- #


class RetailerMGPStrategy(BaseStrategy):
    """
    Adatta `plan_day_ahead` al mondo ASSUME generando `Orderbook` per il mercato MGP.
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: List[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Per ogni slot:
            1. Costruisce gli input usando `RetailerUnit`.
            2. Invoca `plan_day_ahead`.
            3. Applica le salvaguardie locali (se l'unità le espone).
            4. Registra il volume contrattato e traduce il tutto in un `Orderbook`.
        """
        if SHARED_STRATEGY is None:
            raise RuntimeError("Retailer strategy not configured. Call set_shared_strategy() first.")

        orders: Orderbook = []
        day_start = product_tuples[0][0]
        day_end = product_tuples[-1][1]
        min_power, _ = unit.calculate_min_max_power(day_start, day_end)

        for product_slot, fallback_volume in zip(product_tuples, min_power):
            timestamp = pd.Timestamp(product_slot[0])
            mi_reference = _mi_reference_for_timestamp(timestamp)
            inputs = unit.build_strategy_inputs(timestamp, mi_price_market=mi_reference, phase="mgp")
            target_volume = SHARED_STRATEGY.plan_day_ahead(inputs)
            demand_forecast = getattr(unit, "get_demand_forecast", lambda ts: inputs.demand_forecast_MWh)(timestamp)
            regulator = getattr(unit, "regulate_day_ahead_volume", None)
            final_volume = regulator(target_volume, demand_forecast) if callable(regulator) else target_volume
            output_volume = final_volume if not np.isnan(final_volume) else fallback_volume
            unit.record_contracted_volume(timestamp, "MGP", output_volume)

            orders.append(
                {
                    "start_time": product_slot[0],
                    "end_time": product_slot[1],
                    "only_hours": product_slot[2],
                    "price": unit.calculate_marginal_cost(product_slot[0], output_volume, market_config.market_id),
                    "volume": output_volume,
                    "node": unit.node,
                }
            )
        return orders


class RetailerMIStrategy(BaseStrategy):
    """
    Riproduce le correzioni intraday dello standalone chiamando `plan_intraday`.
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: List[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Filtra gli slot non eleggibili, invoca `plan_intraday` e applica i regolatori
        locali dell'unità prima di convertire la decisione in un `Orderbook`.
        """
        if SHARED_STRATEGY is None:
            raise RuntimeError("Retailer strategy not configured. Call set_shared_strategy() first.")

        market_id = (market_config.market_id or "MI1").upper()
        session_index, session_meta = _resolve_session(market_id)
        if session_index > 1 and not SHARED_STRATEGY.supports_second_session():
            return []

        session_name = session_meta.get("name", market_id)
        orders: Orderbook = []

        for product_slot in product_tuples:
            timestamp = product_slot[0]
            eligibility_hook = getattr(unit, "is_session_slot_eligible", None)
            if eligibility_hook is not None and not eligibility_hook(timestamp, session_name):
                continue

            inputs = unit.build_strategy_inputs(timestamp, mi_price_market=market_config.market_id, phase="mi")
            contracted_volume = unit.get_contracted_volume(timestamp)
            demand_forecast = getattr(unit, "get_demand_forecast", lambda ts: 0.0)(timestamp)
            desired_adjustment = SHARED_STRATEGY.plan_intraday(
                inputs,
                contracted_volume_MWh=contracted_volume,
                session=session_index,
            )
            liquidity_hint = getattr(unit, "get_session_liquidity", lambda ts, name: 1.0)(timestamp, session_name)
            regulator = getattr(unit, "regulate_intraday_volume", None)
            if regulator is not None:
                final_volume = regulator(
                    session_index=session_index,
                    desired_volume_MWh=desired_adjustment,
                    contracted_before_MWh=contracted_volume,
                    demand_forecast_MWh=demand_forecast,
                    liquidity_hint=liquidity_hint,
                )
            else:
                final_volume = desired_adjustment

            if abs(final_volume) < 1e-9:
                continue

            unit.record_contracted_volume(timestamp, session_name, final_volume)
            orders.append(
                {
                    "start_time": product_slot[0],
                    "end_time": product_slot[1],
                    "only_hours": product_slot[2],
                    "price": unit.calculate_marginal_cost(product_slot[0], final_volume, market_config.market_id),
                    "volume": final_volume,
                    "node": unit.node,
                }
            )

        return orders


class TernaBalancingStrategy(BaseStrategy):
    """
    Modella la controparte Terna-like che vende capacità residua (massimo power profile).
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: List[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Per ogni slot offre un volume negativo pari alla capacità disponibile, usando il
        prezzo marginale dell'unità. Serve come “supply virtuale” per chiudere il mercato.
        """
        orders: Orderbook = []
        start = product_tuples[0][0]
        end_all = product_tuples[-1][1]
        _, max_power = unit.calculate_min_max_power(start, end_all)
        for product_slot, volume in zip(product_tuples, max_power):
            sell_volume = -abs(volume)
            orders.append(
                {
                    "start_time": product_slot[0],
                    "end_time": product_slot[1],
                    "only_hours": product_slot[2],
                    "price": unit.calculate_marginal_cost(product_slot[0], abs(volume), market_config.market_id),
                    "volume": sell_volume,
                    "node": unit.node,
                }
            )
        return orders


# Backward compatibility per configurazioni legacy.
VirtualSupplyStrategy = TernaBalancingStrategy
