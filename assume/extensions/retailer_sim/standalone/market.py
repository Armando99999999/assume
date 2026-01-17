# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Facciate di mercato e moduli di clearing/MSD per lo scenario standalone.

Walkthrough in stile colloquio tecnico
--------------------------------------
1. ``TradeResult`` e' un semplice contenitore che descrive l'esito di un ordine
   (timestamp, mercato, volume, prezzo e impatto economico).
2. ``ClearingCoordinator`` applica un cap simmetrico ai volumi aggregati MGP/MI e
   tiene conto di un eventuale profilo di interconnessione (baseline import/export).
3. ``DayAheadMarket`` e ``IntraDayMarket`` simulano un clearing lineare con slope
   parametrizzabili; accettano volumi dal retailer e restituiscono ``TradeResult``.
4. ``MarketEnvironment`` e' la facciata che il retailer vede: espone solo i metodi
   `execute_day_ahead_bid` e `execute_intraday_bid`, mantenendo il resto incapsulato.
5. ``MSDSettlement`` calcola prezzo e costo dello sbilanciamento; viene usato sia
   nello standalone sia nel world adapter.
6. ``suggest_market_parameters`` e' una utility offline che aiuta a stimare i cap
   partendo dai dataset storici (utile per calibrare simulazioni rapide).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import pandas as pd


# --------------------------------------------------------------------------- #
# Trade result & Clearing coordinator                                         #
# --------------------------------------------------------------------------- #


@dataclass
class TradeResult:
    """
    Descrive un singolo trade eseguito sul mercato fittizio.

    I campi ``cost_EUR`` e ``revenue_EUR`` vengono popolati in base al segno del
    volume: se il retailer compra energia (volume >= 0) registra un costo, se la
    vende (volume < 0) registra un ricavo.
    """

    timestamp: pd.Timestamp
    market_name: str
    volume_MWh: float
    price_EUR_MWh: float
    cost_EUR: float
    revenue_EUR: float


class ClearingCoordinator:
    """
    Gestisce il cap simmetrico (MWh) condiviso fra MGP e MI.

    - Tiene traccia dei volumi prenotati per ogni timestamp (`_booked`).
    - Permette di caricare un profilo esterno (import/export) che funge da baseline.
    - Il metodo `book` restituisce il volume effettivamente accettato e il grado
      di saturazione (0 = nessun utilizzo, 1 = pieno cap).
    """

    def __init__(self, capacity_MWh: Optional[float] = None) -> None:
        self.capacity = abs(capacity_MWh) if capacity_MWh else None
        self.external_profile: Dict[pd.Timestamp, float] = {}
        self._booked: Dict[pd.Timestamp, float] = {}

    # ----------------------------- API pubblica ----------------------------- #

    def set_external_profile(self, profile: Mapping[pd.Timestamp, float]) -> None:
        """Registra il profilo di scambi con macrozone confinanti (baseline)."""
        normalized = {pd.Timestamp(ts): float(value) for ts, value in profile.items()}
        self.external_profile = normalized
        self.reset()

    def reset(self) -> None:
        """Azzera le prenotazioni mantenendo la baseline degli scambi esterni."""
        self._booked = dict(self.external_profile)

    def book(self, timestamp: pd.Timestamp, request_volume: float) -> tuple[float, float]:
        """
        Applica il cap volumetrico restituendo:
            * cleared_volume -> porzione effettivamente accettata
            * utilization    -> saturazione del cap (0..1)
        """
        if not self.capacity:
            return request_volume, 0.0

        booked = self._booked.get(timestamp, self.external_profile.get(timestamp, 0.0))
        desired = booked + request_volume
        clipped = max(min(desired, self.capacity), -self.capacity)
        cleared = clipped - booked
        self._booked[timestamp] = clipped
        utilization = clipped / self.capacity if self.capacity else 0.0
        return cleared, utilization


# --------------------------------------------------------------------------- #
# Mercati day-ahead e intraday                                                #
# --------------------------------------------------------------------------- #


class DayAheadMarket:
    """
    Rappresentazione monozona del MGP con clearing lineare.

    Il prezzo di clearing viene spostato di una quantità proporzionale alla
    saturazione (`utilization`) e alle slope configurate. Il volume viene
    circonciso dal cap assegnato al MGP o dal clearing coordinator condiviso.
    """

    def __init__(
        self,
        transaction_cost_per_MWh: float = 0.0,
        clearing_capacity_MWh: float | None = None,
        price_slope_low: float = 0.0,
        price_slope_high: Optional[float] = None,
        price_slope_threshold: float = 1.0,
        coordinator: Optional[ClearingCoordinator] = None,
    ) -> None:
        self.transaction_cost_per_MWh = transaction_cost_per_MWh
        self.clearing_capacity_MWh = abs(clearing_capacity_MWh) if clearing_capacity_MWh else None
        self.price_slope_low = float(price_slope_low)
        self.price_slope_high = float(price_slope_high) if price_slope_high is not None else None
        self.price_slope_threshold = float(price_slope_threshold)
        self.coordinator = coordinator

    # ----------------------------- API pubblica ----------------------------- #

    def execute(
        self,
        timestamp: pd.Timestamp,
        volume_MWh: float,
        price_EUR_MWh: float,
    ) -> TradeResult:
        """Esegue l'ordine applicando cap volumetrico e prezzo elastico."""
        cleared_volume, utilization = self._clear_volume(timestamp, volume_MWh)
        cleared_price = self._apply_price_adjustment(price_EUR_MWh, utilization)
        return self._build_trade(timestamp, "MGP", cleared_volume, cleared_price)

    # ----------------------------- Helper interni --------------------------- #

    def _build_trade(self, timestamp: pd.Timestamp, market_name: str, volume: float, price: float) -> TradeResult:
        """Crea la struttura TradeResult aggiungendo i costi di transazione."""
        transaction_cost = abs(volume) * self.transaction_cost_per_MWh
        if volume >= 0:  # acquisto
            cost = volume * price + transaction_cost
            revenue = 0.0
        else:  # vendita
            revenue = -volume * price - transaction_cost
            cost = 0.0
        return TradeResult(timestamp, market_name, volume, price, cost, revenue)

    def _clear_volume(self, timestamp: pd.Timestamp, request: float) -> tuple[float, float]:
        """Applica il cap tramite il coordinatore oppure localmente."""
        if self.coordinator:
            return self.coordinator.book(timestamp, request)
        cleared = self._apply_clearing_capacity(request)
        return cleared, self._compute_utilization(cleared)

    def _apply_clearing_capacity(self, volume: float) -> float:
        if not self.clearing_capacity_MWh:
            return volume
        cap = self.clearing_capacity_MWh
        return max(min(volume, cap), -cap)

    def _compute_utilization(self, cleared_volume: float) -> float:
        if not self.clearing_capacity_MWh:
            return 0.0
        return cleared_volume / self.clearing_capacity_MWh

    def _apply_price_adjustment(self, reference_price: float, utilization: float) -> float:
        if not self.price_slope_low and not self.price_slope_high:
            return reference_price
        slope = self.price_slope_low
        threshold = self.price_slope_threshold
        if self.price_slope_high is not None and threshold < 1.0:
            if utilization > threshold:
                extra = (utilization - threshold) / max(1.0 - threshold, 1e-6)
                mix = min(extra, 1.0)
                slope = self.price_slope_high * mix + self.price_slope_low * (1.0 - mix)
        return reference_price + slope * utilization


class IntraDayMarket(DayAheadMarket):
    """
    Stessa logica del MGP ma con etichetta custom (MI, MI2, ...).

    L'unica differenza e' il parametro `market_label` passato al metodo execute.
    """

    def execute(
        self,
        timestamp: pd.Timestamp,
        volume_MWh: float,
        price_EUR_MWh: float,
        *,
        market_label: str = "MI",
    ) -> TradeResult:
        cleared_volume, utilization = self._clear_volume(timestamp, volume_MWh)
        cleared_price = self._apply_price_adjustment(price_EUR_MWh, utilization)
        return self._build_trade(timestamp, market_label, cleared_volume, cleared_price)


# --------------------------------------------------------------------------- #
# Facciata principale usata dal retailer                                     #
# --------------------------------------------------------------------------- #


class MarketEnvironment:
    """
    Facciata che il retailer riceve in input.

    Incapsula la coppia MGP/MI e il coordinatore di clearing. Gli unici metodi
    esposti sono `execute_day_ahead_bid`, `execute_intraday_bid`, `reset` (per
    resettare il coordinatore) e `set_interconnection_profile`.
    """

    def __init__(
        self,
        day_ahead: DayAheadMarket,
        intraday: IntraDayMarket,
        clearing_coordinator: Optional[ClearingCoordinator] = None,
    ) -> None:
        self.day_ahead = day_ahead
        self.intraday = intraday
        self.clearing_coordinator = clearing_coordinator
        self._interconnection_profile: Dict[pd.Timestamp, float] = {}

    def execute_day_ahead_bid(
        self,
        *,
        timestamp: pd.Timestamp,
        volume_MWh: float,
        price_EUR_MWh: float,
    ) -> TradeResult:
        """Il retailer interagisce con il MGP solo tramite questa facciata."""
        return self.day_ahead.execute(timestamp, volume_MWh, price_EUR_MWh)

    def execute_intraday_bid(
        self,
        *,
        timestamp: pd.Timestamp,
        volume_MWh: float,
        price_EUR_MWh: float,
        market_label: str = "MI",
    ) -> TradeResult:
        """Equivalente del metodo MGP ma per le sessioni MI (MI1, MI2...)."""
        return self.intraday.execute(timestamp, volume_MWh, price_EUR_MWh, market_label=market_label)

    def reset(self) -> None:
        """Resetta lo stato del clearing condiviso (se presente)."""
        if self.clearing_coordinator:
            self.clearing_coordinator.reset()

    def set_interconnection_profile(self, profile: Mapping[pd.Timestamp, float]) -> None:
        """
        Registra il profilo di scambi con altri paesi/macrozone.

        Valori positivi = import (cap occupata); valori negativi = export (cap liberata).
        Il profilo viene inoltrato al coordinatore per ridurre la capacità disponibile.
        """
        self._interconnection_profile = {pd.Timestamp(k): float(v) for k, v in profile.items()}
        if self.clearing_coordinator:
            self.clearing_coordinator.set_external_profile(self._interconnection_profile)


# --------------------------------------------------------------------------- #
# Stima del prezzo/costo MSD                                                  #
# --------------------------------------------------------------------------- #


class MSDSettlement:
    """
    Calcola prezzo e costo MSD usando una formula parametrizzabile.

    Parametri chiave:
        * price_sensitivity: modifica il prezzo rispetto all'entita' dello sbilanciamento
        * additional_penalty: penalita' fissa in €/MWh
        * long_position_credit_factor: fattore con cui si riduce il costo quando lo
          sbilanciamento e' long (energia in surplus).
    """

    def __init__(
        self,
        *,
        price_sensitivity: float = 0.45,
        additional_penalty_EUR_MWh: float = 0.0,
        long_position_credit_factor: float = 1.0,
    ) -> None:
        self.price_sensitivity = price_sensitivity
        self.additional_penalty = additional_penalty_EUR_MWh
        self.long_position_credit_factor = long_position_credit_factor

    def estimate_price(
        self,
        *,
        mgp_price: float,
        macrozone_price: float,
        imbalance_coeff: float,
        imbalance_input_MWh: float,
    ) -> float:
        """Formula semplificata: prezzo = Pz + (Pz - Pmz)*coeff + sens*|sbil| + penalita."""
        delta = mgp_price - macrozone_price
        macro_adjustment = delta * imbalance_coeff
        imbalance_component = self.price_sensitivity * abs(imbalance_input_MWh)
        price = mgp_price + macro_adjustment + imbalance_component + self.additional_penalty
        return max(price, 0.0)

    def cost_from_price(self, imbalance_MWh: float, price_EUR_MWh: float) -> float:
        """
        Converte il prezzo MSD in un costo monetario.

        Quando lo sbilanciamento e' long (volume negativo) il costo viene ridotto
        usando ``long_position_credit_factor`` per imitare il trattamento MSD reale.
        """
        base_cost = abs(imbalance_MWh) * price_EUR_MWh
        if imbalance_MWh < 0:
            return base_cost * self.long_position_credit_factor
        return base_cost

    def evaluate(
        self,
        *,
        imbalance_MWh: float,
        mgp_price: float,
        macrozone_price: float,
        imbalance_coeff: float,
        imbalance_input_MWh: float,
    ) -> tuple[float, float]:
        """Restituisce coppia (prezzo, costo) riutilizzabile in altri contesti."""
        price = self.estimate_price(
            mgp_price=mgp_price,
            macrozone_price=macrozone_price,
            imbalance_coeff=imbalance_coeff,
            imbalance_input_MWh=imbalance_input_MWh,
        )
        return price, self.cost_from_price(imbalance_MWh, price)


# --------------------------------------------------------------------------- #
# Utility offline per calibrare il mercato                                   #
# --------------------------------------------------------------------------- #


def suggest_market_parameters(
    dataset_path: str,
    *,
    time_step_minutes: int = 15,
) -> Dict[str, float]:
    """
    Calcola parametri di massima (cap MGP/MI, sbilanciamento max, slope prezzo).

    L'idea e' analoga a GME/Terna: si osserva l'intero dataset storico PRIMA dei
    filtri e si usano quantili alti per scegliere i cap. Questo aiuta chi monta il
    simulatore a definire valori plausibili con pochi comandi.
    """

    import pandas as pd  # import locale per evitare costi in importazione

    df = pd.read_csv(
        dataset_path,
        usecols=["cluster_total_load_MW", "SBIL_MWH", "MGP_PRICE_NORD"],
    )
    time_step_hours = time_step_minutes / 60

    cap_mgp = df["cluster_total_load_MW"].quantile(0.99) * time_step_hours
    cap_mi = df["cluster_total_load_MW"].quantile(0.95) * time_step_hours
    sbil_max = df["SBIL_MWH"].abs().quantile(0.95)
    price_mean = df["MGP_PRICE_NORD"].mean()
    price_p95 = df["MGP_PRICE_NORD"].quantile(0.95)
    slope = (price_p95 - price_mean) / cap_mgp if cap_mgp else 0.0

    return {
        "cap_mgp_MWh": float(cap_mgp),
        "cap_mi_MWh": float(cap_mi),
        "sbil_max_MWh": float(sbil_max),
        "price_slope_EUR_per_MWh": float(slope),
    }
