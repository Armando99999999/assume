# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Helper che replica l'interazione con Terna sia nello standalone sia nel World."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Set

import pandas as pd


@dataclass
class TernaAdjustment:
    """Raccoglie il risultato di una valutazione Terna (prezzo + segnali)."""

    adjusted_price_EUR_MWh: float
    penalty_EUR_MWh: float
    incentive_EUR_MWh: float


class TernaLikeBalancingAgent:
    """
    Incapsula il comportamento della controparte Terna semplificata.

    Il componente opera in tre modi:
    - penalizza il retailer quando amplifica lo sbilanciamento del sistema;
    - aggiunge una penalità aggiuntiva quando il volume short non viene coperto;
    - incentiva quando si muove in senso opposto rispetto al macro sbilanciamento;
    - adatta prezzi e tariffe al contesto temporale (ore lavorative vs notturne).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Parametri principali (puoi ometterli dal config per usare i default):
        - same/opposite_direction_*: entità della penalità o dell'incentivo;
        - off_hours_penalty_factor: moltiplicatore notturno;
        - reference_volume_MWh: volume di riferimento per normalizzare il residuo;
        - price_floor/price_cap: limiti rigidi sul prezzo MSD;
        - imbalance_pass_through_fraction: quota di costo MSD scaricata in tariffa;
        - coverage_shortfall_*: controllano la penalità per under-coverage (MGP+quota);
        - sale_price_floor/cap: range consentito per la tariffa retail;
        - non_working_hours / working_hours_window: definizione delle ore lente.
        """
        cfg = config or {}
        self.same_direction_penalty = float(cfg.get("same_direction_penalty_EUR_MWh", 18.0))
        self.opposite_direction_bonus = float(cfg.get("opposite_direction_bonus_EUR_MWh", 8.0))
        self.neutral_direction_bonus = float(cfg.get("neutral_direction_bonus_EUR_MWh", 0.0))
        self.off_hours_penalty_factor = float(cfg.get("off_hours_penalty_factor", 0.35))
        self.reference_volume_MWh = max(float(cfg.get("reference_volume_MWh", 80.0)), 1e-3)
        shortfall_markup = cfg.get("coverage_shortfall_markup_EUR_MWh")
        if shortfall_markup is None:
            shortfall_markup = cfg.get("coverage_shortfall_penalty_EUR_MWh", 0.0)
        self.shortfall_markup_EUR_MWh = float(shortfall_markup)
        self.shortfall_reference_volume_MWh = max(
            float(cfg.get("coverage_shortfall_reference_MWh", self.reference_volume_MWh)),
            1e-3,
        )
        self.shortfall_tolerance_MWh = max(float(cfg.get("coverage_shortfall_tolerance_MWh", 0.0)), 0.0)
        self.price_floor = float(cfg.get("price_floor_EUR_MWh", 0.0))
        self.price_cap = float(cfg.get("price_cap_EUR_MWh", 2000.0))
        self.imbalance_pass_through = float(cfg.get("imbalance_pass_through_fraction", 0.4))
        self.sale_price_floor = float(cfg.get("sale_price_floor_EUR_MWh", 50.0))
        self.sale_price_cap = float(cfg.get("sale_price_cap_EUR_MWh", 600.0))
        self.non_working_hours = self._parse_hours(cfg.get("non_working_hours", (0, 1, 2, 3, 4, 5, 6, 22, 23)))
        self.working_hour_span = self._parse_range(cfg.get("working_hours_window", (6, 21)))

    def evaluate(
        self,
        *,
        timestamp: pd.Timestamp | None,
        residual_MWh: float,
        macro_imbalance_MWh: float,
        base_price_EUR_MWh: float,
        mgp_price_EUR_MWh: float | None = None,
    ) -> TernaAdjustment:
        """Calcola la variazione del prezzo MSD in base al residuo e al macro-sbilancio."""
        magnitude = abs(residual_MWh)
        if magnitude <= 1e-9:
            return TernaAdjustment(self._clip_price(base_price_EUR_MWh), 0.0, 0.0)

        same_direction = residual_MWh * macro_imbalance_MWh > 0
        opposite_direction = residual_MWh * macro_imbalance_MWh < 0
        scaling = min(1.0, magnitude / self.reference_volume_MWh)
        off_hour_multiplier = 1.0 + self._off_hour_penalty(timestamp)

        penalty = 0.0
        incentive = 0.0
        if same_direction:
            penalty = self.same_direction_penalty * scaling * off_hour_multiplier
        elif opposite_direction:
            incentive = self.opposite_direction_bonus * scaling * (2.0 - off_hour_multiplier)
        else:
            if self.neutral_direction_bonus > 0 and residual_MWh > 0 and macro_imbalance_MWh <= 0:
                incentive = self.neutral_direction_bonus * scaling

        reference_mgp_price = base_price_EUR_MWh if mgp_price_EUR_MWh is None else mgp_price_EUR_MWh
        if residual_MWh > 0.0:
            penalty += self._coverage_shortfall_penalty(residual_MWh, off_hour_multiplier, reference_mgp_price)

        adjusted = base_price_EUR_MWh + penalty - incentive
        return TernaAdjustment(self._clip_price(adjusted), penalty, incentive)

    def compute_sale_tariff(
        self,
        *,
        purchase_price_EUR_MWh: float,
        imbalance_cost_EUR_MWh: float,
        profit_margin_EUR_MWh: float,
    ) -> float:
        """
        Translate purchase/imbalance costs into the end-customer tariff requested by the user.

        `imbalance_cost_EUR_MWh` rappresenta la sola quota scaricabile (es. costo MSD base
        al netto delle penalità Terna).
        """
        tariff = purchase_price_EUR_MWh + profit_margin_EUR_MWh + imbalance_cost_EUR_MWh * self.imbalance_pass_through
        tariff = max(self.sale_price_floor, tariff)
        return min(self.sale_price_cap, tariff)

    # ------------------------------------------------------------------ helpers

    def _off_hour_penalty(self, timestamp: pd.Timestamp | None) -> float:
        if timestamp is None:
            return 0.0
        hour = int(timestamp.hour)
        if hour in self.non_working_hours:
            return self.off_hours_penalty_factor
        if self.working_hour_span and not (self.working_hour_span[0] <= hour <= self.working_hour_span[1]):
            return self.off_hours_penalty_factor * 0.5
        return 0.0

    def _clip_price(self, price: float) -> float:
        return max(self.price_floor, min(self.price_cap, price))

    def _coverage_shortfall_penalty(
        self,
        residual_MWh: float,
        off_hour_multiplier: float,
        mgp_price_EUR_MWh: float,
    ) -> float:
        if self.shortfall_markup_EUR_MWh <= 0.0:
            return 0.0
        shortfall = residual_MWh - self.shortfall_tolerance_MWh
        if shortfall <= 1e-9:
            return 0.0
        normalized = min(1.0, shortfall / self.shortfall_reference_volume_MWh)
        mgp_component = max(mgp_price_EUR_MWh, 0.0)
        unit_penalty = mgp_component + self.shortfall_markup_EUR_MWh
        if unit_penalty <= 0.0:
            return 0.0
        return unit_penalty * normalized * off_hour_multiplier

    @staticmethod
    def _parse_range(values: Sequence[int] | Sequence[float] | Sequence[str] | None) -> Optional[tuple[int, int]]:
        if not values:
            return None
        if isinstance(values, (tuple, list)) and len(values) >= 2:
            try:
                start = int(float(values[0]))
                end = int(float(values[1]))
                return (max(0, start), min(23, end))
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _parse_hours(values: Iterable[int | float | str]) -> Set[int]:
        hours: Set[int] = set()
        for value in values:
            if isinstance(value, str) and "-" in value:
                try:
                    start_str, end_str = value.split("-", 1)
                    start = int(float(start_str))
                    end = int(float(end_str))
                except ValueError:
                    continue
                for hour in range(start, end + 1):
                    hours.add(hour % 24)
            else:
                try:
                    hours.add(int(float(value)) % 24)
                except (TypeError, ValueError):
                    continue
        return hours
