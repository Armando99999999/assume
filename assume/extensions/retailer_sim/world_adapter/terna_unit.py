"""Terna-like balancing unit usata nel world adapter."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np

from assume.common.base import SupportsMinMax
from assume.common.fast_pandas import FastSeries
from assume.common.forecasts import Forecaster


class TernaBalancingUnit(SupportsMinMax):
    """
    Rappresentazione semplificata della controparte Terna nel world adapter.

    Espone min/max di potenza costanti (o profilati via forecaster) e fornisce i
    prezzi ai mercati MI/MGP per alimentare la strategia di balancing. La logica
    resta volutamente minimale per non duplicare il comportamento del retailer.
    """

    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        bidding_strategies: Dict[str, str],
        max_power: float,
        min_power: float,
        forecaster: Forecaster,
        node: str = "node0",
        location: tuple[float, float] = (0.0, 0.0),
        forecast_column: str = "",
        price_column: str = "mgp_price",
        market_price_columns: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            bidding_strategies=bidding_strategies,
            forecaster=forecaster,
            node=node,
            location=location,
            **kwargs,
        )
        self.max_power = max_power
        self.min_power = min_power
        self.ramp_up = max_power
        self.ramp_down = max_power

        # Profilo di capacità (costante o derivato da una colonna del forecaster).
        self.capacity_profile = (
            self._load_series(forecast_column) if forecast_column else FastSeries(index=self.index, value=max_power)
        )
        self.price_series: Dict[str, FastSeries] = {}
        if price_column:
            self.price_series["MGP"] = self._load_series(price_column)
        for market_id, column in (market_price_columns or {}).items():
            self.price_series[market_id] = self._load_series(column)

    def _load_series(self, column: str) -> FastSeries:
        """Helper interno per standardizzare le serie provenienti dal forecaster."""
        series = self.forecaster[column]
        if isinstance(series, FastSeries):
            return series
        return FastSeries(index=self.index, value=series)

    def execute_current_dispatch(self, start: datetime, end: datetime):
        return self.outputs["energy"].loc[start:end]

    def calculate_min_max_power(
        self,
        start: datetime,
        end: datetime,
        product_type: str = "energy",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Restituisce gli envelope min/max (costanti o legati al profilo capacità)."""
        if end <= start:
            return np.array([]), np.array([])
        end_excl = end - self.index.freq
        capacity = self.capacity_profile.loc[start:end_excl]
        if not isinstance(capacity, np.ndarray):
            horizon = int(round((end - start).total_seconds() / self.index.freq_seconds))
            capacity = np.full(horizon, self.max_power)
        max_profile = np.clip(capacity, self.min_power, self.max_power)
        min_profile = np.full_like(max_profile, self.min_power)
        return min_profile, max_profile

    def calculate_marginal_cost(self, start: datetime, power: float, market_id: str | None = None) -> float:
        """Restituisce il prezzo marginale del mercato richiesto (default MGP)."""
        price_series = self.price_series.get((market_id or "MGP").upper())
        if price_series is None:
            return 0.0
        return float(price_series.at[start])
