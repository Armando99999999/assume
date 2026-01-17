# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Agente retailer che coordina il trading semplificato su MGP/MI (standalone).

Walkthrough in stile colloquio tecnico
--------------------------------------
1. ``HourlyResult`` colleziona tutti i numeri per slot: trade MGP/MI, sbilanciamenti,
   costi MSD e ricavi. Il metodo ``as_dict`` appiattisce i dati per la reportistica.
2. ``Retailer`` e' l'orchestratore: costruisce gli ``StrategyInputs``, invia ordini al
   mercato fittizio e contabilizza sbilanciamenti + surplus. Il comportamento deve
   restare identico al codice originale.
3. Gli helper privati (prefisso ``_``) gestiscono normalizzazioni, calcoli MSD,
   salvaguardie e generazione ordini; in questo refactoring li abbiamo documentati
   e raggruppati in sezioni per facilitarne la lettura.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import logging
import pandas as pd

from .market import MarketEnvironment, MSDSettlement, TradeResult
from .strategies import BiddingStrategy, StrategyInputs
from ..decision import apply_retailer_logic_presets
from ..terna_agent import TernaLikeBalancingAgent


@dataclass
class HourlyResult:
    """
    Snapshot orario di tutti i flussi energetici/economici.

    Questo oggetto tiene traccia dei trade MGP/MI, dello sbilanciamento (con relativo
    costo MSD), dell'eventuale surplus venduto e dei ricavi verso clienti. Rimane una
    struttura dati neutra; la logica di simulazione vive nella classe ``Retailer``.
    """

    timestamp: pd.Timestamp
    mgp_trade: TradeResult
    session_trades: Dict[str, Optional[TradeResult]]
    consumption_forecast_MWh: float
    actual_consumption_MWh: float
    imbalance_MWh: float
    imbalance_cost_EUR: float
    imbalance_price_EUR_MWh: float
    imbalance_value_gap_EUR: float
    imbalance_direction: str
    surplus_sold_MWh: float
    surplus_sale_revenue_EUR: float
    customer_revenue_EUR: float
    purchase_price_EUR_MWh: float
    retail_tariff_EUR_MWh: float
    terna_penalty_EUR: float
    terna_incentive_EUR: float
    session_names: tuple[str, ...]

    def as_dict(self) -> Dict[str, float]:
        """Appiattisce i campi in un dizionario adatto alla creazione di DataFrame/report."""
        contracted = self.mgp_trade.volume_MWh
        session_costs = 0.0
        session_revenues = 0.0
        followup_volume = 0.0
        followup_cost = 0.0
        followup_revenue = 0.0
        followup_price = 0.0
        data: Dict[str, float] = {
            "timestamp": self.timestamp,
            "mgp_volume_MWh": self.mgp_trade.volume_MWh,
            "mgp_price_EUR_MWh": self.mgp_trade.price_EUR_MWh,
            "mgp_cost_EUR": self.mgp_trade.cost_EUR,
            "mgp_revenue_EUR": self.mgp_trade.revenue_EUR,
            "consumption_forecast_MWh": self.consumption_forecast_MWh,
            "actual_consumption_MWh": self.actual_consumption_MWh,
            "imbalance_MWh": self.imbalance_MWh,
            "imbalance_direction": self.imbalance_direction,
            "imbalance_cost_EUR": self.imbalance_cost_EUR,
            "imbalance_price_EUR_MWh": self.imbalance_price_EUR_MWh,
            "imbalance_value_gap_EUR": self.imbalance_value_gap_EUR,
            # Il surplus rappresenta energia immessa nel sistema e valorizzata dall'MSD,
            # non una vendita a un acquirente specifico.
            "surplus_sold_MWh": self.surplus_sold_MWh,
            "surplus_sale_revenue_EUR": self.surplus_sale_revenue_EUR,
            "customer_revenue_EUR": self.customer_revenue_EUR,
            "purchase_price_EUR_MWh": self.purchase_price_EUR_MWh,
            "retail_tariff_EUR_MWh": self.retail_tariff_EUR_MWh,
            "terna_penalty_EUR": self.terna_penalty_EUR,
            "terna_incentive_EUR": self.terna_incentive_EUR,
        }

        for idx, name in enumerate(self.session_names):
            trade = self.session_trades.get(name)
            volume = trade.volume_MWh if trade else 0.0
            price = trade.price_EUR_MWh if trade else 0.0
            cost = trade.cost_EUR if trade else 0.0
            revenue = trade.revenue_EUR if trade else 0.0
            contracted += volume
            session_costs += cost
            session_revenues += revenue
            prefix = name.lower()
            data[f"{prefix}_volume_MWh"] = volume
            data[f"{prefix}_price_EUR_MWh"] = price
            data[f"{prefix}_cost_EUR"] = cost
            data[f"{prefix}_revenue_EUR"] = revenue
            if idx == 0:
                data["mi_volume_MWh"] = volume
                data["mi_price_EUR_MWh"] = price
                data["mi_cost_EUR"] = cost
                data["mi_revenue_EUR"] = revenue
            else:
                followup_volume += volume
                followup_cost += cost
                followup_revenue += revenue
                if volume:
                    followup_price = price

        data.setdefault("mi_volume_MWh", 0.0)
        data.setdefault("mi_price_EUR_MWh", 0.0)
        data.setdefault("mi_cost_EUR", 0.0)
        data.setdefault("mi_revenue_EUR", 0.0)
        data["mi2_volume_MWh"] = followup_volume
        data["mi2_cost_EUR"] = followup_cost
        data["mi2_revenue_EUR"] = followup_revenue
        data["mi2_price_EUR_MWh"] = followup_price if followup_volume else 0.0
        data["contracted_volume_MWh"] = contracted

        revenues = (
            self.mgp_trade.revenue_EUR
            + session_revenues
            + self.surplus_sale_revenue_EUR
            + self.customer_revenue_EUR
        )
        costs = self.mgp_trade.cost_EUR + session_costs + self.imbalance_cost_EUR
        data["hourly_cost_EUR"] = costs
        data["hourly_revenue_EUR"] = revenues
        data["hourly_profit_EUR"] = revenues - costs
        return data


class Retailer:
    """
    Agente retailer semplificato usato al di fuori del clearing World.

    Responsabilità principali:
    - preparare gli StrategyInputs identici a quelli dello standalone;
    - inviare le offerte ai mercati fittizi (MGP + MI) e registrare gli ordini;
    - calcolare sbilanciamenti, costi energetici e ricavi per ogni intervallo orario;
    - quantificare il differenziale di valore dello sbilanciamento e i segnali Terna.

    Nota bene: quando il retailer acquista più energia del necessario il surplus non
    viene “venduto” a un mercato con un vero acquirente ma viene immesso automaticamente
    nel sistema e valorizzato dal gestore di rete (Terna) al prezzo MSD risultante.
    Questo surplus è quindi modellato come sbilanciamento positivo e appare come costo
    (o come credito ridotto) nella rendicontazione MSD.
    """

    def __init__(
        self,
        name: str,
        *,
        strategy: BiddingStrategy,
        market_env: MarketEnvironment,
        timestamp_col: str = "timestamp",
        consumption_col: str = "consumption_forecast_MWh",
        actual_consumption_col: Optional[str] = "actual_consumption_MWh",
        mgp_price_col: str = "price_MGP_EUR_MWh",
        mi_price_col: str = "price_MI_EUR_MWh",
        mi2_price_col: Optional[str] = None,
        imbalance_col: Optional[str] = "imbalance_forecast_MWh",
        macro_imbalance_col: Optional[str] = None,
        macrozone_price_col: str = "price_macrozone_avg_EUR_MWh",
        imbalance_coeff_col: str = "imbalance_coeff",
        imbalance_penalty_cost_per_MWh: float = 0.0,
        retailer_logic: Optional[Dict[str, Any]] = None,
        macro_forecast_cfg: Optional[Dict[str, Any]] = None,
        msd_settlement: Optional[MSDSettlement] = None,
        intraday_sessions: Optional[List[Dict[str, Any]]] = None,
        time_step_minutes: Optional[int] = None,
        real_market: Optional[bool] = None,
    ) -> None:
        self.name = name
        self.strategy = strategy
        self.market_env = market_env
        self.timestamp_col = timestamp_col
        self.consumption_col = consumption_col
        self.actual_consumption_col = actual_consumption_col
        self.mgp_price_col = mgp_price_col
        self.mi_price_col = mi_price_col
        self.mi2_price_col = mi2_price_col or mi_price_col
        self.imbalance_col = imbalance_col
        self.macro_imbalance_col = macro_imbalance_col or imbalance_col
        self.macrozone_price_col = macrozone_price_col
        self.imbalance_coeff_col = imbalance_coeff_col
        self.imbalance_penalty_cost_per_MWh = imbalance_penalty_cost_per_MWh
        self.logic_config = apply_retailer_logic_presets(retailer_logic or {})
        self.logger = logging.getLogger(f"{__name__}.{name}")
        msd_cfg = dict(self.logic_config.get("msd_settlement", {}))
        sensitivity = float(msd_cfg.get("price_sensitivity", self.logic_config.get("msd_price_sensitivity", 0.45)))
        additional_penalty = self.imbalance_penalty_cost_per_MWh + float(
            msd_cfg.get("additional_penalty_EUR_MWh", self.logic_config.get("msd_additional_penalty_EUR_MWh", 0.0))
        )
        credit_factor = float(
            msd_cfg.get("long_position_credit_factor", self.logic_config.get("long_position_credit_factor", 1.0))
        )
        self.msd_settlement = msd_settlement or MSDSettlement(
            price_sensitivity=sensitivity,
            additional_penalty_EUR_MWh=additional_penalty,
            long_position_credit_factor=credit_factor,
        )
        self._hourly_results: List[HourlyResult] = []
        self._hourly_df: Optional[pd.DataFrame] = None
        self.session_definitions = self._normalize_sessions(intraday_sessions)
        self.session_names = tuple(session["name"] for session in self.session_definitions)
        self.time_step_minutes = int(time_step_minutes) if time_step_minutes else None
        self.real_market = bool(real_market)
        self.strategy_label = self._infer_strategy_label(strategy)
        self._orders: List[Dict[str, Any]] = []
        self.terna_agent = TernaLikeBalancingAgent(self.logic_config.get("terna_agent", {}))
        self.macro_forecast_cfg = dict(macro_forecast_cfg or {})
        self.macro_forecast_enabled = bool(self.macro_forecast_cfg.get("enabled"))
        self.macro_forecast_source = self.macro_forecast_cfg.get("source_column") or self.macro_imbalance_col
        self.macro_band_mgp = float(self.macro_forecast_cfg.get("band_mgp", 0.0))
        self.macro_band_mi = float(self.macro_forecast_cfg.get("band_mi", self.macro_band_mgp))
        if self.macro_forecast_enabled:
            self._guard_forecast_column(self.macro_forecast_source)
        self.customer_margin = float(self.logic_config.get("customer_margin_EUR_MWh", 25.0))

    # ------------------------------------------------------------------ #
    # Entry point della simulazione                                      #
    # ------------------------------------------------------------------ #

    def simulate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Esegue la simulazione del retailer sul DataFrame fornito.

        Fasi principali:
            1. Validazione delle colonne richieste e normalizzazione delle serie.
            2. Loop per giorno -> slot: costruzione StrategyInputs + offerte MGP/MI.
            3. Calcolo sbilanciamento, costo MSD (inclusi segnali Terna) e surplus.
        """
        required = {
            self.timestamp_col,
            self.consumption_col,
            self.mgp_price_col,
            self.mi_price_col,
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        working_df = df.copy()
        working_df[self.timestamp_col] = pd.to_datetime(working_df[self.timestamp_col])
        self.market_env.reset()
        self._orders = []
        self._ensure_time_step_minutes(working_df)
        if "slot_hour_float" not in working_df.columns:
            working_df["slot_hour_float"] = (
                working_df[self.timestamp_col].dt.hour + working_df[self.timestamp_col].dt.minute / 60
            )

        macrozone_prices = self._prepare_macrozone_prices(working_df)
        imbalance_coeffs = self._prepare_imbalance_coeffs(working_df)
        macro_imbalances = self._prepare_macro_imbalance(working_df)
        imbalance_forecasts = (
            working_df[self.imbalance_col].astype(float)
            if self.imbalance_col and self.imbalance_col in working_df
            else None
        )
        sale_fraction = min(max(float(self.logic_config.get("surplus_sale_fraction", 1.0)), 0.0), 1.0)

        session_definitions = list(self.session_definitions) or self._normalize_sessions(None)
        if not self.strategy.supports_second_session():
            session_definitions = session_definitions[:1]
        active_session_names = tuple(session["name"] for session in session_definitions)

        inventory = 0.0
        results: List[HourlyResult] = []
        daily_groups = working_df.groupby(working_df[self.timestamp_col].dt.normalize())
        for _, daily_frame in daily_groups:
            slot_records: List[Dict[str, Any]] = []
            for idx, slot_row in daily_frame.iterrows():
                timestamp = pd.Timestamp(slot_row[self.timestamp_col])
                consumption_forecast = float(slot_row[self.consumption_col])
                actual_consumption = consumption_forecast
                if self.actual_consumption_col and self.actual_consumption_col in slot_row:
                    actual_value = slot_row[self.actual_consumption_col]
                    actual_consumption = float(actual_value) if not pd.isna(actual_value) else consumption_forecast
                mgp_price = float(slot_row[self.mgp_price_col])
                mi_price = float(slot_row[self.mi_price_col])
                macro_price = float(macrozone_prices.loc[idx])
                imbalance_coeff = float(imbalance_coeffs.loc[idx])
                macro_mean = float(macro_imbalances.loc[idx])
                macro_sign = self._macro_forecast_sign(macro_mean)
                band = self.macro_band_mgp if self.macro_forecast_enabled else 0.0
                macro_low = macro_mean - band
                macro_high = macro_mean + band
                raw_forecast = float(imbalance_forecasts.loc[idx]) if imbalance_forecasts is not None else float("nan")
                normalized_forecast = self._normalize_imbalance_forecast(raw_forecast, fallback=consumption_forecast)
                msd_price_estimate, _, _, _ = self._compute_imbalance_price(
                    timestamp=timestamp,
                    mgp_price=mgp_price,
                    macrozone_price=macro_price,
                    imbalance_coeff=imbalance_coeff,
                    imbalance_input_MWh=normalized_forecast,
                    macro_imbalance_MWh=macro_mean,
                )
                inputs = StrategyInputs(
                    timestamp=timestamp,
                    demand_forecast_MWh=consumption_forecast,
                    mgp_price_EUR_MWh=mgp_price,
                    mi_price_EUR_MWh=mi_price,
                    macro_price_EUR_MWh=macro_price,
                    imbalance_forecast_MWh=raw_forecast,
                    macro_imbalance_MWh=macro_mean,
                    imbalance_coeff=imbalance_coeff,
                    msd_price_estimate_EUR_MWh=msd_price_estimate,
                    macro_forecast_mean_MWh=macro_mean,
                    macro_forecast_low_MWh=macro_low,
                    macro_forecast_high_MWh=macro_high,
                    macro_forecast_sign=macro_sign,
                )
                slot_records.append(
                    {
                        "idx": idx,
                        "row": slot_row,
                        "timestamp": timestamp,
                        "consumption_forecast": consumption_forecast,
                        "actual_consumption": actual_consumption,
                        "mgp_price": mgp_price,
                        "inputs": inputs,
                    }
                )

            mgp_trades: Dict[int, TradeResult] = {}
            contracted_positions: Dict[int, float] = {}
            for record in slot_records:
                desired_volume = self.strategy.plan_day_ahead(record["inputs"])
                mgp_volume = self._apply_mgp_regulator(
                    desired_volume_MWh=desired_volume,
                    demand_forecast_MWh=record["consumption_forecast"],
                )
                mgp_trade = self.market_env.execute_day_ahead_bid(
                    timestamp=record["timestamp"],
                    volume_MWh=mgp_volume,
                    price_EUR_MWh=record["mgp_price"],
                )
                mgp_trades[record["idx"]] = mgp_trade
                contracted_positions[record["idx"]] = mgp_volume
                self._record_order(
                    market="MGP",
                    timestamp=record["timestamp"],
                    volume_MWh=mgp_trade.volume_MWh,
                    price_EUR_MWh=mgp_trade.price_EUR_MWh,
                    forecast_source="naive",
                    liquidity=None,
                )

            session_trade_map: Dict[str, Dict[int, TradeResult]] = {
                session["name"]: {} for session in session_definitions
            }
            for session_order, session in enumerate(session_definitions, start=1):
                for record in slot_records:
                    slot_row = record["row"]
                    if not self._is_slot_eligible(slot_row, session):
                        continue
                    contracted_before = contracted_positions[record["idx"]]
                    session_price = self._resolve_intraday_price(slot_row, session_order)
                    band_override = self.macro_band_mi if self.macro_forecast_enabled else None
                    inputs_session = record["inputs"].for_intraday(
                        mi_price_EUR_MWh=session_price,
                        macro_band=band_override,
                    )
                    mi_volume = self.strategy.plan_intraday(
                        inputs_session,
                        contracted_volume_MWh=contracted_before,
                        session=session_order,
                    )
                    max_fraction_key = "mi1_max_fraction" if session_order == 1 else "mi2_max_fraction"
                    smoothing_key = "mi1_smoothing" if session_order == 1 else "mi2_smoothing"
                    max_fraction = float(self.logic_config.get(max_fraction_key, 0.6 if session_order == 1 else 0.3))
                    smoothing = float(self.logic_config.get(smoothing_key, 0.95 if session_order == 1 else 0.85))
                    liquidity_hint = self._resolve_session_liquidity(slot_row, session)
                    mi_volume = self._apply_intraday_regulator(
                        desired_volume_MWh=mi_volume,
                        demand_forecast_MWh=record["consumption_forecast"],
                        contracted_before_MWh=contracted_before,
                        max_fraction=max_fraction,
                        smoothing=smoothing,
                        liquidity_hint=liquidity_hint,
                    )
                    if abs(mi_volume) < 1e-9:
                        continue
                    mi_trade = self.market_env.execute_intraday_bid(
                        timestamp=record["timestamp"],
                        volume_MWh=mi_volume,
                        price_EUR_MWh=session_price,
                        market_label=session["name"],
                    )
                    contracted_positions[record["idx"]] = contracted_before + mi_trade.volume_MWh
                    session_trade_map[session["name"]][record["idx"]] = mi_trade
                    self._record_order(
                        market=session["name"],
                        timestamp=record["timestamp"],
                        volume_MWh=mi_trade.volume_MWh,
                        price_EUR_MWh=mi_trade.price_EUR_MWh,
                        forecast_source="updated",
                        liquidity=liquidity_hint,
                    )

            for record in slot_records:
                idx = record["idx"]
                slot_row = record["row"]
                timestamp = record["timestamp"]
                mgp_trade = mgp_trades[idx]
                session_trades = {name: trades.get(idx) for name, trades in session_trade_map.items()}
                contracted_total = contracted_positions[idx]
                actual_consumption = record["actual_consumption"]
                # Sbilanciamento: consumi reali - contratti (short positivo, long negativo).
                imbalance_raw_MWh = actual_consumption - contracted_total
                imbalance_report_MWh = imbalance_raw_MWh
                macro_value = float(macro_imbalances.loc[idx])
                (
                    imbalance_price,
                    terna_penalty_EUR_MWh,
                    terna_incentive_EUR_MWh,
                    base_msd_price,
                ) = self._compute_imbalance_price(
                    timestamp=timestamp,
                    mgp_price=record["mgp_price"],
                    macrozone_price=float(macrozone_prices.loc[idx]),
                    imbalance_coeff=float(imbalance_coeffs.loc[idx]),
                    imbalance_input_MWh=imbalance_raw_MWh,
                    macro_imbalance_MWh=macro_value,
                )
                (
                    imbalance_cost,
                    terna_penalty_EUR,
                    terna_incentive_EUR,
                ) = self._evaluate_imbalance_cost(
                    imbalance_MWh=imbalance_raw_MWh,
                    base_price_EUR_MWh=base_msd_price,
                    penalty_EUR_MWh=terna_penalty_EUR_MWh,
                    incentive_EUR_MWh=terna_incentive_EUR_MWh,
                    macro_imbalance_MWh=macro_value,
                )
                self._record_balancing_order(
                    timestamp=timestamp,
                    volume_MWh=imbalance_raw_MWh,
                    price_EUR_MWh=imbalance_price,
                )

                # Energia acquistata in eccesso: viene registrata come sbilanciamento long
                # e valorizzata dall'MSD (non esiste una vera controparte di vendita).
                # surplus calcolato esplicitamente come (contratti - consumi) se positivo
                surplus_before_sale = max(contracted_total - actual_consumption, 0.0)
                sell_price = self._determine_surplus_price(slot_row)
                surplus_sold = surplus_before_sale * sale_fraction  # liquidato nello stesso slot
                surplus_revenue = surplus_sold * sell_price
                inventory = 0.0  # legacy var, non usata

                imbalance_direction = "short" if imbalance_report_MWh > 0 else ("long" if imbalance_report_MWh < 0 else "balanced")
                purchase_price = self._compute_purchase_price(mgp_trade, session_trades)
                # Differenza economica fra valutazione MSD e costo medio di acquisto.
                imbalance_value_gap = self._compute_imbalance_value_gap(
                    imbalance_MWh=imbalance_raw_MWh,
                    imbalance_price_EUR_MWh=imbalance_price,
                    purchase_price_EUR_MWh=purchase_price,
                )
                imbalance_cost_per_MWh = self._per_MWh(imbalance_cost, imbalance_raw_MWh)
                customer_tariff = max(purchase_price + self.customer_margin, 0.0)
                customer_revenue = customer_tariff * actual_consumption
                purchased_volume, energy_cost = self._aggregate_energy_stats(mgp_trade, session_trades)

                results.append(
                    HourlyResult(
                        timestamp=timestamp,
                        mgp_trade=mgp_trade,
                        session_trades=session_trades,
                        consumption_forecast_MWh=record["consumption_forecast"],
                        actual_consumption_MWh=actual_consumption,
                        imbalance_MWh=imbalance_report_MWh,
                        imbalance_cost_EUR=imbalance_cost,
                        imbalance_price_EUR_MWh=imbalance_price,
                        imbalance_value_gap_EUR=imbalance_value_gap,
                        imbalance_direction=imbalance_direction,
                        surplus_sold_MWh=surplus_sold,
                        surplus_sale_revenue_EUR=surplus_revenue,
                        customer_revenue_EUR=customer_revenue,
                        purchase_price_EUR_MWh=purchase_price,
                        retail_tariff_EUR_MWh=customer_tariff,
                        terna_penalty_EUR=terna_penalty_EUR,
                        terna_incentive_EUR=terna_incentive_EUR,
                        session_names=active_session_names,
                    )
                )

        self._hourly_results = results
        self._hourly_df = pd.DataFrame(res.as_dict() for res in results)
        if self.imbalance_col and self.imbalance_col in working_df.columns:
            self._hourly_df[self.imbalance_col] = working_df[self.imbalance_col].values
        # Conserva il prezzo MI di mercato anche quando non ci sono trade MI (evita zeri in output).
        if self.mi_price_col in working_df.columns:
            mi_market_prices = working_df[self.mi_price_col].values
            self._hourly_df["mi_market_price_EUR_MWh"] = mi_market_prices
            if "mi_price_EUR_MWh" in self._hourly_df.columns:
                zero_mask = self._hourly_df["mi_price_EUR_MWh"] == 0
                self._hourly_df.loc[zero_mask, "mi_price_EUR_MWh"] = mi_market_prices[zero_mask]
        return self._hourly_df.copy()

    # ------------------------------------------------------------------ #
    # Output pubblici (dataframe, ordini, KPI)                           #
    # ------------------------------------------------------------------ #

    def compute_totals(self) -> Dict[str, float]:
        """Somma costi e ricavi orari in un dizionario complessivo."""
        if self._hourly_df is None:
            raise ValueError("Run simulate() first.")
        costs = float(self._hourly_df["hourly_cost_EUR"].sum())
        revenues = float(self._hourly_df["hourly_revenue_EUR"].sum())
        totals = {
            "total_costs_EUR": costs,
            "total_revenues_EUR": revenues,
            "total_profit_EUR": revenues - costs,
        }
        return totals

    def get_hourly_results(self) -> pd.DataFrame:
        """Restituisce il DataFrame orario (copia) generato da simulate()."""
        return self._hourly_df.copy() if self._hourly_df is not None else pd.DataFrame()

    def get_orders(self) -> List[Dict[str, Any]]:
        """Restituisce la lista di ordini generati nei vari mercati."""
        return list(self._orders)

    # ------------------------------------------------------------------ #
    # Helper privati (normalizzazioni, salvaguardie, calcoli MSD)        #
    # ------------------------------------------------------------------ #

    def _infer_strategy_label(self, strategy: BiddingStrategy) -> str:
        return strategy.__class__.__name__.replace("Strategy", "").lower()

    def _normalize_sessions(self, sessions: Optional[List[Dict[str, Any]]]) -> List[Dict[str, float]]:
        """Uniforma la configurazione delle sessioni MI (name, orari, liquidity)."""
        if not sessions:
            sessions = [
                {"name": "MI1", "open_hour": 8.0, "close_hour": 10.0, "coverage_start_hour": 10.0, "coverage_end_hour": 14.0, "liquidity": 0.85},
                {"name": "MI2", "open_hour": 10.0, "close_hour": 12.0, "coverage_start_hour": 14.0, "coverage_end_hour": 28.0, "liquidity": 0.75},
            ]
        normalized: List[Dict[str, float]] = []
        for entry in sessions:
            name = str(entry.get("name", "MI"))
            open_hour = float(entry.get("open_hour", entry.get("coverage_start_hour", 0.0)))
            close_hour = float(entry.get("close_hour", entry.get("coverage_end_hour", 24.0)))
            coverage_start = float(entry.get("coverage_start_hour", open_hour))
            coverage_end = float(entry.get("coverage_end_hour", close_hour))
            liquidity = float(entry.get("liquidity", 1.0))
            normalized.append(
                {
                    "name": name,
                    "open_hour": open_hour,
                    "close_hour": close_hour,
                    "coverage_start_hour": coverage_start,
                    "coverage_end_hour": coverage_end,
                    "liquidity": liquidity,
                }
            )
        return normalized

    def _ensure_time_step_minutes(self, df: pd.DataFrame) -> int:
        """Determina il passo temporale in minuti (ricavato dai timestamp se non noto)."""
        if self.time_step_minutes:
            return self.time_step_minutes
        series = pd.to_datetime(df[self.timestamp_col])
        diffs = series.sort_values().diff().dropna()
        minutes = 60
        if not diffs.empty:
            mode = diffs.mode()
            delta = mode.iloc[0] if not mode.empty else diffs.iloc[0]
            minutes = max(int(delta.total_seconds() // 60) or 1, 1)
        self.time_step_minutes = minutes
        return minutes

    def _slot_in_window(self, hour_value: float, start: float, end: float) -> bool:
        """True se un'ora (in formato float) cade nella finestra [start, end) considerando rollover oltre mezzanotte."""
        if end <= start:
            end += 24
        hour = hour_value
        if hour < start:
            hour += 24
        return start <= hour < end

    def _is_slot_eligible(self, row: pd.Series, session: Dict[str, float]) -> bool:
        """Verifica se uno slot e' eleggibile per la sessione MI (flag serie o fallback su finestre orarie)."""
        flag_col = f"is_{session['name']}_eligible"
        value = row.get(flag_col)
        if value is not None and value == value:
            return bool(value)
        hour_value = float(
            row.get(
                "slot_hour_float",
                row[self.timestamp_col].hour + row[self.timestamp_col].minute / 60,
            )
        )
        return self._slot_in_window(hour_value, session["coverage_start_hour"], session["coverage_end_hour"])

    def _resolve_session_liquidity(self, row: pd.Series, session: Dict[str, float]) -> float:
        """Restituisce l'hint di liquidita' per la sessione, con fallback su valore configurato."""
        col = f"{session['name']}_liquidity_hint"
        value = row.get(col)
        if value is None or not value == value:
            return float(session.get("liquidity", 1.0))
        return float(value) if value > 0 else 0.0

    def _resolve_intraday_price(self, row: pd.Series, session_index: int) -> float:
        """Sceglie il prezzo MI per la sessione (colonna dedicata, MI1, oppure fallback MI2)."""
        if 1 <= session_index <= len(self.session_names):
            session_name = self.session_names[session_index - 1]
            custom_col = f"price_{session_name}_EUR_MWh"
            value = row.get(custom_col)
            if value is not None and value == value:
                return float(value)
        if session_index == 1:
            return float(row[self.mi_price_col])
        fallback = row.get(self.mi2_price_col, row.get(self.mi_price_col, 0.0))
        return float(fallback)

    def _record_order(
        self,
        *,
        market: str,
        timestamp: pd.Timestamp,
        volume_MWh: float,
        price_EUR_MWh: float,
        forecast_source: str,
        liquidity: Optional[float] = None,
    ) -> None:
        """Appende un ordine (MGP o MI) al log `_orders`, convertendo il volume da MWh a MW/slot."""
        if abs(volume_MWh) < 1e-6:
            return
        time_step = self.time_step_minutes or 60
        slot_end = timestamp + pd.Timedelta(minutes=time_step)
        quantity_mw = volume_MWh * (60 / time_step)
        self._orders.append(
            {
                "market": market,
                "time": timestamp.isoformat(),
                "orders": [
                    {
                        "slot": f"{timestamp.strftime('%H:%M')}-{slot_end.strftime('%H:%M')}",
                        "quantity_mw": quantity_mw,
                        "side": "buy" if volume_MWh >= 0 else "sell",
                        "price": price_EUR_MWh,
                    }
                ],
                "metadata": {
                    "strategy": self.strategy_label,
                    "source_forecast": forecast_source,
                    "real_market": self.real_market,
                    "liquidity": liquidity,
                },
            }
        )

    def _record_balancing_order(
        self,
        *,
        timestamp: pd.Timestamp,
        volume_MWh: float,
        price_EUR_MWh: float,
    ) -> None:
        """Registra l'ordine MSD (sbilanciamento) per completare il log ordini."""
        if abs(volume_MWh) < 1e-6:
            return
        time_step = self.time_step_minutes or 60
        slot_end = timestamp + pd.Timedelta(minutes=time_step)
        quantity_mw = volume_MWh * (60 / time_step)
        self._orders.append(
            {
                "market": "MSD",
                "time": timestamp.isoformat(),
                "orders": [
                    {
                        "slot": f"{timestamp.strftime('%H:%M')}-{slot_end.strftime('%H:%M')}",
                        "quantity_mw": quantity_mw,
                        "side": "buy" if volume_MWh >= 0 else "sell",
                        "price": price_EUR_MWh,
                    }
                ],
                "metadata": {
                    "strategy": self.strategy_label,
                    "source_forecast": "actual",
                    "real_market": self.real_market,
                },
            }
        )

    def _prepare_macrozone_prices(self, df: pd.DataFrame) -> pd.Series:
        """Ritorna la serie di prezzo macrozonale (o media mobile dei prezzi MGP se manca la colonna)."""
        if self.macrozone_price_col in df.columns:
            series = df[self.macrozone_price_col].astype(float)
        else:
            window = max(int(self.logic_config.get("macrozone_price_window_hours", 6)), 1)
            series = (
                df[self.mgp_price_col]
                .rolling(window=window, min_periods=1)
                .mean()
            )
        return series.bfill().ffill()

    def _prepare_imbalance_coeffs(self, df: pd.DataFrame) -> pd.Series:
        """Normalizza/riempie i coefficienti di sbilanciamento usati nella stima MSD."""
        if self.imbalance_coeff_col in df.columns:
            series = df[self.imbalance_coeff_col].astype(float)
        else:
            default = float(self.logic_config.get("imbalance_coeff_default", 0.15))
            series = pd.Series(default, index=df.index)
        return series.fillna(float(self.logic_config.get("imbalance_coeff_default", 0.15)))

    def _prepare_macro_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Restituisce la serie macro usata per incentivi/penalità."""
        column = self.macro_forecast_source or self.macro_imbalance_col
        if column and column in df.columns:
            if self.macro_forecast_enabled:
                self._guard_forecast_column(column)
            return df[column].astype(float).fillna(0.0)
        return pd.Series(0.0, index=df.index)

    def _guard_forecast_column(self, column: Optional[str]) -> None:
        """Evita di usare colonne che contengono dati reali anziché previsioni."""
        if not column:
            return
        lowered = str(column).lower()
        if "real" in lowered or "actual" in lowered:
            raise ValueError(f"Macro imbalance forecast column '{column}' contains real data.")

    @staticmethod
    def _macro_forecast_sign(value: float) -> float:
        """Converte il segno dello sbilanciamento macro in {-1, 0, 1}."""
        if value > 0:
            return 1.0
        if value < 0:
            return -1.0
        return 0.0

    def _compute_imbalance_price(
        self,
        *,
        timestamp: pd.Timestamp,
        mgp_price: float,
        macrozone_price: float,
        imbalance_coeff: float,
        imbalance_input_MWh: float,
        macro_imbalance_MWh: float,
    ) -> tuple[float, float, float, float]:
        """
        Restituisce il prezzo MSD finale, i segnali Terna e il prezzo base (pre-incentivi).
        """
        base_price = self.msd_settlement.estimate_price(
            mgp_price=mgp_price,
            macrozone_price=macrozone_price,
            imbalance_coeff=imbalance_coeff,
            imbalance_input_MWh=imbalance_input_MWh,
        )
        adjustment = self.terna_agent.evaluate(
            timestamp=timestamp,
            residual_MWh=imbalance_input_MWh,
            macro_imbalance_MWh=macro_imbalance_MWh,
            base_price_EUR_MWh=base_price,
            mgp_price_EUR_MWh=mgp_price,
        )
        return (
            adjustment.adjusted_price_EUR_MWh,
            adjustment.penalty_EUR_MWh,
            adjustment.incentive_EUR_MWh,
            base_price,
        )

    def _evaluate_imbalance_cost(
        self,
        *,
        imbalance_MWh: float,
        base_price_EUR_MWh: float,
        penalty_EUR_MWh: float,
        incentive_EUR_MWh: float,
        macro_imbalance_MWh: float,
    ) -> tuple[float, float, float]:
        base_cost = self.msd_settlement.cost_from_price(imbalance_MWh, base_price_EUR_MWh)
        volume = abs(imbalance_MWh)
        if volume <= 1e-9:
            return 0.0, 0.0, 0.0
        # Applica il credito solo se la nostra posizione aiuta a ridurre lo sbilanciamento macro.
        helps_macro = (imbalance_MWh > 0 and macro_imbalance_MWh < 0) or (imbalance_MWh < 0 and macro_imbalance_MWh > 0)
        long_factor = self.msd_settlement.long_position_credit_factor if helps_macro else 1.0
        penalty_total = penalty_EUR_MWh * volume * long_factor
        incentive_total = incentive_EUR_MWh * volume * long_factor
        imbalance_cost = base_cost + penalty_total - incentive_total
        return imbalance_cost, penalty_total, incentive_total
    def _determine_surplus_price(self, row: pd.Series) -> float:
        """Scelta del prezzo per il surplus (equilibrium oppure riferimenti MGP/MI/custom)."""
        mode = str(self.logic_config.get("surplus_sale_price", "equilibrium")).lower()
        mgp_price = float(row[self.mgp_price_col])
        mi_price = float(row[self.mi_price_col])
        if mode == "mgp":
            return mgp_price
        if mode == "mi":
            return mi_price
        if mode == "msd":
            return float(row.get("imbalance_price_EUR_MWh", mgp_price))
        if mode == "custom":
            return float(self.logic_config.get("surplus_sale_custom_price", mgp_price))
        return (mgp_price + mi_price) / 2

    def _compute_purchase_price(
        self,
        mgp_trade: TradeResult,
        session_trades: Dict[str, Optional[TradeResult]],
    ) -> float:
        total_volume = max(mgp_trade.volume_MWh, 0.0)
        total_cost = mgp_trade.cost_EUR if mgp_trade.volume_MWh >= 0 else 0.0
        for trade in session_trades.values():
            if trade and trade.volume_MWh >= 0:
                total_volume += trade.volume_MWh
                total_cost += trade.cost_EUR
        if total_volume <= 1e-9:
            return float(mgp_trade.price_EUR_MWh)
        return total_cost / total_volume

    def _compute_imbalance_value_gap(
        self,
        *,
        imbalance_MWh: float,
        imbalance_price_EUR_MWh: float,
        purchase_price_EUR_MWh: float,
    ) -> float:
        """
        Misura il vantaggio/svantaggio economico generato dallo sbilanciamento.

        Valore positivo = il prezzo MSD risulta migliore rispetto al costo medio di acquisto.
        Valore negativo = l'energia eccedente costa di più (o rende di meno) rispetto al prezzo medio.
        """
        if abs(imbalance_MWh) <= 1e-9:
            return 0.0
        price_gap = imbalance_price_EUR_MWh - purchase_price_EUR_MWh
        return -imbalance_MWh * price_gap

    def _aggregate_energy_stats(
        self,
        mgp_trade: TradeResult,
        session_trades: Dict[str, Optional[TradeResult]],
    ) -> tuple[float, float]:
        """Somma i volumi acquistati/venduti per calcolare il costo medio di energia."""
        purchased = max(mgp_trade.volume_MWh, 0.0)
        energy_cost = mgp_trade.cost_EUR - mgp_trade.revenue_EUR
        for trade in session_trades.values():
            if not trade:
                continue
            if trade.volume_MWh >= 0:
                purchased += trade.volume_MWh
                energy_cost += trade.cost_EUR
            else:
                energy_cost -= trade.revenue_EUR
        return purchased, energy_cost

    def _normalize_imbalance_forecast(self, value: float, *, fallback: float) -> float:
        """Se il forecast mancante usa una quota della domanda prevista (default 5%)."""
        if pd.isna(value):
            default_fraction = float(self.logic_config.get("default_imbalance_fraction", 0.05))
            return fallback * default_fraction
        return value

    @staticmethod
    def _per_MWh(total_value: float, volume_MWh: float) -> float:
        """Converte un costo totale nel corrispondente €/MWh gestendo il volume nullo."""
        if abs(volume_MWh) <= 1e-9:
            return 0.0
        return total_value / abs(volume_MWh)

    def _apply_mgp_regulator(self, *, desired_volume_MWh: float, demand_forecast_MWh: float) -> float:
        """
        Applica le salvaguardie MGP (range min/max rispetto alla domanda) evitando ordini short.
        """
        if demand_forecast_MWh <= 0:
            return desired_volume_MWh
        min_frac = float(self.logic_config.get("mgp_min_fraction", 0.82))
        max_frac = float(self.logic_config.get("mgp_max_fraction", 0.95))
        sell_frac = max(float(self.logic_config.get("mgp_sell_fraction", 0.0)), 0.0)
        lower_buy = demand_forecast_MWh * min_frac
        upper_buy = demand_forecast_MWh * max_frac
        sell_cap = demand_forecast_MWh * sell_frac

        regulated = desired_volume_MWh
        if regulated >= 0:
            regulated = max(regulated, lower_buy)
        regulated = min(regulated, upper_buy)
        regulated = max(regulated, -sell_cap)
        # Nel MGP reale il retailer puo' solo acquistare e l'ordine deve essere coerente con la domanda prevista.
        regulated = max(regulated, 0.0)
        if demand_forecast_MWh > 0 and regulated <= 0.0:
            regulated = lower_buy
        return regulated

    def _apply_intraday_regulator(
        self,
        *,
        desired_volume_MWh: float,
        demand_forecast_MWh: float,
        contracted_before_MWh: float,
        max_fraction: float,
        smoothing: float,
        liquidity_hint: float = 1.0,
    ) -> float:
        """
        Limita le correzioni MI usando tolleranza GME, frazioni per sessione, hint di liquidita' e smoothing.
        """
        tolerance = float(self.logic_config.get("gme_balance_tolerance_MWh", 0.1))
        residual = demand_forecast_MWh - contracted_before_MWh
        if abs(residual) <= tolerance:
            return 0.0

        liquidity_hint = min(max(liquidity_hint, 0.0), 1.0)
        allowed = abs(residual) * max_fraction * liquidity_hint
        if allowed <= tolerance:
            return 0.0

        regulated = max(-allowed, min(desired_volume_MWh, allowed))
        if residual > 0 and regulated < 0:
            regulated = 0.0
        elif residual < 0 and regulated > 0:
            regulated = 0.0

        final_position = contracted_before_MWh + regulated
        if residual > 0 and final_position > demand_forecast_MWh:
            regulated = max(0.0, demand_forecast_MWh - contracted_before_MWh)
        elif residual < 0 and final_position < demand_forecast_MWh:
            regulated = min(0.0, demand_forecast_MWh - contracted_before_MWh)

        smoothing = min(max(smoothing, 0.0), 1.0)
        regulated *= smoothing * liquidity_hint
        return regulated
