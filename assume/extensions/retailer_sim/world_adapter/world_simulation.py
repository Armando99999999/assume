# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Costruisce un World ASSUME con il retailer e la relativa controparte virtuale.

Walkthrough in stile colloquio tecnico
--------------------------------------
1. ``DictForecaster`` fornisce un ponte tra un ``DataFrame`` esterno e il sistema di
   forecast di ASSUME, riallineando le serie temporali sull'indice uniforme usato dal
   World.
2. ``register_retailer_extensions`` collega unita' e strategie specifiche del retailer
   agli hook di ASSUME, cosi' la simulazione puo' istanziarle come se fossero native.
3. ``create_world_from_dataframe`` prepara i dati (ordinamento timestamp, riallineamento
   serie, definizione mercati), registra la strategia condivisa e popola il World con
   unita' e mercati pronti a riutilizzare la logica standalone senza differenze
   funzionali.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from assume import World
from assume.common.fast_pandas import FastIndex, FastSeries
from assume.common.forecasts import Forecaster
from assume.common.market_objects import MarketConfig, MarketProduct
from dateutil import rrule as rr
from assume.strategies import bidding_strategies
from assume.units import unit_types

from ..decision import build_strategy
from .retailer_unit import RetailerUnit
from .retailer_strategy import RetailerMGPStrategy, RetailerMIStrategy, TernaBalancingStrategy, set_shared_strategy
from .terna_unit import TernaBalancingUnit


logger = logging.getLogger(__name__)


class DictForecaster(Forecaster):
    """
    Forecaster minimale che avvolge un ``DataFrame`` esterno e lo riallinea all'indice
    uniforme richiesto da ASSUME, cosi' ogni colonna puo' essere interrogata come un
    ``FastSeries``.
    """

    def __init__(self, index: FastIndex, data: Dict[str, pd.Series]):
        """
        Salva l'indice e converte ogni colonna fornita in ``FastSeries``; quando i dati
        arrivano come ``pd.Series`` vengono riindicizzati rispetto alla timeline del
        World per evitare buchi temporali.
        """
        super().__init__(index=index)
        self.data = {}
        date_index = pd.DatetimeIndex(index.get_date_list())
        for key, series in data.items():
            if isinstance(series, pd.Series):
                aligned = series.reindex(date_index, method="nearest")
                values = aligned.values
            else:
                values = series
            self.data[key] = FastSeries(index=index, value=values)

    def __getitem__(self, column: str):
        """Restituisce la serie richiesta o una ``FastSeries`` costante a 0.0 se assente."""
        return self.data.get(column, FastSeries(index=self.index, value=0.0))


def register_retailer_extensions() -> None:
    """
    Registra le unita' e le strategie specifiche del retailer nel registry globale di
    ASSUME, consentendo al World di istanziarle tramite i consueti identificativi
    (`unit_types` e `bidding_strategies`).
    """
    unit_types["retailer_unit"] = RetailerUnit
    unit_types["terna_balancing_unit"] = TernaBalancingUnit
    unit_types["virtual_supply"] = TernaBalancingUnit  # backwards compatibility
    bidding_strategies["retailer_mgp"] = RetailerMGPStrategy
    bidding_strategies["retailer_mi"] = RetailerMIStrategy
    bidding_strategies["terna_balancing"] = TernaBalancingStrategy
    bidding_strategies["virtual_supply"] = TernaBalancingStrategy


def create_world_from_dataframe(df: pd.DataFrame, config: dict) -> World:
    """
    Costruisce un ``World`` ASSUME partendo da un ``DataFrame`` di input e dalla
    configurazione YAML: registra le estensioni, prepara la strategia condivisa, crea il
    forecaster riallineato e aggiunge mercati/unita' che replicano la logica del
    simulatore standalone.
    """
    # --- Registrazione estensioni e lettura configurazioni ------------------
    register_retailer_extensions()
    sim_cfg = config.get("simulation", {})
    decision_cfg = config.get("decision_making", {})
    market_cfg = config.get("market", {})
    retailer_logic = dict(config.get("retailer_logic", {}))
    retailer_logic.setdefault("macro_imbalance_forecast", config.get("macro_imbalance_forecast", {}))
    strategy = build_strategy(decision_cfg, sim_cfg)
    session_definitions = list(config.get("_intraday_sessions", []))
    if not session_definitions:
        session_definitions = [
            {"name": "MI1", "open_hour": 8.0, "close_hour": 10.0, "coverage_start_hour": 10.0, "coverage_end_hour": 14.0, "liquidity": 0.85},
            {"name": "MI2", "open_hour": 10.0, "close_hour": 12.0, "coverage_start_hour": 14.0, "coverage_end_hour": 28.0, "liquidity": 0.75},
        ]
    real_market_flag = bool(sim_cfg.get("real_market", False))
    set_shared_strategy(
        strategy,
        session_definitions=session_definitions,
        real_market=real_market_flag,
    )

    # --- Preparazione indice temporale condiviso -------------------------------
    timestamp_col = sim_cfg.get("timestamp_column", "timestamp")
    actual_col = sim_cfg.get("actual_consumption_col", "actual_consumption_MWh")
    freq = pd.to_timedelta(sim_cfg.get("time_step_minutes", 15), unit="m")
    # Pre-elaboriamo il dataframe per garantire timestamp ordinati e univoci.
    df_local = df.copy()
    df_local[timestamp_col] = pd.to_datetime(df_local[timestamp_col])
    df_local = df_local.sort_values(timestamp_col).drop_duplicates(subset=timestamp_col, keep="last")
    df_indexed = df_local.set_index(timestamp_col)
    start = pd.Timestamp(df_local[timestamp_col].min())
    end = pd.Timestamp(df_local[timestamp_col].max())
    index = FastIndex(start=start, end=end, freq=freq)
    date_index = pd.DatetimeIndex(index.get_date_list())

    # --- Costruzione dataset per il forecaster ---------------------------------
    mi2_col = sim_cfg.get("mi2_price_col", sim_cfg.get("mi_price_col", "price_MI_EUR_MWh"))
    def _series_or_default(column: str, default_value: pd.Series | float) -> pd.Series:
        """Restituisce la colonna richiesta o un fallback (serie o costante) allineato a ``df_indexed``."""
        if column in df_indexed.columns:
            return df_indexed[column]
        if isinstance(default_value, pd.Series):
            return default_value
        return pd.Series(default_value, index=df_indexed.index)

    slot_fallback = pd.Series(
        df_indexed.index.hour + df_indexed.index.minute / 60,
        index=df_indexed.index,
    )

    imbalance_series = df_indexed[sim_cfg.get("imbalance_col", "imbalance_forecast_MWh")]
    forecast_columns = {
        "load_forecast": df_indexed[sim_cfg.get("consumption_col", "consumption_forecast_MWh")],
        "mgp_price": df_indexed[sim_cfg.get("mgp_price_col", "price_MGP_EUR_MWh")],
        "mi_price": df_indexed[sim_cfg.get("mi_price_col", "price_MI_EUR_MWh")],
        "mi2_price": df_indexed[mi2_col],
        "macro_price": df_indexed[sim_cfg.get("macrozone_price_col", "price_macrozone_avg_EUR_MWh")],
        "imbalance_coeff": df_indexed[sim_cfg.get("imbalance_coeff_col", "imbalance_coeff")],
        # For consistency with the standalone config, expose both names.
        "imbalance_forecast": imbalance_series,
        "imbalance_forecast_MWh": imbalance_series,
        "actual_consumption": df_indexed.get(actual_col, df_indexed[sim_cfg.get("consumption_col", "consumption_forecast_MWh")]),
    }
    consumption_fallback = forecast_columns.get("load_forecast", forecast_columns["actual_consumption"])
    forecast_columns["actual_consumption_delayed"] = _series_or_default(
        "actual_consumption_delayed_MWh",
        consumption_fallback,
    )
    forecast_columns["slot_hour_float"] = _series_or_default("slot_hour_float", slot_fallback)
    for session in session_definitions:
        name = session.get("name", "MI1")
        flag_col = f"is_{name}_eligible"
        hint_col = f"{name}_liquidity_hint"
        forecast_columns[flag_col] = _series_or_default(flag_col, 1.0)
        default_liq = float(session.get("liquidity", 1.0))
        forecast_columns[hint_col] = _series_or_default(hint_col, default_liq)
        price_col = f"price_{name}_EUR_MWh"
        if price_col in df_indexed.columns:
            forecast_columns[price_col] = df_indexed[price_col]
    forecast_columns = {k: series.reindex(date_index, method="nearest") for k, series in forecast_columns.items()}
    forecaster = DictForecaster(index=index, data=forecast_columns)

    # --- Istanziamento e setup del World --------------------------------------
    # Il World minimale riusa la cartella di export dello standalone cosi' report e
    # aggregati mantengono la stessa struttura (hourly_results/aggregated_totals).
    world = World(
        export_csv_path=config.get("simulation", {}).get("output_folder", "outputs"),
    )
    world.setup(
        start=start,
        end=end,
        simulation_id="retailer_world",
        save_frequency_hours=24,
        forecaster=forecaster,
    )

    # --- Definizione mercati e prodotti ---------------------------------------
    world.add_market_operator("GME")
    market_products = [
        MarketProduct(
            duration=freq,
            count=1,
            first_delivery=freq,
        )
    ]
    interval_minutes = max(int(freq.total_seconds() / 60), 1)
    opening_hours = rr.rrule(
        rr.MINUTELY,
        interval=interval_minutes,
        dtstart=start,
        until=end,
        cache=True,
    )
    mgp_config = MarketConfig(
        market_id="MGP",
        market_products=market_products,
        product_type="energy",
        opening_hours=opening_hours,
        opening_duration=freq,
        market_mechanism="pay_as_clear",
    )
    world.add_market("GME", mgp_config)
    for session in session_definitions:
        session_id = session.get("name", "MI1")
        session_config = MarketConfig(
            market_id=session_id,
            market_products=market_products,
            product_type="energy",
            opening_hours=opening_hours,
            opening_duration=freq,
            market_mechanism="pay_as_clear",
        )
        world.add_market("GME", session_config)

    # --- Creazione operatori e unita' -----------------------------------------
    world.add_unit_operator("Retailer_Op")
    world.add_unit_operator("Exchange_Op")

    retailer_bidding = {"MGP": "retailer_mgp"}
    retailer_price_columns = {"MGP": "mgp_price"}
    virtual_bidding = {"MGP": "terna_balancing"}
    virtual_price_columns = {"MGP": "mgp_price"}
    for idx, session in enumerate(session_definitions):
        session_name = session.get("name", f"MI{idx + 1}")
        retailer_bidding[session_name] = "retailer_mi"
        virtual_bidding[session_name] = "terna_balancing"
        session_price_column = f"price_{session_name}_EUR_MWh"
        fallback = "mi_price" if idx == 0 else "mi2_price"
        price_column = session_price_column if session_price_column in df_indexed.columns else fallback
        retailer_price_columns[session_name] = price_column
        virtual_price_columns[session_name] = price_column

    # L'unita Retailer replica la logica dello standalone dentro il world.
    world.add_unit(
        id="Retailer",
        unit_type="retailer_unit",
        unit_operator_id="Retailer_Op",
        unit_params={
            "technology": "retailer",
            "max_power": 1e5,
            "min_power": -1e5,
            "bidding_strategies": retailer_bidding,
            "forecast_column": "load_forecast",
            "price_columns": retailer_price_columns,
            "macro_price_column": "macro_price",
            "imbalance_coeff_column": "imbalance_coeff",
            "imbalance_forecast_column": "imbalance_forecast_MWh",
            "macro_imbalance_column": "imbalance_forecast_MWh",
            "actual_consumption_column": "actual_consumption_delayed",
            "retailer_logic": retailer_logic,
            "imbalance_penalty_cost_per_MWh": float(market_cfg.get("imbalance_penalty_cost_per_MWh", 0.0)),
            "intraday_sessions": session_definitions,
        },
        forecaster=forecaster,
    )

    # VirtualSupply funge da supply price-taker che alimenta la simulazione.
    world.add_unit(
        id="TernaAgent",
        unit_type="terna_balancing_unit",
        unit_operator_id="Exchange_Op",
        unit_params={
            "technology": "terna_balancing",
            "max_power": 2000.0,
            "min_power": 0.0,
            "bidding_strategies": virtual_bidding,
            "price_column": "mgp_price",
            "market_price_columns": virtual_price_columns,
        },
        forecaster=forecaster,
    )

    return world
