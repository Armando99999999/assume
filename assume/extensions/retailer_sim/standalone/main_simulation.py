# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Entry point del simulatore standalone del retailer (singolo agente).

Walkthrough in stile colloquio tecnico
--------------------------------------
1. ``run_simulation`` e' il cuore del file: prende il dataframe di input e la
   configurazione YAML, costruisce strategia, mercati e retailer e avvia il loop
   orario (identico a quello della versione originale).
2. Funzioni di supporto:
      - ``save_outputs`` scrive CSV/JSON nelle cartelle di output.
      - ``main`` carica config + dataset e orchestra la simulazione CLI.
3. Tutta la logica esistente e' stata lasciata intatta; abbiamo solo aggiunto
   docstring e commenti per facilitare una spiegazione "da colloquio".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from ..data_utils import filter_dataframe_for_simulation, load_config, load_dataframe
from ..output_utils import write_orders_json, write_totals_csv
from .market import ClearingCoordinator, DayAheadMarket, IntraDayMarket, MarketEnvironment
from .retailer import Retailer
from .strategies import BiddingStrategy, build_strategy

logger = logging.getLogger(__name__)


def run_simulation(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, float], List[Dict[str, Any]]]:
    """
    Crea strategia, mercati e retailer e avvia la simulazione oraria.

    Passaggi principali:
        1. Estrarre le sezioni di configurazione (decisioni, mercato, simulazione).
        2. Costruire la ``BiddingStrategy`` (random o simple) condivisa.
        3. Creare l'ambiente di mercato con i parametri corretti (capacita', slope).
        4. Istanziare il ``Retailer`` e lanciare ``simulate`` sul dataframe.
        5. Restituire i risultati orari, i totali e il log ordini.
    """
    decision_cfg = config.get("decision_making", {})
    market_cfg = config.get("market", {})
    sim_cfg = config.get("simulation", {})

    # ------------------------------------------------------------------ #
    # 1) Strategia di bidding (random o heuristica deterministica)
    # ------------------------------------------------------------------ #
    strategy = build_strategy(decision_cfg, sim_cfg)
    assert isinstance(strategy, BiddingStrategy)

    # ------------------------------------------------------------------ #
    # 2) Ambiente di mercato: day-ahead + intraday coordinati
    # ------------------------------------------------------------------ #
    time_step_minutes = float(sim_cfg.get("time_step_minutes", 60))
    time_step_hours = time_step_minutes / 60

    def per_slot(value: Optional[float]) -> Optional[float]:
        """Normalizza da MWh/h (per hour) al passo di simulazione corrente."""
        if value is None:
            return None
        return float(value) * time_step_hours

    mgp_cfg = market_cfg.get("mgp", {})
    mi_cfg = market_cfg.get("mi", {})

    # Alcune configurazioni specificano direttamente la capacità globale in MWh; in alternativa convertiamo da MW/ora.
    global_capacity = market_cfg.get("global_clearing_capacity_MWh")
    if not global_capacity:
        global_capacity = per_slot(market_cfg.get("global_capacity"))
        if global_capacity:
            market_cfg["global_clearing_capacity_MWh"] = global_capacity

    clearing_coordinator = ClearingCoordinator(global_capacity) if global_capacity else None

    def resolve_capacity(key_base: str, section: Mapping[str, Any]) -> Optional[float]:
        """Helper per ottenere la capacità in MWh, preferendo override a livello top."""
        capacity = market_cfg.get(key_base)
        if capacity:
            return capacity
        per_hour = section.get("capacity")
        return per_slot(per_hour)

    mgp_capacity = resolve_capacity("mgp_clearing_capacity_MWh", mgp_cfg)
    mi_capacity = resolve_capacity("mi_clearing_capacity_MWh", mi_cfg)

    # Le slope definiscono come il prezzo sale quando ci avviciniamo alla capacità; usiamo i fallback MGP/MI se servono.
    slope_scale = float(market_cfg.get("price_slope_scale", 1.0))
    price_threshold = float(market_cfg.get("price_slope_threshold", 1.0))
    mgp_slope_low = 4.0 * slope_scale
    mgp_slope_high = 18.0 * slope_scale
    mi_slope_low = 5.0 * slope_scale
    mi_slope_high = 22.0 * slope_scale

    market_env = MarketEnvironment(
        day_ahead=DayAheadMarket(
            transaction_cost_per_MWh=float(
                market_cfg.get("transaction_cost_per_MWh", market_cfg.get("transaction_cost", 0.0))
            ),
            clearing_capacity_MWh=mgp_capacity,
            price_slope_low=mgp_slope_low,
            price_slope_high=mgp_slope_high,
            price_slope_threshold=price_threshold,
            coordinator=clearing_coordinator,
        ),
        intraday=IntraDayMarket(
            transaction_cost_per_MWh=float(
                market_cfg.get(
                    "transaction_cost_intraday_per_MWh",
                    market_cfg.get("transaction_cost", 0.0),
                )
            ),
            clearing_capacity_MWh=mi_capacity if mi_capacity is not None else mgp_capacity,
            price_slope_low=mi_slope_low,
            price_slope_high=mi_slope_high,
            price_slope_threshold=price_threshold,
            coordinator=clearing_coordinator,
        ),
        clearing_coordinator=clearing_coordinator,
    )

    # Carichiamo (opzionalmente) il profilo di interconnessione per modulare i limiti.
    interconnection_col = sim_cfg.get("interconnection_col", "cross_border_flow_MWh")
    timestamp_col = sim_cfg.get("timestamp_column", "timestamp")
    if interconnection_col in df.columns:
        profile = dict(zip(pd.to_datetime(df[timestamp_col]), df[interconnection_col]))
        market_env.set_interconnection_profile(profile)

    # ------------------------------------------------------------------ #
    # 3) Retailer: aggrega strategia, mercati e regole operative
    # ------------------------------------------------------------------ #
    # Il retailer incapsula logica strategica, interfaccia coi mercati e regole operative.
    retailer = Retailer(
        name=sim_cfg.get("retailer_name", "Retailer Demo"),
        strategy=strategy,
        market_env=market_env,
        timestamp_col=timestamp_col,
        consumption_col=sim_cfg.get("consumption_col", "consumption_forecast_MWh"),
        actual_consumption_col=sim_cfg.get("actual_consumption_col", "actual_consumption_MWh"),
        mgp_price_col=sim_cfg.get("mgp_price_col", "price_MGP_EUR_MWh"),
        mi_price_col=sim_cfg.get("mi_price_col", "price_MI_EUR_MWh"),
        mi2_price_col=sim_cfg.get("mi2_price_col"),
        imbalance_col=sim_cfg.get("imbalance_col", "imbalance_forecast_MWh"),
        macro_imbalance_col=sim_cfg.get("macro_imbalance_col"),
        macrozone_price_col=sim_cfg.get("macrozone_price_col", "price_macrozone_avg_EUR_MWh"),
        imbalance_coeff_col=sim_cfg.get("imbalance_coeff_col", "imbalance_coeff"),
        imbalance_penalty_cost_per_MWh=float(market_cfg.get("imbalance_penalty_cost_per_MWh", 0.0)),
        retailer_logic=config.get("retailer_logic", {}),
        macro_forecast_cfg=config.get("macro_imbalance_forecast", {}),
        intraday_sessions=config.get("_intraday_sessions"),
        time_step_minutes=sim_cfg.get("time_step_minutes"),
        real_market=sim_cfg.get("real_market", False),
    )

    # ------------------------------------------------------------------ #
    # 4) Esecuzione della simulazione e raccolta output
    # ------------------------------------------------------------------ #
    # Simulazione vera e propria: i risultati orari e i totali sono identici alla versione originale.
    hourly_df = retailer.simulate(df)
    totals = retailer.compute_totals()
    orders = retailer.get_orders()
    return hourly_df, totals, orders


def save_outputs(
    hourly_df: pd.DataFrame,
    totals: Dict[str, float],
    orders: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    """
    Persiste i risultati in CSV/JSON nella cartella definita da
    ``simulation.output_folder`` (se presente).
    """
    output_dir = config.get("simulation", {}).get("output_folder")
    if not output_dir:
        return

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    hourly_df.to_csv(path / "hourly_results.csv", index=False)

    totals_json = path / "aggregated_totals.json"
    pd.Series(totals).to_json(totals_json, indent=2)
    write_totals_csv(path / "aggregated_totals.csv", totals)
    write_orders_json(orders, path / "orders.json")


def main() -> None:
    """
    Entry point CLI:
        1. carica configurazione e dataset;
        2. filtra l'intervallo temporale desiderato;
        3. lancia ``run_simulation`` e mostra un breve riepilogo a console;
        4. salva gli output se configurato.
    """
    logging.basicConfig(level=logging.INFO)

    config = load_config()
    df = load_dataframe(config)
    df = filter_dataframe_for_simulation(df, config)

    hourly_df, totals, orders = run_simulation(df, config)

    print("=== Riepilogo Retailer ===")
    print(f"Costi totali   : {totals['total_costs_EUR']:.2f} EUR")
    print(f"Ricavi totali  : {totals['total_revenues_EUR']:.2f} EUR")
    print(f"Profitto totale: {totals['total_profit_EUR']:.2f} EUR")

    save_outputs(hourly_df, totals, orders, config)


if __name__ == "__main__":
    main()
