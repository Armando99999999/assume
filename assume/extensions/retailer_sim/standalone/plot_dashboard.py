# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Genera un dashboard tipo Grafana partendo dai risultati del retailer.

Walkthrough in stile colloquio tecnico
--------------------------------------
1. Il modulo legge i risultati orari (dallo standalone o dal world adapter) e, se
   necessario, ricostruisce il dataframe orario combinando esportazioni multiple.
2. Totali e KPI possono arrivare da un JSON già pronto oppure vengono calcolati
   on-the-fly dalle colonne `hourly_cost_EUR/hourly_revenue_EUR`.
3. ``build_dashboard`` organizza la visualizzazione in quattro pannelli: prezzi,
   volumi, numeri aggregati di cassa e profitti, replicando il layout mostrato
   nella documentazione del simulatore.
4. La CLI (`main`) orchestrace il tutto scegliendo automaticamente i percorsi di
   input/output in base alla configurazione o ai flag passati dall'utente.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt
import pandas as pd

from ..data_utils import load_config
from ..output_utils import build_world_hourly_dataframe


# --------------------------------------------------------------------------- #
# Caricamento dei totali e normalizzazione KPI                               #
# --------------------------------------------------------------------------- #

def load_totals(path: Path | None, hourly_df: pd.DataFrame) -> Dict[str, float]:
    """
    Restituisce i KPI aggregati (costi, ricavi, profitto) leggendo un JSON già
    calcolato oppure, in mancanza, sommando le colonne del dataframe orario.

    Il comportamento replica quello dello script originale: se mancano le colonne
    `hourly_cost_EUR/hourly_revenue_EUR` e non forniamo un JSON, viene lanciato
    un errore esplicito così da evitare dashboard inconsistenti.
    """
    if path and path.exists():
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return {k: float(v) for k, v in data.items()}

    required_columns = {"hourly_cost_EUR", "hourly_revenue_EUR"}
    missing_columns = [col for col in required_columns if col not in hourly_df.columns]
    if missing_columns:
        raise ValueError(
            "Totals JSON not found and required columns are missing from the hourly dataframe "
            f"(needed columns: {sorted(required_columns)}). "
            "Provide --totals when using world exports or rerun the standalone simulator to generate hourly_cost_EUR/hourly_revenue_EUR."
        )

    totals = {
        "total_costs_EUR": float(hourly_df["hourly_cost_EUR"].sum()),
        "total_revenues_EUR": float(hourly_df["hourly_revenue_EUR"].sum()),
    }
    totals["total_profit_EUR"] = totals["total_revenues_EUR"] - totals["total_costs_EUR"]
    return totals


def format_currency(value: float) -> str:
    """
    Formatta i numeri con suffissi leggibili (k/M/B) e il simbolo EUR.

    Questo helper è usato per gli indicatori del dashboard (revenues, costs, profit).
    """
    if abs(value) >= 1e9:
        return f"€{value / 1e9:.1f}B"
    if abs(value) >= 1e6:
        return f"€{value / 1e6:.1f}M"
    if abs(value) >= 1e3:
        return f"€{value / 1e3:.1f}k"
    return f"€{value:,.0f}"


# --------------------------------------------------------------------------- #
# Conversione export World -> dataframe orario                               #
# --------------------------------------------------------------------------- #

def load_world_hourly(world_dir: Path, config: Mapping[str, object]) -> pd.DataFrame:
    """
    Costruisce un dataframe orario compatibile con lo standalone a partire dagli
    export del World adapter. Se il file precomputato esiste lo riusa, altrimenti
    invoca ``build_world_hourly_dataframe`` con i parametri corretti.
    """
    precomputed = world_dir / "world_hourly_results.csv"
    if precomputed.exists():
        return pd.read_csv(precomputed, parse_dates=["timestamp"])

    sim_cfg = config.get("simulation", {}) if isinstance(config, dict) else {}
    session_defs = config.get("_intraday_sessions", []) if isinstance(config, dict) else []
    time_step = int(sim_cfg.get("time_step_minutes", 15)) if isinstance(sim_cfg, dict) else 15
    hourly_df = build_world_hourly_dataframe(
        world_dir,
        time_step_minutes=time_step,
        session_definitions=session_defs,
    )
    return hourly_df


def attach_reference_load(hourly_df: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    """Aggiunge al dataframe orario la serie cluster_total_load_MW presa dal CSV originale."""
    sim_cfg = config.get("simulation", {}) if isinstance(config, dict) else {}
    csv_path = sim_cfg.get("input_csv_path")
    if not csv_path or "timestamp" not in hourly_df.columns:
        return hourly_df

    timestamp_col = sim_cfg.get("timestamp_column", "timestamp")
    reference = pd.read_csv(csv_path, usecols=[timestamp_col, "cluster_total_load_MW"])

    reference = reference.rename(columns={timestamp_col: "timestamp"})
    reference["timestamp"] = pd.to_datetime(reference["timestamp"])
    enriched = hourly_df.copy()
    enriched["timestamp"] = pd.to_datetime(enriched["timestamp"])
    return enriched.merge(reference, on="timestamp", how="left")


# --------------------------------------------------------------------------- #
# Rendering del dashboard                                                     #
# --------------------------------------------------------------------------- #

def build_dashboard(
    hourly_df: pd.DataFrame,
    totals: Dict[str, float],
    *,
    output_path: Path,
) -> None:
    """
    Crea il dashboard Matplotlib che replica l'aspetto della view Grafana:
        * pannello prezzi (MGP/MI/imbalance)
        * pannello volumi (contratti MGP, correzioni MI, eventuale surplus)
        * due pannelli numerici per revenues/costi e profitti/margini
    """
    hourly_df = hourly_df.sort_values("timestamp")
    timestamps = pd.to_datetime(hourly_df["timestamp"])

    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(2, 2, height_ratios=[2.5, 1.5])

    ax_price = fig.add_subplot(grid[0, 0])
    ax_volume = fig.add_subplot(grid[0, 1])
    ax_cash = fig.add_subplot(grid[1, 0])
    ax_profit = fig.add_subplot(grid[1, 1])

    # --- Prezzi --------------------------------------------------------------
    ax_price.plot(timestamps, hourly_df["mgp_price_EUR_MWh"], label="MGP price", color="#F9D648")
    if "mi_price_EUR_MWh" in hourly_df.columns:
        ax_price.plot(timestamps, hourly_df["mi_price_EUR_MWh"], label="MI price", color="#00A676")
    if "imbalance_price_EUR_MWh" in hourly_df.columns:
        ax_price.plot(
            timestamps,
            hourly_df["imbalance_price_EUR_MWh"],
            label="Imbalance price",
            color="#FF9F1C",
            linewidth=1.2,
        )
    ax_price.set_ylabel("€/MWh")
    ax_price.set_title("Bid Prices")
    ax_price.grid(alpha=0.2)
    ax_price.legend(loc="upper right")

    # --- Volumi --------------------------------------------------------------
    ax_volume.plot(
        timestamps,
        hourly_df["mgp_volume_MWh"],
        label="MGP contracted",
        color="#00C4FF",
    )
    ax_volume.plot(
        timestamps,
        hourly_df["mi_volume_MWh"],
        label="MI adjustment",
        color="#FF4E50",
    )
    if "imbalance_MWh" in hourly_df.columns:
        imbalance_series = pd.to_numeric(hourly_df["imbalance_MWh"], errors="coerce").fillna(0.0)
        # Shortfall: consumi > contratti -> imbalance positivo (convenzione consumi - contratti)
        terna_support = imbalance_series.clip(lower=0.0)
        if terna_support.any():
            ax_volume.fill_between(
                timestamps,
                0.0,
                terna_support,
                step="pre",
                alpha=0.25,
                color="#FFB703",
                label="Terna balancing (shortfall)",
            )
    actual_series = None
    actual_label = "Actual demand"
    if "cluster_total_load_MW" in hourly_df.columns:
        actual_series = pd.to_numeric(hourly_df["cluster_total_load_MW"], errors="coerce")
        actual_label = "Cluster total load (MW)"
    elif "actual_load_dataset_MWh" in hourly_df.columns:
        actual_series = pd.to_numeric(hourly_df["actual_load_dataset_MWh"], errors="coerce")
    elif "actual_consumption_MWh" in hourly_df.columns:
        actual_series = pd.to_numeric(hourly_df["actual_consumption_MWh"], errors="coerce")
    if actual_series is not None:
        ax_volume.plot(
            timestamps,
            actual_series,
            label=actual_label,
            color="#2A9D8F",
            linewidth=1.2,
        )
    # Surplus: se non fidiamo del valore pre-calcolato, rigeneriamo da contratti - consumo (solo parte positiva).
    surplus_series = None
    if {"contracted_volume_MWh", "actual_consumption_MWh"} <= set(hourly_df.columns):
        contracted = pd.to_numeric(hourly_df["contracted_volume_MWh"], errors="coerce").fillna(0.0)
        actual = pd.to_numeric(hourly_df["actual_consumption_MWh"], errors="coerce").fillna(0.0)
        surplus_series = (contracted - actual).clip(lower=0.0)
    elif "surplus_sold_MWh" in hourly_df.columns:
        surplus_series = pd.to_numeric(hourly_df["surplus_sold_MWh"], errors="coerce").fillna(0.0)
    if surplus_series is not None and surplus_series.any():
        ax_volume.plot(
            timestamps,
            surplus_series,
            label="Surplus sold",
            color="#9B5DE5",
            linestyle="--",
        )
    ax_volume.set_ylabel("Volume [MWh]")
    ax_volume.set_title("Bid Volumes")
    ax_volume.grid(alpha=0.2)
    ax_volume.legend(loc="upper right")

    # --- KPI numerici --------------------------------------------------------
    ax_cash.axis("off")
    ax_profit.axis("off")
    ax_cash.set_title("Cashflow (Revenues)", loc="left", fontsize=12, color="#F8F8F2")
    ax_profit.set_title("Profit overview", loc="left", fontsize=12, color="#F8F8F2")

    ax_cash.text(
        0.02,
        0.6,
        format_currency(totals["total_revenues_EUR"]),
        fontsize=32,
        color="#FF6B6B",
        weight="bold",
        transform=ax_cash.transAxes,
    )
    ax_cash.text(
        0.02,
        0.15,
        f"Costs: {format_currency(totals['total_costs_EUR'])}",
        fontsize=16,
        color="#E0E0E0",
        transform=ax_cash.transAxes,
    )

    profit_value = totals["total_profit_EUR"]
    revenue_denominator = max(totals["total_revenues_EUR"], 1.0)
    ax_profit.text(
        0.02,
        0.6,
        format_currency(profit_value),
        fontsize=32,
        color="#FFD166",
        weight="bold",
        transform=ax_profit.transAxes,
    )
    ax_profit.text(
        0.02,
        0.32,
        f"Margin: {profit_value / revenue_denominator:.1%}",
        fontsize=16,
        color="#E0E0E0",
        transform=ax_profit.transAxes,
    )
    surplus_total = float(hourly_df.get("surplus_sold_MWh", pd.Series(dtype=float)).sum())
    avg_imb_price = float(hourly_df.get("imbalance_price_EUR_MWh", pd.Series(dtype=float)).mean() or 0.0)
    ax_profit.text(
        0.02,
        0.1,
        f"Surplus sold: {surplus_total:.1f} MWh\nAvg. imbalance €{avg_imb_price:.1f}",
        fontsize=12,
        color="#CFCFCF",
        transform=ax_profit.transAxes,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI helpers                                                                 #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    """Configura e analizza i flag CLI; replica esattamente gli argomenti originali."""
    parser = argparse.ArgumentParser(description="Plot retailer dashboard.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("assume/extensions/retailer_sim/config.yaml"),
        help="Config file, used to infer default output paths.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        help="CSV with hourly results (output of main_simulation).",
    )
    parser.add_argument(
        "--totals",
        type=Path,
        help="Optional JSON with aggregated KPIs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assume/extensions/retailer_sim/outputs/dashboard.png"),
        help="PNG file to save the dashboard.",
    )
    parser.add_argument(
        "--world-export",
        type=Path,
        help="Directory containing ASSUME World exports (retailer_world).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry-point:
        1. carica la configurazione per determinare i percorsi di default;
        2. sceglie la fonte dei dati (standalone vs world);
        3. calcola/legge i totali e genera il dashboard.
    """
    args = parse_args()
    config = load_config(str(args.config))
    sim_cfg = config.get("simulation", {})
    default_output_dir = Path(sim_cfg.get("output_folder", "assume/extensions/retailer_sim/outputs"))
    totals_path = args.totals or default_output_dir / "aggregated_totals.json"

    if args.world_export:
        world_dir = args.world_export
        hourly_df = load_world_hourly(world_dir, config)
        totals_file = args.totals or (world_dir / "world_totals.json")
        totals = load_totals(totals_file if totals_file.exists() else None, hourly_df)
    else:
        results_path = args.results or default_output_dir / "hourly_results.csv"
        hourly_df = pd.read_csv(results_path)
        totals = load_totals(totals_path, hourly_df)

    # Se il dato di carico cluster è già presente nel CSV (world/standalone), non forziamo merge esterni:
    # evitiamo disallineamenti ricaricando dal dataset sorgente.
    if "cluster_total_load_MW" not in hourly_df.columns:
        hourly_df = attach_reference_load(hourly_df, config)
    build_dashboard(hourly_df, totals, output_path=args.output)
    print(f"Dashboard saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
