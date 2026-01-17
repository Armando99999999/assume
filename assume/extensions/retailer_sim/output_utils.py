# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Raccolta di utility per condividere gli stessi output fra standalone e World."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .standalone.market import MSDSettlement
from .terna_agent import TernaLikeBalancingAgent


def ensure_directory(path: Path) -> None:
    """Crea la cartella di destinazione se non esiste (equivalente a mkdir -p)."""
    path.mkdir(parents=True, exist_ok=True)


def write_orders_json(orders: List[Dict[str, Any]], output_path: Path) -> None:
    """Serialize the structured orders to disk."""
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(orders, fh, indent=2)


def write_totals_csv(output_path: Path, totals: Dict[str, float]) -> None:
    """Persist aggregated totals to a CSV file."""
    ensure_directory(output_path.parent)
    pd.Series(totals).to_csv(output_path, header=False)


def _session_column(name: str) -> str:
    """Converte un market_id (es. MI1) in nome colonna dei volumi."""
    return f"{name.lower()}_volume_MWh"


def _session_price_column(name: str) -> str:
    """Converte un market_id in nome colonna dei prezzi."""
    return f"{name.lower()}_price_EUR_MWh"


def _normalize_session_definitions(session_definitions: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Garantisce di avere sempre almeno due sessioni MI standard."""
    if session_definitions:
        return [dict(session) for session in session_definitions]
    return [
        {"name": "MI1", "liquidity": 0.85},
        {"name": "MI2", "liquidity": 0.75},
    ]


def _extract_session_names(session_definitions: Sequence[Dict[str, Any]]) -> List[str]:
    """Ricava l'elenco dei nomi di sessione (es. ["MI1", "MI2"])."""
    names: List[str] = []
    for index, session in enumerate(session_definitions):
        default_name = f"MI{index + 1}"
        name = str(session.get("name") or default_name)
        names.append(name)
    return names


def _align_reference_frame(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Allinea un DataFrame di riferimento (es. input standalone) all'indice temporale World.

    Serve per poter riusare segnali di supporto (macro-prezzi, coefficiente di sbilanciamento)
    durante la ricostruzione oraria del world adapter.
    """
    if df.empty:
        return pd.DataFrame(index=target_index)
    ref = df.copy()
    ref[timestamp_col] = pd.to_datetime(ref[timestamp_col])
    ref = ref.sort_values(timestamp_col).drop_duplicates(subset=timestamp_col, keep="last")
    ref = ref.set_index(timestamp_col)
    return ref.reindex(target_index, method="nearest")


def _compute_purchase_price_from_row(row: pd.Series) -> float:
    """Ricava il prezzo medio di acquisto (€/MWh) a partire da costi/volumi."""
    volume = float(row.get("purchase_volume_MWh", 0.0))
    cost = float(row.get("purchase_cost_EUR", 0.0))
    if volume <= 1e-9:
        return float(row.get("mgp_price_EUR_MWh", 0.0))
    return cost / volume


def _load_dispatch_table(
    world_dir: Path,
    *,
    session_definitions: Sequence[Dict[str, Any]],
    slot_hours: float,
) -> pd.DataFrame:
    """
    Ricostruisce la matrice timestamp x mercato con i volumi contrattati dal retailer.

    Preferisce market_dispatch.csv (più accurato). Se mancante, cade back su market_orders.csv.
    """
    dispatch_path = world_dir / "market_dispatch.csv"
    if dispatch_path.exists():
        dispatch = pd.read_csv(dispatch_path)
        dispatch["datetime"] = pd.to_datetime(dispatch["datetime"])
        dispatch = dispatch[dispatch["unit_id"].str.lower() == "retailer"]
        if not dispatch.empty and dispatch["power"].abs().sum() > 0:
            # Nel dataset gli stessi valori sono già espressi in MWh per slot, evitiamo la conversione MW->MWh.
            dispatch["energy_MWh"] = pd.to_numeric(dispatch["power"], errors="coerce").fillna(0.0)
            return (
                dispatch.pivot_table(
                    values="energy_MWh",
                    index="datetime",
                    columns="market_id",
                    aggfunc="sum",
                    fill_value=0.0,
                ).sort_index()
            )

    orders_path = world_dir / "market_orders.csv"
    if orders_path.exists():
        orders = pd.read_csv(orders_path)
        orders = orders[orders["unit_id"].str.lower() == "retailer"]
        if not orders.empty:
            orders["start_time"] = pd.to_datetime(orders["start_time"])
            orders["energy_MWh"] = pd.to_numeric(orders["volume"], errors="coerce").fillna(0.0)
            return (
                orders.pivot_table(
                    values="energy_MWh",
                    index="start_time",
                    columns="market_id",
                    aggfunc="sum",
                    fill_value=0.0,
                ).sort_index()
            )

    raise ValueError("Unable to derive retailer dispatch from world exports.")


def build_world_hourly_dataframe(
    world_dir: Path,
    *,
    time_step_minutes: int,
    session_definitions: Optional[Sequence[Dict[str, Any]]] = None,
    reference_df: Optional[pd.DataFrame] = None,
    sim_cfg: Optional[Dict[str, Any]] = None,
    retailer_logic: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Ricostruisce il DataFrame orario (stesse colonne dello standalone) a partire dagli export World.

    L'obiettivo è poter riusare dashboard e tooling esistenti senza duplicare logica.
    """
    meta_path = world_dir / "market_meta.csv"
    unit_dispatch_path = world_dir / "unit_dispatch.csv"
    if not meta_path.exists():
        raise FileNotFoundError("Missing market_meta.csv in world exports.")

    session_definitions = _normalize_session_definitions(session_definitions)
    session_names = _extract_session_names(session_definitions)
    slot_hours = float(time_step_minutes) / 60.0

    volume_table = _load_dispatch_table(
        world_dir,
        session_definitions=session_definitions,
        slot_hours=slot_hours,
    )

    volume_columns: Dict[str, str] = {}
    for market_id in volume_table.columns:
        col_name = _session_column(market_id) if market_id.upper().startswith("MI") else f"{market_id.lower()}_volume_MWh"
        volume_columns[market_id] = col_name
    volume_table = volume_table.rename(columns=volume_columns)
    volume_table.index = pd.to_datetime(volume_table.index)
    volume_table.index.name = "timestamp"

    meta = pd.read_csv(meta_path)
    meta["time"] = pd.to_datetime(meta["time"])
    price_table = (
        meta.groupby(["time", "market_id"])["price"].last().unstack().sort_index()
    )
    price_columns: Dict[str, str] = {}
    for market_id in price_table.columns:
        price_columns[market_id] = (
            _session_price_column(market_id) if market_id.upper().startswith("MI") else f"{market_id.lower()}_price_EUR_MWh"
        )
    price_table = price_table.rename(columns=price_columns)
    price_table.index = pd.to_datetime(price_table.index)
    price_table.index.name = "timestamp"

    hourly = volume_table.join(price_table, how="outer").sort_index()
    if "mi_price_EUR_MWh" not in hourly.columns:
        mi_price_column = None
        for session_name in session_names:
            candidate = _session_price_column(session_name)
            if candidate in hourly.columns:
                mi_price_column = candidate
                break
        if mi_price_column is not None:
            hourly["mi_price_EUR_MWh"] = hourly[mi_price_column]
        elif "mgp_price_EUR_MWh" in hourly.columns:
            hourly["mi_price_EUR_MWh"] = hourly["mgp_price_EUR_MWh"]
        else:
            hourly["mi_price_EUR_MWh"] = 0.0

    # 2) Se possibile, portiamo dentro il carico cluster dal CSV originale per usarlo come consumo reale.
    if "cluster_total_load_MW" not in hourly.columns and sim_cfg is not None:
        csv_path = sim_cfg.get("input_csv_path")
        ts_col = sim_cfg.get("timestamp_column", "timestamp")
        if csv_path:
            try:
                ref = pd.read_csv(csv_path, usecols=[ts_col, "cluster_total_load_MW"])
                ref[ts_col] = pd.to_datetime(ref[ts_col])
                ref = ref.set_index(ts_col)
                # usa merge left sulla time index per evitare interpolazioni nearest che appiattiscono i valori
                hourly = hourly.merge(ref, left_index=True, right_index=True, how="left")
            except Exception:
                pass

    # 3) Consumo reale: preferiamo il cluster (se presente), altrimenti il dispatch.
    if "cluster_total_load_MW" in hourly.columns:
        hourly["actual_consumption_MWh"] = pd.to_numeric(hourly["cluster_total_load_MW"], errors="coerce").fillna(0.0)
    elif unit_dispatch_path.exists():
        unit_dispatch = pd.read_csv(unit_dispatch_path)
        unit_dispatch["time"] = pd.to_datetime(unit_dispatch["time"])
        retailer_dispatch = unit_dispatch[unit_dispatch["unit"].str.lower() == "retailer"]
        actual = retailer_dispatch.groupby("time")["power"].sum().reindex(hourly.index, method="nearest").fillna(0.0)
        hourly["actual_consumption_MWh"] = actual
    else:
        hourly["actual_consumption_MWh"] = 0.0

    # 3) Precompiliamo le colonne con valori di default (per evitare KeyError più avanti).
    mgp_col = "mgp_volume_MWh"
    if mgp_col not in hourly.columns:
        hourly[mgp_col] = 0.0

    per_session_cols = []
    for name in session_names:
        col = _session_column(name)
        if col not in hourly.columns:
            hourly[col] = 0.0
        per_session_cols.append(col)

    hourly["mi_volume_MWh"] = hourly[per_session_cols[0]] if per_session_cols else 0.0
    if len(per_session_cols) > 1:
        hourly["mi2_volume_MWh"] = hourly[per_session_cols[1:]].sum(axis=1)
    else:
        hourly["mi2_volume_MWh"] = 0.0

    # 4) Ricostruiamo i principali indicatori standalone-friendly.
    # Se manca il forecast cluster, prova a importarlo dal forecast_inputs.csv usando il timestamp come chiave.
    if "cluster_total_load_forecast_MW" not in hourly.columns:
        try:
            ts_col = sim_cfg.get("timestamp_column", "timestamp") if sim_cfg else "timestamp"
            forecast_path = (
                Path(sim_cfg.get("forecast_inputs_path", "assume/extensions/retailer_sim/outputs/forecast_inputs.csv"))
                if sim_cfg
                else Path("assume/extensions/retailer_sim/outputs/forecast_inputs.csv")
            )
            forecast_df = pd.read_csv(forecast_path, usecols=[ts_col, "load_cluster_forecast"])
            forecast_df[ts_col] = pd.to_datetime(forecast_df[ts_col])
            forecast_df = (
                forecast_df.rename(columns={"load_cluster_forecast": "cluster_total_load_forecast_MW"}).set_index(ts_col)
            )
            hourly = hourly.merge(forecast_df, left_index=True, right_index=True, how="left")
        except Exception:
            pass

    hourly["contracted_volume_MWh"] = hourly[mgp_col] + hourly[per_session_cols].sum(axis=1)
    if "cluster_total_load_forecast_MW" in hourly.columns:
        hourly["consumption_forecast_MWh"] = pd.to_numeric(
            hourly["cluster_total_load_forecast_MW"], errors="coerce"
        )
    else:
        hourly["consumption_forecast_MWh"] = hourly["contracted_volume_MWh"]
    # Sbilanciamento: consumi reali - contratti (short positivo, long negativo).
    raw_imbalance = hourly["actual_consumption_MWh"] - hourly["contracted_volume_MWh"]
    hourly["imbalance_MWh"] = raw_imbalance
    hourly["imbalance_cost_EUR"] = 0.0
    hourly["imbalance_price_EUR_MWh"] = 0.0
    hourly["imbalance_value_gap_EUR"] = 0.0
    hourly["surplus_sold_MWh"] = 0.0
    hourly["surplus_sale_revenue_EUR"] = 0.0
    hourly["inventory_MWh"] = 0.0
    hourly["customer_revenue_EUR"] = 0.0
    hourly["purchase_cost_EUR"] = 0.0
    hourly["purchase_volume_MWh"] = 0.0

    for col in ["hourly_cost_EUR", "hourly_revenue_EUR", "inventory_MWh"]:
        hourly[col] = 0.0

    price_lookup = {
        mgp_col: "mgp_price_EUR_MWh",
    }
    for name in session_names:
        col = _session_column(name)
        price_lookup[col] = _session_price_column(name)

    # 5) Calcoliamo costi/ricavi per acquisti e vendite nei vari mercati.
    for volume_col, price_col in price_lookup.items():
        if volume_col not in hourly.columns:
            continue
        price_series = hourly.get(price_col)
        if price_series is None:
            continue
        volumes = pd.to_numeric(hourly[volume_col], errors="coerce").fillna(0.0)
        prices = pd.to_numeric(price_series, errors="coerce").fillna(0.0)
        buys = volumes.clip(lower=0.0)
        sells = -volumes.clip(upper=0.0)
        hourly["hourly_cost_EUR"] += buys * prices
        hourly["hourly_revenue_EUR"] += sells * prices
        hourly["purchase_volume_MWh"] += buys
        hourly["purchase_cost_EUR"] += buys * prices

    sim_cfg = sim_cfg or {}
    retailer_logic = retailer_logic or {}
    timestamp_col = sim_cfg.get("timestamp_column", "timestamp")
    macro_col = sim_cfg.get("macrozone_price_col", "price_macrozone_avg_EUR_MWh")
    imbalance_coeff_col = sim_cfg.get("imbalance_coeff_col", "imbalance_coeff")
    macro_imbalance_col = sim_cfg.get("macro_imbalance_col") or sim_cfg.get("imbalance_col", "imbalance_forecast_MWh")

    if reference_df is not None:
        aligned = _align_reference_frame(reference_df, timestamp_col=timestamp_col, target_index=hourly.index)
        hourly["macrozone_price_EUR_MWh"] = aligned.get(macro_col, 0.0).astype(float)
        hourly["macro_imbalance_MWh"] = aligned.get(macro_imbalance_col, 0.0).astype(float)
        hourly["imbalance_coeff"] = aligned.get(
            imbalance_coeff_col, retailer_logic.get("imbalance_coeff_default", 0.15)
        ).astype(float)
        forecast_col = sim_cfg.get("consumption_col", "consumption_forecast_MWh")
        if forecast_col in aligned:
            hourly["consumption_forecast_MWh"] = aligned.get(forecast_col, hourly["contracted_volume_MWh"]).astype(float)
    else:
        hourly["macrozone_price_EUR_MWh"] = hourly.get("mgp_price_EUR_MWh", 0.0)
        hourly["macro_imbalance_MWh"] = 0.0
        hourly["imbalance_coeff"] = float(retailer_logic.get("imbalance_coeff_default", 0.15))
        if "consumption_forecast_MWh" not in hourly.columns:
            hourly["consumption_forecast_MWh"] = hourly["contracted_volume_MWh"]

    msd_cfg = dict(retailer_logic.get("msd_settlement", {}))
    settlement = MSDSettlement(
        price_sensitivity=float(msd_cfg.get("price_sensitivity", retailer_logic.get("msd_price_sensitivity", 0.45))),
        additional_penalty_EUR_MWh=float(
            msd_cfg.get("additional_penalty_EUR_MWh", retailer_logic.get("msd_additional_penalty_EUR_MWh", 0.0))
        ),
        long_position_credit_factor=float(
            msd_cfg.get("long_position_credit_factor", retailer_logic.get("long_position_credit_factor", 1.0))
        ),
    )
    terna_agent = TernaLikeBalancingAgent(retailer_logic.get("terna_agent", {}))
    margin = float(retailer_logic.get("customer_margin_EUR_MWh", 25.0))
    align_tariff = bool(retailer_logic.get("align_tariff_with_standalone", True))

    # 6) Ciclo principale: calcoliamo costo sbilanciamento, prezzo retail, tariffe.
    hourly["terna_penalty_EUR"] = 0.0
    hourly["terna_incentive_EUR"] = 0.0
    hourly["purchase_price_EUR_MWh"] = 0.0
    hourly["retail_tariff_EUR_MWh"] = 0.0

    sale_fraction = float(retailer_logic.get("surplus_sale_fraction", 1.0))
    sale_fraction = min(max(sale_fraction, 0.0), 1.0)
    sale_mode = str(retailer_logic.get("surplus_sale_price", "equilibrium")).lower()
    custom_sale_price = float(retailer_logic.get("surplus_sale_custom_price", 0.0))
    inventory = 0.0

    def _determine_surplus_price(row: pd.Series) -> float:
        mgp_price = float(row.get("mgp_price_EUR_MWh", 0.0))
        mi_price = float(
            row.get(
                "mi_price_EUR_MWh",
                row.get(
                    "mi1_price_EUR_MWh",
                    row.get("mi2_price_EUR_MWh", mgp_price),
                ),
            )
        )
        if sale_mode == "mgp":
            return mgp_price
        if sale_mode == "mi":
            return mi_price
        if sale_mode == "msd":
            return float(row.get("imbalance_price_EUR_MWh", mgp_price))
        if sale_mode == "custom":
            return custom_sale_price if custom_sale_price > 0 else mgp_price
        return (mgp_price + mi_price) / 2

    for idx, row in hourly.iterrows():
        timestamp = pd.Timestamp(idx)
        residual = float(row.get("actual_consumption_MWh", 0.0) - row.get("contracted_volume_MWh", 0.0))
        mgp_price = float(row.get("mgp_price_EUR_MWh", 0.0))
        macro_price = float(row.get("macrozone_price_EUR_MWh", mgp_price))
        coeff = float(row.get("imbalance_coeff", retailer_logic.get("imbalance_coeff_default", 0.15)))
        macro_imbalance = float(row.get("macro_imbalance_MWh", 0.0))
        base_price = settlement.estimate_price(
            mgp_price=mgp_price,
            macrozone_price=macro_price,
            imbalance_coeff=coeff,
            imbalance_input_MWh=residual,
        )
        adjustment = terna_agent.evaluate(
            timestamp=timestamp,
            residual_MWh=residual,
            macro_imbalance_MWh=macro_imbalance,
            base_price_EUR_MWh=base_price,
            mgp_price_EUR_MWh=mgp_price,
        )
        long_factor = settlement.long_position_credit_factor if residual < 0 else 1.0
        base_cost = settlement.cost_from_price(residual, base_price)
        penalty_total = adjustment.penalty_EUR_MWh * abs(residual) * long_factor
        incentive_total = adjustment.incentive_EUR_MWh * abs(residual) * long_factor
        imbalance_cost = base_cost + penalty_total - incentive_total
        hourly.at[idx, "imbalance_price_EUR_MWh"] = adjustment.adjusted_price_EUR_MWh
        hourly.at[idx, "imbalance_cost_EUR"] = imbalance_cost
        hourly.at[idx, "terna_penalty_EUR"] = penalty_total
        hourly.at[idx, "terna_incentive_EUR"] = incentive_total
        purchase_price = _compute_purchase_price_from_row(row)
        hourly.at[idx, "purchase_price_EUR_MWh"] = purchase_price
        if abs(residual) > 1e-9:
            hourly.at[idx, "imbalance_value_gap_EUR"] = -residual * (
                adjustment.adjusted_price_EUR_MWh - purchase_price
            )
        else:
            hourly.at[idx, "imbalance_value_gap_EUR"] = 0.0
        abs_imbalance = abs(residual)
        penalty_total = float(row.get("terna_penalty_EUR", 0.0))
        cost_for_pass_through = (
            (imbalance_cost - penalty_total) / abs_imbalance if abs_imbalance > 1e-9 else 0.0
        )
        if align_tariff:
            tariff = purchase_price + margin
        else:
            tariff = terna_agent.compute_sale_tariff(
                purchase_price_EUR_MWh=purchase_price,
                imbalance_cost_EUR_MWh=cost_for_pass_through,
                profit_margin_EUR_MWh=margin,
            )
        hourly.at[idx, "retail_tariff_EUR_MWh"] = tariff
        hourly.at[idx, "customer_revenue_EUR"] = tariff * float(row.get("actual_consumption_MWh", 0.0))

        surplus_before_sale = max(
            float(row.get("contracted_volume_MWh", 0.0)) - float(row.get("actual_consumption_MWh", 0.0)),
            0.0,
        )
        sale_price = _determine_surplus_price(row)
        surplus_sold = surplus_before_sale * sale_fraction  # liquidato nello stesso slot
        surplus_revenue = surplus_sold * sale_price
        hourly.at[idx, "surplus_sold_MWh"] = surplus_sold
        hourly.at[idx, "surplus_sale_revenue_EUR"] = surplus_revenue

    hourly["hourly_cost_EUR"] += hourly["imbalance_cost_EUR"]
    hourly["hourly_revenue_EUR"] += hourly["customer_revenue_EUR"] + hourly["surplus_sale_revenue_EUR"]
    hourly["hourly_profit_EUR"] = hourly["hourly_revenue_EUR"] - hourly["hourly_cost_EUR"]
    hourly = hourly.drop(columns=["purchase_cost_EUR", "purchase_volume_MWh"])

    hourly = hourly.reset_index()
    if "datetime" in hourly.columns:
        hourly = hourly.rename(columns={"datetime": "timestamp"})
    elif "index" in hourly.columns:
        hourly = hourly.rename(columns={"index": "timestamp"})
    elif "time" in hourly.columns:
        hourly = hourly.rename(columns={"time": "timestamp"})

    # In alcuni export world c'e' una riga finale vuota/di chiusura: la rimuoviamo.
    if len(hourly) > 0:
        hourly = hourly.iloc[:-1]
    if "timestamp" not in hourly.columns:
        raise KeyError("Unable to determine timestamp column while rebuilding world hourly dataframe.")
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])
    return hourly.sort_values("timestamp").reset_index(drop=True)


def compute_totals_from_hourly(hourly_df: pd.DataFrame) -> Dict[str, float]:
    """Somma costi, ricavi e profitto a partire dal dataframe orario."""
    total_costs = float(pd.to_numeric(hourly_df.get("hourly_cost_EUR", 0.0), errors="coerce").fillna(0.0).sum())
    total_revenues = float(pd.to_numeric(hourly_df.get("hourly_revenue_EUR", 0.0), errors="coerce").fillna(0.0).sum())
    return {
        "total_costs_EUR": total_costs,
        "total_revenues_EUR": total_revenues,
        "total_profit_EUR": total_revenues - total_costs,
    }


def _determine_forecast_source(market_id: str) -> str:
    """Usato nelle metadata per indicare quale tipo di previsione ha guidato l'ordine."""
    market_upper = market_id.upper()
    if market_upper == "MGP":
        return "naive"
    if market_upper.startswith("MI"):
        return "updated"
    return "actual"


def convert_world_orders_to_json(
    world_dir: Path,
    *,
    time_step_minutes: int,
    session_definitions: Optional[Sequence[Dict[str, Any]]] = None,
    strategy_label: str = "simple",
    real_market: bool = False,
) -> List[Dict[str, Any]]:
    """
    Converte market_orders.csv nel formato JSON atteso dalle pipeline standalone.

    In questo modo dashboard/report possono riusare la stessa struttura output.
    """
    orders_path = world_dir / "market_orders.csv"
    if not orders_path.exists():
        return []

    session_definitions = _normalize_session_definitions(session_definitions)
    session_names = {session.get("name", f"MI{idx + 1}") for idx, session in enumerate(session_definitions)}

    df = pd.read_csv(orders_path)
    if "unit_id" not in df.columns:
        return []
    df = df[df["unit_id"].str.lower() == "retailer"]
    if df.empty:
        return []

    df["start_time"] = pd.to_datetime(df["start_time"])
    freq = pd.to_timedelta(time_step_minutes, unit="m")
    orders: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        market_id = str(row.get("market_id", "MGP"))
        volume = float(row.get("volume", 0.0))
        if abs(volume) < 1e-6:
            continue
        timestamp = pd.Timestamp(row["start_time"])
        slot_end = timestamp + freq
        quantity_mw = abs(volume)
        side = "buy" if volume >= 0 else "sell"
        metadata = {
            "strategy": strategy_label,
            "source_forecast": _determine_forecast_source(market_id),
            "real_market": real_market,
        }
        if market_id.upper().startswith("MI"):
            metadata["session_known"] = market_id.upper() in session_names
        orders.append(
            {
                "market": market_id,
                "time": timestamp.isoformat(),
                "orders": [
                    {
                        "slot": f"{timestamp.strftime('%H:%M')}-{slot_end.strftime('%H:%M')}",
                        "quantity_mw": quantity_mw,
                        "side": side,
                        "price": float(row.get("price", 0.0)),
                    }
                ],
                "metadata": metadata,
            }
        )

    return orders


def export_world_results(
    config: Dict[str, Any],
    *,
    strategy_label: str,
    reference_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Punto di ingresso principale: genera CSV/JSON di output a partire dagli export World.

    Produce:
      - world_hourly_results.csv (stesse colonne dello standalone),
      - world_totals.csv/json,
      - world_orders.json (opzionale se dispatch presente).
    """
    sim_cfg = config.get("simulation", {})
    # Allineiamo il percorso del forecast inputs allo stesso meccanismo usato per il loader:
    # se siamo in modalità advanced con ml_model, preferiamo il file con suffisso del modello.
    forecast_cfg = config.get("forecasting", {})
    base_path = Path(forecast_cfg.get("output_csv", "assume/extensions/retailer_sim/outputs/forecast_inputs.csv"))
    mode = str(forecast_cfg.get("mode", "baseline")).lower()
    ml_model = forecast_cfg.get("ml_model")
    if mode == "advanced" and ml_model:
        candidate = base_path.with_name(f"{base_path.stem}_{str(ml_model).replace(' ', '_')}{base_path.suffix}")
        if candidate.exists():
            base_path = candidate
    sim_cfg = dict(sim_cfg)
    sim_cfg["forecast_inputs_path"] = str(base_path)
    output_dir = Path(sim_cfg.get("output_folder", "outputs"))
    world_dir = output_dir / "retailer_world"
    if not world_dir.exists():
        return

    # Se disponibile, usiamo l'hourly dello standalone come reference per prezzi/forecast
    if reference_df is None:
        hourly_path = output_dir / "hourly_results.csv"
        if hourly_path.exists():
            try:
                reference_df = pd.read_csv(hourly_path)
            except Exception:
                reference_df = None

    session_definitions = config.get("_intraday_sessions", [])
    time_step_minutes = int(sim_cfg.get("time_step_minutes", 15))
    hourly_df = build_world_hourly_dataframe(
        world_dir,
        time_step_minutes=time_step_minutes,
        session_definitions=session_definitions,
        reference_df=reference_df,
        sim_cfg=sim_cfg,
        retailer_logic=config.get("retailer_logic", {}),
    )

    hourly_path = world_dir / "world_hourly_results.csv"
    ensure_directory(hourly_path.parent)
    hourly_df.to_csv(hourly_path, index=False)

    totals = compute_totals_from_hourly(hourly_df)
    totals_path = world_dir / "world_totals.json"
    with totals_path.open("w", encoding="utf-8") as fh:
        json.dump(totals, fh, indent=2)
    write_totals_csv(world_dir / "world_totals.csv", totals)

    orders = convert_world_orders_to_json(
        world_dir,
        time_step_minutes=time_step_minutes,
        session_definitions=session_definitions,
        strategy_label=strategy_label,
        real_market=bool(sim_cfg.get("real_market", False)),
    )
    if orders:
        write_orders_json(orders, world_dir / "world_orders.json")
