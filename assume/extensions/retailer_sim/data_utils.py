# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared helpers for loading configuration and data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from .standalone.market import MSDSettlement, suggest_market_parameters
from .config_audit import audit_config


@dataclass
class IntradaySession:
    """Metadati essenziali di una sessione MI."""

    name: str
    open_hour: float
    close_hour: float
    coverage_start_hour: float
    coverage_end_hour: float
    liquidity: float = 1.0


def _hour(hour: int, minute: int = 0) -> float:
    return hour + minute / 60


def _build_intraday_sessions(real_market: bool) -> List[IntradaySession]:
    if real_market:
        return [
            IntradaySession("MI1", _hour(8), _hour(10), _hour(10), _hour(14), liquidity=0.85),
            IntradaySession("MI2", _hour(10), _hour(12), _hour(14), _hour(18), liquidity=0.8),
            IntradaySession("MI3", _hour(14), _hour(15, 45), _hour(15, 45), _hour(20), liquidity=0.7),
            IntradaySession("MI4", _hour(16), _hour(17, 45), _hour(17, 45), _hour(21), liquidity=0.6),
            IntradaySession("MI5", _hour(18), _hour(19, 45), _hour(19, 45), _hour(23), liquidity=0.5),
            IntradaySession("MI6", _hour(20), _hour(21, 45), _hour(21, 45), _hour(24), liquidity=0.4),
            IntradaySession("MI7", _hour(22), _hour(23, 45), _hour(23, 45), _hour(28), liquidity=0.35),
        ]
    # Versione compatta: due sessioni che coprono l'intera giornata mantenendo il ritardo minimo di 4 ore.
    return [
        IntradaySession("MI1", _hour(8), _hour(10), _hour(10), _hour(14), liquidity=0.85),
        IntradaySession("MI2", _hour(10), _hour(12), _hour(14), _hour(28), liquidity=0.75),
    ]


def _session_mask(slot_hours: pd.Series, start: float, end: float) -> pd.Series:
    if end <= start:
        end += 24
    base_mask = (slot_hours >= start) & (slot_hours < end)
    if end > 24:
        rollover_mask = ((slot_hours + 24) >= start) & ((slot_hours + 24) < end)
        return base_mask | rollover_mask
    return base_mask


def _load_forecast_inputs(
    config: Dict[str, Any],
    *,
    timestamp_col: str,
    period_hours: float,
    sessions: List[IntradaySession],
) -> Optional[pd.DataFrame]:
    forecast_cfg = config.get("forecasting", {})
    output_path = forecast_cfg.get("output_csv")
    if not output_path:
        return None
    base_path = Path(output_path)
    path = base_path
    mode = str(forecast_cfg.get("mode", "baseline")).lower()
    ml_model = forecast_cfg.get("ml_model")
    # In modalità advanced preferiamo il file specifico del modello se presente, anche se il base esiste.
    if mode == "advanced" and ml_model:
        suffix_name = f"{base_path.stem}_{str(ml_model).replace(' ', '_')}{base_path.suffix}"
        candidate = base_path.with_name(suffix_name)
        if candidate.exists():
            path = candidate
    if not path.exists():
        return None

    forecast_df = pd.read_csv(path)
    if timestamp_col not in forecast_df.columns:
        if "timestamp" in forecast_df.columns:
            forecast_df = forecast_df.rename(columns={"timestamp": timestamp_col})
        else:
            raise ValueError("forecast_inputs.csv must include a timestamp column.")
    forecast_df[timestamp_col] = pd.to_datetime(forecast_df[timestamp_col])

    payload = pd.DataFrame({timestamp_col: forecast_df[timestamp_col]})
    if "load_cluster_forecast" in forecast_df.columns:
        payload["load_cluster_forecast_MW"] = pd.to_numeric(
            forecast_df["load_cluster_forecast"], errors="coerce"
        )
        payload["consumption_forecast_MWh"] = payload["load_cluster_forecast_MW"]
    if "sbil_forecasted" in forecast_df.columns:
        payload["imbalance_forecast_MWh"] = pd.to_numeric(
            forecast_df["sbil_forecasted"], errors="coerce"
        )
    if "mgp" in forecast_df.columns:
        payload["price_MGP_EUR_MWh"] = pd.to_numeric(forecast_df["mgp"], errors="coerce")
        payload["price_MGP_forecast_EUR_MWh"] = payload["price_MGP_EUR_MWh"]

    for idx, session in enumerate(sessions, start=1):
        forecast_col = f"mi{idx}"
        if forecast_col not in forecast_df.columns:
            continue
        column_name = f"price_{session.name}_EUR_MWh"
        payload[column_name] = pd.to_numeric(forecast_df[forecast_col], errors="coerce")

    return payload

def load_config(path: str = "assume/extensions/retailer_sim/config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    audit_config(config)
    return config


def get_frequency_string(sim_cfg: Dict[str, Any]) -> str:
    minutes = sim_cfg.get("time_step_minutes")
    if minutes:
        return f"{int(minutes)}T"
    hours = float(sim_cfg.get("time_step_hours", 1))
    total_minutes = max(int(round(hours * 60)), 1)
    if total_minutes % 60 == 0:
        return f"{int(total_minutes / 60)}H"
    return f"{total_minutes}T"


def build_synthetic_dataframe(sim_cfg: Dict[str, Any]) -> pd.DataFrame:
    start = pd.Timestamp(sim_cfg.get("start_datetime", "2024-01-01 00:00:00"))
    end = pd.Timestamp(sim_cfg.get("end_datetime", "2024-01-02 23:00:00"))
    freq = get_frequency_string(sim_cfg)
    index = pd.date_range(start, end, freq=freq)
    rng = np.random.default_rng(0)
    hours = index.hour.values + index.minute.values / 60
    consumption_actual = 30 + 6 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 1, size=len(index))
    price_mgp_real = 120 + 15 * np.sin(2 * np.pi * hours / 24 + 0.2)
    price_mi_real = price_mgp_real + rng.normal(0, 3, size=len(index))
    price_macro_real = price_mgp_real - rng.normal(0, 2, size=len(index))
    imbalance_actual = rng.normal(0, 1, size=len(index))
    imbalance_coeff = rng.uniform(0.05, 0.25, size=len(index))

    return pd.DataFrame(
        {
            sim_cfg.get("timestamp_column", "timestamp"): index,
            "actual_consumption_MWh": consumption_actual + rng.normal(0, 0.5, size=len(index)),
            "actual_imbalance_MWh": imbalance_actual,
            sim_cfg.get("mgp_price_col", "price_MGP_EUR_MWh"): price_mgp_real,
            sim_cfg.get("mi_price_col", "price_MI_EUR_MWh"): price_mi_real,
            sim_cfg.get("macrozone_price_col", "price_macrozone_avg_EUR_MWh"): price_macro_real,
            sim_cfg.get("imbalance_coeff_col", "imbalance_coeff"): imbalance_coeff,
            "mgp_price_real": price_mgp_real,
            "mi_price_real": price_mi_real,
            "macrozone_price_real": price_macro_real,
        }
    )


def apply_naive_forecasts(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    mapping = config.get("forecast_mapping", {})
    if not mapping:
        return df
    sim_cfg = config.get("simulation", {})
    lag = max(int(mapping.get("naive_lag_steps", 1)), 1)

    def naive_from(actual_col: str, target_col: str, *, lag_override: Optional[int] = None) -> None:
        if actual_col and target_col and actual_col in df.columns:
            lag_steps = lag_override if lag_override is not None else lag
            df[target_col] = (
                df[actual_col]
                .shift(lag_steps)
                .bfill()
                .ffill()
            )

    consumption_col = sim_cfg.get("consumption_col", "consumption_forecast_MWh")
    actual_consumption_col = sim_cfg.get("actual_consumption_col") or "actual_consumption_MWh"
    actual_consumption_col = sim_cfg.get("actual_consumption_col")
    if not actual_consumption_col:
        actual_consumption_col = "actual_consumption_MWh"
    imbalance_col = sim_cfg.get("imbalance_col", "imbalance_forecast_MWh")
    mgp_col = sim_cfg.get("mgp_price_col", "price_MGP_EUR_MWh")
    mi_col = sim_cfg.get("mi_price_col", "price_MI_EUR_MWh")
    macro_col = sim_cfg.get("macrozone_price_col", "price_macrozone_avg_EUR_MWh")
    mi2_col = sim_cfg.get("mi2_price_col", mi_col)

    naive_from(mapping.get("load_actual_col"), consumption_col)
    naive_from(mapping.get("imbalance_actual_col"), imbalance_col)
    naive_from(mapping.get("mi_price_actual_col"), mi_col, lag_override=lag)
    mi2_actual_src = mapping.get("mi2_price_actual_col", mapping.get("mi_price_actual_col"))
    naive_from(mi2_actual_src, mi2_col, lag_override=lag * 2)
    naive_from(mapping.get("macrozone_price_actual_col"), macro_col)

    mgp_actual = mapping.get("mgp_price_actual_col")
    if mgp_actual and mgp_actual in df.columns:
        df[mgp_col] = df[mgp_actual].astype(float)

    if mi2_col not in df.columns:
        df[mi2_col] = df[mi_col]
    df[mi2_col] = df[mi2_col].ffill().bfill()

    return df


def _infer_period_minutes(timestamps: pd.Series) -> int:
    diffs = timestamps.sort_values().diff().dropna()
    if diffs.empty:
        return 60
    mode = diffs.mode()
    if mode.empty:
        minutes = int(diffs.iloc[0].total_seconds() // 60 or 60)
    else:
        minutes = int(mode.iloc[0].total_seconds() // 60 or 60)
    return max(minutes, 1)


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def load_dataframe(config: Dict[str, Any]) -> pd.DataFrame:
    sim_cfg = config.get("simulation", {})
    csv_path = sim_cfg.get("input_csv_path")
    timestamp_col = sim_cfg.get("timestamp_column", "timestamp")
    if csv_path:
        df = pd.read_csv(csv_path)
        if timestamp_col not in df.columns:
            raise ValueError(f"CSV must contain '{timestamp_col}'.")
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    else:
        df = build_synthetic_dataframe(sim_cfg)

    df = df.sort_values(timestamp_col).reset_index(drop=True)

    period_minutes = _infer_period_minutes(df[timestamp_col])
    config.setdefault("simulation", {})["time_step_minutes"] = period_minutes
    period_hours = period_minutes / 60

    real_market = bool(sim_cfg.get("real_market", False))
    sessions = _build_intraday_sessions(real_market)
    config.setdefault("_intraday_sessions", [asdict(session) for session in sessions])

    forecast_cfg = config.get("forecasting", {})
    forecast_path = forecast_cfg.get("output_csv")
    same_source = False
    if csv_path and forecast_path:
        try:
            same_source = Path(csv_path).resolve() == Path(forecast_path).resolve()
        except Exception:
            same_source = False
    if not same_source:
        forecast_payload = _load_forecast_inputs(
            config,
            timestamp_col=timestamp_col,
            period_hours=period_hours,
            sessions=sessions,
        )
        if forecast_payload is not None:
            df = df.merge(forecast_payload, on=timestamp_col, how="left")

    market_cfg = config.setdefault("market", {})
    if csv_path and not market_cfg.get("mgp"):
        stats = suggest_market_parameters(csv_path, time_step_minutes=period_minutes)
        market_cfg.setdefault("mgp_clearing_capacity_MWh", stats["cap_mgp_MWh"])
        market_cfg.setdefault("mi_clearing_capacity_MWh", stats["cap_mi_MWh"])
        market_cfg.setdefault("global_clearing_capacity_MWh", stats["cap_mgp_MWh"])
        market_cfg.setdefault("global_clearing_price_slope", stats["price_slope_EUR_per_MWh"])
        market_cfg.setdefault("mgp_clearing_price_slope", stats["price_slope_EUR_per_MWh"])
        market_cfg.setdefault("mi_clearing_price_slope", stats["price_slope_EUR_per_MWh"] * 0.8)

    start_filter = sim_cfg.get("start_datetime")
    end_filter = sim_cfg.get("end_datetime")
    if start_filter:
        df = df[df[timestamp_col] >= pd.Timestamp(start_filter)]
    if end_filter:
        df = df[df[timestamp_col] <= pd.Timestamp(end_filter)]
    df = df.reset_index(drop=True)

    consumption_col = sim_cfg.get("consumption_col", "consumption_forecast_MWh")
    actual_consumption_col = sim_cfg.get("actual_consumption_col") or "actual_consumption_MWh"
    imbalance_col = sim_cfg.get("imbalance_col", "imbalance_forecast_MWh")
    mgp_col = sim_cfg.get("mgp_price_col", "price_MGP_EUR_MWh")
    mi_col = sim_cfg.get("mi_price_col", "price_MI_EUR_MWh")
    macro_col = sim_cfg.get("macrozone_price_col", "price_macrozone_avg_EUR_MWh")
    coeff_col = sim_cfg.get("imbalance_coeff_col", "imbalance_coeff")
    mi2_col = sim_cfg.get("mi2_price_col", mi_col)

    # Assicuriamoci che le colonne chiave esistano anche se non sono presenti nel CSV iniziale.
    def ensure_column(column: str, *, source: Optional[str] = None, default: float = 0.0) -> None:
        if column in df.columns:
            return
        if source and source in df.columns:
            df[column] = pd.to_numeric(df[source], errors="coerce")
            return
        df[column] = default

    default_coeff = float(config.get("retailer_logic", {}).get("imbalance_coeff_default", 0.15))
    ensure_column(mgp_col, source="MGP_PRICE_NORD")
    ensure_column(mi_col, source="MI_PRICE_NORD")
    ensure_column(macro_col, source="MGP_PRICE_FORECAST", default=0.0)
    ensure_column(coeff_col, default=default_coeff)
    ensure_column(mi2_col, source=mi_col)
    # Non copiare il carico reale nel forecast: se manca, lasciamo NaN cosÛ da usare il file di forecast.
    ensure_column("cluster_total_load_forecast_MW", default=np.nan)
    ensure_column(consumption_col, default=np.nan)
    ensure_column(imbalance_col, source="SBIL_MWH", default=0.0)
    if actual_consumption_col:
        ensure_column(actual_consumption_col, source="actual_consumption_MWh")

    # Allinea il forecast di carico indipendentemente dal nome colonna (con o senza suffisso _MW).
    forecast_source = None
    if "load_cluster_forecast_MW" in df.columns:
        forecast_source = "load_cluster_forecast_MW"
    elif "load_cluster_forecast" in df.columns:
        forecast_source = "load_cluster_forecast"
    if forecast_source:
        df["cluster_load_forecast_value"] = pd.to_numeric(df[forecast_source], errors="coerce")
        df["cluster_total_load_forecast_MW"] = df["cluster_load_forecast_value"]
        df[consumption_col] = df["cluster_load_forecast_value"].fillna(df[consumption_col])

    # Riempie eventuali buchi residui di domanda con forward/backward fill per evitare NaN che generano volumi MGP nulli.
    df[consumption_col] = df[consumption_col].ffill().bfill()

    if "consumption_actual_available_MWh" in df.columns and actual_consumption_col:
        df["consumption_actual_available_MWh"] = pd.to_numeric(
            df["consumption_actual_available_MWh"], errors="coerce"
        )
        df[actual_consumption_col] = df["consumption_actual_available_MWh"].fillna(df[actual_consumption_col])

    # Clip gli sbilanciamenti previsti per evitare outlier che esplodono il prezzo MSD.
    max_forecast = float(config.get("retailer_logic", {}).get("max_imbalance_forecast_MWh", 0.0))
    if max_forecast <= 0:
        max_forecast = float(df[imbalance_col].abs().quantile(0.99))
    if max_forecast > 0:
        df[imbalance_col] = df[imbalance_col].clip(lower=-max_forecast, upper=max_forecast)

    session_price_columns: List[str] = []
    for session in sessions:
        column_name = f"price_{session.name}_EUR_MWh"
        if column_name in df.columns:
            df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
            session_price_columns.append(column_name)
    if session_price_columns:
        df[mi_col] = df[session_price_columns[0]]
    if len(session_price_columns) > 1:
        df[mi2_col] = df[session_price_columns[1]]

    load_sources = [
        "consumption_actual_available_MWh",
        "cluster_total_load_MW",
        "ACTUAL_TOTAL_LOAD_MW_NORD",
        "TOTAL_LOAD_MW",
    ]
    actual_load = None
    for col in load_sources:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            actual_load = df[col]
            break
    if actual_load is None:
        actual_load = pd.Series(0.0, index=df.index)
    df["actual_consumption_MWh"] = actual_load
    delay_slots = max(int(sim_cfg.get("real_consumption_delay_slots", 4)), 0)
    if delay_slots:
        df["actual_consumption_delayed_MWh"] = df["actual_consumption_MWh"].shift(delay_slots)
    else:
        df["actual_consumption_delayed_MWh"] = df["actual_consumption_MWh"]

    _coerce_numeric(df, ["SBIL_MWH", "MGP_PRICE_NORD", "MI_PRICE_NORD", "MGP_PRICE_FORECAST"])

    # Calcola un prezzo MSD "reale" usando il modello semplificato e lo userà come base per la previsione naive.
    msd_cfg = config.get("retailer_logic", {}).get("msd_settlement", {})
    settlement = MSDSettlement(
        price_sensitivity=float(msd_cfg.get("price_sensitivity", config.get("retailer_logic", {}).get("msd_price_sensitivity", 0.45))),
        additional_penalty_EUR_MWh=float(
            msd_cfg.get(
                "additional_penalty_EUR_MWh",
                config.get("retailer_logic", {}).get("msd_additional_penalty_EUR_MWh", 0.0),
            )
        ),
        long_position_credit_factor=float(
            msd_cfg.get("long_position_credit_factor", config.get("retailer_logic", {}).get("long_position_credit_factor", 1.0))
        ),
    )
    mgp_actual_series = df.get("MGP_PRICE_NORD", df[mgp_col])
    macro_actual_series = df.get("MGP_PRICE_FORECAST", df[macro_col])
    coeff_series = df.get(coeff_col, default_coeff)
    imbalance_actual_series = df.get("SBIL_MWH", df[imbalance_col])
    delta = mgp_actual_series - macro_actual_series
    df["msd_price_real_tmp"] = (
        mgp_actual_series
        + delta * coeff_series
        + settlement.price_sensitivity * imbalance_actual_series.abs()
        + settlement.additional_penalty
    )
    # Usiamo il campo macro come "previsione" MSD (naive di un passo).
    df["MGP_PRICE_FORECAST"] = df["msd_price_real_tmp"]

    df = apply_naive_forecasts(df, config)

    df[consumption_col] = df[consumption_col].fillna(df["actual_consumption_MWh"])
    df[imbalance_col] = df[imbalance_col].fillna(df.get("SBIL_MWH")).fillna(0.0)
    if mgp_col in df.columns:
        df[mgp_col] = pd.to_numeric(df[mgp_col], errors="coerce")
    if "price_MGP_forecast_EUR_MWh" not in df.columns and "MGP_PRICE_NORD" in df.columns:
        df[mgp_col] = df["MGP_PRICE_NORD"]
    df[mgp_col] = df[mgp_col].ffill().bfill()
    df[mi_col] = df[mi_col].fillna(df[mgp_col]).ffill().bfill()
    # Se i prezzi MI o per sessione sono zero/non finiti, rimpiazza con il MGP.
    zero_mask = (df[mi_col] == 0) | (~np.isfinite(df[mi_col]))
    df.loc[zero_mask, mi_col] = df.loc[zero_mask, mgp_col]
    if mi2_col in df.columns:
        zero_mask_mi2 = (df[mi2_col] == 0) | (~np.isfinite(df[mi2_col]))
        df.loc[zero_mask_mi2, mi2_col] = df.loc[zero_mask_mi2, mgp_col]
    for col in session_price_columns:
        mask_col = (df[col] == 0) | (~np.isfinite(df[col]))
        df.loc[mask_col, col] = df.loc[mask_col, mgp_col]
    df[macro_col] = df[macro_col].fillna(df[mgp_col]).ffill().bfill()
    default_coeff = float(config.get("retailer_logic", {}).get("imbalance_coeff_default", 0.15))
    if coeff_col not in df.columns:
        df[coeff_col] = default_coeff
    df[coeff_col] = df[coeff_col].fillna(default_coeff)

    interconnection_cols = [
        "AUSTRIA_MWQH",
        "FRANCE_MWQH",
        "SLOVENIA_MWQH",
        "SWITZERLAND_MWQH",
        "SCHEDULED_INTERNAL_EXCHANGE_MW",
    ]
    cross_border_series = pd.Series(0.0, index=df.index)
    available_intercols = []
    for col in interconnection_cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            cross_border_series = cross_border_series.add(series, fill_value=0.0)
            available_intercols.append(col)
    df["cross_border_flow_MWh"] = cross_border_series
    
    slot_hours = df[timestamp_col].dt.hour + df[timestamp_col].dt.minute / 60
    df["slot_hour_float"] = slot_hours
    for session in sessions:
        mask = _session_mask(slot_hours, session.coverage_start_hour, session.coverage_end_hour)
        df[f"is_{session.name}_eligible"] = mask.astype(int)
        df[f"{session.name}_liquidity_hint"] = mask.astype(float) * session.liquidity

    ordered_candidates = [
        timestamp_col,
        consumption_col,
        imbalance_col,
        mgp_col,
        mi_col,
        mi2_col,
        macro_col,
        coeff_col,
        "cross_border_flow_MWh",
        "slot_hour_float",
    ]
    if actual_consumption_col:
        ordered_candidates.insert(2, actual_consumption_col)
        ordered_candidates.insert(3, "actual_consumption_delayed_MWh")
    for session in sessions:
        ordered_candidates.append(f"is_{session.name}_eligible")
        ordered_candidates.append(f"{session.name}_liquidity_hint")

    ordered_cols: list[str] = []
    for col in ordered_candidates:
        if col in df.columns and col not in ordered_cols:
            ordered_cols.append(col)

    remaining_cols = [col for col in df.columns if col not in ordered_cols]
    ordered_cols.extend(remaining_cols)

    df = df[ordered_cols].copy()
    return df


def determine_simulation_window(config: Dict[str, Any]) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Resolve the effective simulation window combining `simulation` dates and the ML test range.

    The test range (when provided) restricts the simulation to the evaluation horizon, ensuring
    standalone and World runs operate on the same window used for forecast validation.
    """
    sim_cfg = config.get("simulation", {})
    start_value = sim_cfg.get("start_datetime")
    end_value = sim_cfg.get("end_datetime")
    start_ts = pd.Timestamp(start_value) if start_value else None
    end_ts = pd.Timestamp(end_value) if end_value else None

    forecast_cfg = config.get("forecasting", {})
    advanced_cfg = forecast_cfg.get("advanced_settings", {})
    test_range = advanced_cfg.get("test_range")
    if test_range:
        try:
            test_start = pd.Timestamp(test_range[0])
            test_end = pd.Timestamp(test_range[1])
        except Exception as exc:  # pragma: no cover - defensive programming
            raise ValueError("Invalid forecasting.test_range configuration.") from exc
        start_ts = test_start if start_ts is None else max(start_ts, test_start)
        end_ts = test_end if end_ts is None else min(end_ts, test_end)

    if start_ts and end_ts and start_ts > end_ts:
        raise ValueError("Computed simulation window is empty. Check start/end or test_range settings.")
    return start_ts, end_ts


def filter_dataframe_for_simulation(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Slice the provided dataframe to the effective simulation window.

    Ensures both standalone and World environments replay only the desired period (e.g. ML test set).
    """
    start_ts, end_ts = determine_simulation_window(config)
    if start_ts is None and end_ts is None:
        return df

    timestamp_col = config.get("simulation", {}).get("timestamp_column", "timestamp")
    if timestamp_col not in df.columns:
        return df

    df_local = df.copy()
    df_local[timestamp_col] = pd.to_datetime(df_local[timestamp_col])
    if start_ts:
        df_local = df_local[df_local[timestamp_col] >= start_ts]
    if end_ts:
        df_local = df_local[df_local[timestamp_col] <= end_ts]
    return df_local.reset_index(drop=True)
