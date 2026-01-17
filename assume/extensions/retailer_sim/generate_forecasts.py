# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Utility che allena modelli ML/Neural, sceglie il migliore per RMSE e
produce un CSV compatibile con il simulatore del retailer.

Il file risultante contiene esclusivamente previsioni utilizzabili dal retailer:
    timestamp,
    load_cluster_forecast,
    sbil_forecasted,
    mi_price_session1..N (in base alle sessioni MI configurate),
    mgp_price,
    eventuali colonne reali rese disponibili secondo i ritardi configurati.
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .MLFORECASTCopia import MLFORECAST_Predictor
except ModuleNotFoundError as MLFORECAST_IMPORT_ERROR:  # pragma: no cover - dipendenza opzionale
    MLFORECAST_Predictor = None  # type: ignore
    MLFORECAST_IMPORT_ERROR_MESSAGE = str(MLFORECAST_IMPORT_ERROR)
else:
    MLFORECAST_IMPORT_ERROR_MESSAGE = ""

try:
    from .NEURALFORECASTCopia import NEURALFORECAST_Predictor
except ModuleNotFoundError as NEURALFORECAST_IMPORT_ERROR:  # pragma: no cover - dipendenza opzionale
    NEURALFORECAST_Predictor = None  # type: ignore
    NEURALFORECAST_IMPORT_ERROR_MESSAGE = str(NEURALFORECAST_IMPORT_ERROR)
else:
    NEURALFORECAST_IMPORT_ERROR_MESSAGE = ""

from .data_utils import load_config, load_dataframe


@dataclass
class TargetConfig:
    name: str
    column: str


def _rmse(pred: pd.Series, actual: pd.Series) -> float:
    mask = pred.notna()
    if mask.sum() == 0:
        return math.inf
    diff = pred[mask] - actual[mask]
    return float(np.sqrt(np.mean(np.square(diff))))


def _mae(pred: pd.Series, actual: pd.Series) -> float:
    mask = pred.notna()
    if mask.sum() == 0:
        return math.inf
    diff = pred[mask] - actual[mask]
    return float(np.mean(np.abs(diff)))


def _run_ml_model(
    series: Dict[str, Optional[pd.DataFrame]],
    freq: str,
    seasonal_period: int,
    horizon: int,
    model_name: str,
    optimization: bool,
    use_exog: bool,
) -> pd.Series:
    if MLFORECAST_Predictor is None:
        test_df = series.get("test")
        idx = test_df.index if isinstance(test_df, pd.DataFrame) else None
        return pd.Series(np.nan, index=idx, name="target")
    predictor = MLFORECAST_Predictor(
        run_mode="simulation",
        target_column="target",
        data_freq=freq,
        seasonal_period=seasonal_period,
        optimization=optimization,
        use_exog=use_exog,
        horizon=horizon,
    )
    predictor.prepare_data(
        train=series["train"],
        valid=series.get("valid"),
        test=series["test"],
    )
    predictor.train_model(model_name=model_name, optimization=optimization)
    forecasts = predictor.backtest(horizon=horizon)
    return forecasts["target"]


def _run_neural_model(
    series: Dict[str, Optional[pd.DataFrame]],
    freq: str,
    seasonal_period: int,
    horizon: int,
    optimization: bool,
    use_exog: bool,
) -> pd.Series:
    if NEURALFORECAST_Predictor is None:
        # Dipendenza opzionale mancante: restituisce una serie NaN per ignorare l'LSTM in ranking.
        test_df = series.get("test")
        idx = test_df.index if isinstance(test_df, pd.DataFrame) else None
        return pd.Series(np.nan, index=idx, name="target")
    predictor = NEURALFORECAST_Predictor(
        run_mode="simulation",
        target_column="target",
        data_freq=freq,
        seasonal_period=seasonal_period,
        optimization=optimization,
        use_exog=use_exog,
        verbose=False,
    )
    predictor.prepare_data(
        train=series["train"],
        valid=series.get("valid"),
        test=series["test"],
    )
    predictor.train_model("LSTM", horizon=horizon, step_size=horizon, optimization=optimization)
    forecasts = predictor.backtest(horizon=horizon)
    return forecasts["target"]


def _slice_range(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    exog_cols: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    columns: List[str] = []

    def _append_unique(column: Optional[str]) -> None:
        if column and column in df.columns and column not in columns:
            columns.append(column)

    _append_unique(timestamp_col)
    _append_unique(target_col)
    for col in exog_cols:
        # Evitiamo duplicati (es. target ripetuto tra le esogene).
        _append_unique(col)

    mask = (df[timestamp_col] >= start) & (df[timestamp_col] <= end)
    subset = df.loc[mask, columns].copy()
    subset = subset.rename(columns={timestamp_col: "timestamp", target_col: "target"})
    subset["timestamp"] = pd.to_datetime(subset["timestamp"])
    subset = subset.sort_values("timestamp").reset_index(drop=True)
    subset = subset.dropna(subset=["target"])
    return subset


def _prepare_series(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    exog_cols: List[str],
    ranges: Mapping[str, Tuple[pd.Timestamp, pd.Timestamp]],
) -> Dict[str, Optional[pd.DataFrame]]:
    series: Dict[str, Optional[pd.DataFrame]] = {}
    for key, value in ranges.items():
        if value is None:
            series[key] = None
            continue
        start, end = value
        subset = _slice_range(df, timestamp_col, target_col, exog_cols, start, end)
        series[key] = subset if not subset.empty else None

    if series["train"] is None or series["test"] is None:
        raise ValueError("Train/test windows must contain at least one sample.")
    return series


def _build_legacy_series(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    exog_cols: List[str],
    train_fraction: float,
) -> Dict[str, Optional[pd.DataFrame]]:
    columns = [timestamp_col, target_col] + [c for c in exog_cols if c in df.columns]
    subset = df[columns].dropna(subset=[target_col]).copy()
    subset = subset.rename(columns={timestamp_col: "timestamp", target_col: "target"})
    subset["timestamp"] = pd.to_datetime(subset["timestamp"])
    subset = subset.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(subset) * train_fraction)
    split_idx = min(max(split_idx, 1), len(subset) - 1)
    train = subset.iloc[:split_idx].copy()
    test = subset.iloc[split_idx:].copy()
    return {"train": train, "valid": None, "test": test}


def _select_actual_series(
    df_indexed: pd.DataFrame,
    candidates: List[Optional[str]],
) -> Optional[pd.Series]:
    for col in candidates:
        if col and col in df_indexed.columns:
            return pd.to_numeric(df_indexed[col], errors="coerce")
    return None


def _align_series(
    series: pd.Series,
    target_index: pd.DatetimeIndex,
    *,
    delay_slots: int = 0,
) -> pd.Series:
    offset = int(delay_slots)
    shifted = series.shift(offset)
    aligned = shifted.reindex(target_index)
    return aligned.reset_index(drop=True)


def _resolve_session_delay_hours(
    delay_cfg: object,
    *,
    session_name: str,
    index: int,
    default_hours: float,
) -> float:
    if delay_cfg is None:
        return default_hours
    try:
        if isinstance(delay_cfg, Mapping):
            key_candidates = (
                session_name,
                session_name.upper(),
                f"MI{index + 1}",
                str(index + 1),
            )
            for key in key_candidates:
                if key in delay_cfg:
                    return float(delay_cfg[key])
        elif isinstance(delay_cfg, Sequence) and not isinstance(delay_cfg, (str, bytes)):
            if 0 <= index < len(delay_cfg):
                return float(delay_cfg[index])
    except (TypeError, ValueError):
        return default_hours
    return default_hours


def _attach_actual_signals(
    frame: pd.DataFrame,
    *,
    df: pd.DataFrame,
    timestamp_col: str,
    sim_cfg: Mapping[str, object],
    info_cfg: Mapping[str, object],
    mi_columns: List[str],
    sessions: Sequence[Mapping[str, object]],
) -> Tuple[pd.DataFrame, List[str]]:
    frame_local = frame.copy()
    frame_ts_col = timestamp_col if timestamp_col in frame_local.columns else "timestamp"
    if frame_ts_col not in frame_local.columns:
        return frame_local, []

    frame_local[frame_ts_col] = pd.to_datetime(frame_local[frame_ts_col])
    target_index = pd.DatetimeIndex(frame_local[frame_ts_col])
    df_indexed = df.set_index(timestamp_col)

    freq_minutes = int(sim_cfg.get("time_step_minutes", 60) or 60)
    default_consumption_delay = int(sim_cfg.get("real_consumption_delay_slots", 4) or 0)
    consumption_delay = int(info_cfg.get("consumption_delay_slots", default_consumption_delay) or 0)
    imbalance_delay = int(info_cfg.get("imbalance_delay_slots", consumption_delay) or 0)

    mi_delay_hours = float(info_cfg.get("mi_price_delay_hours", 4.0) or 0.0)
    mgp_delay_hours = float(info_cfg.get("mgp_price_delay_hours", 0.0) or 0.0)
    mi_delay_slots = max(int(round(mi_delay_hours * 60 / max(freq_minutes, 1))), 0)
    mgp_delay_slots = max(int(round(mgp_delay_hours * 60 / max(freq_minutes, 1))), 0)
    session_list: List[Mapping[str, object]] = list(sessions or [])
    mi_delay_cfg = info_cfg.get("mi_session_delays_hours")

    availability_cols: List[str] = []

    load_series = _select_actual_series(
        df_indexed,
        [
            "cluster_total_load_MW",
            "ACTUAL_TOTAL_LOAD_MW_NORD",
            "TOTAL_LOAD_MW",
        ],
    )
    if load_series is not None:
        load_available = _align_series(load_series, target_index, delay_slots=consumption_delay)
        frame_local["cluster_load_actual_available_MW"] = load_available
        frame_local["consumption_actual_available_MWh"] = load_available
        availability_cols.extend(
            [
                "cluster_load_actual_available_MW",
                "consumption_actual_available_MWh",
            ]
        )

    imbalance_series = _select_actual_series(df_indexed, ["SBIL_MWH"])
    if imbalance_series is not None:
        frame_local["macro_imbalance_actual_available_MWh"] = _align_series(
            imbalance_series,
            target_index,
            delay_slots=imbalance_delay,
        )
        availability_cols.append("macro_imbalance_actual_available_MWh")

    mgp_series = _select_actual_series(
        df_indexed,
        [
            sim_cfg.get("mgp_price_col"),
            "MGP_PRICE_NORD",
        ],
    )
    if mgp_series is not None:
        mgp_realized = _align_series(mgp_series, target_index, delay_slots=mgp_delay_slots)
        frame_local["price_MGP_realized_EUR_MWh"] = mgp_realized
        availability_cols.append("price_MGP_realized_EUR_MWh")
        if "mgp" in frame_local.columns:
            frame_local["mgp_signal_EUR_MWh"] = mgp_realized.combine_first(frame_local["mgp"])
            availability_cols.append("mgp_signal_EUR_MWh")

    mi_series = _select_actual_series(
        df_indexed,
        [
            sim_cfg.get("mi_price_col"),
            "MI_PRICE_NORD",
        ],
    )
    mi_realized_reference: Optional[str] = None
    if mi_series is not None:
        for idx, column in enumerate(mi_columns):
            session = session_list[idx] if idx < len(session_list) else {}
            session_name = str(session.get("name", f"MI{idx + 1}"))
            delay_hours = _resolve_session_delay_hours(
                mi_delay_cfg,
                session_name=session_name,
                index=idx,
                default_hours=mi_delay_hours,
            )
            delay_slots = max(int(round(delay_hours * 60 / max(freq_minutes, 1))), 0)
            mi_realized = _align_series(mi_series, target_index, delay_slots=delay_slots)
            realized_col = f"{column}_realized_EUR_MWh"
            frame_local[realized_col] = mi_realized
            availability_cols.append(realized_col)
            session_real_col = f"price_{session_name}_realized_EUR_MWh"
            frame_local[session_real_col] = mi_realized
            availability_cols.append(session_real_col)
            if column in frame_local.columns:
                signal_col = f"{column}_signal_EUR_MWh"
                frame_local[signal_col] = frame_local[realized_col].combine_first(frame_local[column])
                availability_cols.append(signal_col)
            if mi_realized_reference is None:
                mi_realized_reference = realized_col
    if mi_realized_reference and mi_realized_reference in frame_local.columns:
        frame_local["price_MI_realized_EUR_MWh"] = frame_local[mi_realized_reference]
        availability_cols.append("price_MI_realized_EUR_MWh")

    availability_cols.extend(
        _annotate_mgp_deadline(
            frame_local,
            frame_ts_col=frame_ts_col,
            info_cfg=info_cfg,
        )
    )

    return frame_local, availability_cols


def _annotate_mgp_deadline(
    frame: pd.DataFrame,
    *,
    frame_ts_col: str,
    info_cfg: Mapping[str, object],
) -> List[str]:
    if frame_ts_col not in frame.columns:
        return []
    column_list: List[str] = []
    day_offset = int(info_cfg.get("mgp_offer_day_offset", -1) or -1)
    deadline_hour = info_cfg.get("mgp_offer_deadline_hour")
    if deadline_hour is None:
        return column_list
    try:
        hour_value = float(deadline_hour)
    except (TypeError, ValueError):
        hour_value = 14.0
    day_delta = pd.to_timedelta(day_offset, unit="D")
    hour_delta = pd.to_timedelta(hour_value, unit="h")
    deadline_col = "mgp_offer_deadline_ts"
    frame[deadline_col] = frame[frame_ts_col].dt.normalize() + day_delta + hour_delta
    column_list.append(deadline_col)
    offset_col = "mgp_offer_deadline_hours_before_delivery"
    frame[offset_col] = (frame[frame_ts_col] - frame[deadline_col]).dt.total_seconds() / 3600.0
    column_list.append(offset_col)
    return column_list


def _select_best_model(
    series: Dict[str, Optional[pd.DataFrame]],
    freq: str,
    seasonal_period: int,
    horizon: int,
    ml_models: Sequence[str],
    optimization: bool,
    use_exog: bool,
    include_lstm: bool,
) -> Tuple[str, pd.Series, float, float]:
    actual = series["test"]["target"].reset_index(drop=True)
    candidates: List[Tuple[str, pd.Series]] = []

    lstm_added = False
    for name in ml_models:
        name_upper = str(name).upper()
        if name_upper != "LSTM" and MLFORECAST_Predictor is None:
            continue  # skip ML se pacchetto assente
        if name_upper == "LSTM":
            neural_pred = _run_neural_model(series, freq, seasonal_period, horizon, optimization, use_exog)
            candidates.append(("LSTM", neural_pred.reset_index(drop=True)))
            lstm_added = True
            continue
        ml_pred = _run_ml_model(series, freq, seasonal_period, horizon, name, optimization, use_exog)
        candidates.append((name, ml_pred.reset_index(drop=True)))

    if include_lstm and not lstm_added:
        if NEURALFORECAST_Predictor is not None:
            neural_pred = _run_neural_model(series, freq, seasonal_period, horizon, optimization, use_exog)
            candidates.append(("LSTM", neural_pred.reset_index(drop=True)))

    best_name = ""
    best_rmse = math.inf
    best_mae = math.inf
    best_series: Optional[pd.Series] = None
    for name, pred in candidates:
        rmse_value = _rmse(pred, actual)
        mae_value = _mae(pred, actual)
        if rmse_value < best_rmse:
            best_rmse = rmse_value
            best_mae = mae_value
            best_name = name
            best_series = pred

    if best_series is None:
        # fallback naive: replica l'ultimo valore di train
        train_df = series.get("train")
        test_df = series.get("test")
        if isinstance(train_df, pd.DataFrame) and not train_df.empty and isinstance(test_df, pd.DataFrame):
            last_val = float(train_df["target"].dropna().iloc[-1]) if not train_df["target"].dropna().empty else np.nan
            naive = pd.Series(last_val, index=test_df.index, name="target").reset_index(drop=True)
            return ("NAIVE", naive, math.inf, _mae(naive, actual))
        raise RuntimeError("No forecast produced by the candidate models.")

    timestamps = series["test"]["timestamp"].reset_index(drop=True)
    best_series = best_series.reset_index(drop=True)
    if len(best_series) != len(timestamps):
        min_len = min(len(best_series), len(timestamps))
        best_series = best_series.iloc[:min_len]
        timestamps = timestamps.iloc[:min_len]
    best_series.index = timestamps
    return best_name, best_series, best_rmse, best_mae


def _normalize_dataframe(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    frame = df.copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col])
    return frame.sort_values(timestamp_col).reset_index(drop=True)


def _build_full_config(config: Dict[str, Any]) -> Dict[str, Any]:
    config_full = copy.deepcopy(config)
    sim_cfg = config_full.setdefault("simulation", {})
    sim_cfg.pop("start_datetime", None)
    sim_cfg.pop("end_datetime", None)
    return config_full


def generate_forecast_inputs() -> Path:
    config = load_config()
    sim_cfg = config.get("simulation", {})
    forecast_cfg = config.get("forecasting", {})

    timestamp_col = sim_cfg.get("timestamp_column", "timestamp")
    mgp_col = sim_cfg.get("mgp_price_col", "price_MGP_EUR_MWh")
    mi_col = sim_cfg.get("mi_price_col", "price_MI_EUR_MWh")
    macro_col = sim_cfg.get("macrozone_price_col", "price_macrozone_avg_EUR_MWh")

    freq_minutes = int(sim_cfg.get("time_step_minutes", 60))
    freq = f"{freq_minutes}min"
    seasonal_period = max(int(round(24 * 60 / freq_minutes)), 1)
    horizon = int(forecast_cfg.get("horizon_steps", 1))
    ml_model_name = str(forecast_cfg.get("ml_model", "LightGBM"))
    single_model_requested = "ml_model" in forecast_cfg
    enable_opt = bool(forecast_cfg.get("enable_model_optimization", False))
    ml_method_flag = forecast_cfg.get("ml_method")
    mode = str(forecast_cfg.get("mode", "baseline")).lower()
    ml_method = bool(ml_method_flag) if ml_method_flag is not None else (mode == "advanced")
    output_csv = Path(forecast_cfg.get("output_csv", "assume/extensions/retailer_sim/outputs/forecast_inputs.csv"))
    exog_cols = forecast_cfg.get("exogenous_columns", [])
    advanced_cfg = forecast_cfg.get("advanced_settings", {})

    default_targets = {
        "load_cluster": "cluster_total_load_MW",
        "imbalance": "SBIL_MWH",
    }

    df_filtered = load_dataframe(config)
    df_local = _normalize_dataframe(df_filtered, timestamp_col)
    if ml_method:
        full_config = _build_full_config(config)
        df_training_raw = load_dataframe(full_config)
        df_training = _normalize_dataframe(df_training_raw, timestamp_col)
    else:
        df_training = df_local

    session_definitions = config.get("_intraday_sessions") or []
    available_sessions = len(session_definitions) if session_definitions else (7 if bool(sim_cfg.get("real_market", False)) else 2)
    max_sessions = int(forecast_cfg.get("max_intraday_sessions", 2))
    session_count = available_sessions if max_sessions <= 0 else min(available_sessions, max_sessions)
    mi_columns = [f"mi{idx}" for idx in range(1, session_count + 1)]
    mgp_real_candidates = [
        forecast_cfg.get("mgp_real_column"),
        forecast_cfg.get("mgp_actual_column"),
        "MGP_PRICE_NORD",
        mgp_col,
    ]

    def _resolve_mgp(series: pd.DataFrame) -> pd.Series:
        for candidate in mgp_real_candidates:
            if candidate and candidate in series.columns:
                return pd.to_numeric(series[candidate], errors="coerce")
        return pd.to_numeric(series[mgp_col], errors="coerce")

    def _parse_range(config_section: Mapping[str, List[str]], key: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        rng = config_section.get(key)
        if not rng:
            return None
        return pd.Timestamp(rng[0]), pd.Timestamp(rng[1])

    if not ml_method:
        # Caso più semplice: usa previsioni naive (valori shiftati). Utile per
        # verificare il simulatore senza addestrare modelli complessi.
        target_map = advanced_cfg.get("target_columns", default_targets)
        lag_steps = max(int(forecast_cfg.get("baseline_lag_steps", 1)), 1)
        time_range = advanced_cfg.get("test_range")
        if time_range:
            start, end = pd.Timestamp(time_range[0]), pd.Timestamp(time_range[1])
            mask = (df_local[timestamp_col] >= start) & (df_local[timestamp_col] <= end)
            df_window = df_local.loc[mask].copy()
        else:
            df_window = df_local.copy()
        frame = pd.DataFrame({"timestamp": df_window[timestamp_col].values})
        frame["load_cluster_forecast"] = (
            df_window[target_map.get("load_cluster", default_targets["load_cluster"])].shift(lag_steps).bfill().values
        )
        frame["sbil_forecasted"] = (
            df_window[target_map.get("imbalance", default_targets["imbalance"])].shift(lag_steps).bfill().values
        )
        mi_prices = df_window[mi_col].astype(float)
        for idx, column in enumerate(mi_columns, start=1):
            frame[column] = mi_prices.shift(idx).bfill().values
        frame["mgp"] = _resolve_mgp(df_window).values
        macro_series = (
            df_window[macro_col] if macro_col in df_window.columns else df_window.get("MGP_PRICE_FORECAST")
        )
        if macro_series is not None:
            frame["msd_forecast_EUR_MWh"] = macro_series.shift(lag_steps).bfill().values
    else:
        target_map = advanced_cfg.get("target_columns", default_targets)
        exog_cols = advanced_cfg.get("exogenous_columns", [])
        ranges = {
            "train": _parse_range(advanced_cfg, "train_range"),
            "valid": _parse_range(advanced_cfg, "validation_range"),
            "test": _parse_range(advanced_cfg, "test_range"),
        }
        use_exog = bool(exog_cols)
        ml_models_cfg = forecast_cfg.get("ml_models")
        if ml_models_cfg:
            ml_models = list(ml_models_cfg)
        elif single_model_requested:
            ml_models = [ml_model_name]
        else:
            ml_models = ["LightGBM", "XGBoost", "RandomForest", "LinearRegression"]
        include_lstm = bool(forecast_cfg.get("include_lstm", True))
        targets = [
            TargetConfig(name="load_cluster_forecast", column=target_map.get("load_cluster", default_targets["load_cluster"])),
            TargetConfig(name="sbil_forecasted", column=target_map.get("imbalance", default_targets["imbalance"])),
        ]
        outputs: Dict[str, pd.Series] = {}
        best_models: Dict[str, str] = {}
        for target in targets:
            series = _prepare_series(df_training, timestamp_col, target.column, exog_cols, ranges)
            best_name, pred_series, best_rmse, best_mae = _select_best_model(
                series=series,
                freq=freq,
                seasonal_period=seasonal_period,
                horizon=horizon,
                ml_models=ml_models,
                optimization=enable_opt,
                use_exog=use_exog,
                include_lstm=include_lstm,
            )
            print(
                f"[Forecast] Target={target.name} best_model={best_name} "
                f"RMSE={best_rmse:.3f} MAE={best_mae:.3f}"
            )
            outputs[target.name] = pred_series
            best_models[target.name] = best_name

        load_series = outputs["load_cluster_forecast"]
        sbil_series = outputs["sbil_forecasted"].reindex(load_series.index).ffill().bfill()
        forecast_index = load_series.index
        frame = pd.DataFrame({"timestamp": forecast_index})
        frame["load_cluster_forecast"] = load_series.values
        frame["sbil_forecasted"] = sbil_series.values
        mi_series = df_local[[timestamp_col, mi_col]].set_index(timestamp_col)[mi_col].astype(float)
        mgp_series = _resolve_mgp(df_local)
        mgp_series.index = df_local[timestamp_col]
        macro_series = df_local[[timestamp_col, macro_col]].set_index(timestamp_col)[macro_col] if macro_col in df_local.columns else None
        for idx, column in enumerate(mi_columns, start=1):
            frame[column] = (
                mi_series.shift(idx)
                .reindex(forecast_index)
                .bfill()
                .reset_index(drop=True)
            )
        frame["mgp"] = mgp_series.reindex(forecast_index).bfill().reset_index(drop=True)
        if macro_series is not None:
            frame["msd_forecast_EUR_MWh"] = (
                macro_series.reindex(forecast_index).bfill().reset_index(drop=True)
            )

    frame, availability_columns = _attach_actual_signals(
        frame=frame,
        df=df_local,
        timestamp_col=timestamp_col,
        sim_cfg=sim_cfg,
        info_cfg=config.get("information_flow", {}),
        mi_columns=mi_columns,
        sessions=session_definitions,
    )

    if timestamp_col != "timestamp" and "timestamp" in frame.columns:
        frame = frame.rename(columns={"timestamp": timestamp_col})

    ordered_columns = [
        timestamp_col,
        "load_cluster_forecast",
        "sbil_forecasted",
        *mi_columns,
        "mgp",
        "msd_forecast_EUR_MWh",
        *availability_columns,
    ]
    existing_columns = [col for col in ordered_columns if col in frame.columns]
    frame = frame[existing_columns].copy()

    # Se siamo in modalità avanzata e c'è un modello ML, usa il nome del modello nel file di output.
    if mode == "advanced" and ml_method:
        if single_model_requested:
            label = ml_model_name
        else:
            label = best_models.get("load_cluster_forecast", "")
        if label:
            safe_label = str(label).replace(" ", "_")
            output_csv = output_csv.with_name(f"{output_csv.stem}_{safe_label}{output_csv.suffix}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    print(f"[Forecast] Saved forecast inputs to {output_csv.resolve()}")
    return output_csv


def main() -> None:
    generate_forecast_inputs()


if __name__ == "__main__":
    main()
