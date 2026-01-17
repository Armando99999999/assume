"""Forecasting orchestrator for retailer datasets."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from Predictors.MLFORECASTCopia import MLFORECAST_Predictor
from Predictors.NAIVE_model import NAIVE_Predictor
from Predictors.NEURALFORECASTCopia import NEURALFORECAST_Predictor


class RetailerForecastRunner:
    """
    Run coordinated forecasts for:
      - retailer cluster load (cluster_total_load_MW)
      - macro-zonal imbalance (SBIL_MWH)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        datetime_col: str = "ORAINI",
    ) -> None:
        self.datetime_col = datetime_col
        self.df = df.copy()
        self.df[datetime_col] = pd.to_datetime(self.df[datetime_col])
        self.df = self.df.sort_values(datetime_col).reset_index(drop=True)

        self.model_candidates: List[Dict[str, Any]] = [
            {"name": "LightGBM", "kind": "mlforecast", "model_name": "LightGBM"},
            {"name": "XGBoost", "kind": "mlforecast", "model_name": "XGBoost"},
            {"name": "RandomForest", "kind": "mlforecast", "model_name": "RandomForest"},
            {"name": "LinearRegression", "kind": "mlforecast", "model_name": "LinearRegression"},
            {"name": "NaivePersistence", "kind": "naive", "forecast_type": "multi_step_persistence"},
            {"name": "NaiveMean", "kind": "naive", "forecast_type": "multi_step_mean"},
            {"name": "NaiveSeasonal", "kind": "naive_seasonal"},
            {"name": "NeuralForecast", "kind": "neuralforecast", "model_name": "NeuralForecast"},
        ]

        self.last_model_report: Dict[str, Any] = {}

    # --------------------------------------------------------------------- #
    # Feature selection
    # --------------------------------------------------------------------- #
    def _select_exog_columns(
        self,
        target_col: str,
        *,
        max_features: int,
        extra_preferred: Optional[Sequence[str]] = None,
        excluded_cols: Optional[Sequence[str]] = None,
    ) -> List[str]:
        numeric_df = self.df.select_dtypes(include=[np.number]).copy()
        excluded = set(excluded_cols or [])
        excluded.update({target_col})
        numeric_df = numeric_df.drop(columns=[col for col in numeric_df.columns if col in excluded], errors="ignore")

        if numeric_df.empty:
            return []

        target_series = self.df[target_col]
        corr = numeric_df.corrwith(target_series).abs().dropna().sort_values(ascending=False)
        selected = corr.head(max_features).index.tolist()

        if extra_preferred:
            for col in extra_preferred:
                if col in numeric_df.columns and col not in selected:
                    selected.append(col)

        return selected

    def _prepare_target_frame(
        self,
        target_col: str,
        feature_cols: Sequence[str],
    ) -> pd.DataFrame:
        cols = [self.datetime_col, target_col, *feature_cols]
        df_target = self.df[cols].dropna(subset=[target_col]).copy()
        df_target = df_target.sort_values(self.datetime_col).reset_index(drop=True)
        return df_target

    # --------------------------------------------------------------------- #
    # Model evaluation helpers
    # --------------------------------------------------------------------- #
    def _forecast_with_model(
        self,
        cfg: Dict[str, Any],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        target_col: str,
        data_freq: str,
        seasonal_period: int,
        horizon: int,
        use_exog: bool,
    ) -> pd.Series:
        if cfg["kind"] == "mlforecast":
            predictor = MLFORECAST_Predictor(
                run_mode="backtest",
                target_column=target_col,
                data_freq=data_freq,
                seasonal_period=seasonal_period,
                optimization=False,
                use_exog=use_exog,
                horizon=horizon,
                verbose=False,
            )
            predictor.prepare_data(train=train_df, valid=None, test=test_df)
            predictor.train_model(model_name=cfg["model_name"], optimization=False)
            preds = predictor.backtest(horizon=horizon)[target_col]
            preds.index = pd.DatetimeIndex(preds.index)
            return preds

        history = train_df.set_index(self.datetime_col)[[target_col]]
        history = history.sort_index()
        naive = NAIVE_Predictor(run_mode="backtest", target_column=target_col, data_freq=data_freq)
        if cfg["kind"] == "naive":
            preds = naive.forecast(
                history,
                horizon=horizon,
                forecast_type=cfg["forecast_type"],
            )
        elif cfg["kind"] == "naive_seasonal":
            preds = naive.seasonal_forecast(
                history,
                horizon=horizon,
                period=seasonal_period,
                mode="multi_step",
            )
        else:
            raise ValueError(f"Tipo modello non supportato: {cfg}")

        return preds

    def _evaluate_candidates(
        self,
        df_target: pd.DataFrame,
        *,
        target_col: str,
        horizon: int,
        data_freq: str,
        seasonal_period: int,
        use_exog: bool,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        if len(df_target) <= horizon:
            raise ValueError(f"Dataset troppo corto ({len(df_target)}) rispetto all'horizon {horizon}.")

        train = df_target.iloc[:-horizon].copy()
        test = df_target.iloc[-horizon:].copy()
        actual = test.set_index(self.datetime_col)[target_col]

        best_metric = np.inf
        best_pred: Optional[pd.Series] = None
        best_model: Optional[str] = None
        evaluations: List[Dict[str, Any]] = []

        for cfg in self.model_candidates:
            try:
                preds = self._forecast_with_model(
                    cfg,
                    train,
                    test,
                    target_col=target_col,
                    data_freq=data_freq,
                    seasonal_period=seasonal_period,
                    horizon=horizon,
                    use_exog=use_exog,
                )
                preds = preds.reindex(actual.index)
                metric = mean_absolute_error(actual, preds)
                evaluations.append({"model": cfg["name"], "mae": float(metric)})
                if metric < best_metric:
                    best_metric = metric
                    best_pred = preds
                    best_model = cfg["name"]
            except Exception as exc:  # pragma: no cover - defensive
                evaluations.append({"model": cfg["name"], "mae": np.inf, "error": str(exc)})
                continue

        if best_pred is None or best_model is None:
            raise RuntimeError("Nessun modello è riuscito a produrre previsioni valide.")

        report = {
            "best_model": best_model,
            "best_mae": float(best_metric),
            "evaluations": evaluations,
        }
        return best_pred, report

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def run_all_forecasts(
        self,
        *,
        horizon: int = 96,
        data_freq: str = "15min",
        seasonal_period: int = 96,
        max_exog_features: int = 20,
        cluster_target: str = "cluster_total_load_MW",
        sbil_target: str = "SBIL_MWH",
        extra_cluster_exog: Optional[Sequence[str]] = None,
        extra_sbil_exog: Optional[Sequence[str]] = None,
        ml_method: bool = True,
        ml_preferred_model: Optional[str] = None,
    ) -> pd.DataFrame:
        if cluster_target not in self.df.columns:
            raise KeyError(f"Colonna '{cluster_target}' non trovata nel dataset.")
        if sbil_target not in self.df.columns:
            raise KeyError(f"Colonna '{sbil_target}' non trovata nel dataset.")

        exclude_cols = [self.datetime_col, "cluster_load_hat", "sbil_macro_hat", "cluster_imbalance_hat"]

        cluster_features = self._select_exog_columns(
            cluster_target,
            max_features=max_exog_features,
            extra_preferred=extra_cluster_exog,
            excluded_cols=exclude_cols,
        )
        sbil_features = self._select_exog_columns(
            sbil_target,
            max_features=max_exog_features,
            extra_preferred=extra_sbil_exog,
            excluded_cols=exclude_cols,
        )

        df_cluster = self._prepare_target_frame(cluster_target, cluster_features)
        df_sbil = self._prepare_target_frame(sbil_target, sbil_features)

        if not ml_method:
            naive = NAIVE_Predictor(run_mode="backtest", target_column=cluster_target, data_freq=data_freq)
            history_cluster = df_cluster.set_index(self.datetime_col)[[cluster_target]].sort_index()
            history_sbil = df_sbil.set_index(self.datetime_col)[[sbil_target]].sort_index()
            cluster_pred = naive.forecast(history_cluster, horizon=horizon, forecast_type="multi_step_persistence")
            cluster_pred.index = history_cluster.index[-horizon:]

            sbil_naive = NAIVE_Predictor(run_mode="backtest", target_column=sbil_target, data_freq=data_freq)
            sbil_pred = sbil_naive.forecast(history_sbil, horizon=horizon, forecast_type="multi_step_persistence")
            sbil_pred.index = history_sbil.index[-horizon:]

            cluster_report = {"best_model": "NaivePersistence", "best_mae": np.nan, "evaluations": []}
            sbil_report = {"best_model": "NaivePersistence", "best_mae": np.nan, "evaluations": []}
        else:
            cluster_pred, cluster_report = self._evaluate_candidates(
                df_cluster,
                target_col=cluster_target,
                horizon=horizon,
                data_freq=data_freq,
                seasonal_period=seasonal_period,
                use_exog=bool(cluster_features),
            )
            sbil_pred, sbil_report = self._evaluate_candidates(
                df_sbil,
                target_col=sbil_target,
                horizon=horizon,
                data_freq=data_freq,
                seasonal_period=seasonal_period,
                use_exog=bool(sbil_features),
            )

        df_out = self.df.set_index(self.datetime_col)
        df_out.loc[cluster_pred.index, "cluster_load_hat"] = cluster_pred
        df_out.loc[sbil_pred.index, "sbil_macro_hat"] = sbil_pred
        if "cluster_total_load_MW" in df_out.columns:
            df_out["cluster_imbalance_hat"] = (
                df_out["cluster_load_hat"] - df_out["cluster_total_load_MW"]
            )

        # Previsione cluster/sbilancio allineata alle colonne storiche del CSV naive
        df_out["load_cluster_forecast"] = df_out["cluster_load_hat"]
        df_out["consumption_forecast_MWh"] = df_out["cluster_load_hat"]
        df_out["sbil_forecasted"] = df_out["sbil_macro_hat"]

        # Forecast naive di prezzo MGP/MI per mantenere le stesse colonne in input
        if "price_MGP_EUR_MWh" in df_out.columns:
            mgp_forecast = df_out["price_MGP_EUR_MWh"].shift(1).ffill()
            df_out["price_MGP_forecast_EUR_MWh"] = mgp_forecast
            df_out["mgp"] = mgp_forecast
        if "price_MI_EUR_MWh" in df_out.columns:
            mi_forecast = df_out["price_MI_EUR_MWh"].shift(1).ffill()
            df_out["price_MI_forecast_EUR_MWh"] = mi_forecast
            df_out["mi1"] = mi_forecast
        if "price_MI2_EUR_MWh" in df_out.columns:
            mi2_forecast = df_out["price_MI2_EUR_MWh"].shift(1).ffill()
        else:
            source = None
            if "price_MI_EUR_MWh" in df_out.columns:
                source = df_out["price_MI_EUR_MWh"]
            elif "mi1" in df_out.columns:
                source = df_out["mi1"]
            mi2_forecast = source.shift(1).ffill() if source is not None else pd.Series(np.nan, index=df_out.index)
        df_out["mi2"] = mi2_forecast

        df_out = df_out.reset_index().rename(columns={"index": self.datetime_col})
        cluster_report["selected_features"] = cluster_features if ml_method else []
        sbil_report["selected_features"] = sbil_features if ml_method else []
        self.last_model_report = {
            cluster_target: cluster_report,
            sbil_target: sbil_report,
        }
        # Importante: restituiamo l'intero dataframe con le nuove colonne cosi' da
        # preservare forma/ordine del CSV originale (stesso intervallo temporale).
        return df_out


__all__ = ["RetailerForecastRunner"]
