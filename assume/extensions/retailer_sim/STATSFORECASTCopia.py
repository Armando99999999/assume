from statsforecast import StatsForecast
from statsforecast.models import (
    WindowAverage,
    Naive,
    SeasonalNaive,
    RandomWalkWithDrift,
    HistoricAverage,
    AutoETS,
    ARIMA,
)
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import itertools

class STATSFORECAST_Predictor:
    """
    Predictor basato su StatsForecast integrato con il framework principale.

    - prepare_data: gestisce train / valid / test, rename delle colonne e uso di esogene.
    - train_model: istanzia e allena il modello StatsForecast scelto.
    - forecast:
        * per modelli semplici (senza esogene) → usa cross_validation su train+test,
          come nel codice originale;
        * per modelli con esogene (ARIMA, AutoARIMA, SARIMA con use_exog=True)
          → forecast iterativo one-step, aggiornando la storia con target + esogene reali,
          senza X_df futuro “a blocchi”.
    - unscale_predictions, plot_predictions, save_metrics: utility per il main.
    """

    def __init__(self, run_mode, target_column, data_freq, seasonal_period,horizon,step_size,
                 use_exog=False,verbose=False):

        self.run_mode = run_mode
        self.target_column = target_column
        self.data_freq = data_freq
        self.seasonal_period = seasonal_period
        self.use_exog = use_exog
        self.horizon = horizon
        self.step_size = step_size
        self.verbose = verbose

        self.train = None
        self.valid = None
        self.test = None
        self.exog_cols = []
        self.model = None
        self.model_type = None

    def _anomaly_removing(self, df):
        """Remove inf/NaN from the slices passed to StatsForecast."""
        if df is None:
            return None

        df = df.replace([np.inf, -np.inf], np.nan)
        cols_to_check = ['y']

        if self.use_exog and self.exog_cols:
            present_exog = [c for c in self.exog_cols if c in df.columns]
            if present_exog:
                df[present_exog] = df[present_exog].ffill().bfill()
                cols_to_check += present_exog

        df = df.dropna(subset=cols_to_check)
        return df.reset_index(drop=True)

    #########################################################
    # DATA PREPARATION
    #########################################################
    def prepare_data(self, train=None, valid=None, test=None):
        """
        Prepara i dataset per StatsForecast.

        - Individua la colonna datetime.
        - Rinomina: date_col -> 'ds', target -> 'y'.
        - Aggiunge 'unique_id'.
        - Se use_exog=False tiene solo ['unique_id', 'ds', 'y'].
        """

        self.train = train.copy()
        self.valid = valid.copy() if valid is not None else None
        self.test = test.copy() if test is not None else None

        # ---- TRAIN ----
        date_col_train = self.train.select_dtypes(include=["datetime64[ns]"]).columns[0]
        self.train = self.train.assign(unique_id='time_series')
        self.train = self.train.rename(columns={date_col_train: 'ds', self.target_column: 'y'})
        self.train = self.train.sort_values('ds').reset_index(drop=True)

        # ---- VALIDATION (se usata in futuro) ----
        if self.valid is not None:
            date_col_valid = self.valid.select_dtypes(include=["datetime64[ns]"]).columns[0]
            self.valid = self.valid.assign(unique_id='time_series')
            self.valid = self.valid.rename(columns={date_col_valid: 'ds', self.target_column: 'y'})
            self.valid = self.valid.sort_values('ds').reset_index(drop=True)

        # ---- TEST ----
        if self.test is not None:
            date_col_test = self.test.select_dtypes(include=["datetime64[ns]"]).columns[0]
            self.test = self.test.assign(unique_id='time_series')
            self.test = self.test.rename(columns={date_col_test: 'ds', self.target_column: 'y'})
            self.test = self.test.sort_values('ds').reset_index(drop=True)

        # ---- ESOGENE / COVARIATE ----
        if not self.use_exog:
            # tengo solo la serie principale
            self.train = self.train[['unique_id', 'ds', 'y']]
            if self.valid is not None:
                self.valid = self.valid[['unique_id', 'ds', 'y']]
            if self.test is not None:
                self.test = self.test[['unique_id', 'ds', 'y']]
            self.exog_cols = []
        else:
            # tengo tutte le colonne, salvo la lista delle esogene
            self.exog_cols = [
                c for c in self.train.columns
                if c not in ['unique_id', 'ds', 'y']
            ]

        self.train = self._anomaly_removing(self.train)
        if self.valid is not None:
            self.valid = self._anomaly_removing(self.valid)
        if self.test is not None:
            self.test = self._anomaly_removing(self.test)

    #########################################################
    # MODEL TRAINING
    #########################################################
    def train_model(self, model_type,optimization):
        """
        Allena il modello StatsForecast.

        model_type (come da main):
        'Naive', 'SeasonalNaive', 'win_avg', 'random_walk_with_drift',
        'hist_avg', 'AutoETS', 'ARIMA', 'AutoARIMA', 'SARIMA'.
        """
        
        p_values = range(0, 4)
        d_values = range(0, 1)
        q_values = range(0, 4)
        P_range=range(0,2)
        D_range=range(0,2) 
        Q_range=range(0,2)
        param_grid = list(itertools.product(p_values, d_values, q_values))
        seasonal_grid = list(itertools.product(P_range, D_range, Q_range))

        self.model_type = model_type

        model_dict = {
            'Naive': Naive(),
            'SeasonalNaive': SeasonalNaive(self.seasonal_period),
            'win_avg': WindowAverage(self.seasonal_period),
            'random_walk_with_drift': RandomWalkWithDrift(),
            'hist_avg': HistoricAverage(),
            'AutoETS': AutoETS(season_length=self.seasonal_period),
            'ARIMA': [ARIMA(order=(p,d,q)) for p,d,q in param_grid],
            'SARIMA':[ARIMA(order=(p,d,q),
                        seasonal_order=(P,D,Q),
                        season_length=self.seasonal_period
            )
            for (p,d,q) in param_grid 
            for (P,D,Q) in seasonal_grid]
        }

        if model_type not in model_dict:
            raise ValueError(f"Model type '{model_type}' non riconosciuto in STATSFORECAST.")

        base_model = model_dict[model_type]

        # Modelli che possono usare esogene
        models_with_exog = {'ARIMA', 'SARIMA'}
        if model_type in models_with_exog:
            candidate_models = base_model if isinstance(base_model, list) else [base_model]
            for idx, mdl in enumerate(candidate_models):
                alias = f"{model_type}_{idx}"
                setattr(mdl, "alias", alias)

            if optimization:
                best_model = self.cross_valid(
                    self.train,
                    candidate_models,
                    self.horizon,
                    self.step_size,
                    self.data_freq,
                    verbose=False,
                )
                selected_model = best_model
            else:
                if model_type == 'ARIMA':
                    selected_model = ARIMA(order=(1, 1, 1))
                else:  # SARIMA
                    selected_model = ARIMA(
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1),
                        season_length=self.seasonal_period,
                    )

            self.model = StatsForecast(models=[selected_model], freq=self.data_freq)
        else:
            models = [base_model]

            self.model = StatsForecast(
                models=models,
                freq=self.data_freq,
            )
        if self.valid is not None:
            full_train = pd.concat([self.train, self.valid], ignore_index=True)
        else:
            full_train = self.train.copy()

        full_train = full_train.sort_values('ds').reset_index(drop=True)

        # --- Limito la coda anche per il fit finale (come in CV) ---
        n_total = len(full_train)
        if self.valid is not None:
            valid_len = len(self.valid)
            max_points_fit = min(2 * valid_len, 20_000)   # es. stessa logica di max_points
        else:
            max_points_fit = min(n_total, 20_000)

        if n_total > max_points_fit:
            full_train = full_train.tail(max_points_fit).copy()

        # (opzionale ma consigliato) se hai esogene, puliscile come in CV
        if self.use_exog and self.exog_cols:
            full_train[self.exog_cols] = full_train[self.exog_cols].replace([np.inf, -np.inf], np.nan)
            full_train[self.exog_cols] = full_train[self.exog_cols].ffill().bfill()

        # --- Fit finale del modello StatsForecast ---
        self.model.fit(
            df=full_train,
            id_col='unique_id',
            time_col='ds',
            target_col='y',
        )

        
        print(f"[STATSFORECAST] Modello '{model_type}' addestrato.")

        return self.model

    #########################################################
    # FORECAST: SIMPLE MODELS → CROSS_VALIDATION, EXOG MODELS → ITERATIVO
    #########################################################
    def forecast(self, horizon, step_size,window):
        """
        Per i modelli semplici (o senza esogene) → cross_validation su train+test,
        come nel codice originale.

        Per i modelli con esogene (ARIMA, AutoARIMA, SARIMA con use_exog=True)
        → forecast iterativo one-step, senza X_df futuro a blocchi.
        """

        # Modelli che possono usare esogene
        models_with_exog = {'ARIMA', 'SARIMA'}

        ################################################################
        # MODELLI SEMPLICI (o SENZA ESOGENE) → CROSS_VALIDATION
        ################################################################
        if (self.model_type not in models_with_exog) :
            test_size = len(self.test)

            remainder = (test_size - horizon) % step_size
            if remainder != 0:
                adjusted_test_size = test_size - remainder
                if adjusted_test_size < horizon:
                    raise ValueError(
                        "Impossibile aggiustare il test_size per soddisfare i vincoli di cross_validation."
                    )
                self.test = self.test.tail(adjusted_test_size).copy()
                test_size = adjusted_test_size
                if self.verbose:
                    print(f"[STATSFORECAST] Adjusted test_size to {test_size} for CV.")

            # df per CV: train + test
            test_df = pd.concat([self.train, self.test], ignore_index=True)

            cv_df = self.model.cross_validation(
                df=test_df,
                refit=False,
                h=horizon,
                step_size=step_size,
                n_windows=None,
                test_size=test_size,
                id_col='unique_id',
                time_col='ds',
                target_col='y',
            )

            candidate_cols = [
                col for col in cv_df.columns
                if col not in {'unique_id', 'cutoff', 'ds', 'y'}
            ]
            if not candidate_cols:
                raise ValueError("cross_validation non ha restituito colonne di previsione.")

            pred_col = candidate_cols[0]

            preds = cv_df[['unique_id', 'ds', pred_col]].copy()
            preds = preds.drop_duplicates(subset=['unique_id', 'ds'], keep='last')
            test_ds = self.test[['unique_id', 'ds']].copy()
            preds = test_ds.merge(preds, on=['unique_id', 'ds'], how='left')
            preds = preds.rename(columns={pred_col: self.target_column})
            preds = preds.drop(columns=['unique_id'])
            preds = preds.set_index('ds')
            preds = preds.sort_index().tail(len(self.test))

            return preds

        ################################################################
        # CASO 2: MODELLI CON ESOGENE -> CROSS_VALIDATION CON TEST_SIZE
        ################################################################
        train = self.train.copy()
        tail_len = window * self.seasonal_period
        if len(self.train) > tail_len:
            train = self.train.tail(tail_len).copy()
        test_size = len(self.test)

        remainder = (test_size - horizon) % step_size
        if remainder != 0:
            adjusted_test_size = test_size - remainder
            self.test = self.test.tail(adjusted_test_size).copy()
            test_size = adjusted_test_size
        
        full_df = pd.concat([train, self.test], ignore_index=True)

        cv_df = self.model.cross_validation(
            df=full_df,
            refit=False,
            h=horizon,
            step_size=step_size,
            test_size=test_size,
            n_windows=None,
            id_col='unique_id',
            time_col='ds',
            target_col='y',
        )

        pred_cols = [c for c in cv_df.columns if c not in ['unique_id', 'ds', 'cutoff', 'y']]
        if not pred_cols:
            raise ValueError("cross_validation non ha restituito colonne di previsione.")
        model_col_name = pred_cols[0]

        preds = cv_df[['unique_id', 'ds', model_col_name]].copy()
        preds = preds.drop_duplicates(subset=['unique_id', 'ds'], keep='last')
        test_ds = self.test[['unique_id', 'ds']].copy()
        preds = test_ds.merge(preds, on=['unique_id', 'ds'], how='left')
        preds = preds.rename(columns={model_col_name: self.target_column})
        preds = preds.drop(columns=['unique_id']).set_index('ds')
        preds = preds.sort_index().tail(len(self.test))

        return preds


    #########################################################
    # UTILITY
    #########################################################
    def unscale_predictions(self, predictions, folder_path):
        with open(f"{folder_path}/target_scaler.pkl", "rb") as file:
            target_scaler = pickle.load(file)

        predictions = np.array(predictions).reshape(-1, 1)
        predictions = target_scaler.inverse_transform(predictions).flatten()
        return pd.Series(predictions)

    def plot_predictions(self, predictions, test_data):
        test_series = test_data[self.target_column]
        pred_series = pd.Series(predictions)

        align_len = min(len(pred_series), len(test_series))
        if align_len <= 0:
            raise ValueError("Nessun punto valido per il plot delle previsioni.")

        test_series = test_series.tail(align_len)
        pred_series = pred_series.tail(align_len)

        plt.plot(test_series.index, test_series, 'b-', label='Test Set')
        plt.plot(test_series.index, pred_series.values, 'r--', label='Forecast')
        plt.title(f'STATSFORECAST prediction for feature: {self.target_column}')
        plt.xlabel('Time index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def save_metrics(self, path, metrics):
        with open(f"{path}/model_details_BASELINE.txt", "w") as file:
            file.write(f"Test Info:\n")
            file.write(f"Model Performance: {metrics}\n")
            file.write(f"Launch Command Used:{sys.argv[1:]}\n")
        return self.model

    def cross_valid(self, train, models, horizon, step_size, data_freq, verbose=False):
        """
        Cross-validation rolling-origin con StatsForecast.cross_validation.

        - Usa SOLO il train (no valid nella CV, quindi niente leakage).
        - Limita la coda usata per la CV.
        - Usa test_size (non n_windows).
        - Calcola l'RMSE a mano per scegliere il modello migliore.
        """

        id_col = 'unique_id'
        time_col = 'ds'
        target_col = 'y'

        # 1) Ordino il train e prendo solo la coda (per velocizzare)
        df = train.sort_values(time_col).reset_index(drop=True)
        n_total = len(df)

        if self.valid is not None:
            valid_len = len(self.valid)
            max_points = min(2 * valid_len, 20_000)
        else:
            max_points = min(n_total, 20_000)

        if n_total > max_points:
            df = df.tail(max_points).copy()

        train_size = len(df)
        if train_size <= horizon:
            raise ValueError(
                f"Train troppo corto per horizon={horizon}: train_size={train_size}"
            )

        # 2) Aggiusto per coerenza con step_size
        remainder = (train_size - horizon) % step_size
        if remainder != 0:
            df = df.tail(train_size - remainder).copy()
            train_size = len(df)

        # 3) test_size: quanti punti di coda usare per la CV
        if self.valid is not None:
            test_size = min(len(self.valid), train_size - horizon)
        else:
            test_size = max(horizon, train_size // 3)

        if test_size <= 0:
            raise ValueError(
                f"test_size non valido ({test_size}). "
                f"horizon={horizon}, train_size={train_size}"
            )
        remainder = (test_size - horizon) % step_size
        if remainder != 0:
            adjusted_test_size = test_size - remainder
            if adjusted_test_size < horizon:
                raise ValueError(
                    "Impossibile aggiustare il test_size per soddisfare i vincoli della cross_valid."
                )
            test_size = adjusted_test_size
            if verbose:
                print(f"[CV] Adjusted test_size to {test_size} for step_size compatibility.")

        if verbose:
            print(
                f"[CV] train_size={train_size}, horizon={horizon}, "
                f"step_size={step_size}, test_size={test_size}"
            )

        # 4) Cross-validation StatsForecast
        sf = StatsForecast(models=models, freq=data_freq)
        #input_size=max_points//self.seasonal_period  # default

        cv_df = sf.cross_validation(
            df=df,
            h=horizon,
            step_size=step_size,
            test_size=test_size,
            n_windows=None,        # viene dedotto da test_size e step_size
            refit=False,           # un solo fit per modello su df
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )

        if verbose:
            print(f"[CV] cv_df shape={cv_df.shape}")

        # 5) RMSE per ogni modello (con evaluate)
        metrics_df = evaluate(
            cv_df,
            metrics=[rmse],
            id_col=id_col,
            target_col=target_col,
        )

        metric_cols = [c for c in metrics_df.columns if c != id_col]
        if not metric_cols:
            raise ValueError("cross_validation non ha prodotto colonne di previsione.")

        rmse_per_model = {}
        for col in metric_cols:
            rmse_val = float(metrics_df[col].mean())
            rmse_per_model[col] = rmse_val
            if verbose:
                print(f"[CV] Modello {col}: RMSE={rmse_val:.6f}")

        # 6) Miglior modello
        best_model_alias = min(rmse_per_model, key=rmse_per_model.get)
        best_rmse = rmse_per_model[best_model_alias]

        best_model = None
        for m in models:
            alias = getattr(m, "alias", repr(m))
            if alias == best_model_alias:
                best_model = m
                break

        if verbose:
            alias = getattr(best_model, "alias", repr(best_model))
            msg = f"[CV] Miglior modello (RMSE): {alias} | RMSE={best_rmse:.6f}"
            if hasattr(best_model, 'order'):
                msg += f" | order={best_model.order}"
            if hasattr(best_model, 'seasonal_order'):
                msg += f", seasonal_order={best_model.seasonal_order}"
            print(msg)

        return best_model
