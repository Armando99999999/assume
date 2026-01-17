import pandas as pd
try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - opzionale in produzione headless
    plt = None
import numpy as np
import pickle
import sys
from mlforecast import MLForecast
from utilsforecast.evaluation import evaluate
from mlforecast.lag_transforms import RollingMean
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class MLFORECAST_Predictor:

    def __init__(self, run_mode, target_column, data_freq,
                 seasonal_period, optimization, use_exog,horizon, verbose=False):
        """
        Constructs all the necessary attributes for the MLFORECAST_Predictor object.

        :param run_mode: The mode in which the predictor runs
        :param target_column: The target column of the DataFrame to predict
        :param verbose: If True, prints detailed outputs during the execution of methods
        """

        self.run_mode = run_mode
        self.verbose = verbose
        self.target_column = target_column
        self.data_freq = data_freq
        self.seasonal_period = seasonal_period
        self.optimization = optimization
        # Se True, mantiene e usa covariate/esogene
        self.use_exog = use_exog
        self.horizon = horizon
        self.original_test_ds = None

        # niente date features per ora (le togliamo per evitare l'errore sulle lambda)
        # se in futuro ti serviranno, basterà definire funzioni normali (non lambda)
        # e passarle a MLForecast(date_features=[funzione1, funzione2, ...])

    ####################################################
    # DATA PREPARATION
    ####################################################
    def prepare_data(self, train=None, valid=None, test=None):
        """
        Prepara i dataset per MLForecast.

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
            self.original_test_ds = self.test['ds'].copy()

        # Allineo ogni blocco alla frequenza dichiarata per evitare buchi temporali
        self.train = self.date_check(self.train)
        if self.valid is not None:
            self.valid = self.date_check(self.valid)
        if self.test is not None:
            self.test = self.date_check(self.test)

        if self.use_exog:
            frames = [self.train]
            if self.valid is not None:
                frames.append(self.valid)
            if self.test is not None:
                frames.append(self.test)
            self.exog_reference = pd.concat(frames, ignore_index=True)
        else:
            self.exog_reference = None

        if not self.use_exog:
            # uso solo la serie principale
            self.train = self.train[['unique_id', 'ds', 'y']]
            if self.valid is not None:
                self.valid = self.valid[['unique_id', 'ds', 'y']]
            if self.test is not None:
                self.test = self.test[['unique_id', 'ds', 'y']]

    ####################################################
    # FEATURE ENGINEERING PER GRID SEARCH
    ####################################################
    def prepare_features_for_cv(self, input_data):

        fcst = MLForecast(
            models={self.model_name: self.estimator},
            freq=self.data_freq,
            lags=[1],#[self.horizon, 4 * self.horizon, self.seasonal_period * self.horizon,2*self.seasonal_period*self.horizon,7*self.seasonal_period*self.horizon],  # e volendo anche 192, 288...
            lag_transforms={
            1: [RollingMean(window_size=self.seasonal_period),
            RollingMean(window_size= 2 * self.seasonal_period)]
            }
        )
        features = fcst.preprocess(
            input_data.assign(unique_id='time_series'),
            id_col='unique_id',
            time_col='ds',
            target_col='y',
            static_features=[]
        )
        features = features.dropna()
        X = features.drop(['y', 'unique_id', 'ds'], axis=1)
        y = features['y']
        X = X.loc[:, X.nunique(dropna=True) > 1]
        return X, y

    ####################################################
    # TRAIN MODEL
    ####################################################
    def train_model(self, model_name, optimization):

        model_dict = {
            'LightGBM': lgb.LGBMRegressor(random_state=0, verbosity=-1),
            'XGBoost': xgb.XGBRegressor(random_state=0, verbosity=0, tree_method='hist', n_jobs=1),
            'RandomForest': RandomForestRegressor(random_state=0),
            'LinearRegression': LinearRegression(),
        }

        # salva nome/estimator base per uso in prepare_features_for_cv
        self.model_name = model_name
        self.estimator = model_dict[model_name]

        # calcolo i lag UNA VOLTA per tutti i modelli
        lags = self._compute_lags()

        if model_name == 'LightGBM' and optimization:
            best_estimator = self.optimize_lgbm(self.train)
        elif model_name == 'XGBoost' and optimization:
            best_estimator = self.optimize_xgb(self.train)
        else:
            best_estimator = model_dict[model_name]

        self.estimator = best_estimator

        # uso gli stessi lags e le stesse rolling per tutti i modelli, compresa LinearRegression
        self.model = MLForecast(
            models={self.model_name: self.estimator},
            freq=self.data_freq,
            lags=[1],
            lag_transforms={
                1: [
                    RollingMean(window_size=4),
                    RollingMean(window_size=8),
                ]
            }
        )

        train_df = self.train
        if self.valid is not None and not self.valid.empty:
            train_df = pd.concat([train_df, self.valid], ignore_index=True)

        self.model.fit(
            df=train_df,
            id_col='unique_id',
            time_col='ds',
            target_col='y',
            static_features=[]
        )

        return self.model

    ####################################################
    # BACKTEST ROLLING
    ####################################################
    def backtest(self, horizon):

        trained_model = self.model
        total_predictions = pd.DataFrame()

        

        for step in range(0, len(self.test), horizon):

            new_observations = self.test.iloc[step:step + horizon].copy()

            if new_observations.empty:
                break
            h_step = len(new_observations)
            if h_step == 0:
                break

            # Exog future solo se use_exog=True
            new_observations, X_df = self._prepare_future_exog(trained_model, new_observations, h_step)

            if X_df is not None:
                forecasts = trained_model.predict(h=h_step, X_df=X_df)
            else:
                forecasts = trained_model.predict(h=h_step)

            total_predictions = pd.concat([total_predictions, forecasts], ignore_index=True)

            # aggiorno il modello con le osservazioni reali (già interpolate se serviva)
            trained_model.update(new_observations)

        total_predictions = total_predictions.rename(columns={self.model_name: self.target_column})
        if 'unique_id' in total_predictions.columns:
            total_predictions = total_predictions.drop(columns=['unique_id'])

        # Allineo le previsioni alle date del test per evitare mismatch di dimensione
        if self.original_test_ds is not None:
            test_ds = pd.DataFrame({'ds': self.original_test_ds}).reset_index(drop=True)
        else:
            test_ds = self.test[['ds']].copy()
        total_predictions = test_ds.merge(total_predictions, on='ds', how='left')
        total_predictions = total_predictions.set_index('ds')

        return total_predictions

    ####################################################
    # UNSCALE, PLOT, SAVE METRICS
    ####################################################
    def unscale_predictions(self, predictions, folder_path):
        """
        Unscales the predictions using the scaler saved during model training.
        """
        with open(f"{folder_path}/target_scaler.pkl", "rb") as file:
            target_scaler = pickle.load(file)

        predictions = np.array(predictions)
        predictions = predictions.reshape(-1, 1)
        predictions = target_scaler.inverse_transform(predictions)
        predictions = predictions.flatten()
        predictions = pd.Series(predictions)

        return predictions

    def plot_predictions(self, predictions, test_data):

        if plt is None:
            return

        horizon = test_data.shape[0]
        test = test_data[:horizon][self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, predictions, 'r--', label=self.model_name)
        plt.title(f'MLForecast prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def save_metrics(self, path, metrics):
        # Save test info
        with open(f"{path}/model_details_MLFORECAST.txt", "w") as file:
            file.write(f"Test Info:\n")
            file.write(f"Model Performance: {metrics}\n")
            file.write(f"Launch Command Used:{sys.argv[1:]}\n")
        return self.model

    ####################################################
    # HYPERPARAMETER OPTIMIZATION
    ####################################################
    def optimize_lgbm(self, input_data):

        X, y = self.prepare_features_for_cv(input_data)
        param_grid = {
            'num_leaves': list(range(15, 100, 15)),
            'learning_rate': list(np.arange(0.001, 0.012, 0.005)),
            'n_estimators': list(range(70, 401, 50)),
        }
        model = lgb.LGBMRegressor(random_state=0, verbosity=-1, n_jobs=1)
        tscv = TimeSeriesSplit(n_splits=3)
        grid = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
        grid.fit(X, y)
        print("Best params:", grid.best_params_)
        return grid.best_estimator_

    def optimize_xgb(self, input_data):
        X, y = self.prepare_features_for_cv(input_data)
        param_grid = {
            # intensità del boosting
            'n_estimators': [1500,2000],
            'learning_rate': [1e-4,0.005, 0.01],
            'max_depth': [6,12, None],
            'min_child_weight': [10, 12],
            'reg_lambda': [0.0,1.0],
            'max_bin': [255,511],
        }
        X = X.loc[:, X.nunique(dropna=True) > 1]
        model = xgb.XGBRegressor(random_state=0, verbosity=0, tree_method='hist', n_jobs=1)
        tscv = TimeSeriesSplit(n_splits=4)
        grid = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
        grid.fit(X, y)
        print("Best params:", grid.best_params_)
        return grid.best_estimator_
    def date_check(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ricampiona la serie sulla frequenza dichiarata e riempie i buchi."""
        if df is None or df.empty:
            return df

        df = df.sort_values('ds')
        if df['ds'].duplicated().any():
            df = df.drop_duplicates(subset='ds', keep='last')
        full_idx = pd.date_range(
            start=df['ds'].iloc[0],
            end=df['ds'].iloc[-1],
            freq=self.data_freq
        )

        df = df.set_index('ds').reindex(full_idx)
        df.index.name = 'ds'

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            interpolated = (
                df[numeric_cols]
                .interpolate(method='linear', limit_direction='both')
                .ffill()
                .bfill()
            )
            for col in numeric_cols:
                df[col] = interpolated[col].to_numpy()

        other_cols = [c for c in df.columns if c not in numeric_cols]
        if other_cols:
            df[other_cols] = df[other_cols].ffill().bfill()

        return df.reset_index().rename(columns={'index': 'ds'})

    def _fill_missing_exog(self, df, exog_cols):
        if not exog_cols:
            return df

        ref_df = getattr(self, 'exog_reference', None)
        for col in exog_cols:
            if col not in df.columns:
                continue
            if df[col].isna().any():
                fallback_val = None
                if ref_df is not None and col in ref_df.columns:
                    valid_vals = ref_df[col].dropna()
                    if not valid_vals.empty:
                        fallback_val = valid_vals.iloc[-1]
                if fallback_val is None:
                    fallback_val = 0.0
                df[col] = df[col].fillna(fallback_val)
        return df

    def _prepare_future_exog(self, trained_model, observations, h_step):
        """Intercetta, ripulisce e prepara gli esogeni futuri per il backtest rolling."""
        if not self.use_exog:
            return observations, None

        base_cols = ['unique_id', 'ds', 'y']
        exog_cols = [c for c in observations.columns if c not in base_cols]
        if not exog_cols:
            return observations, None

        clean_obs = observations.copy()
        clean_obs[exog_cols] = (
            clean_obs[exog_cols]
            .interpolate(method='linear', limit_direction='both')
            .ffill()
            .bfill()
        )
        clean_obs = self._fill_missing_exog(clean_obs, exog_cols)

        future_df = trained_model.make_future_dataframe(h=h_step)
        future_df = future_df.merge(
            clean_obs[['unique_id', 'ds'] + exog_cols],
            on=['unique_id', 'ds'],
            how='left'
        )
        if future_df[exog_cols].isna().any().any():
            future_df = future_df.sort_values(['unique_id', 'ds'])
            for col in exog_cols:
                future_df[col] = future_df.groupby('unique_id')[col].transform(
                    lambda s: s.interpolate(method='linear', limit_direction='both').ffill().bfill()
                )
        future_df = self._fill_missing_exog(future_df, exog_cols)
        if future_df[exog_cols].isna().any().any():
            missing_rows = trained_model.get_missing_future(h=h_step, X_df=future_df)
            raise ValueError(
                "Missing exogenous values for some future steps anche dopo interpolazione.\n"
                f"Combinazioni mancanti:\n{missing_rows}"
            )

        return clean_obs, future_df
    ####################################################
    # UTILITY: LAG ADATTIVI
    ####################################################
    def _compute_lags(self):
        """
        Calcola in modo adattivo i lag a partire da seasonal_period.
        - Sempre lag 1
        - Un lag intra-giornaliero (~ stagione/4)
        - Un lag giornaliero = seasonal_period
        Niente lag settimanale per evitare sparsità/rumore.
        """
        sp = int(self.seasonal_period) if self.seasonal_period is not None else None

        lags = [1]

        if sp is not None and sp > 1:
            # lag intra-giornaliero (~ stagione/4) ma almeno 2
            intra = max(2, sp // 4) if sp >= 4 else 2
            lags.append(intra)

            # lag giornaliero
            lags.append(sp)

        # rimuovi duplicati, ordina e assicurati che siano interi positivi
        lags = sorted({int(l) for l in lags if l >= 1})
        return lags


#def get_date_features(self):
        """Returns list of ordinal + cyclical date features."""
        return [
            self.hour_feature,
            self.minute_feature,
            self.dayofweek_feature,
            self.month_feature,
            lambda dates: np.sin(dates.hour / 24 * 2 * np.pi),
            lambda dates: np.cos(dates.hour / 24 * 2 * np.pi),
            lambda dates: np.sin(dates.minute / 60 * 2 * np.pi),
            lambda dates: np.cos(dates.minute / 60 * 2 * np.pi),
            lambda dates: np.sin(dates.dayofweek / 7 * 2 * np.pi),
            lambda dates: np.cos(dates.dayofweek / 7 * 2 * np.pi),
            lambda dates: np.sin(dates.month / 12 * 2 * np.pi),
            lambda dates: np.cos(dates.month / 12 * 2 * np.pi),
        ]
    #if self.model_name == 'LinearRegression' and len(self.test) > 0:
           # target_windows = max(1, len(self.test) // max(1, self.seasonal_period * window))
           # block_size = max(1, len(self.test) // target_windows)
       # else:
           # block_size = horizon if horizon > 0 else len(self.test)
