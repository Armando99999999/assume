
import pandas as pd
try:
    from matplotlib import pyplot as plt
except ImportError:  # opzionale: il training non richiede Matplotlib
    plt = None
import numpy as np
import pickle
import sys
try:  # opzionale, non necessario per l'LSTM
    from mlforecast import MLForecast  # type: ignore
    from mlforecast.lag_transforms import RollingMean  # type: ignore
except ImportError:
    MLForecast = None  # pragma: no cover
    RollingMean = None  # pragma: no cover
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM

# utilsforecast è opzionale: se manca forniamo fallback minimi (rmse + evaluate)
try:
    from utilsforecast.losses import rmse  # type: ignore
    from utilsforecast.evaluation import evaluate  # type: ignore
except ImportError:
    def rmse(y_true, y_pred):
        arr = np.asarray(y_true) - np.asarray(y_pred)
        return float(np.sqrt(np.nanmean(arr ** 2)))

    def evaluate(df, metrics, id_col="unique_id", target_col="y"):
        cols = [c for c in df.columns if c not in {id_col, "ds", target_col, "cutoff"}]
        out = {}
        for col in cols:
            out[col] = [rmse(df[target_col], df[col])]
        return pd.DataFrame(out)


class NEURALFORECAST_Predictor():


    def __init__(self, run_mode, target_column, data_freq, seasonal_period,optimization, use_exog,
                 verbose=False):
        """
        Constructs all the necessary attributes for the NAIVE_Predictor object.

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
        self.use_exog= use_exog
        self.lstm_best_params_ = {}



    def prepare_data(self, train=None, valid=None, test=None):
        """
        Prepares train/valid/test for the forecasting model, con validazione opzionale.

        :param train: Training dataset
        :param valid: Validation dataset (optional)
        :param test: Testing dataset
        """
        self.train = train.copy()
        self.valid = valid.copy() if valid is not None else None
        self.test = test.copy() if test is not None else None

        # ---- TRAIN ----
        date_col_train = self.train.select_dtypes(include=["datetime64[ns]"]).columns[0]
        self.train = self.train.assign(unique_id='time_series')
        self.train = self.train.rename(columns={date_col_train: 'ds', self.target_column: 'y'})

        # ---- VALIDATION ----
        if self.valid is not None:
            date_col_valid = self.valid.select_dtypes(include=["datetime64[ns]"]).columns[0]
            self.valid = self.valid.assign(unique_id='time_series')
            self.valid = self.valid.rename(columns={date_col_valid: 'ds', self.target_column: 'y'})

        # ---- TEST ----
        if self.test is not None:
            date_col_test = self.test.select_dtypes(include=["datetime64[ns]"]).columns[0]
            self.test = self.test.assign(unique_id='time_series')
            self.test = self.test.rename(columns={date_col_test: 'ds', self.target_column: 'y'})
            self.original_test_ds = self.test['ds'].copy()

        # Allineamento temporale per ogni blocco
        self.train = self.date_check(self.train)
        if self.valid is not None:
            self.valid = self.date_check(self.valid)
        if self.test is not None:
            self.test = self.date_check(self.test)

        if self.use_exog:
            # tengo tutte le colonne (inclusi esogeni/covariate)
            self.train = self.train.sort_values('ds').reset_index(drop=True)
            if self.valid is not None:
                self.valid = self.valid.sort_values('ds').reset_index(drop=True)
            if self.test is not None:
                self.test = self.test.sort_values('ds').reset_index(drop=True)
            # salva la lista delle colonne esogene (tutto tranne id, ds, y)
            self.exog_cols = [c for c in self.train.columns if c not in ['unique_id', 'ds', 'y']]
            frames = [self.train]
            if self.valid is not None:
                frames.append(self.valid)
            if self.test is not None:
                frames.append(self.test)
            self.exog_reference = pd.concat(frames, ignore_index=True)
        else:
            # uso solo serie principale
            self.train = self.train[['unique_id', 'ds', 'y']].sort_values('ds').reset_index(drop=True)
            if self.valid is not None:
                self.valid = self.valid[['unique_id', 'ds', 'y']].sort_values('ds').reset_index(drop=True)
            if self.test is not None:
                self.test = self.test[['unique_id', 'ds', 'y']].sort_values('ds').reset_index(drop=True)
            self.exog_cols = []
            self.exog_reference = None

        
    def train_model(self, model_name, horizon, step_size, optimization):

        exog_list = getattr(self, "exog_cols", [])
        model_h = int(horizon or self.seasonal_period)

        if model_name == 'LSTM' and optimization:
            self.model_name = 'LSTM'
            self.estimator = None

            best_input_size = self.optimize_lstm(horizon=model_h)
            best_cfg = self.lstm_best_params_ or {}

            input_size = best_cfg.get('input_size', best_input_size or 2 * model_h)
            max_steps = best_cfg.get('max_steps', 400)
            enc_hidden = best_cfg.get('encoder_hidden_size', 128)
            dec_hidden = best_cfg.get('decoder_hidden_size', enc_hidden)
            enc_layers = best_cfg.get('encoder_n_layers', 2)
            dec_layers = best_cfg.get('decoder_layers', enc_layers)
            dropout = best_cfg.get('encoder_dropout', 0.0)
            learning_rate = best_cfg.get('learning_rate', 1e-3)
            batch = best_cfg.get('batch_size', 32)

            self.model = NeuralForecast(
                models=[
                    LSTM(
                        h=model_h,
                        input_size=input_size,
                        max_steps=max_steps,
                        encoder_hidden_size=enc_hidden,
                        decoder_hidden_size=dec_hidden,
                        encoder_n_layers=enc_layers,
                        decoder_layers=dec_layers,
                        encoder_dropout=dropout,
                        learning_rate=learning_rate,
                        batch_size=batch,
                        hist_exog_list=exog_list if exog_list else None,
                    )
                ],
                freq=self.data_freq,
            )
            if self.valid is not None and len(self.valid) > 0:
            # unisco train + valid, così NeuralForecast può usare gli ultimi punti come validation
                total_df = pd.concat([self.train, self.valid], ignore_index=True)
                self.model.fit(
                    df=total_df,
                    val_size=len(self.valid),   # ultimi N punti usati per early stopping
                )
            else:
                # nessun validation set: fit "normale"
                self.model.fit(df=self.train)

                return self.model

        # ramo non ottimizzato: LSTM semplice con impostazioni conservative
            # ramo non ottimizzato: LSTM semplice ma un po' più sensato per h=1
        else:
            self.model_name = 'LSTM'
            self.estimator = None

            base_len = max(24, self.seasonal_period // 4)  # ad es. 24 se sp=96
            input_size = base_len

            self.model = NeuralForecast(
                models=[
                    LSTM(
                        h=model_h,
                        input_size=input_size,
                        max_steps=300,
                        hist_exog_list=exog_list if exog_list else None,
                    )
                ],
                freq=self.data_freq,
            )

            if self.valid is not None and len(self.valid) > 0:
                total_df = pd.concat([self.train, self.valid], ignore_index=True)
                self.model.fit(df=total_df, val_size=len(self.valid))
            else:
                self.model.fit(df=self.train)

            return self.model

    
    def backtest(self, horizon):
        
        test_len = len(self.test)
        if test_len == 0:
            return pd.DataFrame(columns=['ds', self.target_column])

        # concatena train+test e assicura che gli esogeni siano puliti prima della CV
        if self.valid is not None:
            combined = pd.concat([self.train, self.valid, self.test], ignore_index=True)
            combined = self._prepare_exog_block(combined)
        else:
            combined = pd.concat([self.train, self.test], ignore_index=True)
            combined = self._prepare_exog_block(combined)

        step = 1  # usa passo unitario per evitare vincoli divisibili su (test_size - h)
        cv = self.model.cross_validation(
            df=combined,
            val_size=len(self.valid) if self.valid is not None else 0,
            test_size=test_len,
            step_size=step,
            n_windows=None,
            refit=False,
        )

        preds = cv[['unique_id', 'ds', self.model_name]]
        preds = preds.drop_duplicates(subset=['unique_id', 'ds'], keep='last')
        test_ds = self.test[['unique_id', 'ds']].copy()
        preds = test_ds.merge(preds, on=['unique_id', 'ds'], how='left')
        preds = preds.rename(columns={self.model_name: self.target_column})
        preds = preds.sort_values('ds')
        preds = preds.drop(columns=['unique_id']).set_index('ds')
        preds = preds.reindex(test_ds['ds']).tail(test_len)
        return preds
    
    
    def optimize_lstm(
        self,
        horizon: int,
        n_trials: int = 8,
        seed: int = 0,
    ):
        """
        Random search sugli iperparametri LSTM con rolling cross-validation.
        Tutto il flusso (campionamento -> windowing -> valutazione -> logging)  gestito qui
        per mantenere il codice lineare ma la ricerca completa.
        """

        rng = np.random.default_rng(seed)
        exog_list = getattr(self, "exog_cols", [])
        # per h=1 uso una finestra legata ma più corta della stagione
        # es: 6 ore (24 step) o max 1 giorno (96 step)
        sp = int(self.seasonal_period)
        if horizon <= 4:
            base_len = max(24, sp // 4)   # es: 96/4=24 step
        else:
            base_len = max(horizon, sp // 2)


        search_space = {
            "input_size": [base_len // 2, base_len, base_len * 2],
            "tail_factor": [4, 6],
            "max_steps": [200, 300, 400],
            "hidden": [64, 128, 256],
            "layers": [1, 2, 3],
            "dropout": [0.0, 0.1, 0.2, 0.3],
            "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
            "batch": [32, 64, 128],
        }

        best_score = float("inf")
        best_cfg = None

        for trial_idx in range(int(n_trials) + 1):
            # Primo passaggio: baseline deterministica che replica il comportamento precedente.
            if trial_idx == 0:
                raw_cfg = {
                    "input_size": base_len,
                    "tail_factor": 6,
                    "max_steps": 200,
                    "hidden": 128,
                    "layers": 2,
                    "dropout": 0.0,
                    "learning_rate": 1e-3,
                    "batch": 32,
                }
            else:
                # Campionamento casuale dagli spazi definiti.
                raw_cfg = {
                    "input_size": int(rng.choice(search_space["input_size"])),
                    "tail_factor": int(rng.choice(search_space["tail_factor"])),
                    "max_steps": int(rng.choice(search_space["max_steps"])),
                    "hidden": int(rng.choice(search_space["hidden"])),
                    "layers": int(rng.choice(search_space["layers"])),
                    "dropout": float(rng.choice(search_space["dropout"])),
                    "learning_rate": float(rng.choice(search_space["learning_rate"])),
                    "batch": int(rng.choice(search_space["batch"])),
                }

            # Seleziona la coda di train per la finestra di cross-validation.
           # coda base = frazione del train
            base_tail = len(self.train) // raw_cfg["tail_factor"]

            # almeno k stagioni (es. 7 giorni = 7 * seasonal_period)
            min_tail = 7 * int(self.seasonal_period)

            # almeno quello che serve per input_size + horizon
            need_tail = raw_cfg["input_size"] + horizon + 1

            tail_len = max(base_tail, min_tail, need_tail)
            tail_len = min(tail_len, len(self.train))  # non superare la dimensione del train

            train_tail = self.train.iloc[-tail_len:]

            # Richiede almeno qualche punto oltre all'orizzonte.
            avail = len(train_tail) - horizon
            if avail <= 2:
                continue

            trial_cfg = raw_cfg.copy()
            trial_cfg["input_size"] = min(trial_cfg["input_size"], max(2, avail - 1))

            # Fit + CV direttamente con il set di iperparametri corrente.
            nf = NeuralForecast(
                models=[
                    LSTM(
                        h=horizon,
                        input_size=trial_cfg["input_size"],
                        max_steps=trial_cfg["max_steps"],
                        encoder_hidden_size=trial_cfg["hidden"],
                        decoder_hidden_size=trial_cfg["hidden"],
                        encoder_n_layers=trial_cfg["layers"],
                        decoder_layers=trial_cfg["layers"],
                        encoder_dropout=trial_cfg["dropout"],
                        learning_rate=trial_cfg["learning_rate"],
                        batch_size=trial_cfg["batch"],
                        hist_exog_list=exog_list if exog_list else None,
                    )
                ],
                freq=self.data_freq,
            )
            if self.valid is not None:
                len_validation = len(self.valid)
                total_df = pd.concat([train_tail, self.valid], ignore_index=True)

                cv = nf.cross_validation(
                    df=total_df,
                    step_size=1,          # passo unitario per rispettare i vincoli di divisibilità
                    val_size=len_validation,
                    test_size=max(1, horizon),  # deve essere >= h
                    n_windows=None,       # richiesto da NeuralForecast quando si passa val_size
                )
            else:
                hz = max(1, horizon)
                max_windows = (len(train_tail) - (trial_cfg["input_size"] + hz)) // hz
                if max_windows < 1:
                    continue
                n_windows = min(3, max_windows)
                cv = nf.cross_validation(
                    df=train_tail,
                    n_windows=n_windows,
                    step_size=1,  # passo unitario per evitare constraint `test_size - h`
                )

            metrics = evaluate(
                cv.drop(columns="cutoff"),
                metrics=[rmse],
                id_col="unique_id",
                target_col="y",
            )
            lstm_cols = [c for c in metrics.columns if "LSTM" in c]
            if not lstm_cols:
                continue
            score = float(metrics[lstm_cols[0]].mean())

            if score < best_score:
                best_score = score
                best_cfg = {
                    "input_size": trial_cfg["input_size"],
                    "tail_factor": trial_cfg["tail_factor"],
                    "max_steps": trial_cfg["max_steps"],
                    "encoder_hidden_size": trial_cfg["hidden"],
                    "decoder_hidden_size": trial_cfg["hidden"],
                    "encoder_n_layers": trial_cfg["layers"],
                    "decoder_layers": trial_cfg["layers"],
                    "encoder_dropout": trial_cfg["dropout"],
                    "learning_rate": trial_cfg["learning_rate"],
                    "batch_size": trial_cfg["batch"],
                }

        self.lstm_best_params_ = best_cfg or {}
        return (best_cfg or {}).get("input_size", None)


    def unscale_predictions(self, predictions, folder_path):
        """
        Unscales the predictions using the scaler saved during model training.

        :param predictions: The scaled predictions that need to be unscaled
        :param folder_path: Path to the folder containing the scaler object
        """
        # Load scaler for unscaling data
        with open(f"{folder_path}/target_scaler.pkl", "rb") as file:
            target_scaler = pickle.load(file)

        # Unscale predictions
        predictions = np.array(predictions)
        predictions = predictions.reshape(-1, 1)
        predictions = target_scaler.inverse_transform(predictions)
        predictions = predictions.flatten()
        predictions = pd.Series(predictions)

        return predictions

    def plot_predictions(self, predictions, test_data):
        """Plotta test e previsioni allineando la lunghezza dei vettori."""
        if plt is None:
            return
        test_series = test_data[self.target_column]
        pred_series = pd.Series(predictions)

        align_len = min(len(test_series), len(pred_series))
        if align_len <= 0:
            raise ValueError("Nessun punto valido per il plot delle previsioni.")

        test_series = test_series.tail(align_len)
        pred_series = pred_series.tail(align_len)

        plt.plot(test_series.index, test_series, 'b-', label='Test Set')
        plt.plot(test_series.index, pred_series.values, 'r--', label='Forecast')
        plt.title(f'baseline prediction for feature: {self.target_column}')
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

    def _prepare_exog_block(self, df):
        """Interpolazione e riempimento degli esogeni per evitare NaN prima della predict."""
        if not self.use_exog or df is None or df.empty:
            return df

        base_cols = ['unique_id', 'ds', 'y']
        exog_cols = [c for c in df.columns if c not in base_cols]
        if not exog_cols:
            return df

        clean = df.copy()
        clean[exog_cols] = (
            clean[exog_cols]
            .interpolate(method='linear', limit_direction='both')
            .ffill()
            .bfill()
        )
        clean = self._fill_missing_exog(clean, exog_cols)
        return clean

    def date_check(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ricampiona la serie sulla frequenza dichiarata e riempie eventuali buchi temporali."""
        if df is None or df.empty:
            return df

        df = df.sort_values('ds')
        if df['ds'].duplicated().any():
            df = df.drop_duplicates(subset='ds', keep='last')

        full_idx = pd.date_range(start=df['ds'].iloc[0], end=df['ds'].iloc[-1], freq=self.data_freq)
        df = df.set_index('ds').reindex(full_idx)
        df.index.name = 'ds'

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols):
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')

        other_cols = [c for c in df.columns if c not in numeric_cols]
        if other_cols:
            df[other_cols] = df[other_cols].ffill().bfill()

        return df.reset_index().rename(columns={'index': 'ds'})

        
