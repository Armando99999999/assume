
from utilsforecast.losses import rmse 
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pickle
import sys
from mlforecast import MLForecast
from utilsforecast.evaluation import evaluate
from mlforecast.lag_transforms import RollingMean
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM


class NEURALFORECAST_Predictor2():


    def __init__(self, run_mode, target_column, data_freq, seasonal_period,optimization, 
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



    def prepare_data(self, train=None, valid=None, test=None):
        """
        Prepares the data for the baseline forecasting model.

        :param train: Training dataset
        :param valid: Validation dataset (optional)
        :param test: Testing dataset
        """
        self.valid = valid
        self.train = train
        self.test = test

        self.train = self.train.assign(unique_id='time_series')
        self.train = self.train.rename(columns={'date': 'ds', self.target_column: 'y'})
        self.train = self.train[['unique_id', 'ds', 'y']].sort_values('ds').reset_index(drop=True)

        self.test = self.test.assign(unique_id='time_series')
        self.test = self.test.rename(columns={'date': 'ds', self.target_column: 'y'})
        self.test = self.test[['unique_id', 'ds', 'y']].sort_values('ds').reset_index(drop=True)

        
    def train_model(self, model_name, optimization):


        if model_name == 'LSTM' and optimization == True:  # ADD — ramo LSTM
            self.model_name = 'LSTM'
            self.estimator = None
            best_input_size = self.optimize_lstm(horizon=self.seasonal_period)
            h = int(self.seasonal_period)

            best_input_size = best_input_size or 2*h

            # 2) fit finale su TUTTO il train con i migliori iperparametri trovati
           
            self.model = NeuralForecast(
                models=[LSTM(
                    h=self.seasonal_period,
                    input_size=best_input_size,
                    max_steps=self.lstm_best_params_.get('max_steps', 400),
                )],
                freq=self.data_freq
            )
            self.model.fit(df=self.train)
        
        else :
                self.model_name = 'LSTM'
                h = int(self.seasonal_period)
                self.estimator = None
                # es. al più 2*stagionalità ma non oltre il possibile
                self.model = NeuralForecast(
                    models=[LSTM(h=h, input_size=2*h, 
                                 max_steps=100)
                ],
                    freq=self.data_freq
                )
                self.model.fit(df=self.train)

        return self.model
    
    def backtest(self, horizon):
        
        trained_model = self.model 

        total_predictions = pd.DataFrame()

        new_train = self.train

        for step in range(0, len(self.test), horizon):

            forecasts = trained_model.predict(df = new_train, h = horizon)
            total_predictions = pd.concat([total_predictions, forecasts], ignore_index=True)
            new_observations = self.test.iloc[step:step + horizon]
            new_train = pd.concat([new_train, new_observations])

            if new_observations.empty:
                break

        total_predictions = total_predictions.rename(columns={self.model_name: self.target_column})


        return total_predictions
    
    
    def optimize_lstm(
        self,
        horizon: int,
        n_trials: int = 5,
        input_mult_range = [1,2],
        tail_factor_range = [4, 6],
        max_steps_range = [100, 150],
        seed: int = 0,
    ):
        """
        SUMMARY
        -------
        Randomized search of LSTM hyperparameters on the TRAIN set using rolling
        cross-validation. We sample (input_size, tail_factor, max_steps), build a
        recent tail for CV (to avoid leakage and reduce cost), evaluate with RMSE
        averaged over multiple non-overlapping forecast windows (step_size = h),
        keep the best configuration in self.lstm_best_params_, and return its input_size.
        """

        ### 1) Setup random generator and hyperparameter ranges
        rng = np.random.default_rng(seed)
        #We keep the setup explicit. The rng (with seed) ensures reproducibility;
        # the three ranges bound the search space so we don't explore absurd values

        ### 2) Derive a base input length for scaling
        base_len = max(int(horizon), int(self.seasonal_period))
        best_score = float("inf")
        best_cfg = None

        # input_size should at least cover the forecast horizon or a full seasonal cycle.
        # Using the max(h, seasonality) keeps the LSTM context informative without exploding
        # the window length on weakly seasonal series.

        ### 3) Random search loop over n_trials
        for _ in range(int(n_trials)):

            # 3.1) Sample a candidate configuration
            k = int(rng.choice(input_mult_range))
            input_size = int(k * base_len)                   # LSTM lookback window
            tail_factor = int(rng.choice(tail_factor_range))
            max_steps = int(rng.choice(max_steps_range))

            # 3.2) Build a recent training tail for CV
            tail_len = max(int(tail_factor * horizon), int(2 * self.seasonal_period), int(input_size + horizon + 1))
            train_tail = self.train.iloc[-tail_len:]

        # 3.3) Convert to NeuralForecast format

        # NeuralForecast expects columns ['unique_id','ds','y'] sorted by time and
        # without missing/duplicate timestamps in the evaluation window.

        # Clamp input_size to available history and ensure we can build at least one window
            avail = len(train_tail) - horizon
            if avail <= 2:
                # series too short for a training window, skip this trial
                continue
            # input_size cannot exceed available history
            input_size = min(input_size, max(2, avail - 1))

            # 3.4) Decide the number of rolling CV windows (non-overlapping tests)
            # We need enough history for training AFTER removing the test tail used by CV.
            # cross_validation(no_refit) sets test_size = h + step_size*(n_windows-1) = h * n_windows (since step_size=h)
            # Ensure that (len(Y_df) - test_size) >= (input_size + h) =>
            # n_windows <= (len(Y_df) - (input_size + h)) // h
            max_windows = (len(train_tail) - max(1, input_size)) // max(1, horizon)
            max_train_windows = (len(train_tail) - (input_size + horizon)) // max(1, horizon)
            if max_train_windows < 1:
                # not enough history to leave a train window after reserving test tail
                continue
            n_windows = min(3, max(1, min(max_windows, max_train_windows)))

                #Using step_size = h (see below) yields non-overlapping test blocks of length h.

                # 3.5) Build the LSTM candidate
            nf = NeuralForecast(
                    models=[LSTM(h=horizon, input_size=input_size, max_steps=max_steps)],
                    freq=self.data_freq
                )

            # WHY: We evaluate exactly the horizon we will use at inference; input_size
            # is the candidate lookback; max_steps is our compute budget for this trial.

            # 3.6) Rolling-origin cross-validation
            cv = nf.cross_validation(
                df=train_tail,
                n_windows=n_windows,
                step_size=horizon,
            )

            # Rolling origin enforces temporal causality. With step_size = h, each test
            # window is a fresh block of size h, yielding independent-like errors and a cleaner
            # estimate of future performance.

            # 3.7) Compute RMSE over all windows
            metrics = evaluate(
                cv.drop(columns='cutoff'),
                metrics=[rmse],
                id_col='unique_id',
                target_col='y'
            )
            lstm_cols = [c for c in metrics.columns if 'LSTM' in c]
            if not lstm_cols:   # CHG: safety check to avoid index error
                continue
            score = float(metrics[lstm_cols[0]].mean())

            # 3.8) Keep the best configuration so far
            if score < best_score:
                best_score = score
                best_cfg = {
                    "input_size": input_size,
                    "tail_factor": tail_factor,
                    "max_steps": max_steps,
                    "n_windows": n_windows,
                }

            # we store all fields needed to reproduce the final training configuration.

        ### 4) Persist best params and return input_size
        self.lstm_best_params_ = best_cfg or {}
        return (best_cfg or {}).get("input_size", None)
        # configuration remains available in self.lstm_best_params_ for the final fit.


    def unscale_predictions(self, predictions, folder_path):
        """
        Unscales the predictions using the scaler saved during model training.

        :param predictions: The scaled predictions that need to be unscaled
        :param folder_path: Path to the folder containing the scaler object
        """
        # Load scaler for unscaling data
        with open(f"{folder_path}/scaler.pkl", "rb") as file:
            scaler = pickle.load(file)

        # Unscale predictions
        predictions = np.array(predictions)
        predictions = predictions.reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        predictions = predictions.flatten()
        predictions = pd.Series(predictions)

        return predictions

    def plot_predictions(self, predictions, test_data):
      
        horizon = test_data.shape[0]
        test = test_data[:horizon][self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, predictions, 'r--', label='Naive')
        plt.title(f'baseline prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
    #    plt.savefig("./predictions_NAIVE.png", format='png', dpi=300)
        plt.show()

    def save_metrics(self, path, metrics):
        # Save test info
        with open(f"{path}/model_details_MLFORECAST.txt", "w") as file:
            file.write(f"Test Info:\n")
            file.write(f"Model Performance: {metrics}\n")
            file.write(f"Launch Command Used:{sys.argv[1:]}\n")
        return self.model

        
