from statsforecast import StatsForecast
from statsforecast.models import SklearnModel
from statsforecast.utils import ConformalIntervals
from sklearn.linear_model import LinearRegression
from statsforecast.models import (
    WindowAverage,
    Naive,
    SeasonalNaive,
    RandomWalkWithDrift,
    HistoricAverage,
    AutoETS,
    ARIMA,
    AutoARIMA,

)
from utilsforecast.losses import rmse
from IPython.display import display
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pickle
import sys


class STATSFORECAST_Predictor():


    def __init__(self, run_mode, target_column, data_freq, seasonal_period, use_exog=False,
                 verbose=False):
        """
        Constructs all the necessary attributes for the STATSFORECAST object.

        :param run_mode: The mode in which the predictor runs
        :param target_column: The target column of the DataFrame to predict
        :param verbose: If True, prints detailed outputs during the execution of methods
        """

        self.run_mode = run_mode
        self.verbose = verbose
        self.target_column = target_column
        self.data_freq = data_freq
        self.seasonal_period = seasonal_period
        self.use_exog = use_exog



    def prepare_data(self, train=None, valid=None, test=None):
        """
        Prepares the data for the forecasting model.

        :param train: Training dataset
        :param valid: Validation dataset (optional)
        :param test: Testing dataset
        """
        self.valid = valid
        self.train = train
        self.test = test

        # trova automaticamente la colonna delle date
        date_col = self.train.select_dtypes(include=["datetime64[ns]"]).columns[0]

        self.train = self.train.assign(unique_id='time_series')
        self.train = self.train.rename(columns={date_col: 'ds', self.target_column: 'y'})
        self.test = self.test.assign(unique_id='time_series')
        self.test = self.test.rename(columns={date_col: 'ds', self.target_column: 'y'})

        if self.use_exog:
            # tengo tutte le colonne (inclusi esogeni/covariate)
            self.train = self.train.sort_values('ds').reset_index(drop=True)
            self.test = self.test.sort_values('ds').reset_index(drop=True)
        else:
            # uso solo serie principale
            self.train = self.train[['unique_id', 'ds', 'y']].sort_values('ds').reset_index(drop=True)
            self.test = self.test[['unique_id', 'ds', 'y']].sort_values('ds').reset_index(drop=True)
        

    def train_model(self, model_type, date_feature=False):
        
        model_dict = {
                      'Naive'       :   Naive(),
                      'SeasonalNaive' : SeasonalNaive(self.seasonal_period),
                      'win_avg' : WindowAverage(self.seasonal_period),
                      'random_walk_with_drift': RandomWalkWithDrift(),
                      'hist_avg': HistoricAverage(),
                      'AutoETS' : AutoETS(season_length = self.seasonal_period),
                      'ARIMA': ARIMA(order = (1,0,1)),
                      'AutoARIMA': AutoARIMA(),
                      'linear_regression': LinearRegression(fit_intercept=True),
                      'SARIMA': ARIMA(
                                        order=(1, 1, 1),           # (p, d, q)
                                        seasonal_order=(1, 1, 1),  # (P, D, Q)
                                        season_length=self.seasonal_period
                                    )
                      }
        
        self.model_type = model_type
    
        if self.model_type == 'linear_regression':
            if not self.use_exog:
                raise ValueError("linear_regression richiede colonne esogene: abilita il flag use_exog.")
            sklearn_model = SklearnModel(model_dict[model_type], alias=self.model_type)
            self.model = StatsForecast(models=[sklearn_model], freq=self.data_freq)
        else:
            self.model = StatsForecast(models=[model_dict[model_type]], freq=self.data_freq)

        self.model.fit(df = self.train, time_col = 'ds', target_col = 'y')

        return self.model


    def forecast(self, horizon, step_size):

        if self.model_type == 'SARIMA':
            model_name = 'ARIMA'  # in order to find the ARIMA column in predictions
            context_len = 2 * self.seasonal_period

        else:
            model_name = self.model_type
            context_len = horizon

        test_size = len(self.test)
        if test_size < horizon:
            raise ValueError("Test set length must be at least equal to the forecast horizon.")
        remainder = (test_size - horizon) % step_size
        if remainder != 0:
            adjusted_test_size = test_size - remainder
            if adjusted_test_size < horizon:
                raise ValueError("Unable to adjust test_size to satisfy StatsForecast constraints.")
            drop = test_size - adjusted_test_size
            # keep only the most recent rows to satisfy StatsForecast's requirement
            self.test = self.test.tail(adjusted_test_size).copy()
            test_size = adjusted_test_size
            print(f"Adjusted test_size to {test_size} to satisfy step_size/h requirements.")


        #test_df = pd.concat([self.train.iloc[-context_len:], self.test])
        test_df = pd.concat([self.train, self.test])
        test_df = test_df.assign(unique_id='time_series')

        """# Compute test_size from your test_df
        test_size = test_df['date'].count()

        # Adjust test_size to satisfy the constraint
        remainder = (test_size - horizon) % step_size
        if remainder != 0:
            test_size = test_size - remainder
            
        # Crop test_df to the adjusted test_size
        test_df = test_df.tail(test_size)"""

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

        candidate_cols = [col for col in cv_df.columns if col not in {'unique_id', 'cutoff', 'ds'}]
        if not candidate_cols:
            raise ValueError("StatsForecast cross_validation did not return model predictions.")
        pred_col = candidate_cols[0]

        predictions = cv_df[['ds', pred_col]].copy()
        predictions = predictions.set_index('ds')
        predictions.columns = ['y']
        predictions['y'] = predictions['y'].interpolate(method='linear').bfill().ffill()
        predictions = predictions.rename(columns={'y': self.target_column})

        expected_len = len(self.test)
        if len(predictions) != expected_len:
            aligned_len = min(len(predictions), expected_len)
            predictions = predictions.tail(aligned_len)
            self.test = self.test.tail(aligned_len).copy()


        return predictions

        
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
        test_series = test_data[self.target_column]
        pred_series = pd.Series(predictions)
        if len(pred_series) != len(test_series):
            align_len = min(len(pred_series), len(test_series))
            pred_series = pred_series.tail(align_len)
            test_series = test_series.tail(align_len)
        plt.plot(test_series.index, test_series, 'b-', label='Test Set')
        plt.plot(test_series.index, pred_series, 'r--', label='Naive')
        plt.title(f'baseline prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
    #    plt.savefig("./predictions_NAIVE.png", format='png', dpi=300)
        plt.show()

    def save_metrics(self, path, metrics):
        # Save test info
        with open(f"{path}/model_details_BASELINE.txt", "w") as file:
            file.write(f"Test Info:\n")
            file.write(f"Model Performance: {metrics}\n")
            file.write(f"Launch Command Used:{sys.argv[1:]}\n")
