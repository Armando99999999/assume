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


class MLFORECAST_Predictor2():


    def __init__(self, run_mode, target_column, data_freq, 
                 seasonal_period, optimization, verbose=False):
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

        # date features
        self.hour_feature = lambda dates: dates.hour
        self.minute_feature = lambda dates: dates.minute
        self.dayofweek_feature = lambda dates: dates.dayofweek
        self.month_feature = lambda dates: dates.month


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

        self.train = self.train.assign(unique_id='time_series')
        self.train = self.train.rename(columns={'date': 'ds', self.target_column: 'y'})
        self.train = self.train[['unique_id', 'ds', 'y']].sort_values('ds').reset_index(drop=True)

        self.test = self.test.assign(unique_id='time_series')
        self.test = self.test.rename(columns={'date': 'ds', self.target_column: 'y'})
        self.test = self.test[['unique_id', 'ds', 'y']].sort_values('ds').reset_index(drop=True)


    def prepare_features_for_cv(self, input_data):

        fcst = MLForecast(
            models=[],  # Nessun modello, solo feature engineering
            freq=self.data_freq,
            lags = [1, self.seasonal_period, 2*self.seasonal_period, 3*self.seasonal_period],
            lag_transforms = {1: [RollingMean(window_size=self.seasonal_period),
                      RollingMean(window_size=2*self.seasonal_period)]}

        )
        features = fcst.preprocess(
            input_data.assign(unique_id='time_series'),
            id_col='unique_id', time_col='ds', target_col='y'
        )
        features = features.dropna()
        X = features.drop(['y', 'unique_id', 'ds'], axis=1)
        y = features['y']
        X = X.loc[:, X.nunique(dropna=True) > 1]
        return X, y

    
    def train_model(self, model_name, optimization, date_features):

        
        model_dict = {
            'LightGBM': lgb.LGBMRegressor(random_state=0,verbosity=-1),
            'XGBoost': xgb.XGBRegressor(random_state=0,verbosity=0,tree_method='auto',subsample=0.8, colsample_bytree=0.8, n_jobs=1),
            'RandomForest': RandomForestRegressor(random_state=0),
        }
         
        if model_name == 'LightGBM' and optimization:
                best_estimator = self.optimize_lgbm(self.train)
        elif model_name == 'XGBoost'and optimization:
                best_estimator = self.optimize_xgb(self.train)
        else :
                best_estimator = model_dict[model_name]
        

        self.model_name = model_name
        self.estimator = best_estimator

        if date_features:
            data_cyclical_characteristics = self.get_date_features(include_ordinal=False)
            self.model = MLForecast(
            models={self.model_name: self.estimator}, 
            freq=self.data_freq,
            lags=[1, self.seasonal_period, 2*self.seasonal_period],
            lag_transforms={1: [RollingMean(window_size=self.seasonal_period),
                            RollingMean(window_size=2*self.seasonal_period)]},
            data_cyclical_characteristics=data_cyclical_characteristics  # Pass cyclical features here
        )

        else:       
            # MLForecast take model as a list
            self.model = MLForecast(
            models={self.model_name: self.estimator}, 
            freq=self.data_freq,
            lags = [1, self.seasonal_period, 2*self.seasonal_period],
            lag_transforms = {1: [RollingMean(window_size=self.seasonal_period),
                        RollingMean(window_size=2*self.seasonal_period)]
                        }
        )
            
        
        self.model.fit(
            df=self.train,
            id_col='unique_id',
            time_col='ds',
            target_col='y'
            )

        return self.model
    

    def backtest(self, horizon):
        
        trained_model = self.model 

        total_predictions = pd.DataFrame()

        for step in range(0, len(self.test), horizon):

            forecasts = trained_model.predict(h = horizon)
            total_predictions = pd.concat([total_predictions, forecasts], ignore_index=True)
            new_observations = self.test.iloc[step:step + horizon]
            trained_model.update(new_observations)

            if new_observations.empty:
                break

        total_predictions = total_predictions.rename(columns={self.model_name: self.target_column})


        return total_predictions


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


    def optimize_lgbm(self, input_data):

        X, y = self.prepare_features_for_cv(input_data)
        param_grid = {
            'num_leaves': list(range(15, 100, 15)),
            'learning_rate': list(np.arange(0.01, 0.22, 0.05)),
            'n_estimators': list(range(70, 401, 50)),
        }
        model = lgb.LGBMRegressor(random_state=0,verbosity=-1,n_jobs=1)
        tscv = TimeSeriesSplit(n_splits=3)
        grid = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
        grid.fit(X, y)
        print("Best params:", grid.best_params_)
        return grid.best_estimator_
    

    def optimize_xgb(self, input_data):
        X, y = self.prepare_features_for_cv(input_data)
        param_grid = {
            'max_depth': list(range(2, 21, 4)),
            'learning_rate': list(np.arange(0.01, 0.18, 0.04)),
            'n_estimators': list(range(70, 401, 100)),
            'min_child_weight': list(range(2, 18, 5)),
        }
        X = X.loc[:, X.nunique(dropna=True) > 1]
        model = xgb.XGBRegressor(random_state=0,verbosity=0 ,tree_method='auto',n_jobs=1)
        tscv = TimeSeriesSplit(n_splits=2)
        grid = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
        grid.fit(X, y)
        print("Best params:", grid.best_params_)
        return grid.best_estimator_
        

    def get_date_features(self):
        """Returns list of all date features including cyclical encodings"""
        return [
            # Ordinal features
            self.hour_feature,
            self.minute_feature,
            self.dayofweek_feature, 
            self.month_feature,
            
            # Cyclical features (semplificato - senza sklearn)
            lambda dates: np.sin(dates.hour / 24 * 2 * np.pi),
            lambda dates: np.cos(dates.hour / 24 * 2 * np.pi),
            lambda dates: np.sin(dates.minute / 60 * 2 * np.pi),
            lambda dates: np.cos(dates.minute / 60 * 2 * np.pi),
            lambda dates: np.sin(dates.dayofweek / 7 * 2 * np.pi),
            lambda dates: np.cos(dates.dayofweek / 7 * 2 * np.pi),
            lambda dates: np.sin(dates.month / 12 * 2 * np.pi),
            lambda dates: np.cos(dates.month / 12 * 2 * np.pi)
        ]

            