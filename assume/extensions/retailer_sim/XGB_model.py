from datetime import datetime
import pandas as pd
import numpy as np
import datetime as datetime
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  

from xgboost import XGBRegressor,plot_importance


import shap
import sklearn
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_selection import RFECV

import xgboost as xgb
from xgboost import plot_importance, plot_tree
from Predictors.Predictor import Predictor

import random
import copy
import sys

class XGB_Predictor(Predictor):
    """
    A class used to predict time series data using XGBoost, a gradient boosting framework.
    """

    def __init__(self, run_mode, target_column=None, 
                 verbose=False,  seasonal_model=False, input_len = None, output_len= 24, forecast_type= None, period=24):

        super().__init__(verbose=verbose)  

        self.run_mode = run_mode
        self.verbose = verbose
        self.target_column = target_column
        self.seasonal_model = seasonal_model
        self.input_len = input_len
        self.output_len = output_len
        self.forecast_type = forecast_type
        self.period = period
        self.model = None

    def train_model(self):

        for df in (self.train, self.test):
            # Existing time features
            df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

            df['week_of_year_sin'] = np.sin(2 * np.pi * df.index.isocalendar().week / 52)
            df['week_of_year_cos'] = np.cos(2 * np.pi * df.index.isocalendar().week / 52)
            df['week_day_sin'] = np.sin(2 * np.pi * df.index.weekday / 7)
            df['week_day_cos'] = np.cos(2 * np.pi * df.index.weekday / 7)

            df['hour_day_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            df['hour_day_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

            df['day_sin'] = np.sin(2 * np.pi * df.index.day / df.index.days_in_month)
            df['day_cos'] = np.cos(2 * np.pi * df.index.day / df.index.days_in_month)

        # Aggiornamento dell'elenco delle caratteristiche esogene
        self.exog_features = [
            #'month_sin', 
            #'month_cos',
            #'week_of_year_sin',
            #'week_of_year_cos',
            'week_day_sin',
            'week_day_cos',
            'hour_day_sin',
            'hour_day_cos',
            'day_sin',
            'day_cos',
        ]

        X_train = self.train[self.exog_features]
        y_train = self.train[self.target_column]

        model = XGBRegressor(
            n_estimators=1000,  # Number of boosting rounds (you can tune this)
            learning_rate=0.01,   # Learning rate (you can tune this)
            max_depth=5,          # Maximum depth of the trees (you can tune this)
            min_child_weight=1,   # Minimum sum of instance weight needed in a child
            gamma=0,              # Minimum loss reduction required to make a further partition
            subsample=0.8,        # Fraction of samples used for training
            colsample_bytree=0.8, # Fraction of features used for training
            reg_alpha=0,          # L1 regularization term on weights
            reg_lambda=1,         # L2 regularization term on weights
            objective='reg:squarederror',  # Objective function for regression
            random_state=42,      # Seed for reproducibility
            eval_metric=['rmse', 'mae'],

            # use this two lines to enable GPU
            # tree_method  = 'hist',
            # device       = 'cuda',
                            )

        model.fit(X_train, y_train)
        return model
    

    def test_model(self, model):

        X_test = self.test[self.exog_features]
        y_pred = model.predict(X_test)

        predictions = pd.DataFrame(
            data=y_pred,
            index=X_test.index,
            columns=[self.target_column]
        )

        return predictions

    def plot_predictions(self, predictions):
        """
        Plots the XGB model predictions against the test data.

        :param predictions: The predictions made by the LSTM model
        """
        test = self.test[self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, predictions, 'k--', label='XGB')
        plt.title(f'XGB prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()


    def save_model(self, path):
        # Save model
        print()
    
    def save_metrics(self, path, metrics):
        # Save test info
        with open(f"{path}/model_details_XGB.txt", "w") as file:
            file.write(f"Test Info:\n")
            file.write(f"Model Performance: {metrics}\n") 
            file.write(f"Launch Command Used:{sys.argv[1:]}\n")





