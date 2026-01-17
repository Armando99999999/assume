import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.deterministic import Fourier


# pmdarima
import pmdarima
from pmdarima import ARIMA
from pmdarima import auto_arima

from tqdm import tqdm
import pickle
from Predictors.Predictor import Predictor



class SARIMA_Predictor(Predictor):
    """
    A class used to predict time series data using Seasonal ARIMA (SARIMA) models.
    """

    def __init__(self, run_mode, target_column=None, period = 24,
                 verbose=False):
        """
        Constructs all the necessary attributes for the SARIMA_Predictor object.

        :param run_mode: The mode in which the predictor runs
        :param target_column: The target column of the DataFrame to predict
        :param period: Seasonal period of the SARIMA model
        :param verbose: If True, prints detailed outputs during the execution of methods
        """

        super().__init__(verbose=verbose)  

        self.run_mode = run_mode
        self.verbose = verbose
        self.target_column = target_column
        self.period = period
        self.SARIMA_order = []
        self.model = None
        

    def train_model(self):
        """
        Trains a SARIMAX model using the training dataset and exogenous variables, if specified.

        :return: A tuple containing the trained model, validation metrics, and the index of the last training/validation timestep
        """
        try:    


            # Selection of the model with best AIC score
            """model = auto_arima(
                        y=self.train[self.target_column],
                        start_p=0,
                        start_q=0,
                        max_p=10,
                        max_q=10,
                        start_P=0,
                        start_Q=0,
                        max_P=10,
                        max_Q=4,
                        seasonal=True,
                        m = self.period,
                        test='adf',
                        d=None,  # Let auto_arima determine the optimal 'd'
                        D=None,  # Let auto_arima determine the optimal 'D'
                        trace=True,
                        error_action='warn',  # Show warnings for troubleshooting
                        suppress_warnings=False,
                        stepwise=True # If True, reduces computation time by avoiding full grid search
                        )"""

            # for debug
            order = (4, 1, 4)
            seasonal_order = (4, 1, 4, self.period)
            
            """order = model.order
            seasonal_order = model.seasonal_order"""

            best_order = (order, seasonal_order)
            print(f"Best order found: {best_order}")

            self.SARIMA_order = best_order
            regressor = SARIMAX(endog = self.train[self.target_column],
                                order = order,
                                seasonal_order = seasonal_order)

            print("\nTraining the SARIMAX model...")

            regressor = regressor.fit()
            # Save the model for later use
            self.model = regressor

            # Running the LJUNG-BOX test for residual correlation
            #residuals = model.resid()
            #perform ljung_box_test 
            # (... to be implemented)
            valid_metrics = None
            last_index = self.train.index[-1]
            print("Model successfully trained.")

            return regressor, valid_metrics, last_index
        
        except Exception as e:
                print(f"An error occurred during the model training: {e}")
                return None

    def test_model(self, model, forecast_type, output_len, ol_refit=False):
        """
        Tests a SARIMAX model by performing one-step ahead predictions and optionally refitting the model.

        :param model: The SARIMAX model to be tested
        :param last_index: Index of last training/validation timestep
        :param forecast_type: Type of forecasting ('ol-one' for open-loop one-step ahead, 'cl-multi' for closed-loop multi-step)
        :param ol_refit: Boolean indicating whether to refit the model after each forecast
        :return: A pandas Series of the predictions
        """
        try:
            print("\nTesting SARIMA model...\n")

            self.steps_ahead = self.test.shape[0]
            self.forecast_type = forecast_type



            if self.forecast_type == 'ol-one':
                predictions = []
                steps = 1
                for t in range(0, self.steps_ahead):
                    # Forecast one step at a time
                    y_hat = self.model.forecast()
                    # Append the forecast to the list
                    predictions.append(y_hat)
                    # Take the actual value from the test set to predict the next
                    y = self.test.iloc[t, self.test.columns.get_loc(self.target_column)]
                    # Update the model with the actual value
                    if ol_refit:
                        self.model = self.model.append([y], refit=True)
                    else:
                        self.model = self.model.append([y], refit=False)
                print("Model testing successful.")

            elif self.forecast_type == 'ol-multi':
                predictions = self.model.forecast(steps=len(self.test))
                predictions = predictions.reset_index(drop=True).to_frame(name=self.target_column)
                print("Model testing successful.")

            return predictions

        except Exception as e:
            print(f"An error occurred during the model test: {e}")
            return None
        

    def plot_predictions(self, predictions):
        """
        Plots the SARIMA model predictions against the test data.

        :param predictions: The predictions made by the SARIMA model
        """
        test = self.test[:self.steps_ahead][self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, predictions, 'k--', label='SARIMA')
        plt.title(f'SARIMA prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()


    def save_model(self, path):
        # Save model
        #save_forecaster(self.model, f"{path}/SARIMA.joblib", verbose=False)
        # Save training info
        with open(f"{path}/model_details_SARIMA.txt", "w") as file:
            file.write(f"Training Info:\n")
            file.write(f"Best Order: {self.SARIMA_order}\n")
            file.write(f"End Index: {len(self.train)}\n")
            file.write(f"Target_column: {self.target_column}\n")
    
    def save_metrics(self, path, metrics):
        file_mode = "a" if os.path.exists(f"{path}/model_details_SARIMA.txt") else "w"
        # Save test info
        with open(f"{path}/model_details_SARIMA.txt", file_mode) as file:
            file.write(f"Test Info:\n")
            file.write(f"Model Performance: {metrics}\n") 
            file.write(f"Launch Command Used:{sys.argv[1:]}\n")

    
