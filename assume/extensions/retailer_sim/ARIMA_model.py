import pandas as pd
import matplotlib.pyplot as plt
import sys, os
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from Predictors.Predictor import Predictor


class ARIMA_Predictor(Predictor):
    """
    A class used to predict time series data using the ARIMA model.
    """

    def __init__(self, run_mode, target_column=None, data_freq = 'H',
                 verbose=False):
        """
        Initializes an ARIMA_Predictor object with specified settings.

        :param run_mode: The mode in which the predictor runs
        :param target_column: The target column of the DataFrame to predict
        :param verbose: If True, prints detailed outputs during the execution of methods
        """
        super().__init__(verbose=verbose)  

        self.run_mode = run_mode
        self.verbose = verbose
        self.target_column = target_column
        self.ARIMA_order = []
        self.model = None
        self.data_freq = data_freq

        


    def train_model(self):
        """
        Trains an ARIMA model using the training dataset.

        :return: A tuple containing the trained model, validation metrics, and the index of the last training/validation timestep
        """
        try:


            self.train.index = pd.date_range(start=self.train.index[0],
                                 periods=len(self.train),
                                 freq=self.data_freq)

            # Selection of the model with best AIC score
            """model = auto_arima(
                        y=self.train[self.target_column],
                        start_p=0,
                        start_q=0,
                        max_p=10,
                        max_q=10,
                        seasonal=False,
                        test='adf',
                        d=None,  # Let auto_arima determine the optimal 'd'
                        trace=True,
                        error_action='warn',  # Show warnings for troubleshooting
                        suppress_warnings=False,
                        stepwise=True # If True, reduces computation time by avoiding full grid search
                        )"""
            
            # for debug
            order = (4, 1, 4)

            #order = model.order

            print(f"Best order found: {order}")
            self.ARIMA_order = order

            regressor = SARIMAX(endog = self.train[self.target_column], order=self.ARIMA_order)
             
            # Training the model with the best parameters found
            print("\nTraining the ARIMA model...")
            regressor = regressor.fit()

            # Save the model for later use
            self.model = regressor

            """# Running the LJUNG-BOX test for residual correlation
            residuals = model.resid()
            ljung_box_test(residuals)"""

            print("Model successfully trained.")
            valid_metrics = None
            last_index = self.train.index[-1]

            return regressor, valid_metrics, last_index
 
        except Exception as e:
            print(f"An error occurred during the model training: {e}")
            return None
        
        
    def test_model(self, model, forecast_type, output_len, ol_refit = False):
        """
        Tests an ARIMA model by performing one-step ahead predictions and optionally refitting the model.

        :param model: The ARIMA model to be tested
        :param last_index: Index of last training/validation timestep
        :param forecast_type: Type of forecasting ('ol-one' for open-loop one-step ahead, 'cl-multi' for closed-loop multi-step)
        :param ol_refit: Boolean indicating whether to refit the model after each forecast
        :return: A pandas Series of the predictions
        """
        try:
            print("\nTesting ARIMA model...\n")

            self.test.index = pd.date_range(start=self.train.index[-1] + pd.tseries.frequencies.to_offset(self.data_freq),
                                periods=len(self.test),
                                freq=self.data_freq)

            self.len_test = self.test.shape[0]
            self.forecast_type = forecast_type

            if self.forecast_type == 'ol-one':
                predictions = []
                for t in range(0, self.len_test):
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

                predictions = pd.DataFrame( [y_hat.iloc[0] for y_hat in predictions],
                                            columns=[self.target_column]
                                          )
                print("Model testing successful.")

            elif self.forecast_type == 'ol-multi':

                horizon = output_len
                predictions = []

                # Prima del test, allinea l'indice del test set

                last_training_index = self.model.model.data.row_labels[-1]
                
                for start_idx in range(0, self.len_test, horizon):
                    # Definisce la finestra di previsione
                    end_idx = min(start_idx + horizon, self.len_test)

                    # Effettua la previsione multi-step
                    y_hat = self.model.forecast(steps=end_idx - start_idx)
                    predictions.extend(y_hat.tolist())

                    # Se abilitato, rifitta il modello ogni blocco di horizon timesteps
                    if ol_refit:
                        y_actual = self.test.iloc[start_idx:end_idx, self.test.columns.get_loc(self.target_column)]
                        self.model = self.model.append(y_actual, refit=True)
                    else:
                        y_actual = self.test.iloc[start_idx:end_idx, self.test.columns.get_loc(self.target_column)]
                        self.model = self.model.append(y_actual, refit=False)

                # Conversione finale in DataFrame
                predictions = pd.DataFrame(predictions, columns=[self.target_column]).reset_index(drop=True)
                
                print("Model testing successful.")

            return predictions
            
        except Exception as e:
            print(f"An error occurred during the model test: {e}")
            return None
        
    def plot_predictions(self, predictions):
        """
        Plots the ARIMA model predictions against the test data.

        :param predictions: The predictions made by the ARIMA model
        """
        test = self.test[:self.len_test][self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, predictions, 'k--', label='ARIMA')
        plt.title(f'ARIMA prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        # Save model
        #save_forecaster(self.model, f"{path}/ARIMA.joblib", verbose=False)
        # Save training info
        with open(f"{path}/model_details_ARIMA.txt", "w") as file:
            file.write(f"Training Info:\n")
            file.write(f"Best Order: {self.ARIMA_order}\n")
            file.write(f"End Index: {len(self.train)}\n")
            file.write(f"Target_column: {self.target_column}\n")
    
    def save_metrics(self, path, metrics):
        file_mode = "a" if os.path.exists(f"{path}/model_details_ARIMA.txt") else "w"
        # Save test info
        with open(f"{path}/model_details_ARIMA.txt", file_mode) as file:
            file.write(f"Test Info:\n")
            file.write(f"Model Performance: {metrics}\n") 
            file.write(f"Launch Command Used:{sys.argv[1:]}\n")

    
