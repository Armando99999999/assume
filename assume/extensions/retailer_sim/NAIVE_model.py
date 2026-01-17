import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pickle
import sys


class NAIVE_Predictor():
    """
    A class used to predict time series data using simple naive methods.
    """

    def __init__(self, run_mode, target_column, data_freq,
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



    def prepare_data(self, train=None, valid=None, test=None):
        """
        Prepares the data for the naive forecasting model.

        :param train: Training dataset
        :param valid: Validation dataset (optional)
        :param test: Testing dataset
        """
        self.valid = valid
        self.test = test

    def forecast(self, input_data, horizon, forecast_type,
             use_first_as_start=False, initial_value=None, test_data=None):
        """
        Previsione naive per diverse modalità:
        - multi_step_mean: media dei valori di input_data
        - multi_step_persistence: ultimo valore di input_data ripetuto
        - one_step_recursive: usa i valori reali (train o test) passo-passo
        """
        try:
            target_col = self.target_column

            delta_t = input_data.index[1] - input_data.index[0]

            inferred_freq = pd.infer_freq(input_data.index)

            forecast_index_multi_step = pd.date_range(
                start=input_data.index[-1] + delta_t,
                periods=horizon,
                freq=inferred_freq)

            if forecast_type == "multi_step_mean":
                # --- usa solo il train ---
                mean_value = input_data[target_col].mean()
                predictions = pd.Series(
                    [mean_value] * horizon,
                    index=forecast_index_multi_step
                )
                return predictions

            elif forecast_type == "multi_step_persistence":
                # --- usa solo l'ultimo valore del train ---
                last_value = input_data.iloc[-1][target_col]
                predictions = pd.Series(
                    [last_value] * horizon,
                    index=forecast_index_multi_step
                )
                return predictions

            # === One-step recursive forecast ===
            elif forecast_type == "one_step_recursive":
                if use_first_as_start:
                    # autoregressivo interno: usa i valori reali del train
                    vals = input_data[target_col].values
                    seq = [vals[0]] + [vals[t - 1] for t in range(1, len(vals))]
                    predictions = pd.Series(seq, index=input_data.index[:len(seq)])
                else:
                    initial_value = input_data[self.target_column].iloc[-1]
                    if test_data is None:
                        raise ValueError("Serve 'test_data' per usare i valori reali del test.")
                    test_vals = test_data[target_col].values
                    preds = [initial_value]
                    for t in range(1, horizon):
                        preds.append(test_vals[t - 1])  # usa y_test reale precedente
                    predictions = pd.Series(preds, index=test_data.index[:horizon])

            else:
                raise ValueError(f"forecast_type '{forecast_type}' non riconosciuto.")

            return predictions

        except Exception as e:
            print(f"Errore nella funzione forecast(): {e}")
            raise

        

    def seasonal_forecast(self, input_data, horizon=None, period=24, from_start=False, mode="multi_step"):


        target_series = input_data[self.target_column]
        n = len(target_series)

        if from_start:
            # Restituisce la serie spostata di un periodo
            shifted = target_series.shift(period).iloc[period:]
            shifted.index = input_data.index[period:]
            return shifted

        delta_t = input_data.index[1] - input_data.index[0]

        inferred_freq = pd.infer_freq(input_data.index)

        forecast_index = pd.date_range(
                start=input_data.index[-1] + delta_t,
                periods=horizon,
                freq=inferred_freq)
                                

        if mode == "multi_step":
            # Copia l'ultimo ciclo osservato (forecast statico)
            last_cycle = target_series.iloc[-period:]
            if horizon <= period:
                forecast_values = last_cycle.iloc[:horizon].values
            else:
                reps = -(-horizon // period)  # ceil division
                forecast_values = (last_cycle.tolist() * reps)[:horizon]
            return pd.Series(forecast_values, index=forecast_index)


        
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

    def plot_predictions(self, naive_predictions, test_data):
        """
        Plots naive predictions against the test data.

        :param naive_predictions: The naive predictions to plot.
        """

        horizon = test_data.shape[0]
        test = test_data[:horizon][self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, naive_predictions, 'r--', label='Naive')
        plt.title(f'Naive prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig("./predictions_NAIVE.png", format='png', dpi=300)
        plt.show()

    def save_metrics(self, path, metrics):
        # Save test info
        with open(f"{path}/model_details_NAIVE.txt", "w") as file:
            file.write(f"Test Info:\n")
            file.write(f"Model Performance: {metrics}\n")
            file.write(f"Launch Command Used:{sys.argv[1:]}\n")