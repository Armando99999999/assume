import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error

from Predictors.Predictor import Predictor
from torch.utils.data import DataLoader, TensorDataset

from hyperopt import fmin, tpe, hp, Trials

import sys
import random

import warnings
warnings.filterwarnings("ignore", module="matplotlib")


class LSTM_Predictor(Predictor):
    def __init__(self, run_mode, target_column=None,
                 verbose=False, input_len=None, output_len=None, validation=False):

        super().__init__(verbose=verbose)

        self.device = 'cpu'
        self.run_mode = run_mode
        self.verbose = verbose
        self.target_column = target_column
        self.input_len = input_len
        self.output_len = output_len
        self.validation = validation
        self.model = None

    def train_model(self):

        try:

            if self.validation:

                # LSTM WITH REPEATED HOLDOUT VALIDATION

                best_params, trials = self.bayesian_optimization(n_reps=10, max_evals=10)
                print("Migliori iperparametri trovati:", best_params)
                lstm = LSTM_Network(
                    output_len=self.output_len,
                    num_layers=best_params['num_layers'],
                    learning_rate=best_params['learning_rate'],
                    num_epochs=best_params['num_epochs'],
                    batch_size=best_params['batch_size'],
                    hidden_dim=best_params['hidden_dim']
                )
                lstm.to(self.device)

            else:

                # LSTM WITHOUT VALIDATION

                # Instantiate LSTM
                lstm = LSTM_Network(
                    output_len=self.output_len,
                    hidden_dim=64,
                    num_layers=2,
                    learning_rate=0.001,
                    num_epochs=100,
                    batch_size=128
                )
                lstm.to(self.device)



            # TRAINING LSTM

            # Create time windows for the training set
            X_train, y_train = self.data_windowing(self.train[self.target_column])

            # Convert to PyTorch tensors
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)

            # Torch DataLoader
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(dataset=train_dataset, batch_size=lstm.batch_size, shuffle=False)

            tbar = tqdm(range(lstm.num_epochs))

            # Optimizer
            optimizer = torch.optim.Adam(lstm.parameters(), lr=lstm.learning_rate)

            total_loss = 0
            num_batches = 0

            epoch_losses = []


            for epoch in tbar:

                total_loss = 0
                num_batches = 0

                for inputs, targets in train_loader:

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    optimizer.zero_grad()
                    predictions = lstm(inputs)
                    loss = lstm.loss_function(predictions, targets)
                    loss.backward()

                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                #self.plot_grad_flow(lstm.named_parameters())

                #self.plot_grad_flow_hist(lstm.named_parameters())

                # Calcola la loss media per l'epoca e aggiorna la barra di progresso
                average_loss = total_loss / num_batches
                epoch_losses.append(average_loss)  # Aggiungi la loss media alla lista
                tbar.set_description(f"Epoch {epoch + 1}/{lstm.num_epochs} Average Loss: {average_loss:.4f}")

                if self.verbose:
                    print(f"Epoch {epoch + 1}, Loss: {lstm.loss.item()}")

            plt.figure(figsize=(10, 5))
            plt.plot(epoch_losses, marker='o', linestyle='-', color='b')
            plt.title('Training Loss per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            plt.grid(True)
            plt.show()

            return lstm

        except Exception as e:
            print(f"An error occurred during the model training: {e}")
            return None

    def test_model(self, model):
        try:

            X_test, y_test = self.data_windowing(self.test[self.target_column])

            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)

            test_dataset = TensorDataset(X_test, y_test)
            test_loader = DataLoader(dataset=test_dataset, batch_size=model.batch_size, shuffle=False)

            model.eval()

            predictions = []
            with torch.no_grad():
                for inputs, _ in test_loader:
                    inputs = inputs.to(self.device)

                    batch_predictions = model(inputs)
                    predictions.append(batch_predictions.cpu())

            predictions = torch.cat(predictions, dim=0)

            predictions = predictions.numpy()
            return predictions, y_test.numpy()

        except Exception as e:
            print(f"An error occurred during the model test: {e}")
            return None

    def repeated_holdout_validation(self,
                                    hidden_dim=64,
                                    num_layers=2,
                                    learning_rate=0.001,
                                    num_epochs=400,
                                    batch_size=128,
                                    nreps=10):
        """
        Esegue un repeated holdout validation con nreps ripetizioni,
        calcolando l'RMSE medio su diverse subset di training/validation.
        Restituisce il valore medio di RMSE.
        """

        train_size = int(0.6 * len(self.train))
        val_size = int(0.1 * len(self.train))

        target_train = self.train[[self.target_column]]

        rmse_values = []

        for i in range(nreps):
            # Istanziamo la LSTM con gli iperparametri forniti
            lstm = LSTM_Network(output_len=self.output_len,
                                hidden_dim=hidden_dim,
                                num_layers=num_layers,
                                learning_rate=learning_rate,
                                num_epochs=num_epochs,
                                batch_size=batch_size)
            lstm.to(self.device)

            # Optimizer
            optimizer = torch.optim.Adam(lstm.parameters(), lr=lstm.learning_rate)

            # Controllo finestra
            t_min = train_size
            t_max = len(self.train) - val_size
            if t_min >= t_max:
                raise ValueError("Non c'è abbastanza spazio per allocare training e validation.")

            # Scelta casuale del punto di split
            t = random.randint(t_min, t_max)

            # Subset training
            train_subset = target_train.iloc[t - train_size:t]
            # Subset validation
            val_subset = target_train.iloc[t:t + val_size]

            # Finestra temporale su train
            X_train, y_train = self.data_windowing(train_subset)
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)

            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=lstm.batch_size,
                                      shuffle=False)

            # Finestra temporale su validation
            X_val, y_val = self.data_windowing(val_subset)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)

            # Training sulla subset
            for epoch in range(lstm.num_epochs):
                total_loss = 0
                num_batches = 0
                for inputs, targets in train_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    optimizer.zero_grad()
                    predictions = lstm(inputs)
                    loss = lstm.loss_function(predictions, targets)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

            # Validation
            lstm.eval()
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(dataset=val_dataset,
                                    batch_size=lstm.batch_size,
                                    shuffle=False)

            predictions_list = []
            with torch.no_grad():
                for inputs, _ in val_loader:
                    inputs = inputs.to(self.device)
                    batch_predictions = lstm(inputs)
                    predictions_list.append(batch_predictions.cpu())

            predictions = torch.cat(predictions_list, dim=0).numpy()
            targets = y_val.squeeze().numpy()

            mse = mean_squared_error(targets, predictions)
            rmse = np.sqrt(mse)
            rmse_values.append(rmse)

        # Restituisce la media dell'RMSE sulle nreps
        return float(np.mean(rmse_values))

    def bayesian_optimization(self, n_reps=3, max_evals=20):
        """
        Esegue ottimizzazione bayesiana sugli iperparametri LSTM,
        minimizzando l'RMSE ritornato da repeated_holdout_validation.
        n_reps: numero di volte che eseguiamo la holdout validation per ogni set di iperparametri
        max_evals: numero di iterazioni di Hyperopt
        """

        # Definiamo uno spazio di ricerca degli iperparametri
        search_space = {
            'num_layers': hp.choice('num_layers', [1, 2, 3]),
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
            'num_epochs': hp.choice('num_epochs', [100, 200, 400]),
            'batch_size': hp.choice('batch_size', [32, 64, 128]),
            'hidden_dim': hp.choice('hidden_dim', [32, 64, 128])
        }


        def objective(params):
            # params è un dizionario con i valori campionati da Hyperopt
            num_layers = int(params['num_layers'])
            learning_rate = float(params['learning_rate'])
            num_epochs = int(params['num_epochs'])
            batch_size = int(params['batch_size'])
            hidden_dim = int(params['hidden_dim'])

            # Calcoliamo la metrica (RMSE) su repeated_holdout_validation
            rmse = self.repeated_holdout_validation(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                batch_size=batch_size,
                nreps=n_reps
            )
            # La funzione obiettivo dev'essere MINIMIZZATA,
            # quindi restituiamo semplicemente il valore di RMSE
            return rmse

        # L'oggetto Trials ci consente di tenere traccia di tutte le prove
        trials = Trials()

        # Avviamo l'ottimizzazione
        best = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(0)  # seed per riproducibilità
        )

        # 'best' contiene la combinazione di iperparametri ottima trovata,
        # ma attenzione che hp.choice restituisce l'indice della lista,
        # quindi va "convertito" correttamente.
        # Se vogliamo recuperare i valori veri:
        num_layers_candidates = [1, 2, 3]
        num_epochs_candidates = [100, 200, 400]
        batch_size_candidates = [32, 64, 128]
        hidden_dim_candidates = [32, 64, 128]

        best_params = {
            'num_layers': num_layers_candidates[best['num_layers']],
            'learning_rate': float(best['learning_rate']),
            'num_epochs': num_epochs_candidates[best['num_epochs']],
            'batch_size': batch_size_candidates[best['batch_size']],
            'hidden_dim': hidden_dim_candidates[best['hidden_dim']]
        }

        return best_params, trials


    def data_windowing(self, series):

        stride = 1 if self.output_len == 1 else self.output_len
        X, y = [], []
        indices = []

        if len(series) < self.input_len + self.output_len:
            print("Data is too short for creating windows")
            return None
        else:

            for i in range(0, len(series) - self.input_len - self.output_len + 1, stride):
                X.append(series.iloc[i:i + self.input_len].values)
                y.append(series.iloc[i + self.input_len:i + self.input_len + self.output_len].values)
                indices.append(i)

        # Conversione in array e ridimensionamento
        X, y = np.array(X), np.array(y)

        # Reshape dei dati di input per includere tutte le feature nel modello
        X = np.reshape(X, (X.shape[0], self.input_len, 1))

        return X, y

    def plot_grad_flow(self, named_parameters):
        ave_grads = []
        layers = []
        for n, p in named_parameters:
            if p.requires_grad and "bias" not in n:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().item())
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.gca().set_yscale('log')  # Impostazione della scala logaritmica

    def plot_grad_flow_hist(self, named_parameters):
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if p.requires_grad and "bias" not in n:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().item())
                max_grads.append(p.grad.abs().max().item())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.gca().set_yscale('log')  # Impostazione della scala logaritmica

    def plot_predictions(self, predictions, y_test):
        """
        Plots the LSTM model predictions against the test data.

        :param predictions: The predictions made by the LSTM model
        """

        """# Plot the first days of the test series
        n_days = 3
        day_timesteps = 24
        end_index = n_days * day_timesteps
        plt.plot(range(end_index), y_test[:end_index].flatten(), 'b-', label='Test Set')
        plt.plot(range(end_index), predictions[:end_index], 'k--', label='LSTM')
        plt.title(f'Previsione LSTM per i primi {n_days} giorni', fontsize=18)
        plt.xlabel('Indice della serie temporale', fontsize=14)
        plt.ylabel('Valore', fontsize=14)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()"""

        # Plot the whole time series (funziona solo per previsioni one step, il plot per le multi step
        #                                                                               va implementato)
        plt.plot(range(len(y_test)), y_test.flatten(), 'b-', label='Test Set')
        plt.plot(range(len(y_test)), predictions, 'k--', label='LSTM')
        plt.title(f'LSTM prediction', fontsize=18)
        plt.xlabel('Time series index', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def save_metrics(self, path, metrics):
        # Save test info
        with open(f"{path}/model_details_LSTM.txt", "w") as file:
            file.write(f"Test Info:\n")
            file.write(f"Model Performance: {metrics}\n")
            file.write(f"Launch Command Used:{sys.argv[1:]}\n")



class LSTM_Network(nn.Module):

    def __init__(self,
                 output_len,
                 input_dim=1,
                 hidden_dim=64,
                 num_layers=2,
                 learning_rate=0.001,
                 num_epochs=400,
                 batch_size=128
                 ):

        super(LSTM_Network, self).__init__()

        # Parametri di training (li spostiamo qui per comodità)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Parametri di rete
        self.input_dim = input_dim
        self.output_dim = 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.output_len = output_len

        # Loss Function
        self.loss_function = nn.MSELoss()

        # LSTM e Layer Finale
        self._lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bias=True
        )

        self._fc = nn.Linear(self.hidden_dim, self.output_dim * self.output_len)

    def forward(self, inputs: torch.Tensor):
        predictions, _ = self._lstm(inputs)
        predictions = self._fc(predictions[:, -1, :])
        return predictions


class MLP_Network(nn.Module):
    def __init__(self, output_len, input_len):
        """MLP network for test purposes"""
        super(MLP_Network, self).__init__()

        self.num_epochs = 400
        self.batch_size = 128
        self.learning_rate = 0.001

        self.output_len = output_len
        self.input_len = input_len
        self.loss_function = nn.MSELoss()

        # Aggiornamento della dimensione di input
        self.input_dim = input_len * 1  # Moltiplicato per il numero di feature per timestep

        # Definizione dell'architettura MLP
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.input_dim, 64),  # primo strato nascosto
            nn.ReLU(),
            nn.Linear(64, 64),  # secondo strato nascosto
            nn.ReLU(),
            nn.Linear(64, self.output_len)  # strato di output
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # appiattimento dell'input
        output = self.hidden_layers(x)
        return output










