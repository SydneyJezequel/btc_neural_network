import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt






class PredictionsGenerator:



    def __init__(self, model, test_data, scaler, time_step=30):
        """ Constructeur """
        self.model = model
        self.test_data = test_data
        self.time_step = time_step
        self.scaler = scaler



    def prepare_data_for_prediction(self):
        """ Préparation des données pour la prédiction """
        # x_input = self.test_data[-self.time_step:, :2].reshape(1, -1)  # Sélectionnez uniquement les 2 premières colonnes
        x_input = self.test_data[-self.time_step:].reshape(1, -1)  # Sélectionnez toutes les colonnes
        print("x_input shape : ", x_input.shape)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()
        return temp_input



    def generate_predictions(self, pred_days=30):
        """ Génération de prédictions """
        temp_input = self.prepare_data_for_prediction()
        lst_output = []
        n_steps = self.time_step
        print("time step : ", self.time_step)
        print("n_steps : ", n_steps)
        i = 0

        while i < pred_days:
            if len(temp_input) > self.time_step:
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                # Assurez-vous que la taille de x_input est correcte
                if x_input.shape[1] != n_steps:
                    x_input = x_input[:, :n_steps]
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = self.model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
                i += 1
            else:
                x_input = np.array(temp_input).reshape((1, n_steps, 1))
                yhat = self.model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i += 1

        # Inverser la transformation pour obtenir les valeurs réelles
        lst_output = np.array(lst_output).reshape(-1, 1)
        # Ajoutez une colonne fictive pour correspondre à la forme attendue par le scaler
        lst_output = np.hstack((lst_output, np.zeros((lst_output.shape[0], 7))))  # Ajustez ici pour correspondre à 8 colonnes
        lst_output = self.scaler.inverse_transform(lst_output)
        print("données renvoyées : ", lst_output[:, 0].reshape(-1, 1))
        return lst_output[:, 0].reshape(-1, 1)  # Retournez uniquement la première colonne



    def plot_predictions(self, predictions):
        """ Affichage des prédictions """
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(predictions)), predictions, label='Prédictions')
        plt.title('Prédictions pour les 30 prochains jours')
        plt.xlabel('Jour')
        plt.ylabel('Valeur')
        plt.legend()
        plt.show()
