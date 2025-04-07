import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt






class PredictionsGenerator:



    def __init__(self, model, test_data, scaler, time_step=15):
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



    def generate_predictions(self, pred_days=15):
        """ Génération de prédictions """
        temp_input = self.prepare_data_for_prediction()
        lst_output = []
        n_steps = self.time_step
        print("time step : ", self.time_step)
        print("n_steps : ", n_steps)
        print("temp_input : ", temp_input)
        i = 0

        #  Le code va effectuer des prédictions pour un certain nombre de jours défini par pred_days :
        while i < pred_days:
            # S'il y a suffisamment de données pour faire une prédiction basée sur les time_step derniers éléments :
            if len(temp_input) > self.time_step:
                # Convertit temp_input (sauf le premier élément) en un tableau NumPy :
                x_input = np.array(temp_input[1:])
                # Redimensionne x_input en un tableau 2D avec une seule ligne :
                x_input = x_input.reshape(1, -1)
                # Si la taille de x_input diffère des steps :
                if x_input.shape[1] != n_steps:
                    x_input = x_input[:, :n_steps]
                # Redimensionne x_input pour qu'il ait la forme (1, n_steps, 1) (forme attendue par les modèles de séries temporelles LSTM) :
                x_input = x_input.reshape((1, n_steps, 1))
                print("x_input : ", x_input)
                print("x_input : ", x_input.shape)
                # Prédiction basé sur x_input :
                yhat = self.model.predict(x_input, verbose=0)
                # Ajoute la prédiction à temp_input pour l'utiliser dans la prochaine itération :
                temp_input.extend(yhat[0].tolist())
                # Supprime le premier élément pour conserver une fenêtre glissante de taille constante :
                temp_input = temp_input[1:]
                # Ajoute la prédiction à lst_output, qui stocke toutes les prédiction :
                lst_output.extend(yhat.tolist())
                i += 1
            else:
                # prépare les données pour être utilisées comme entrée dans un modèle de séries temporelles (forme : (1, n_steps, 1))
                x_input = np.array(temp_input).reshape((1, n_steps, 1))
                # Génération de la prédiction :
                yhat = self.model.predict(x_input, verbose=0)
                # Convertit la première prédiction de 'yhat' en une liste Python et l'ajoute à 'temp_input' :
                temp_input.extend(yhat[0].tolist())
                # Convertit toutes les prédictions de 'yhat' en une liste Python et les ajoute à 'lst_output' :
                lst_output.extend(yhat.tolist())
                i += 1

        #  Convertit lst_output en un tableau NumPy et le redimensionne :
        lst_output = np.array(lst_output).reshape(-1, 1)
        # Ajustez le nombre de colonnes fictives en fonction du nombre de colonnes d'origine
        num_original_columns = self.scaler.n_features_in_  # Nombre de colonnes d'origine
        lst_output = np.hstack((lst_output, np.zeros((lst_output.shape[0], num_original_columns - 1))))
        lst_output = self.scaler.inverse_transform(lst_output)
        return lst_output[:, 0].reshape(-1, 1)


    def plot_predictions(self, predictions):
        """ Affichage des prédictions """
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(predictions)), predictions, label='Prédictions')
        plt.title('Prédictions pour les 15 prochains jours')
        plt.xlabel('Jour')
        plt.ylabel('Valeur')
        plt.legend()
        plt.show()
