import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score, mean_gamma_deviance, mean_poisson_deviance
import parameters






""" ************************* Méthodes ************************* """

def get_fitted_scaler(tmp_dataset):
    """ Méthode pour obtenir le scaler ajusté """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(tmp_dataset)
    return scaler



def load_model_and_predict(file_path, input_data, scaler):
    """ Méthode qui charge le modèle """
    model = Sequential()
    model.add(LSTM(50, input_shape=(input_data.shape[1], 1), activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_regularizer=l2(0.01)))
    model.compile(loss="mean_squared_error", optimizer="adam")

    # Charger les poids du modèle
    model.load_weights(file_path)

    # Mettre à l'échelle les données d'entrée
    input_data_scaled = scaler.transform(input_data.reshape(-1, 1)).reshape(input_data.shape)

    # Faire des prédictions
    predictions = model.predict(input_data_scaled)

    # Redimensionner les données pour qu'elles aient deux dimensions
    input_data_reshaped = input_data.reshape(-1, 1)
    predictions_reshaped = predictions.reshape(-1, 1)

    # Inverser la transformation
    original_input_data = scaler.inverse_transform(input_data_reshaped)
    predictions_inversed = scaler.inverse_transform(predictions_reshaped)

    # Calculer les métriques
    rmse = np.sqrt(mean_squared_error(original_input_data, predictions_inversed))
    mse = mean_squared_error(original_input_data, predictions_inversed)
    mae = mean_absolute_error(original_input_data, predictions_inversed)
    evs = explained_variance_score(original_input_data, predictions_inversed)
    r2 = r2_score(original_input_data, predictions_inversed)

    # Vérifier si les valeurs sont strictement positives avant de calculer la déviance gamma et la déviance de Poisson
    if np.all(original_input_data > 0) and np.all(predictions_inversed > 0):
        mgd = mean_gamma_deviance(original_input_data, predictions_inversed)
        mpd = mean_poisson_deviance(original_input_data, predictions_inversed)
    else:
        mgd, mpd = np.nan, np.nan

    # Retourner les prédictions et les métriques
    return predictions_inversed, rmse, mse, mae, evs, r2, mgd, mpd






""" ************************* Chargement du modèle ************************* """
file_path = parameters.LOAD_MODEL_PATH+'best_model_weights4.weights.h5'
dataset_for_prediction = parameters.DATASET_FOR_PREDICTIONS
scaler = get_fitted_scaler(dataset_for_prediction)
input_data = ...  # Vos données d'entrée ici






""" ************************* Génération des prédictions ************************* """
predictions, rmse, mse, mae, evs, r2, mgd, mpd = load_model_and_predict(file_path, dataset_for_prediction, scaler)

print("Prédictions :", predictions)
print("RMSE :", rmse)
print("MSE :", mse)
print("MAE :", mae)
print("Explained Variance Score :", evs)
print("R2 Score :", r2)
print("Mean Gamma Deviance :", mgd)
print("Mean Poisson Deviance :", mpd)
