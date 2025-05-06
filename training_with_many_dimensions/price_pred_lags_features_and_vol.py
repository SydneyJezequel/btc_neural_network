import pprint
import pandas as pd
import numpy as np
from BO.metrics_callback import MetricsCallback
from service.display_results_service import DisplayResultsService
from service.prepare_dataset_service import PrepareDatasetService
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import parameters




""" ************************* Paramètres ************************* """

DATASET_PATH = parameters.DATASET_PATH
TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE
SAVED_MODEL = parameters.SAVED_MODEL




""" ************* Préparation du dataset ************* """

prepare_dataset = PrepareDatasetService()

# Chargement du dataset :
initial_dataset = pd.read_csv(TRAINING_DATASET_FILE)

# Préparation of the Dataset :
tmp_dataset = prepare_dataset.format_dataset(initial_dataset)
tmp_dataset = prepare_dataset.delete_columns(tmp_dataset)

# Ajout de la volatilité historique :
tmp_dataset['Historical_Volatility'] = prepare_dataset.calculate_historical_volatility(tmp_dataset)

# Ajout des caractéristiques de lag :
lags = [1, 7]
# lags = [1, 7, 30, 60, 90, 180, 365]
tmp_dataset = prepare_dataset.add_lag_features(tmp_dataset, lags)
# Supprimer les lignes avec des valeurs NaN introduites par les lags :
tmp_dataset = tmp_dataset.dropna()

# Définir une date de coupure pour séparer les anciennes et récentes données :
cutoff_date = '2020-01-01'

# Appliquer le sous-échantillonnage :
tmp_dataset = prepare_dataset.subsample_old_data(tmp_dataset, cutoff_date, fraction=0.1)

# Normalisation :
tmp_dataset_copy = tmp_dataset.copy()
columns_to_normalize = ['Dernier']
scaler = prepare_dataset.get_fitted_scaler(tmp_dataset_copy[columns_to_normalize])
model_dataset = tmp_dataset
normalized_datas = prepare_dataset.normalize_datas(tmp_dataset_copy[columns_to_normalize], scaler)
model_dataset[columns_to_normalize] = normalized_datas
print("dataset d'entrainement normalisé :", model_dataset)
print("model_dataset shape : ", model_dataset.shape)

# Sauvegarde du dataset pour contrôle :
model_dataset.to_csv(DATASET_PATH + 'dataset_modified_with_date.csv', index=False)

# Suppression de la colonne date :
del model_dataset['Date']

# Création des datasets d'entrainement et test :
x_train, y_train, x_test, y_test, test_data, dates, scaler = prepare_dataset.prepare_dataset(initial_dataset, cutoff_date)
"""
train_data, test_data = prepare_dataset.create_train_and_test_dataset(model_dataset)
time_step = 15
x_train, y_train = prepare_dataset.create_dataset(train_data, time_step)
x_test, y_test = prepare_dataset.create_dataset(test_data, time_step)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
"""
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)




""" ************* Définition du modèle ************* """

# Définition du nombre de timesteps et de features :
nb_timesteps = x_train.shape[1]
nb_features = x_train.shape[2]
print("nb_timesteps : ", nb_timesteps)
print("nb_features : ", nb_features)

# Création du réseau de neurones :
model = Sequential()
model.add(LSTM(10, input_shape=(nb_timesteps, nb_features), activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
"""
# Création du modèle amélioré
model = Sequential()
model.add(LSTM(20, input_shape=(None, 1), activation="tanh", return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(20, activation="tanh"))
model.add(Dropout(0.2))
model.add(Dense(1))
# Compilation du modèle
optimizer = Adam(learning_rate=0.001)
model.compile(loss="mean_squared_error", optimizer=optimizer)
"""




""" ************* Initialisation des métriques ************* """

# Stockage des métriques :
metrics_history = {
    "epoch": [],
    "train_rmse": [],
    "train_mse": [],
    "train_mae": [],
    "test_rmse": [],
    "test_mse": [],
    "test_mae": [],
    "train_explained_variance": [],
    "test_explained_variance": [],
    "train_r2": [],
    "test_r2": [],
    "train_mgd": [],
    "test_mgd": [],
    "train_mpd": [],
    "test_mpd": [],
}

# Callback pour stocker les métriques toutes les 50 epochs :
metrics_callback = MetricsCallback(x_train, y_train, x_test, y_test, metrics_history)




""" ************* Entrainement du modèle ************* """

# Entraînement du modèle :
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=800,
    batch_size=32,
    verbose=1,
    callbacks=[metrics_callback]
)

# Sauvegarde du modèle :
model.save_weights(SAVED_MODEL)




""" ************* Affichage des métriques ************* """

# Affichage des métriques :
pprint.pprint(metrics_history)

# Affichage des métriques durant les époques :
display_results = DisplayResultsService()
display_results.plot_metrics_history(metrics_history, metrics_to_plot=["rmse", "mse", "mae", "explained_variance", "r2", "mgd", "mpd"])




""" ************* Affichage des résultats ************* """

display_results = DisplayResultsService()

# Affichage des courbes de pertes :
display_results.plot_loss(history)

# Affichage des courbes de pertes zoomées :
display_results.zoom_plot_loss(history)

# Affichage des sur et sous apprentissage :
loss = history.history['loss']
val_loss = history.history['val_loss']
loss_array = np.array(loss)
val_loss_array = np.array(val_loss)




""" ************* Controle du surapprentissage ************* """

# Calcul des prédictions :
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

# Mise en forme du dataset :
train_predict = train_predict.reshape(-1, 1)
test_predict = test_predict.reshape(-1, 1)

scaler = prepare_dataset.get_fitted_scaler(train_predict)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

# Affichage des résidus :
display_results.plot_residuals(original_ytrain, train_predict, 'Training Residuals')
display_results.plot_residuals(original_ytest, test_predict, 'Test Residuals')
