import pprint
import pandas as pd
import numpy as np
from service.display_results_service import DisplayResultsService
from service.prepare_dataset_service import PrepareDatasetService
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import parameters
from BO.metrics_callback import MetricsCallback




""" ************* Paramètres ************* """

TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE
SAVED_MODEL = parameters.SAVED_MODEL




""" ************* Préparation du dataset ************* """

prepare_dataset = PrepareDatasetService()

# Chargement du dataset :
initial_dataset = pd.read_csv(TRAINING_DATASET_FILE)

# Préparation du dataset pré-entrainement :
cutoff_date = '2020-01-01'
x_train, y_train, x_test, y_test, test_data,  dates, scaler = prepare_dataset.prepare_one_dimension_dataset(initial_dataset, cutoff_date)




""" ************* Définition du modèle ************* """

# Création du réseau de neurones :
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(None, 1), activation="relu"))
model.add(LSTM(25, activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")




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

#early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# Entraînement du modèle :
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=200,
    batch_size=32,
    verbose=1,
    callbacks=[metrics_callback] # [metrics_callback , early_stopping]
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

# Mise en forme des datasets :
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
