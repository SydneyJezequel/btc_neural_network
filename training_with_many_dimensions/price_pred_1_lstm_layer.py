import pprint
import pandas as pd
from BO.metrics_callback import MetricsCallback
from service.display_results_service import DisplayResultsService
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import parameters
from service.generate_prediction_service import GeneratePredictionService
from service.prepare_dataset_service import PrepareDatasetService






""" ************* Paramètres ************* """

TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE
DATASET_FOR_PREDICTIONS = parameters.DATASET_FOR_PREDICTIONS
FORMATED_BTC_COTATIONS_FILE = parameters.FORMATED_BTC_COTATIONS_FILE
SAVED_MODEL = parameters.SAVED_MODEL
TIME_STEP = parameters.TIME_STEP
TRAIN_PREDICT_START_INDEX = 2000
TEST_PREDICT_START_INDEX = 3200




""" ************* Préparation du dataset ************* """

prepare_dataset = PrepareDatasetService()

# Chargement du dataset :
initial_dataset = pd.read_csv(TRAINING_DATASET_FILE)

# Préparation du dataset pré-entrainement :
cutoff_date = '2020-01-01'
x_train, y_train, x_test, y_test, test_data, dates, scaler = prepare_dataset.prepare_many_dimensions_dataset(initial_dataset, cutoff_date)

# Affichage du dataset :
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)




""" ************* Définition du modèle ************* """

# Définition du nombre de timesteps et de features.
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
model = Sequential()
model.add(LSTM(10, input_shape=(None, 1), activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
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

# early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# Entraînement du modèle :
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=400,
    batch_size=32,
    verbose=1,
    callbacks=[metrics_callback] # [metrics_callback, early_stopping]
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

# Affichage des courbes de pertes :
display_results.plot_loss(history)

# Affichage des courbes de pertes (zoom) :
display_results.zoom_plot_loss(history)




""" ************* Génération des résidus d'entrainements ************* """

# Génération des prédictions :
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

# Dénormalisation :
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Mise en forme du dataset :
original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

# Affichage des résidus :
display_results.plot_residuals(original_ytrain, train_predict, 'Training Residuals')
display_results.plot_residuals(original_ytest, test_predict, 'Test Residuals')




""" ************* Charger le modèle sauvegardé ************* """
"""
model.save_weights(...)
model = create_model()  # fonction qui redéfinit la même architecture
model.load_weights('model.weights.h5')
"""




""" ************* Prédictions sur un dataset indépendant ************* """

# Chargement du dataset :
dataset_for_predictions = '../dataset/btc_cotations_for_predictions.csv'
dataset_for_predictions = pd.read_csv(dataset_for_predictions)

# Génération des prédictions :
generate_prediction_service = GeneratePredictionService()
time_step = TIME_STEP

# predictions = generate_prediction_service.predict_on_new_data(dataset_for_predictions, model, time_step)
predictions = generate_prediction_service.predict_on_new_data(test_data, model, scaler, time_step)
print("predictions : ", predictions)

# Affichage des prédictions :
display_results.plot_predictions(dates, predictions, time_step)




""" ************* Affichage du dataset d'origine et des prédictions ************* """

# Chargement du dataset initial :
formated_dataset = pd.read_csv(FORMATED_BTC_COTATIONS_FILE)
formated_dataset['Date'] = pd.to_datetime(formated_dataset['Date'])

# Préparation des datasets pour l'affichage :
formatted_dataset_last_date = formated_dataset['Date'].max()
num_days = len(predictions)
new_dates = [formatted_dataset_last_date + pd.Timedelta(days=i) for i in range(1, num_days + 1)]
print("last date : ", formatted_dataset_last_date)
print("new_dates : ", new_dates)
predictions_dataset = pd.DataFrame({
    'Date': new_dates,
    'Dernier': predictions.flatten()
})

# Affichage des prédictions :
display_results.display_all_dataset(predictions_dataset)

# Affichage du dataset initial et des prédictions :
display_results.display_dataset_and_predictions(formated_dataset, predictions_dataset)




""" ************* Affichage des prédictions d'entrainement et test VS le dataset d'origine  ************* """

# Alignement des datasets :
dataset_length = len(initial_dataset)
train_predict_plot = generate_prediction_service.insert_with_padding(train_predict, TRAIN_PREDICT_START_INDEX, dataset_length)
test_predict_plot = generate_prediction_service.insert_with_padding(test_predict, TEST_PREDICT_START_INDEX, dataset_length)

# Affichages des prédictions :
display_results.plot_initial_dataset_and_predictions(initial_dataset, formated_dataset, train_predict_plot, test_predict_plot)
