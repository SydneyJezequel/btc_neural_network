import pandas as pd
import numpy as np
import math
from keras.src.utils.audio_dataset_utils import prepare_dataset
from service.display_results_service import DisplayResultsService
from service.prepare_dataset_service import PrepareDatasetService
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import parameters
from tensorflow.keras.callbacks import Callback
from service.generate_prediction_service import GeneratePredictionService






""" ************************* Paramètres ************************* """

DATASET_PATH = parameters.DATASET_PATH
PATH_TRAINING_DATASET = parameters.PATH_TRAINING_DATASET
TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE
DATASET_FOR_MODEL = parameters.DATASET_FOR_MODEL
DATASET_FOR_PREDICTIONS = parameters.DATASET_FOR_PREDICTIONS
FORMATED_BTC_COTATIONS = parameters.FORMATED_BTC_COTATIONS
TRAIN_PREDICT_START_INDEX = 2000
TEST_PREDICT_START_INDEX = 3200







""" ************************* Préparation du dataset ************************* """

prepare_dataset = PrepareDatasetService()

# Loading dataset :
initial_dataset = pd.read_csv(PATH_TRAINING_DATASET + TRAINING_DATASET_FILE)

# Préparation du dataset pré-entrainement :
cutoff_date = '2020-01-01'
x_train, y_train, x_test, y_test, scaler = prepare_dataset.prepare_dataset(initial_dataset, cutoff_date)

# Affichage du dataset :
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)








""" ************************* Définition du modèle ************************* """

# Création du modèle :
model = Sequential()
model.add(LSTM(10, input_shape=(None, 1), activation="relu"))
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

# Initialisation des tableaux pour stocker les métriques :
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
class MetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 50 == 0:
            train_predict = self.model.predict(x_train)
            test_predict = self.model.predict(x_test)

            train_predict = train_predict.reshape(-1, 1)
            test_predict = test_predict.reshape(-1, 1)

            scaler = prepare_dataset.get_fitted_scaler(train_predict)
            train_predict = scaler.inverse_transform(train_predict)
            test_predict = scaler.inverse_transform(test_predict)

            original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
            original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

            metrics_history["epoch"].append(epoch + 1)
            metrics_history["train_rmse"].append(math.sqrt(mean_squared_error(original_ytrain, train_predict)))
            metrics_history["train_mse"].append(mean_squared_error(original_ytrain, train_predict))
            metrics_history["train_mae"].append(mean_absolute_error(original_ytrain, train_predict))
            metrics_history["test_rmse"].append(math.sqrt(mean_squared_error(original_ytest, test_predict)))
            metrics_history["test_mse"].append(mean_squared_error(original_ytest, test_predict))
            metrics_history["test_mae"].append(mean_absolute_error(original_ytest, test_predict))
            metrics_history["train_explained_variance"].append(explained_variance_score(original_ytrain, train_predict))
            metrics_history["test_explained_variance"].append(explained_variance_score(original_ytest, test_predict))
            metrics_history["train_r2"].append(r2_score(original_ytrain, train_predict))
            metrics_history["test_r2"].append(r2_score(original_ytest, test_predict))
            # Vérification des valeurs strictement positives avant le calcul de la déviance gamma :
            if np.all(original_ytrain > 0) and np.all(train_predict > 0):
                mgd = mean_gamma_deviance(original_ytrain, train_predict)
                metrics_history["train_mgd"].append(mgd)
            else:
                metrics_history["train_mgd"].append(np.nan)

            if np.all(original_ytest > 0) and np.all(test_predict > 0):
                mgd = mean_gamma_deviance(original_ytest, test_predict)
                metrics_history["test_mgd"].append(mgd)
            else:
                metrics_history["test_mgd"].append(np.nan)
            # Vérification des valeurs strictement positives avant le calcul de la déviance poisson :
            if np.all(original_ytrain > 0) and np.all(train_predict > 0):
                mpd = mean_poisson_deviance(original_ytrain, train_predict)
                metrics_history["train_mpd"].append(mpd)
            else:
                metrics_history["train_mpd"].append(np.nan)

            if np.all(original_ytest > 0) and np.all(test_predict > 0):
                mpd = mean_poisson_deviance(original_ytest, test_predict)
                metrics_history["test_mpd"].append(mpd)
            else:
                metrics_history["test_mpd"].append(np.nan)







""" ************************* Entrainement du modèle ************************* """

#early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# Entraînement du modèle :
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=200,
    batch_size=32,
    verbose=1,
    callbacks=[MetricsCallback()] # [MetricsCallback(), early_stopping]
)

# Sauvegarde du modèle :
model.save_weights(parameters.SAVE_MODEL_PATH + f'model.weights.h5')








""" ************************* Affichage des résultats ************************* """

# Affichage des métriques stockées :
print("Metrics History:")
for metric, values in metrics_history.items():
    print(f"{metric}: {values}")

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
# Affichage des tableaux :
print("Loss Array:", loss_array)
print("Validation Loss Array:", val_loss_array)








""" ***************** Charger le modèle sauvegardé ***************** """
"""
model.save_weights(...)
model = create_model()  # fonction qui redéfinit la même architecture
model.load_weights('model.weights.h5')
"""











""" ***************** Prédictions sur un dataset indépendant ***************** """


""" Prédictions """
# Chargement du dataset :
dataset_for_predictions = DATASET_FOR_PREDICTIONS
dataset_for_predictions = pd.read_csv(dataset_for_predictions)

# Conserver les dates avant de les supprimer :
dates = dataset_for_predictions['Date'].values

# Prédictions :
generate_prediction_service = GeneratePredictionService()
time_step=15
predictions = generate_prediction_service.predict_on_new_data(dataset_for_predictions, model, time_step)

# Affichage des prédictions :
display_results.plot_predictions(dates, predictions, time_step)



""" Affichage des prédictions à la suite du dataset d'origine """
formated_dataset = pd.read_csv(FORMATED_BTC_COTATIONS)
display_results.display_all_dataset(formated_dataset)

# Conversion de la date :
formated_dataset['Date'] = pd.to_datetime(formated_dataset['Date'])

# Récupération de la dernière date :
formatted_dataset_last_date = formated_dataset['Date'].max()
print("last date : ", formatted_dataset_last_date)

# Numéro de la nouvelle date :
num_new_dates = len(formated_dataset)
num_days = len(predictions)
print("nombre de dates générées : ", num_new_dates)
print("nombre de predictions : ", len(predictions))

# Créer de nouvelles dates :
new_dates = [formatted_dataset_last_date + pd.Timedelta(days=i) for i in range(1, num_days + 1)]
print("new_dates : ", new_dates)

# Affichage pour vérification
for d in new_dates:
    print(d.date())

# Création du dataset de prédictions (Date + Dernier) :
predictions_dataset = pd.DataFrame({
    'Date': new_dates,
    'Dernier': predictions.flatten()
})
print("predictions_dataset : ", predictions_dataset)

# Affichage des prédictions :
display_results.display_all_dataset(predictions_dataset)

# Ajout des prédictions au dataset :
display_results.display_dataset_and_predictions(formated_dataset, predictions_dataset)




""" Affichage des prédictions d'entrainement et test VS le dataset d'origine """
# Prédictions :
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

# Dénormalisation :
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Création des tableaux alignés :
dataset_length = len(initial_dataset)
train_predict_plot = generate_prediction_service.insert_with_padding(train_predict, TRAIN_PREDICT_START_INDEX, dataset_length)
test_predict_plot = generate_prediction_service.insert_with_padding(test_predict, TEST_PREDICT_START_INDEX, dataset_length)

# Tracer la courbe :
display_results.plot_initial_dataset_and_predictions(initial_dataset, formated_dataset, train_predict_plot, test_predict_plot)

