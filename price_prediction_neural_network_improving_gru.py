import pandas as pd
import numpy as np
import math
from keras.src.layers import GRU
from keras.src.utils.audio_dataset_utils import prepare_dataset
from service.display_results_service import DisplayResultsService
from service.prepare_dataset_service import PrepareDatasetService
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import parameters
from tensorflow.keras.callbacks import Callback











""" ************************* Paramètres ************************* """

DATASET_PATH = parameters.DATASET_PATH
PATH_TRAINING_DATASET = parameters.PATH_TRAINING_DATASET
TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE
DATASET_FOR_MODEL = parameters.DATASET_FOR_MODEL













""" ************************* Préparation du dataset ************************* """

prepare_dataset = PrepareDatasetService()


# Loading dataset :
initial_dataset = pd.read_csv(PATH_TRAINING_DATASET + TRAINING_DATASET_FILE)


# Préparation du dataset pré-entrainement :
cutoff_date = '2020-01-01'
x_train, y_train, x_test, y_test = prepare_dataset.prepare_dataset(initial_dataset, cutoff_date)


# Affichage du dataset :
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)















""" ************************* Définition du modèle ************************* """

# Création du modèle :
model = Sequential()
model.add(GRU(10, activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")


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














""" ************************* Controle du surapprentissage ************************* """

# Calcul des prédictions :
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

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



























""" ******************************** Autres ******************************** """

""" Faire des prédictions sur un dataset indépendant """

def predict_on_new_data(new_data_path, model, scaler, time_step=15):
    """Faire des prédictions sur un dataset indépendant"""

    # Charger et préparer le nouveau dataset
    new_dataset = pd.read_csv(new_data_path)
    new_dataset = format_dataset(new_dataset)
    new_dataset = delete_columns(new_dataset)

    # Ajouter les indicateurs techniques
    prepare_dataset = PrepareDataset()
    new_dataset = prepare_dataset.add_technicals_indicators(new_dataset)

    # Supprimer les lignes avec des valeurs manquantes
    new_dataset = new_dataset.dropna()

    # Normaliser les données
    columns_to_normalize = ['Dernier', 'MA_150', 'MA_100', 'MA_50', 'MA_50_supérieure_MA_150', 'MA_100_supérieure_MA_150', 'MA_50_supérieure_MA_100']

    # Vérifiez que toutes les colonnes à normaliser existent
    missing_columns = [col for col in columns_to_normalize if col not in new_dataset.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in new dataset: {missing_columns}")

    # Utiliser la méthode normalize_datas pour normaliser les données
    normalized_datas = prepare_dataset.normalize_datas(new_dataset[columns_to_normalize], scaler)
    new_dataset[columns_to_normalize] = normalized_datas

    # Suppression de la colonne date :
    del new_dataset['Date']

    # Créer le dataset pour la prédiction
    x_new, _ = create_dataset(new_dataset, time_step)
    x_new = x_new.reshape(x_new.shape[0], x_new.shape[1], 1)

    # Faire des prédictions
    new_predictions = model.predict(x_new)
    new_predictions = scaler.inverse_transform(new_predictions)

    return new_predictions



"""
def predict_on_new_data(new_data_path, model, scaler, time_step=15):
    # Faire des prédictions sur un dataset indépendant

    # Charger et préparer le nouveau dataset
    new_dataset = pd.read_csv(new_data_path)
    new_dataset = format_dataset(new_dataset)
    new_dataset = delete_columns(new_dataset)

    # Ajouter les indicateurs techniques
    prepare_dataset = PrepareDataset()
    new_dataset = prepare_dataset.add_technicals_indicators(new_dataset)

    # Supprimer les lignes avec des valeurs manquantes
    new_dataset = new_dataset.dropna()

    # Vérifiez que le dataset n'est pas vide après le traitement
    if new_dataset.empty:
        raise ValueError("The dataset is empty after processing.")

    # Normaliser les données
    columns_to_normalize = ['Dernier', 'MA_150', 'MA_100', 'MA_50', 'MA_50_supérieure_MA_150', 'MA_100_supérieure_MA_150', 'MA_50_supérieure_MA_100', 'Historical_Volatility'] + [f'Lag_{lag}' for lag in lags]

    # Vérifiez que toutes les colonnes à normaliser existent et ne sont pas vides
    for col in columns_to_normalize:
        if col not in new_dataset.columns or new_dataset[col].isnull().all():
            raise ValueError(f"Column {col} is missing or empty in the new dataset.")

    # Utiliser la méthode normalize_datas pour normaliser les données
    normalized_datas = prepare_dataset.normalize_datas(new_dataset[columns_to_normalize], scaler)
    new_dataset[columns_to_normalize] = normalized_datas

    # Créer le dataset pour la prédiction
    x_new, _ = create_dataset(new_dataset, time_step)
    x_new = x_new.reshape(x_new.shape[0], x_new.shape[1], 1)

    # Faire des prédictions
    new_predictions = model.predict(x_new)
    new_predictions = scaler.inverse_transform(new_predictions)

    return new_predictions
"""



# Exemple d'utilisation de la fonction pour faire des prédictions sur un nouveau dataset
DATASET_FILE_FOR_TEST_PREDICTIONS = parameters.DATASET_FILE_FOR_TEST_PREDICTIONS
dataset_for_test_predictions = PATH_TRAINING_DATASET + DATASET_FILE_FOR_TEST_PREDICTIONS
new_predictions = predict_on_new_data(dataset_for_test_predictions, model, scaler)
print("New Predictions:", new_predictions)



