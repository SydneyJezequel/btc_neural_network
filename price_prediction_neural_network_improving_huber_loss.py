import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.src.utils.audio_dataset_utils import prepare_dataset
from tensorflow.python.keras.losses import Huber
from tensorflow.python.keras.optimizer_v1 import Adam

from BO.prepare_dataset import PrepareDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import parameters
from tensorflow.keras.callbacks import Callback
import joblib
from tensorflow.keras.callbacks import EarlyStopping





""" ************************* Paramètres ************************* """

DATASET_PATH = parameters.DATASET_PATH
PATH_TRAINING_DATASET = parameters.PATH_TRAINING_DATASET
TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE
DATASET_FOR_MODEL = parameters.DATASET_FOR_MODEL




""" ************************* Méthodes ************************* """

def ma(df, n):
    """ Calcul des moyennes mobiles """
    return pd.Series(df['Dernier'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))


def rsi(df, period):
    """ Calcul du RSI """
    delta = df['Dernier'].diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean(u[:period])
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean(d[:period])
    d = d.drop(d.index[:(period-1)])
    rs = u.ewm(com=period-1, adjust=False).mean() / d.ewm(com=period-1, adjust=False).mean()
    return 100 - 100 / (1 + rs)


def calculate_signal(dataset, taille_sma1, taille_sma2):
    """ Calcul des signaux de croisement des moyennes mobiles """
    sma1_col = 'MA_' + str(taille_sma1)
    sma2_col = 'MA_' + str(taille_sma2)
    signal_col = sma1_col + '_supérieure_' + sma2_col
    dataset[sma1_col] = ma(dataset, taille_sma1)
    dataset[sma2_col] = ma(dataset, taille_sma2)
    dataset[signal_col] = np.where(dataset[sma1_col] > dataset[sma2_col], 1.0, 0.0)
    return dataset


def format_dataset(initial_dataset):
    """ Préparation des données """
    tmp_dataset = initial_dataset.copy()
    tmp_dataset['Date'] = pd.to_datetime(initial_dataset['Date'], format='%d/%m/%Y', errors='coerce')
    tmp_dataset = tmp_dataset.sort_values(by='Date')
    numeric_columns = ["Dernier", "Ouv.", " Plus Haut", "Plus Bas", "Variation %"]
    for col in numeric_columns:
        tmp_dataset.loc[:, col] = tmp_dataset[col].str.replace('.', ' ').str.replace(' ', '').str.replace(',', '.')
    for col in numeric_columns:
        tmp_dataset[col] = pd.to_numeric(tmp_dataset[col], errors='coerce')
    return tmp_dataset


def delete_columns(tmp_dataset):
    """ Suppression des colones du dataset d'origine """
    tmp_dataset = tmp_dataset.drop(columns=['Vol.', 'Variation %', 'Ouv.', ' Plus Haut', 'Plus Bas'])
    return tmp_dataset


def add_technicals_indicators(tmp_dataset):
    """ Ajout des indicateurs techniques dans le dataset """
    tmp_dataset['MA_150'] = ma(tmp_dataset, 150)
    tmp_dataset['MA_100'] = ma(tmp_dataset, 100)
    tmp_dataset['MA_50'] = ma(tmp_dataset, 50)
    tmp_dataset['RSI'] = rsi(tmp_dataset, 14)
    calculate_signal(tmp_dataset, 50, 150)
    calculate_signal(tmp_dataset, 100, 150)
    calculate_signal(tmp_dataset, 50, 100)
    date_column = tmp_dataset['Date']
    tmp_dataset = tmp_dataset.drop(columns=['Date'])
    imputer = SimpleImputer(strategy='mean')
    tmp_dataset_imputed = imputer.fit_transform(tmp_dataset)
    tmp_dataset = pd.DataFrame(tmp_dataset_imputed, columns=tmp_dataset.columns)
    tmp_dataset['Date'] = date_column
    return tmp_dataset


def create_train_and_test_dataset(model_dataset):
    """ Création des datasets d'entrainement et tests """
    training_size = int(len(model_dataset) * 0.60)
    train_data, test_data = model_dataset.iloc[0:training_size, :], model_dataset.iloc[training_size:len(model_dataset), :]
    return train_data, test_data


def create_dataset(dataset, time_step=1):
    """ Méthode qui génère les datasets d'entrainement et de test """
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset.iloc[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset.iloc[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


def create_data_matrix(train_data, test_data, create_dataset):
    """ Création des matrices pour les datasets d'entrainement et test """
    time_step = 15
    x_train, y_train = create_dataset(train_data, time_step)
    x_test, y_test = create_dataset(test_data, time_step)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    return x_train, y_train, x_test, y_test


def subsample_old_data(tmp_dataset, cutoff_date, fraction=0.1):
    """ Sous-échantillonnage des anciennes données """
    old_data = tmp_dataset[tmp_dataset['Date'] < cutoff_date]
    recent_data = tmp_dataset[tmp_dataset['Date'] >= cutoff_date]
    old_data_sampled = old_data.sample(frac=fraction, random_state=42)
    combined_data = pd.concat([old_data_sampled, recent_data])
    combined_data = combined_data.sort_values(by='Date').reset_index(drop=True)
    return combined_data





""" ************************* Exécution du script principal ************************* """

prepare_dataset = PrepareDataset()


print(" ************ Etape 1 : Loading dataset ************ ")
# initial_dataset = pd.read_csv(DATASET_PATH + DATASET_FILE)
initial_dataset = pd.read_csv(PATH_TRAINING_DATASET + TRAINING_DATASET_FILE)


print(" ************ Etape 2 : Preparation of the Dataset ************ ")
tmp_dataset = format_dataset(initial_dataset)
tmp_dataset = delete_columns(tmp_dataset)
""" MIS DE COTE :
tmp_dataset = prepare_dataset.add_technicals_indicators(tmp_dataset)
"""

# Définir une date de coupure pour séparer les anciennes et récentes données :
cutoff_date = '2020-01-01'


# Appliquer le sous-échantillonnage :
tmp_dataset = subsample_old_data(tmp_dataset, cutoff_date, fraction=0.1)


# Normalisation :
tmp_dataset_copy = tmp_dataset.copy()
""" MIS DE COTE :
columns_to_normalize = ['Dernier', 'MA_150', 'MA_100', 'MA_50', 'MA_50_supérieure_MA_150', 'MA_100_supérieure_MA_150', 'MA_50_supérieure_MA_100']
"""
columns_to_normalize = ['Dernier']

scaler = prepare_dataset.get_fitted_scaler(tmp_dataset_copy[columns_to_normalize])
joblib.dump(scaler, 'scaler.save')
model_dataset = tmp_dataset
print("dataset")
normalized_datas = prepare_dataset.normalize_datas(tmp_dataset_copy[columns_to_normalize], scaler)
model_dataset[columns_to_normalize] = normalized_datas
print("dataset d'entrainement normalisé :", model_dataset)
print("model_dataset shape : ", model_dataset.shape)


# Sauvegarde du dataset pour contrôle :
model_dataset.to_csv(PATH_TRAINING_DATASET + 'dataset_modified_with_date.csv', index=False)


# Suppression de la colonne date :
del model_dataset['Date']


# Création des datasets d'entrainement et test :
train_data, test_data = create_train_and_test_dataset(model_dataset)
print("train_data type  : ", type(train_data))
print("test_data type  : ", type(test_data))
print("COLONNES DE train_data:", train_data.columns.tolist())
print("COLONNES DE test_data:", test_data.columns.tolist())
time_step = 15
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


# Création du modèle :
model = Sequential()
model.add(LSTM(10, input_shape=(None, 1), activation="relu"))
model.add(Dense(1))
model.compile(loss=Huber(delta=1.0), optimizer="adam")


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


# Affichage des métriques stockées :
print("Metrics History:")
for metric, values in metrics_history.items():
    print(f"{metric}: {values}")


# Évaluation du sur-apprentissage :
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()
plt.show()


# Évaluation du sur-apprentissage avec agrandissement des zones ou se trouvent les courbes :
def plot_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    loss_array = np.array(loss)
    val_loss_array = np.array(val_loss)
    plt.figure(figsize=(12, 6))
    plt.plot(loss_array, label='Training Loss', color='red')
    plt.plot(val_loss_array, label='Validation Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Zoomed)')
    plt.legend()
    plt.ylim(0, 0.003)  # Zoom sur la zone des pertes basses
    plt.grid(True)
    plt.show()
plot_loss(history)


# Affichage des sur et sous apprentissage :
# Conversion en tableaux NumPy :
loss_array = np.array(loss)
val_loss_array = np.array(val_loss)
# Affichage des tableaux :
print("Loss Array:", loss_array)
print("Validation Loss Array:", val_loss_array)






""" ************************* Controle du surapprentissage ************************* """


""" Affichage des résidus """
# Visualiser les résidus de tests et d'entrainements (différence valeur prédite et réelle) :
def plot_residuals(y_true, y_pred, title):
    # Trace les résidus
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.plot(residuals, label='Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(title)
    plt.xlabel('Observations')
    plt.ylabel('Residuals')
    plt.legend()
    plt.show()


# Tracer les résidus
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

train_predict = train_predict.reshape(-1, 1)
test_predict = test_predict.reshape(-1, 1)

scaler = prepare_dataset.get_fitted_scaler(train_predict)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

plot_residuals(original_ytrain, train_predict, 'Training Residuals')
plot_residuals(original_ytest, test_predict, 'Test Residuals')






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



