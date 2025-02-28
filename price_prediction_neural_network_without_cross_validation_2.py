import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.src.utils.audio_dataset_utils import prepare_dataset
from BO.prepare_dataset import PrepareDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import plotly.express as px
import parameters
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from itertools import cycle





""" ****************************** Paramètres ****************************** """
DATASET_PATH = parameters.DATASET_PATH
PATH_TRAINING_DATASET = parameters.PATH_TRAINING_DATASET
DATASET_FILE = parameters.DATASET_FILE
DATASET_FOR_MODEL = parameters.DATASET_FOR_MODEL





""" ****************************** Classe technical indicators ****************************** """

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
    u[u.index[period - 1]] = np.mean(u[:period])  # first value is sum of avg gains
    u = u.drop(u.index[:(period - 1)])
    d[d.index[period - 1]] = np.mean(d[:period])  # first value is sum of avg losses
    d = d.drop(d.index[:(period - 1)])
    rs = u.ewm(com=period - 1, adjust=False).mean() / d.ewm(com=period - 1, adjust=False).mean()
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
    # Convertir la colonne "Date" au format datetime :
    tmp_dataset['Date'] = pd.to_datetime(initial_dataset['Date'], format='%d/%m/%Y', errors='coerce')
    # Trier le dataset par date en ordre chronologique :
    tmp_dataset = tmp_dataset.sort_values(by='Date')
    print("tmp_dataset trié : ", tmp_dataset)
    # errors='coerce' --> Les valeurs non converties sont remplacées par NaN.
    # Remplacer les points par un espace, puis suppression de l'espace, enfin remplacement de la virgule par un point :
    numeric_columns = ["Dernier", "Ouv.", " Plus Haut", "Plus Bas", "Variation %"]
    for col in numeric_columns:
        tmp_dataset.loc[:, col] = tmp_dataset[col].str.replace('.', ' ').str.replace(' ', '').str.replace(',', '.')
        # .loc[:, col] est utilisé pour sélectionner toutes les lignes (:) de la colonne spécifiée par col.
    # Conversion des colonnes numériques en float :
    for col in numeric_columns:
        tmp_dataset[col] = pd.to_numeric(tmp_dataset[col], errors='coerce')
    return tmp_dataset


def delete_columns(tmp_dataset):
    """ Suppression des colones du dataset d'origine """
    # Suppression des colonnes de départ :
    tmp_dataset = tmp_dataset.drop(columns=['Vol.', 'Variation %', 'Ouv.', ' Plus Haut', 'Plus Bas'])
    print('Dataset transformé :', tmp_dataset)
    return tmp_dataset


def add_technicals_indicators(tmp_dataset):
    """ Ajout des indicateurs techniques dans le dataset """
    # Ajout des indicateurs dans les colonnes :
    tmp_dataset['MA_150'] = ma(tmp_dataset, 150)
    tmp_dataset['MA_100'] = ma(tmp_dataset, 100)
    tmp_dataset['MA_50'] = ma(tmp_dataset, 50)
    tmp_dataset['RSI'] = rsi(tmp_dataset, 14)
    # Ajout des signaux générés par les indicateurs :
    calculate_signal(tmp_dataset, 50, 150)
    calculate_signal(tmp_dataset, 100, 150)
    calculate_signal(tmp_dataset, 50, 100)
    # Suppression de la colonne 'Date' :
    date_column = tmp_dataset['Date']
    tmp_dataset = tmp_dataset.drop(columns=['Date'])
    # Remplir les valeurs NaN avec la moyenne des colonnes :
    imputer = SimpleImputer(strategy='mean')
    tmp_dataset_imputed = imputer.fit_transform(tmp_dataset)
    # Reconversion en DataFrame avec les noms de colonnes d'origine :
    tmp_dataset = pd.DataFrame(tmp_dataset_imputed, columns=tmp_dataset.columns)
    # Réintégration de la colonne 'Date' :
    tmp_dataset['Date'] = date_column
    return tmp_dataset


def create_train_and_test_dataset(model_dataset):
    """ Création des datasets d'entrainement et tests """
    # Création des datasets d'entrainement et de test
    training_size = int(len(model_dataset) * 0.60)
    train_data, test_data = model_dataset.iloc[0:training_size, :], model_dataset.iloc[training_size:len(model_dataset),                                                                  :1]
    return train_data, test_data


def create_dataset(dataset, time_step=1):
    """ Méthode qui génère les datasets d'entrainement et de test """
    dataX, dataY = [], []
    # Boucle sur le dataset pour créer des séquences de longueur time_step :
    for i in range(len(dataset) - time_step - 1):
        # Extrait une séquence de longueur time_step à partir de l'index i
        a = dataset.iloc[i:(i + time_step), 0]
        # Ajoute la séquence à dataX :
        dataX.append(a)
        # Ajoute la valeur cible correspondante à dataY :
        dataY.append(dataset.iloc[i + time_step, 0])
    # Convertit les listes dataX et dataY en arrays numpy pour faciliter leur utilisation dans les modèles de machine learning :
    return np.array(dataX), np.array(dataY)


def create_data_matrix(train_data, test_data, create_dataset):
    """ Création des matrices pour les datasets d'entrainement et test """
    # Définition du time_step (longueur des séquences ex : Si 15, chaque séquence d'entrée contiendra 3 valeurs du dataset) :
    time_step = 15
    # Création des ensembles de données d'entraînement et de test en utilisant la fonction create_dataset :
    x_train, y_train = create_dataset(train_data, time_step)
    x_test, y_test = create_dataset(test_data, time_step)
    # Remodelage de X_train pour obtenir la forme [échantillons, time steps, caractéristiques]
    # Cela est nécessaire pour que les données soient compatibles avec les couches LSTM :
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    # Remodelage de X_test pour obtenir la forme [échantillons, time steps, caractéristiques]
    # Cela est nécessaire pour que les données soient compatibles avec les couches LSTM :
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    # Affichage des dimensions des ensembles de données d'entraînement et de test après remodelage :
    print("X_train: ", x_train.shape)
    print("X_test: ", x_test.shape)
    return x_train, y_train, x_test, y_test





""" **************************** Exécution du script principal **************************** """

# Initialisation de la classe qui prépare le dataset :
prepare_dataset = PrepareDataset()


initial_dataset = pd.read_csv(DATASET_PATH + DATASET_FILE)


# formatage des colonnes :
tmp_dataset = format_dataset(initial_dataset)


# Suppression des colonnes :
tmp_dataset = delete_columns(tmp_dataset)


# Contrôle des modifications :
print("En-tête du dataset d'entrainement : ", tmp_dataset.head())
print("dataset d'entrainement modifié (dernières lignes) pour vérifier si mes indicateurs sont bien calculés : ",
      tmp_dataset.tail())
print('Null Values dataset final : ', tmp_dataset.isnull().values.sum())
print('NA values dataset final :', tmp_dataset.isnull().values.any())


# Ajout des indicateurs techniques :
tmp_dataset = prepare_dataset.add_technicals_indicators(tmp_dataset)
print(" forme tmp_dataset shape : ", tmp_dataset.shape)
print(" forme tmp_dataset : ", tmp_dataset)


# Normalisation du dataset :
tmp_dataset_copy = tmp_dataset.copy()
columns_to_normalize = ['Dernier', 'MA_150', 'MA_100', 'MA_50', 'MA_50_supérieure_MA_150', 'MA_100_supérieure_MA_150',
                        'MA_50_supérieure_MA_100']
scaler = prepare_dataset.get_fitted_scaler(tmp_dataset_copy[columns_to_normalize])
joblib.dump(scaler, 'scaler.save')
model_dataset = tmp_dataset
print("dataset")
normalized_datas = prepare_dataset.normalize_datas(tmp_dataset_copy[columns_to_normalize], scaler)
model_dataset[columns_to_normalize] = normalized_datas
print("dataset d'entrainement normalisé :", model_dataset)
print("model_dataset shape : ", model_dataset.shape)


# Vérification du dataset après la normalisation :
model_dataset.to_csv(PATH_TRAINING_DATASET + 'dataset_modified_with_date.csv', index=False)
# Tracé de l'évolution du prix en fonction des dates :
import seaborn as sns
features_to_plot = ['Dernier', 'MA_150', 'MA_100', 'MA_50', 'RSI']
for feature in features_to_plot:
    plt.figure(figsize=(10, 6))
    sns.histplot(tmp_dataset[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()
plt.figure(figsize=(14, 7))
plt.plot(tmp_dataset['Date'], tmp_dataset['Dernier'], label='Prix de clôture', color='b')
plt.title('Évolution du Prix de Clôture en Fonction des Dates')
plt.xlabel('Date')
plt.ylabel('Prix de Clôture')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Suppression de la colonne 'Date' :
date_column = model_dataset['Date']
del tmp_dataset['Date']


# Contrôle : Dataset avant la scission en dataset de train et de test :
model_dataset.to_csv(PATH_TRAINING_DATASET + 'dataset_modified_for_model.csv', index=False)
print("model_dataset shape juste avant traitement : ", model_dataset.shape)


# Création des datasets d'entrainement et de test :
train_data, test_data = create_train_and_test_dataset(model_dataset)


# Contrôle : Sauvegarde des Datasets de 'Train' et 'Test' :
train_data.to_csv(PATH_TRAINING_DATASET + 'train_data.csv', index=False)
test_data.to_csv(PATH_TRAINING_DATASET + 'test_data.csv', index=False)


# Conversion des arrays en matrice :
time_step = 15
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

"""
# ************************* TESTS ************************* #
if (y_train < 0).any():
    print("Il y a des valeurs négatives dans y_train")
if (y_test < 0).any():
    print("Il y a des valeurs négatives dans y_test")
if (x_train < 0).any():
    print("Il y a des valeurs négatives dans x_train")
if (x_test < 0).any():
    print("Il y a des valeurs négatives dans x_test")

print("x_train : ", x_train)
print("x_test : ", x_test)
print("y_train : ", y_train)
print("y_test : ", y_test)


# Aplatir les données 3D en 2D
x_train_2d = x_train.reshape(x_train.shape[0], -1)
x_test_2d = x_test.reshape(x_test.shape[0], -1)
# Convertir en DataFrame
x_train_df = pd.DataFrame(x_train_2d)
x_test_df = pd.DataFrame(x_test_2d)
y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)
# Sauvegarder en format CSV
x_train_df.to_csv('x_train.csv', index=False)
y_train_df.to_csv('y_train.csv', index=False)
x_test_df.to_csv('x_test.csv', index=False)
y_test_df.to_csv('y_test.csv', index=False)
# ************************* TESTS ************************* #
"""

# Vérification du nombre de dimensions :
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


# Création du modèle :
model = Sequential()
model.add(LSTM(10, input_shape=(None, 1), activation="relu"))
# model.add(Dropout(0.2))
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


# Callback personnalisé pour stocker les métriques :
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


            # Contrôle des datasets :
            original_ytrain = pd.DataFrame(original_ytrain)
            original_ytest = pd.DataFrame(original_ytest)
            train_predict = pd.DataFrame(train_predict)
            test_predict = pd.DataFrame(test_predict)
            """
            if (original_ytrain < 0).any():
                print("Il y a des valeurs négatives dans original_ytrain")
            else:
                print("pas de valeurs négatives dans original_ytrain")
            if (original_ytest < 0).any():
                print("Il y a des valeurs négatives dans original_ytest")
            else :
                print("pas de valeurs négatives dans original_ytest")
            print('Null Values dataset final original_ytrain : ', original_ytrain.isnull().values.sum())
            print('NA values dataset final original_ytest :', original_ytest.isnull().values.any())
            if (train_predict < 0).any():
                print("Il y a des valeurs négatives dans train_predict")
            if (test_predict < 0).any():
                print("Il y a des valeurs négatives dans test_predict")

            print('Null Values dataset final : ', train_predict.isnull().values.sum())
            print('NA values dataset final :', test_predict.isnull().values.any())
            """
            original_ytrain.to_csv(PATH_TRAINING_DATASET + 'original_ytrain.csv', index=False)
            original_ytest.to_csv(PATH_TRAINING_DATASET + 'original_ytest.csv', index=False)
            train_predict.to_csv(PATH_TRAINING_DATASET + 'train_predict.csv', index=False)
            test_predict.to_csv(PATH_TRAINING_DATASET + 'test_predict.csv', index=False)


            # Enregistrement des métriques :
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


# Définition du EarlyStopping :
# early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)


# Entraînement du modèle avec le callback :
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=200,
    batch_size=32,
    verbose=1,
    callbacks=[MetricsCallback()]  # [MetricsCallback(), early_stopping] #
)


# Sauvegarde du modèle :
model.save_weights(parameters.SAVE_MODEL_PATH + f'model.weights.h5')


# Affichage des métriques :
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



""" ******************** Comparaison entre le prix original et les prédictions ******************** """
print(" ******************** Comparaison entre le prix original et les prédictions ******************** ")

# Génération des prédictions :
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)


# Utilisez uniquement la colonne pertinente pour les prédictions :
train_predict = scaler.inverse_transform(np.hstack([train_predict] * len(columns_to_normalize)))
test_predict = scaler.inverse_transform(np.hstack([test_predict] * len(columns_to_normalize)))


look_back = time_step
trainPredictPlot = np.empty_like(model_dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, 0] = train_predict[:, 0]
print("Données prédites d'entraînement :", trainPredictPlot.shape)


testPredictPlot = np.empty_like(model_dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(model_dataset) - 1, 0] = test_predict[:, 0]
print("Données prédites de test :", testPredictPlot.shape)


names = cycle(['Prix de clôture original', 'Prix de clôture prédit (entraînement)', 'Prix de clôture prédit (test)'])
close_stock = model_dataset.copy()
close_stock['Date'] = date_column
plotdf = pd.DataFrame({
    'original_close': close_stock['Dernier'],
    'train_predicted_close': trainPredictPlot[:, 0].tolist(),
    'test_predicted_close': testPredictPlot[:, 0].tolist()
})
fig = px.line(
    plotdf,
    x=plotdf.index,
    y=['original_close', 'train_predicted_close', 'test_predicted_close'],
    labels={'value': 'Prix de l\'action', 'index': 'Date'}
)
fig.update_layout(
    title_text='Comparaison entre le prix de clôture original et le prix de clôture prédit',
    plot_bgcolor='white',
    font_size=15,
    font_color='black',
    legend_title_text='Prix de clôture'
)
fig.for_each_trace(lambda t: t.update(name=next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()





""" ******************** Prédictions des 30 prochains jours ******************** """
print(" ******************** Prédictions des 30 prochains jours ******************** ")

# Sélectionner les dernières 'time_step' valeurs des données de test pour l'entrée initiale :
x_input = test_data[len(test_data) - time_step:].reshape(1, -1)


# Convertir l'entrée en une liste :
temp_input = list(x_input)


# Convertir la liste en une liste de listes (pour manipulation ultérieure) :
temp_input = temp_input[0].tolist()


# Initialiser la liste pour stocker les prédictions :
lst_output = []


# Définir le nombre de pas de temps (time_step) :
n_steps = time_step


# Initialiser le compteur de boucle :
i = 0


# Définir le nombre de jours à prédire :
pred_days = 30


# Boucle de prédiction pour les prochains jours :
while i < pred_days:
    if len(temp_input) > time_step:
        # Préparer l'entrée pour la prédiction en utilisant les 'time_step' dernières valeurs :
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        # Prédire la valeur suivante en utilisant le modèle :
        yhat = model.predict(x_input, verbose=0)
        # Ajouter la prédiction à temp_input et mettre à jour temp_input :
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        # Ajouter la prédiction à la liste des sorties :
        lst_output.extend(yhat.tolist())
        # Incrémenter le compteur de boucle :
        i += 1
    else:
        # Préparer l'entrée pour la prédiction en utilisant les valeurs initiales :
        x_input = x_input.reshape((1, n_steps, 1))
        # Prédire la valeur suivante en utilisant le modèle :
        yhat = model.predict(x_input, verbose=0)
        # Ajouter la prédiction à temp_input :
        temp_input.extend(yhat[0].tolist())
        # Ajouter la prédiction à la liste des sorties :
        lst_output.extend(yhat.tolist())
        # Incrémenter le compteur de boucle :
        i += 1


# Afficher le nombre de prédictions générées :
print("Nombre de prédictions pour les prochains jours :", len(lst_output))
