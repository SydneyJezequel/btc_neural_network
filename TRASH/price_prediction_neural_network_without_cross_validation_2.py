import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.src.utils.audio_dataset_utils import prepare_dataset
from service.prepare_dataset_service import PrepareDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import parameters
from tensorflow.keras.callbacks import Callback
import joblib

""" ****************************** Paramètres ****************************** """
DATASET_PATH = parameters.DATASET_PATH
PATH_TRAINING_DATASET = parameters.PATH_TRAINING_DATASET
DATASET_FILE = parameters.DATASET_FILE
DATASET_FOR_MODEL = parameters.DATASET_FOR_MODEL










""" ****************************** Classe technical indicators ****************************** """

def ma(df, n):
    """ Calcul de la moyenne mobile """
    return pd.Series(df['Dernier'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))


def rsi(df, period):
    """ Calcul du RSI """
    delta = df['Dernier'].diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean(u[:period])  # first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean(d[:period])  # first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = u.ewm(com=period-1, adjust=False).mean() / d.ewm(com=period-1, adjust=False).mean()
    return 100 - 100 / (1 + rs)



def calculate_signal(dataset, taille_sma1, taille_sma2):
    """ Calcul du signal croisement des sma """
    sma1_col = 'MA_' + str(taille_sma1)
    sma2_col = 'MA_' + str(taille_sma2)
    signal_col = sma1_col +  '_supérieure_'  + sma2_col
    dataset[sma1_col] = ma(dataset, taille_sma1)
    dataset[sma2_col] = ma(dataset, taille_sma2)
    dataset[signal_col] = np.where(dataset[sma1_col] > dataset[sma2_col], 1.0, 0.0)
    return dataset








""" ****************************** Classe prepare_dataset ****************************** """

def format_dataset(initial_dataset):
    """ Méthode format_dataset() """
    tmp_dataset = initial_dataset.copy()
    # Convertir la colonne "Date" au format datetime :
    tmp_dataset['Date'] = pd.to_datetime(initial_dataset['Date'], format='%d/%m/%Y', errors='coerce')
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
    """ Méthode delete_columns() """
    # Suppression des colonnes de départ :
    tmp_dataset = tmp_dataset.drop(columns=['Vol.', 'Variation %', 'Ouv.', ' Plus Haut', 'Plus Bas'])
    print('Dataset transformé :', tmp_dataset)
    return tmp_dataset



def add_technicals_indicators(tmp_dataset):
    """ Méthode add_technicals_indicators() """
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



"""
def normalize_datas(tmp_dataset):
    #  Méthode normalize_data()
    # Suppression de la colonne 'Date' avant la normalisation
    tmp_dataset = tmp_dataset.drop(columns=['Date'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(tmp_dataset)
"""
"""
La méthode MinMaxScaler de la bibliothèque scikit-learn est utilisée pour normaliser
les caractéristiques (features) d'un jeu de données.
Elle met à l'échelle chaque caractéristique dans une plage spécifiée, généralement entre 0 et 1.
On peut faire la même chose avec un StandardScaler().
"""



def create_train_and_test_dataset(model_dataset):
    """ Méthode create_train_and_test_dataset() """
    # Création des datasets d'entrainement et de test
    training_size = int(len(model_dataset) * 0.60)
    test_size = len(model_dataset) - training_size
    train_data, test_data = model_dataset[0:training_size, :], model_dataset[training_size:len(model_dataset), :1]
    # datas_for_model[0:training_size,:] : Sélectionne les training_size premières lignes de l'array.
    # datas_for_model[training_size:len(datas_for_model),:1] : Sélectionne toutes les lignes à partir de training_size jusqu'à la fin de l'array.
    print("dataset d'entrainement :", train_data.shape)
    print("dataset de test :", test_data.shape)
    return train_data, test_data



def create_dataset(dataset, time_step=1):
    """ Méthode create_dataset() """
    dataX, dataY = [], []
    # Boucle sur le dataset pour créer des séquences de longueur time_step :
    for i in range(len(dataset) - time_step - 1):
        # Extrait une séquence de longueur time_step à partir de l'index i
        a = dataset[i:(i + time_step), 0]
        # Ajoute la séquence à dataX :
        dataX.append(a)
        # Ajoute la valeur cible correspondante à dataY :
        dataY.append(dataset[i + time_step, 0])
    # Convertit les listes dataX et dataY en arrays numpy pour faciliter leur utilisation dans les modèles de machine learning :
    return np.array(dataX), np.array(dataY)



def create_data_matrix(train_data, test_data, create_dataset):
    """ Méthode create_data_matrix() """
    # Définition du time_step (longueur des séquences ex : Si 15, chaque séquence d'entrée contiendra 3 valeurs du dataset) :
    time_step = 15
    # Création des ensembles de données d'entraînement et de test en utilisant la fonction create_dataset :
    x_train, y_train = create_dataset(train_data, time_step)
    x_test, y_test = create_dataset(test_data, time_step)
    # Affichage des dimensions des ensembles de données d'entraînement et de test :
    print("X_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    print("X_test: ", x_test.shape)
    print("y_test: ", y_test.shape)
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



print(" ************ Etape 1 : Loading dataset ************ ")
initial_dataset = pd.read_csv(DATASET_PATH+DATASET_FILE)

print('Total number of days present in the dataset: ', initial_dataset.shape[0])
print('Total number of fields present in the dataset: ', initial_dataset.shape[1])
print('Nombre de lignes et colonnes: ', initial_dataset.shape)
print('En-têtes du dataset: ', initial_dataset.head())
print('initial_dataset.tail(): ', initial_dataset.tail())
print('initial_dataset.info(): ', initial_dataset.info())
print('initial_dataset.describe()', initial_dataset.describe())
print(" ************ Vérification des données manquantes ************ ")
print('Null Values:', initial_dataset.isnull().values.sum())
print('NA values:', initial_dataset.isnull().values.any())
ed=initial_dataset.iloc[0][0]
sd=initial_dataset.iloc[-1][0]
print('Starting Date : ',sd)
print('Ending Date : ',ed)





print(" ************ Etape 2 : Preparation of the Dataset ************ ")
# formatage des colonnes :
tmp_dataset = format_dataset(initial_dataset)


# Suppression des colonnes :
tmp_dataset = delete_columns(tmp_dataset)


# Ajout des indicateurs techniques :
# tmp_dataset = prepare_dataset.add_technicals_indicators(tmp_dataset)
# print("tmp_datastet AVEC SMA : ", tmp_dataset)


# Enregistrement du dataset au format csv :
tmp_dataset.to_csv(PATH_TRAINING_DATASET+DATASET_FOR_MODEL, index=False)


# Contrôle des modifications :
print("En-tête du dataset d'entrainement : ", tmp_dataset.head())
print("dataset d'entrainement modifié (dernières lignes) pour vérifier si mes indicateurs sont bien calculés : ", tmp_dataset.tail())
print('Null Values dataset final : ', tmp_dataset.isnull().values.sum())
print('NA values dataset final :', tmp_dataset.isnull().values.any())


tmp_dataset = prepare_dataset.add_technicals_indicators(tmp_dataset)
print(" forme tmp_dataset : ", tmp_dataset.shape)
# joblib.dump(tmp_dataset, 'tmp_dataset.save')
print(" dataset avant transformation : ", tmp_dataset)
tmp_dataset.to_csv(PATH_TRAINING_DATASET+DATASET_FOR_MODEL, index=False)

# Normalise dataset :
# model_dataset = normalize_datas(tmp_dataset)
# Obtenir le scaler ajusté :
scaler = prepare_dataset.get_fitted_scaler(tmp_dataset)
joblib.dump(scaler, '../scaler.save')


# Normalise dataset :
model_dataset = prepare_dataset.normalize_datas(tmp_dataset, scaler)
# NOUVELLE VERSION :
# columns_to_scale = tmp_dataset.columns[:-3]
# model_dataset = prepare_dataset.normalize_datas2(tmp_dataset, scaler, columns_to_scale)
print("model_dataset : ", model_dataset )
print("dataset d'entrainement normalisé :", model_dataset)
print("model_dataset shape : ", model_dataset.shape)
print("dataset retransformé : ", scaler.inverse_transform(model_dataset))
print("dataset d'entrainement normalisé :", model_dataset)


# Méthode create_dataset :
train_data, test_data = create_train_and_test_dataset(model_dataset)


# Sauvegarde du dataset de test pour les prédictions :
joblib.dump(test_data, 'test_data.save')


# Conversion des arrays en matrice
time_step = 15
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


# Création du modèle
model = Sequential()
model.add(LSTM(10, input_shape=(None, 1), activation="relu"))
# model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")


# Initialisation des tableaux pour stocker les métriques
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


# Callback personnalisé pour stocker les métriques toutes les 50 epochs
class MetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 50 == 0:
            train_predict = self.model.predict(x_train)
            test_predict = self.model.predict(x_test)

            print("train_predict.shape, test_predict.shape : ", train_predict.shape, test_predict.shape)

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
            metrics_history["train_mgd"].append(mean_gamma_deviance(original_ytrain, train_predict))
            metrics_history["test_mgd"].append(mean_gamma_deviance(original_ytest, test_predict))
            metrics_history["train_mpd"].append(mean_poisson_deviance(original_ytrain, train_predict))
            metrics_history["test_mpd"].append(mean_poisson_deviance(original_ytest, test_predict))


# Définition du EarlyStopping :
# early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)


# Entraînement du modèle avec le callback
print("SHAPE : ")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=200,
    batch_size=32,
    verbose=1,
    callbacks=[MetricsCallback()] # [MetricsCallback(), early_stopping] #
)


# Sauvegarde du modèle
model.save_weights(parameters.SAVE_MODEL_PATH + f'model.weights.h5')


# Affichage des métriques stockées :
print("Metrics History:")
for metric, values in metrics_history.items():
    print(f"{metric}: {values}")


# Évaluation du sur-apprentissage
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()
plt.show()

