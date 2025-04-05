""" First we will import the necessary Library """
import os
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt
""" Evalution library """
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
""" For model building we will use these library """
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
""" For Plotting we will use these library """
import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import parameters
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2








""" ****************************** Paramètres ****************************** """

DATASET_PATH = parameters.DATASET_PATH
PATH_TRAINING_DATASET = parameters.PATH_TRAINING_DATASET
TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE
DATASET_FOR_MODEL = parameters.DATASET_FOR_MODEL
DATASET_FOR_PREDICTIONS = parameters.DATASET_FOR_PREDICTIONS








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
    signal_col = 'signal_' + sma1_col + '_' + sma2_col
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
    # Ignorer les valeurs NAN (Remplir les valeurs NaN avec la moyenne des colonnes) :
    del tmp_dataset['Date']
    imputer = SimpleImputer(strategy='mean')
    tmp_dataset = imputer.fit_transform(tmp_dataset)
    return tmp_dataset



def normalize_datas(tmp_dataset):
    """ Méthode normalize_data() """
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(np.array(tmp_dataset).reshape(-1, 1))
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




print(" ************ Etape 1 : Loading dataset ************ ")
# initial_dataset = pd.read_csv(DATASET_PATH+DATASET_FILE)
initial_dataset = pd.read_csv(PATH_TRAINING_DATASET + TRAINING_DATASET_FILE)







print(" ************ Etape 2 : Preparation of the Dataset ************ ")
# formatage des colonnes :
tmp_dataset = format_dataset(initial_dataset)

# Suppression des colonnes :
tmp_dataset = delete_columns(tmp_dataset)

# Affichage des données :
fig = px.line(tmp_dataset, x=tmp_dataset.Date, y=tmp_dataset.Dernier,labels={'Date':'date','Dernier':'Close Stock'})
fig.update_traces(marker_line_width=2, opacity=0.8, marker_line_color='orange')
fig.update_layout(title_text='Whole period of timeframe of Bitcoin close price 2014-2025', plot_bgcolor='white',
                  font_size=15, font_color='black')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()





# Proposition de Mistral AI :
import plotly.graph_objects as go

def display_all_dataset(dataset, additional_datasets=None):
    """
    Affichage de l'intégralité du dataset principal avec la possibilité d'ajouter d'autres courbes de prix.

    :param dataset: DataFrame principal contenant les données de prix.
    :param additional_datasets: Liste de DataFrames supplémentaires à ajouter au graphique.
    """
    # Création de la figure avec la première courbe de prix
    fig = go.Figure()

    # Ajout de la première courbe de prix
    fig.add_trace(go.Scatter(
        x=dataset['Date'],
        y=dataset['Dernier'],
        mode='lines',
        name='Bitcoin',
        line=dict(color='orange', width=2)
    ))

    # Ajout des courbes supplémentaires si fournies
    if additional_datasets:
        for i, additional_dataset in enumerate(additional_datasets, start=2):
            fig.add_trace(go.Scatter(
                x=additional_dataset['Date'],
                y=additional_dataset['Dernier'],
                mode='lines',
                name=f'Courbe {i}',
                line=dict(width=2)
            ))

    # Mise à jour de la mise en page
    fig.update_layout(
        title_text='Whole period of timeframe of Bitcoin close price 2014-2025',
        plot_bgcolor='white',
        font_size=15,
        font_color='black',
        xaxis_title='Date',
        yaxis_title='Close Price'
    )

    # Suppression des grilles
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Affichage du graphique
    fig.show()



import pandas as pd

# Exemple de données pour Bitcoin
data_bitcoin = {
    'Date': pd.date_range(start='2020-01-01', periods=10, freq='D'),
    'Dernier': [30000, 31000, 30500, 32000, 31500, 33000, 32500, 34000, 33500, 35000]
}
df_bitcoin = pd.DataFrame(data_bitcoin)

# Exemple de données pour Ethereum
data_ethereum = {
    'Date': pd.date_range(start='2020-01-01', periods=10, freq='D'),
    'Dernier': [1000, 1050, 1020, 1100, 1080, 1150, 1120, 1200, 1180, 1250]
}
df_ethereum = pd.DataFrame(data_ethereum)

# Exemple de données pour Litecoin
data_litecoin = {
    'Date': pd.date_range(start='2020-01-01', periods=10, freq='D'),
    'Dernier': [100, 105, 102, 110, 108, 115, 112, 120, 118, 125]
}
df_litecoin = pd.DataFrame(data_litecoin)

# Appel de la méthode pour afficher les courbes
display_all_dataset(df_bitcoin, additional_datasets=[df_ethereum, df_litecoin])






# Test :
display_all_dataset(tmp_dataset, additional_datasets=[df_ethereum, df_litecoin])








