import os
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import parameters
from itertools import cycle
from BO.prepare_dataset import PrepareDataset


# Paramètres
DATASET_PATH = parameters.DATASET_PATH
PATH_TRAINING_DATASET = parameters.PATH_TRAINING_DATASET
DATASET_FILE = parameters.DATASET_FILE
DATASET_FOR_MODEL = parameters.DATASET_FOR_MODEL


# Fonctions pour les indicateurs techniques
def ma(df, n):
    return pd.Series(df['Dernier'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))


def rsi(df, period):
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
    sma1_col = 'MA_' + str(taille_sma1)
    sma2_col = 'MA_' + str(taille_sma2)
    signal_col = sma1_col + '_supérieure_' + sma2_col
    dataset[sma1_col] = ma(dataset, taille_sma1)
    dataset[sma2_col] = ma(dataset, taille_sma2)
    dataset[signal_col] = np.where(dataset[sma1_col] > dataset[sma2_col], 1.0, 0.0)
    return dataset


# Fonctions pour la préparation des données
def format_dataset(initial_dataset):
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
    tmp_dataset = tmp_dataset.drop(columns=['Vol.', 'Variation %', 'Ouv.', ' Plus Haut', 'Plus Bas'])
    return tmp_dataset


def add_technicals_indicators(tmp_dataset):
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
    training_size = int(len(model_dataset) * 0.60)
    train_data, test_data = model_dataset.iloc[0:training_size, :], model_dataset.iloc[training_size:len(model_dataset), :]
    return train_data, test_data











# Exécution du script principal
prepare_dataset = PrepareDataset()

print(" ************ Etape 1 : Loading dataset ************ ")
initial_dataset = pd.read_csv(DATASET_PATH + DATASET_FILE)

print(" ************ Etape 2 : Preparation of the Dataset ************ ")
tmp_dataset = format_dataset(initial_dataset)
tmp_dataset = delete_columns(tmp_dataset)
tmp_dataset = prepare_dataset.add_technicals_indicators(tmp_dataset)

tmp_dataset_copy = tmp_dataset.copy()
columns_to_normalize = ['Dernier', 'MA_150', 'MA_100', 'MA_50', 'MA_50_supérieure_MA_150', 'MA_100_supérieure_MA_150', 'MA_50_supérieure_MA_100']
scaler = prepare_dataset.get_fitted_scaler(tmp_dataset_copy[columns_to_normalize])
joblib.dump(scaler, 'scaler.save')
model_dataset = tmp_dataset

print("dataset")
normalized_datas = prepare_dataset.normalize_datas(tmp_dataset_copy[columns_to_normalize], scaler)
model_dataset[columns_to_normalize] = normalized_datas
print("dataset d'entrainement normalisé :", model_dataset)
print("model_dataset shape : ", model_dataset.shape)

model_dataset.to_csv(PATH_TRAINING_DATASET + 'dataset_modified_with_date.csv', index=False)

train_data, test_data = create_train_and_test_dataset(model_dataset)

print("train_data type  : ", type(train_data))
print("test_data type  : ", type(test_data))
print("COLONNES DE train_data:", train_data.columns.tolist())
print("COLONNES DE test_data:", test_data.columns.tolist())

# Préparer les données pour le Gradient Boosting
X_train = train_data.drop(columns=['Date']).values
y_train = train_data['Dernier'].values
X_test = test_data.drop(columns=['Date']).values
y_test = test_data['Dernier'].values

# Création du modèle Gradient Boosting
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Évaluation du modèle
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calcul des métriques
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
print(f"Train R2: {train_r2}")
print(f"Test R2: {test_r2}")

# Sauvegarde du modèle
joblib.dump(model, 'gradient_boosting_model.pkl')

# Visualisation des résultats
plt.figure(figsize=(14, 7))
plt.plot(train_data['Date'], y_train, label='Train Actual')
plt.plot(train_data['Date'], y_train_pred, label='Train Predicted', alpha=0.7)
plt.plot(test_data['Date'], y_test, label='Test Actual')
plt.plot(test_data['Date'], y_test_pred, label='Test Predicted', alpha=0.7)
plt.title('Gradient Boosting Regression Results')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
