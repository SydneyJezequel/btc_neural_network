import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
import joblib
import parameters
from BO.prepare_dataset import PrepareDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance
from sklearn.model_selection import GridSearchCV












""" ****************************** Paramètres ****************************** """
DATASET_PATH = parameters.DATASET_PATH
PATH_TRAINING_DATASET = parameters.PATH_TRAINING_DATASET
TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE
# DATASET_FILE = parameters.DATASET_FILE
DATASET_FOR_MODEL = parameters.DATASET_FOR_MODEL
SAVE_MODEL_PATH = parameters.SAVE_MODEL_PATH









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
    """ Suppression des colonnes du dataset d'origine """
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


def add_lag_features(dataset, lags):
    """ Ajout des caractéristiques de lag """
    for lag in lags:
        dataset[f'Lag_{lag}'] = dataset['Dernier'].shift(lag)
    return dataset


def create_train_and_test_dataset(model_dataset):
    """ Création des datasets d'entrainement et tests """
    training_size = int(len(model_dataset) * 0.60)
    train_data, test_data = model_dataset.iloc[0:training_size, :], model_dataset.iloc[training_size:len(model_dataset), :]
    return train_data, test_data















""" ************************* Préparation du dataset ************************* """

prepare_dataset = PrepareDataset()


initial_dataset = pd.read_csv(PATH_TRAINING_DATASET + TRAINING_DATASET_FILE)


# Formatage du dataset :
tmp_dataset = prepare_dataset.format_dataset(initial_dataset)


# Suppression des colonnes du dataset initial :
tmp_dataset = prepare_dataset.delete_columns(tmp_dataset)


# Ajout des indicateurs techniques et signaux :
tmp_dataset = prepare_dataset.add_technicals_indicators(tmp_dataset)


# Ajout des caractéristiques de lag :
lags = [1, 7, 30]  # Lag pour 1 jour, 1 semaine, 1 mois
tmp_dataset = prepare_dataset.add_lag_features(tmp_dataset, lags)
tmp_dataset = tmp_dataset.dropna()  # Supprimer les lignes avec des valeurs NaN introduites par les caractéristiques de lag


# Normalisation du dataset :
tmp_dataset_copy = tmp_dataset.copy()
columns_to_normalize = ['Dernier', 'MA_150', 'MA_100', 'MA_50', 'MA_50_supérieure_MA_150', 'MA_100_supérieure_MA_150', 'MA_50_supérieure_MA_100'] + [f'Lag_{lag}' for lag in lags]
scaler = prepare_dataset.get_fitted_scaler(tmp_dataset_copy[columns_to_normalize])
joblib.dump(scaler, '../../scaler.save')
model_dataset = tmp_dataset
print("dataset")
normalized_datas = prepare_dataset.normalize_datas(tmp_dataset_copy[columns_to_normalize], scaler)
model_dataset[columns_to_normalize] = normalized_datas
print("dataset d'entrainement normalisé :", model_dataset)
print("model_dataset shape : ", model_dataset.shape)


# Contrôle : Sauvegarde du dataset :
model_dataset.to_csv(PATH_TRAINING_DATASET + 'dataset_modified_with_date.csv', index=False)


# Création des datasets d'entrainement et de test pour le modèle :
train_data, test_data = prepare_dataset.create_train_and_test_dataset(model_dataset)
print("train_data type  : ", type(train_data))
print("test_data type  : ", type(test_data))
print("COLONNES DE train_data:", train_data.columns.tolist())
print("COLONNES DE test_data:", test_data.columns.tolist())
X_train = train_data.drop(columns=['Date']).values
y_train = train_data['Dernier'].values
X_test = test_data.drop(columns=['Date']).values
y_test = test_data['Dernier'].values














""" ************************* Définition du modèle ************************* """

# Définir les paramètres pour la recherche de grille
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0],
    'min_samples_split': [2, 5, 10]
}


# Utiliser GridSearchCV pour trouver les meilleurs hyperparamètres
grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
                          param_grid=param_grid,
                          scoring='neg_mean_squared_error',
                          cv=3,
                          verbose=1,
                          n_jobs=-1)















""" ************************* Entrainement du modèle ************************* """

grid_search.fit(X_train, y_train)


# Meilleurs paramètres trouvés :
best_params = grid_search.best_params_
print("Meilleurs paramètres trouvés :", best_params)


# Entraîner le modèle avec les meilleurs paramètres :
model = grid_search.best_estimator_


# Sauvegarde du modèle
joblib.dump(model, SAVE_MODEL_PATH+'gradient_boosting_model.pkl')

















""" ************************* Affichage des résultats ************************* """

# Génération des prédictions  :
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# Mise en forme des prédictions :
train_predict = train_predict.reshape(-1, 1)
test_predict = test_predict.reshape(-1, 1)
scaler = prepare_dataset.get_fitted_scaler(train_predict)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))


# Calcul des métriques :
train_rmse = np.sqrt(mean_squared_error(original_ytrain, train_predict))
test_rmse = np.sqrt(mean_squared_error(original_ytest, test_predict))
train_mse = mean_squared_error(original_ytrain, train_predict)
test_mse = mean_squared_error(original_ytest, test_predict)
train_mae = mean_absolute_error(original_ytrain, train_predict)
test_mae = mean_absolute_error(original_ytest, test_predict)
train_evs = explained_variance_score(original_ytrain, train_predict)
test_evs = explained_variance_score(original_ytest, test_predict)
train_r2 = r2_score(original_ytrain, train_predict)
test_r2 = r2_score(original_ytest, test_predict)
train_mgd = mean_gamma_deviance(original_ytrain, train_predict)
test_mgd = mean_gamma_deviance(original_ytest, test_predict)
train_mpd = mean_poisson_deviance(original_ytrain, train_predict)
test_mpd = mean_poisson_deviance(original_ytest, test_predict)


# Affichage des métriques :
print("train_rmse : ", train_rmse)
print("test_rmse : ", test_rmse)
print("train_mse : ", train_mse)
print("test_mse : ", test_mse)
print("train_mae : ", train_mae)
print("test_mae : ", test_mae)
print("train_evs : ", train_evs)
print("test_evs : ", test_evs)
print("train_r2 : ", train_r2)
print("test_r2 : ", test_r2)
print("train_mgd : ", train_mgd)
print("test_mgd : ", test_mgd)
print("train_mpd : ", train_mpd)
print("test_mpd : ", test_mpd)


# Visualisation des résultats
plt.figure(figsize=(14, 7))
plt.plot(train_data['Date'], y_train, label='Train Actual')
plt.plot(train_data['Date'], train_predict, label='Train Predicted', alpha=0.7)
plt.plot(test_data['Date'], y_test, label='Test Actual')
plt.plot(test_data['Date'], test_predict, label='Test Predicted', alpha=0.7)
plt.title('Gradient Boosting Regression Results')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
