import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import parameters
from service.prepare_dataset_service import PrepareDatasetService
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance
from sklearn.model_selection import GridSearchCV




""" ************* Paramètres ************* """

DATASET_PATH = parameters.DATASET_PATH
TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE
GRADIENT_BOOSTING_SAVED_MODEL = parameters.GRADIENT_BOOSTING_SAVED_MODEL
MODEL_PATH = parameters.MODEL_PATH




""" ************* Préparation du dataset ************* """

prepare_dataset = PrepareDatasetService()

# Chargement du dataset :
initial_dataset = pd.read_csv(TRAINING_DATASET_FILE)

# Formatage du dataset :
tmp_dataset = prepare_dataset.format_dataset(initial_dataset)
tmp_dataset = prepare_dataset.delete_columns(tmp_dataset)

# Ajout des caractéristiques de lag :
lags = [1, 7, 30]  # Lag pour 1 jour, 1 semaine, 1 mois
tmp_dataset = prepare_dataset.add_lag_features(tmp_dataset, lags)
tmp_dataset = tmp_dataset.dropna()

# Normalisation du dataset :
tmp_dataset_copy = tmp_dataset.copy()
columns_to_normalize = ['Dernier'] + [f'Lag_{lag}' for lag in lags]
scaler = prepare_dataset.get_fitted_scaler(tmp_dataset_copy[columns_to_normalize])
joblib.dump(scaler, '../../scaler.save')
model_dataset = tmp_dataset
normalized_datas = prepare_dataset.normalize_datas(tmp_dataset_copy[columns_to_normalize], scaler)
model_dataset[columns_to_normalize] = normalized_datas
print("dataset d'entrainement normalisé :", model_dataset)
print("model_dataset shape : ", model_dataset.shape)

# Contrôle : Sauvegarde du dataset :
model_dataset.to_csv(DATASET_PATH + 'dataset_modified_with_date.csv', index=False)

# Création des datasets d'entrainement et de test pour le modèle :
train_data, test_data = prepare_dataset.create_train_and_test_dataset(model_dataset)
X_train = train_data.drop(columns=['Date']).values
y_train = train_data['Dernier'].values
X_test = test_data.drop(columns=['Date']).values
y_test = test_data['Dernier'].values




""" ************* Définition du modèle ************* """

# Définir les paramètres pour la recherche de grille :
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0],
    'min_samples_split': [2, 5, 10]
}

# Utiliser GridSearchCV pour trouver les meilleurs hyperparamètres :
grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
                          param_grid=param_grid,
                          scoring='neg_mean_squared_error',
                          cv=3,
                          verbose=1,
                          n_jobs=-1)




""" ************* Entrainement du modèle ************* """

grid_search.fit(X_train, y_train)

# Meilleurs paramètres trouvés :
best_params = grid_search.best_params_
print("Meilleurs paramètres trouvés :", best_params)

# Entraîner le modèle avec les meilleurs paramètres :
model = grid_search.best_estimator_

# Sauvegarde du modèle
joblib.dump(model, MODEL_PATH +'gradient_boosting_model.pkl')




""" ************* Affichage des résultats ************* """

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

# Visualisation des résultats :
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
