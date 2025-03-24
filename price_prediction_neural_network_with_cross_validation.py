import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
# For model building :
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score, mean_gamma_deviance, mean_poisson_deviance
# For Plotting :
import plotly.express as px
import parameters
from sklearn.model_selection import TimeSeriesSplit
from BO.prepare_dataset import PrepareDataset
import joblib





""" ****************************** Paramètres ****************************** """
DATASET_PATH = parameters.DATASET_PATH
PATH_TRAINING_DATASET = parameters.PATH_TRAINING_DATASET
TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE
DATASET_FOR_MODEL = parameters.DATASET_FOR_MODEL
SAVE_MODEL_PATH = parameters.SAVE_MODEL_PATH
MODEL_FOR_PREDICTIONS_PATH = parameters.MODEL_FOR_PREDICTIONS_PATH

"""
DATASET_PATH = parameters.DATASET_PATH
PATH_TRAINING_DATASET = parameters.PATH_TRAINING_DATASET
TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE
DATASET_FOR_MODEL = parameters.DATASET_FOR_MODEL

"""




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
    u[u.index[period-1]] = np.mean(u[:period])  # first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean(d[:period])  # first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = u.ewm(com=period-1, adjust=False).mean() / d.ewm(com=period-1, adjust=False).mean()
    return 100 - 100 / (1 + rs)


def calculate_signal(dataset, taille_sma1, taille_sma2):
    """ Calcul des signaux de croisement des moyennes mobiles """
    sma1_col = 'MA_' + str(taille_sma1)
    sma2_col = 'MA_' + str(taille_sma2)
    signal_col = 'signal_' + sma1_col + '_' + sma2_col
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


def get_fitted_scaler(tmp_dataset):
    """ Méthode pour obtenir le scaler ajusté """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(tmp_dataset)
    return scaler


def normalize_datas(tmp_dataset, scaler):
    """ Méthode normalize_data() """
    return scaler.transform(tmp_dataset)


def create_train_and_test_dataset(model_dataset):
    """ Création des datasets d'entrainement et tests """
    # Création des datasets d'entrainement et de test
    training_size = int(len(model_dataset) * 0.60)
    test_size = len(model_dataset) - training_size
    train_data, test_data = model_dataset[0:training_size, :], model_dataset[training_size:len(model_dataset), :1]
    print("dataset d'entrainement :", train_data.shape)
    print("dataset de test :", test_data.shape)
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


def create_data_matrix(model_dataset, time_step=15):
    """ Création des matrices pour les datasets d'entrainement et test """
    # Création des ensembles de données en utilisant la fonction create_dataset :
    x, y = create_dataset(model_dataset, time_step)
    # Remodelage de X pour obtenir la forme [échantillons, time steps, caractéristiques]
    # Cela est nécessaire pour que les données soient compatibles avec les couches LSTM :
    x = x.reshape(x.shape[0], x.shape[1], 1)
    # Affichage des dimensions des ensembles de données après remodelage :
    print("dataset x: ", x.shape)
    # On ne modifie pas la forme du dataset y car elle sert de valeur cible à comparer avec le dataset x :
    print("dataset y: ", y.shape)
    return x, y















""" ************************* Préparation du dataset ************************* """

print(" ************ Etape 1 : Loading dataset ************ ")
initial_dataset = pd.read_csv(PATH_TRAINING_DATASET+TRAINING_DATASET_FILE)


print(" ************ Etape 2 : Preparation of the Dataset ************ ")
# Formatage des colonnes :
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


# Contrôle des modifications :
print("En-tête du dataset d'entrainement : ", tmp_dataset.head())
print("dataset d'entrainement modifié (dernières lignes) pour vérifier si mes indicateurs sont bien calculés : ", tmp_dataset.tail())


# Ajout des indicateurs techniques :
prepare_dataset = PrepareDataset()
tmp_dataset = prepare_dataset.add_technicals_indicators(tmp_dataset)
print(" forme tmp_dataset shape : ", tmp_dataset.shape)
print(" forme tmp_dataset : ", tmp_dataset)


# # Contrôle : Enregistrement du dataset au format csv :
tmp_dataset.to_csv(PATH_TRAINING_DATASET+DATASET_FOR_MODEL, index=False)


# Normalisation des colonnes :
tmp_dataset_copy = tmp_dataset.copy()
columns_to_normalize = ['Dernier', 'MA_150', 'MA_100', 'MA_50', 'MA_50_supérieure_MA_150', 'MA_100_supérieure_MA_150', 'MA_50_supérieure_MA_100']
prepare_dataset = PrepareDataset()
scaler = prepare_dataset.get_fitted_scaler(tmp_dataset_copy[columns_to_normalize])
joblib.dump(scaler, 'scaler.save')
model_dataset = tmp_dataset
print("dataset")
normalized_datas = prepare_dataset.normalize_datas(tmp_dataset_copy[columns_to_normalize], scaler)
model_dataset[columns_to_normalize] = normalized_datas
print("dataset d'entrainement normalisé :", model_dataset)
print("model_dataset shape : ", model_dataset.shape)


# Contrôle : Sauvegarde du dataset retraité pour traitement par le modèle :
model_dataset.to_csv(PATH_TRAINING_DATASET+'dataset_modified_with_date.csv', index=False)
print("MODEL DATASET  : ", type(model_dataset))
print("COLONNES DE MODEL DATASET :", tmp_dataset.columns.tolist())


# Suppression de la colonne 'Date' du dataset :
date_column = model_dataset['Date']
del tmp_dataset['Date']
model_dataset.to_csv(PATH_TRAINING_DATASET+'dataset_modified_for_model.csv', index=False)
print("model_dataset shape juste avant traitement : ", model_dataset.shape)
print("COLONNES DE MODEL DATASET :", tmp_dataset.columns.tolist())


# Création des matrices de données pour l'entraînement et le test :
x_train, y_train = create_data_matrix(model_dataset)
x_test, y_test = create_data_matrix(model_dataset)


print(" ************ Etape 3 : Create and train model ************ ")
# Initialisation du TimeSeriesSplit avec le nombre de splits souhaité :
tscv = TimeSeriesSplit(n_splits=5)


# Initialiser des listes pour stocker les résultats et les métriques
results = []
rmse_results = []
mse_results = []
mae_results = []
evs_results = []
r2_results = []
mgd_results = []
mpd_results = []
training_loss_results = []
validation_loss_results = []


# Initialisation du compteur :
cpt = 1














""" ************************* Définition du modèle ************************* """

for train_index, val_index in tscv.split(x_train):

    """ Entrainement du modèle """
    print("n° du tour de boucle : ", cpt)

    x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    print("shape of datasets : ")
    print("x_train_fold : ", x_train_fold.shape)
    print("y_train_fold : ", y_train_fold.shape)
    print("x_val_fold : ", x_val_fold.shape)
    print("y_val_fold : ", y_val_fold.shape)

    model = Sequential()
    model.add(LSTM(50, input_shape=(x_train_fold.shape[1], 1), activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_regularizer=l2(0.01)))
    model.compile(loss="mean_squared_error", optimizer="adam")

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        x_train_fold, y_train_fold,
        validation_data=(x_val_fold, y_val_fold),
        epochs=200,
        batch_size=32,
        verbose=1,
        callbacks=[early_stopping]
    )


    # Enregistrement des poids du modèle :
    print("Enregistrement du modèle.")
    model.save_weights(SAVE_MODEL_PATH+f'best_model_weights{cpt}.weights.h5')


    # Évaluation du modèle sur les données de validation :
    val_loss = model.evaluate(x_val_fold, y_val_fold, verbose=0)
    results.append(val_loss)


    # Prédiction et évaluation des métriques de performance :
    val_predict = model.predict(x_val_fold)


    # Redimensionner les données pour qu'elles aient deux dimensions :
    y_val_fold_reshaped = y_val_fold.reshape(-1, 1)
    print("y_val_fold_reshaped : ", y_val_fold_reshaped.shape)
    val_predict_reshaped = val_predict.reshape(-1, 1)
    print("val_predict_reshaped : ", val_predict_reshaped.shape)


    # Ajustement du scaler sur les données redimensionnées :
    scaler.fit(y_val_fold_reshaped)


    # Inversion de la transformation :
    original_yval = scaler.inverse_transform(y_val_fold_reshaped)
    val_predict_inversed = scaler.inverse_transform(val_predict_reshaped)


    # Calcul des métriques :
    rmse = np.sqrt(mean_squared_error(original_yval, val_predict_inversed))
    mse = mean_squared_error(original_yval, val_predict_inversed)
    mae = mean_absolute_error(original_yval, val_predict_inversed)
    evs = explained_variance_score(original_yval, val_predict_inversed)
    r2 = r2_score(original_yval, val_predict_inversed)


    # Vérifier si les valeurs sont strictement positives avant de calculer la déviance gamma et la déviance de Poisson :
    if np.all(original_yval > 0) and np.all(val_predict_inversed > 0):
        mgd = mean_gamma_deviance(original_yval, val_predict_inversed)
        mpd = mean_poisson_deviance(original_yval, val_predict_inversed)
    else:
        mgd, mpd = np.nan, np.nan


    # Ajout des résultats aux listes :
    rmse_results.append(rmse)
    mse_results.append(mse)
    mae_results.append(mae)
    evs_results.append(evs)
    r2_results.append(r2)
    mgd_results.append(mgd)
    mpd_results.append(mpd)


    # Incrément du compteur de tours de boucle
    cpt += 1

















""" ************************* Affichage des résultats ************************* """

# Conversion des listes en arrays numpy
rmse_results = np.array(rmse_results)
mse_results = np.array(mse_results)
mae_results = np.array(mae_results)
evs_results = np.array(evs_results)
r2_results = np.array(r2_results)
mgd_results = np.array(mgd_results)
mpd_results = np.array(mpd_results)
training_loss_results = np.array(training_loss_results)
validation_loss_results = np.array(validation_loss_results)


# Affichage des résultats :
print("Validation RMSE: ", rmse_results)
print("Validation MSE: ", mse_results)
print("Validation MAE: ", mae_results)
print("Validation Explained Variance Score: ", evs_results)
print("Validation R2 Score: ", r2_results)
print("Validation MGD: ", mgd_results)
print("Validation MPD: ", mpd_results)
print("Validation Loss for each fold: ", validation_loss_results)
print("Mean Validation Loss: ", np.mean(validation_loss_results))
print("Training Loss for each fold: ", training_loss_results)
print("Mean Training Loss: ", np.mean(training_loss_results))


# Vérification de l'existence du fichier :
weights_path = os.path.join(MODEL_FOR_PREDICTIONS_PATH, 'model.weights.h5')
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Le fichier de poids n'existe pas à l'emplacement spécifié : {weights_path}")


# Chargement des poids :
model.load_weights(weights_path)


# Évaluer le modèle sur l'ensemble de test
test_loss = model.evaluate(x_test, y_test, verbose=0)


# Prédire et évaluer les métriques de performance sur l'ensemble de test
test_predict = model.predict(x_test)


# Redimensionner les données pour qu'elles aient deux dimensions :
y_test_reshaped = y_test.reshape(-1, 1)
test_predict_reshaped = test_predict.reshape(-1, 1)


# Ajuster le scaler sur les données redimensionnées :
scaler.fit(y_test_reshaped)


# Inverser la transformation :
original_ytest = scaler.inverse_transform(y_test_reshaped)
test_predict_inversed = scaler.inverse_transform(test_predict_reshaped)


# Calcul des métriques :
rmse_test = np.sqrt(mean_squared_error(original_ytest, test_predict_inversed))
mse_test = mean_squared_error(original_ytest, test_predict_inversed)
mae_test = mean_absolute_error(original_ytest, test_predict_inversed)
evs_test = explained_variance_score(original_ytest, test_predict_inversed)
r2_test = r2_score(original_ytest, test_predict_inversed)


# Vérifier si les valeurs sont strictement positives avant de calculer la déviance gamma et la déviance de Poisson
if np.all(original_ytest > 0) and np.all(test_predict_inversed > 0):
    mgd_test = mean_gamma_deviance(original_ytest, test_predict_inversed)
    mpd_test = mean_poisson_deviance(original_ytest, test_predict_inversed)
else:
    mgd_test, mpd_test = np.nan, np.nan


# Affichage des résultats
print("Test RMSE: ", rmse_test)
print("Test MSE: ", mse_test)
print("Test MAE: ", mae_test)
print("Test Explained Variance Score: ", evs_test)
print("Test R2 Score: ", r2_test)
print("Test MGD: ", mgd_test)
print("Test MPD: ", mpd_test)

















































" ******************** Evaluation de la fonction de perte / du sur-entrainement ******************** "
print(" ************** Etape 5 : Evaluation de la fonction de perte / du sur-entrainement ************* ")
"""
print("Training loss 1 : ")
# Exemple de pertes d'entraînement (à remplir avec vos valeurs réelles)
training_loss = [0.001, 0.0005, 0.0003, 0.0002, 0.0001]

# Pertes de validation
validation_loss = [0.0020137971732765436, 0.0030710133723914623, 0.0003042859607376158, 0.0002075933152809739, 0.00011482657282613218]

# Tracer les pertes
plt.plot(training_loss_results, label='Training Loss')
plt.plot(validation_loss_results, label='Validation Loss')
plt.xlabel('Fold')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Comparison')
plt.legend()
plt.show()
"""





" ******************** Evaluation Globale du modèle / Métriques ******************** "

print(" ******************** Evaluation Globale du modèle / Métriques ******************** ")






