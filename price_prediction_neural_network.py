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

# Affichage des données :
fig = px.line(tmp_dataset, x=tmp_dataset.Date, y=tmp_dataset.Dernier,labels={'Date':'date','Dernier':'Close Stock'})
fig.update_traces(marker_line_width=2, opacity=0.8, marker_line_color='orange')
fig.update_layout(title_text='Whole period of timeframe of Bitcoin close price 2014-2025', plot_bgcolor='white',
                  font_size=15, font_color='black')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# Ajout des indicateurs techniques :
add_technicals_indicators(tmp_dataset)

# Enregistrement du dataset au format csv :
tmp_dataset.to_csv(PATH_TRAINING_DATASET+DATASET_FOR_MODEL, index=False)

# Contrôle des modifications :
print("En-tête du dataset d'entrainement : ", tmp_dataset.head())
print("dataset d'entrainement modifié (dernières lignes) pour vérifier si mes indicateurs sont bien calculés : ", tmp_dataset.tail())

# Normalise dataset :
model_dataset = normalize_datas(tmp_dataset)
print("dataset d'entrainement normalisé :", model_dataset)

# Méthode create_dataset :
train_data, test_data = create_train_and_test_dataset(model_dataset)

# Conversion des arrays en matrice :
x_train, y_train, x_test, y_test = create_data_matrix(train_data, test_data, create_dataset)

# NOTION DE VALEUR CIBLE :
"""
Une valeur cible, également appelée étiquette ou label, est la valeur que l'on souhaite prédire ou prévoir
dans un problème de machine learning. Elle représente l'output ou la sortie attendue pour une entrée donnée.
Les valeurs cibles sont utilisées
pour entraîner les modèles de machine learning en leur fournissant des exemples de ce qu'ils doivent
apprendre à prédire.
"""

print("NaN in X_train:", np.isnan(x_train).sum())
print("NaN in y_train:", np.isnan(y_train).sum())
print("NaN in X_test:", np.isnan(x_test).sum())
print("NaN in y_test:", np.isnan(y_test).sum())

print('En-têtes du dataset: ', initial_dataset.head())
print('initial_dataset.tail(): ', initial_dataset.tail())




































""" ************************************ Créer une classe qui encapsule le modèle (Réseau de neurones) ************************************ """
""" NOM DE LA CLASSE : neural_network """

print(" ******************** Création et entrainement du modèle ******************** ")

""" 
----------------------------------------------------------------
# Ajout d'une couche LSTM (Long Short-Term Memory) au modèle :
# - 10 unités LSTM
# - input_shape=(None, 1) : La couche LSTM a une sortie de forme (None, 10), ce qui signifie qu'elle produit 10 valeurs pour chaque séquence d'entrée.
# - activation="relu" : Fonction d'activation ReLU (Rectified Linear Unit)
# :  La couche dense a une sortie de forme (None, 1), ce qui signifie qu'elle produit une seule valeur de sortie pour chaque séquence d'entrée.
----------------------------------------------------------------
"""


# Création du modèle :
# Initialisation d'un modèle séquentiel :
model=Sequential()
model.add(LSTM(10,input_shape=(None,1), activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1))
# model.add(Dense(1), kernel_regularizer=l2(0.01)) ==> NE FONCTIONNE PAS.
model.compile(loss="mean_squared_error", optimizer="adam")


# Exécution du modèle :
# Compilation du modèle :
# - loss="mean_squared_error" : Utilisation de la moyenne des erreurs quadratiques comme fonction de perte (écart prévision/résultat).
# - optimizer="adam" : Utilisation de l'optimiseur Adam pour gérer la descente de gradient (algorithme d'optimisation utilisé pour minimiser la fonction de perte. Elle ajuste les paramètres du modèle (comme les poids et les biais dans un réseau de neurones) de manière itérative pour réduire la valeur de la fonction de perte.).
# model.compile(loss="mean_squared_error",optimizer="adam")


# Entrainement du modèle :
# - X_train, y_train : données et cibles d'entraînement.
# - validation_data=(X_test, y_test) : données et cibles de validation.
# - epochs=200 : nombre d'époques (passages complets sur l'ensemble des données d'entraînement).
# - batch_size=32 : taille des lots (batch size) pour l'entraînement.
# - verbose=1 : mode verbeux pour afficher les informations de progression pendant l'entraînement.
history = model.fit(
    x_train,y_train,
    validation_data=(x_test,y_test),
    epochs=10,
    batch_size=32,
    verbose=1
)

# Sauvegarde du modèle :
model.save_weights(parameters.SAVE_MODEL_PATH + f'best_model_weights_without_cross_validation.weights.h5')








"""
----------------------------------------------------------------
==> COMMENT EVALUER LES METRIQUES :
----------------------------------------------------------------

Plus je vais avancer dans mon entrainement, au fur et à mesure des folds,
plus les performances doivent évoluer comme suit.
Néanmoins, il faut l'amélioration des métriques n'est pas toujours linéaire et peut varier 
en fonction de plusieurs facteurs, notamment la complexité du modèle, 
la qualité des données, et la configuration de l'entraînement.

- Validation RMSE (Root Mean Squared Error) / Mesure l'écart moyen entre les prédictions et les valeurs réelles :
Plus l'entrainement va progresser au fur et à mesure des folds, plus la valeur sera basse.
Mais il est possible que la RMSE atteigne un plateau ou même augmente légèrement 
si le modèle commence à surapprendre.

- Validation MSE (Mean Squared Error) / Mesure la moyenne des carrés des erreurs :
Plus l'entrainement va progresser au fur et à mesure des folds, plus la valeur sera basse.
Mais il peut aussi atteindre un plateau ou augmenter en cas de surapprentissage.

- Validation MAE (Mean Absolute Error) / Mesure l'erreur absolue moyenne entre les prédictions et les valeurs réelles : 
Plus l'entrainement va progresser au fur et à mesure des folds, plus la valeur sera basse.
Mais il peut aussi atteindre un plateau ou augmenter en cas de surapprentissage.

- Validation Explained Variance Score / Mesure la proportion de la variance dans les valeurs réelles qui est expliquée par le modèle : 
Plus l'entrainement va progresser au fur et à mesure des folds, plus la valeur sera proche de 1.
Mais il peut aussi atteindre un plateau puis diminuer en cas de surraprentissage.

- Validation R2 Score / Mesure la proportion de la variance dans les valeurs réelles qui est prédite par le modèle : 
Plus l'entrainement va progresser au fur et à mesure des folds, plus la valeur sera proche de 1.
Mais il peut aussi atteindre un plateau puis diminuer en cas de surraprentissage.

- Validation MGD (Mean Gamma Deviance) : Mesure la déviance moyenne pour une distribution gamma : 
Plus l'entrainement va progresser au fur et à mesure des folds, plus la valeur sera basse.
Mais il peut aussi atteindre un plateau ou augmenter en cas de surapprentissage.

- Validation MPD (Mean Poisson Deviance) / Mesure la déviance moyenne pour une distribution de Poisson : 
Plus l'entrainement va progresser au fur et à mesure des folds, plus la valeur sera basse.



----------------------------------------------------------------
==> POINTS A CONSIDERER :
----------------------------------------------------------------
Points à Considerer
- SURAPPRENTISSAGE : Si le modèle commence à surapprendre, les métriques de performance 
sur les données de validation  peuvent se dégrader (augmenter pour les erreurs, 
diminuer pour les scores de variance et R²). C'est pourquoi il est important de surveiller 
ces métriques et d'utiliser des techniques comme l'arrêt précoce (early stopping) 
pour éviter le surapprentissage.

- PLATEAU : Les métriques peuvent atteindre un plateau, où elles cessent de s'améliorer même 
avec un entraînement supplémentaire. Cela peut indiquer que le modèle a atteint ses limites 
de performance avec les données et la configuration actuelles.

- VARIABILITE : Les métriques peuvent varier d'un fold à l'autre en raison de la variabilité des données. 
C'est pourquoi la validation croisée est utile pour obtenir une estimation plus robuste de la performance du modèle.
"""










print(" ******************** Evaluation du sur-apprentissage ******************** ")

"""
==> COMMENT ANALYSER CE GRAPHE ?
Plus les 2 courbes se suivent : Moins il y a d'overfitting.
Si la perte d'entrainement continue à diminuer alors que la perte de validation augmente, cela 
signifie que le modèle commence à surajuster les données d'entrainement).
Si les 2 courbes suivent une tendance similaires, cela signifie que le modèle est moins susceptible de
surajuster.
"""

loss = history.history['loss']
val_loss = history.history['val_loss']
print("*************** DEBUG ****************")
print("history : ", history)
print("loss : ", loss)
print("val_loss : ", val_loss)
print("*************** DEBUG ****************")
epochs = range(len(loss))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()
plt.show()






print(" ******************** Génération de prédiction par le modèle ******************** ")

train_predict=model.predict(x_train)
test_predict=model.predict(x_test)
train_predict.shape, test_predict.shape



""" ************************************ Créer une classe qui encapsule le modèle (Réseau de neurones) ************************************ """








































""" ************************************ Créer une classe qui évalue le modèle (Réseau de neurones) ************************************ """
""" NOM DE LA CLASSE : evaluate_neural_network_training """
print(" ******************** Evaluation du modèle ******************** ")


# Les datasets d'entrainements et de tests sont ramenées à leur échelle d'origine :
scaler = MinMaxScaler(feature_range=(0, 1))
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Inverse la transformation des valeurs cibles d'entraînement pour les ramener à leur échelle d'origine
# Note : y_train et t_train sont remodelées en matrices colonnes pour correspondre à la forme attendue par inverse_transform :
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 







""" ******************** Evaluation des metrics RMSE and MAE******************** """
print(" ******************** Evaluation des metrics RMSE and MAE ******************** ")

"""
-------------------------------------------------------------------------------
==> COMMENT INTERPRETER LES RESULTATS ?

Plus les résultats des la RMSE, MSE et MAE seront proches entre les données de test et d'entrainement :
Moins il y aura de surajustement. Une grande disparité (RMSE, MSE et MAE fortes pour les données 
d'entrainement et faible pour les données de test) indique que le modèle réagit bien aux données 
d'entrainement et moins aux données de test.
-------------------------------------------------------------------------------
"""




# Calcule et affiche la RMSE (Root Mean Squared Error) pour les données d'entraînement.
# La RMSE ou racine carrée de la moyenne des carrés des erreurs.
# Elle donne une idée de la magnitude des erreurs.
# Une RMSE plus faible indique un meilleur ajustement du modèle aux données :
print("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain, train_predict)))

# Calcule et affiche la MSE (Mean Squared Error) pour les données d'entraînement.
# La MSE est la moyenne des carrés des erreurs.
# Elle pénalise plus les grandes erreurs que les petites.
# Une MSE plus faible indique un meilleur ajustement du modèle aux données :
print("Train data MSE: ", mean_squared_error(original_ytrain, train_predict))

# Calcule et affiche la MAE (Mean Absolute Error) pour les données d'entraînement.
# La MAE est la moyenne des valeurs absolues des erreurs.
# Elle est moins sensible aux grandes erreurs que la MSE.
# Une MAE plus faible indique un meilleur ajustement du modèle aux données :
print("Train data MAE: ", mean_absolute_error(original_ytrain, train_predict))

print("-------------------------------------------------------------------------------------")

# Calcule et affiche la RMSE (Root Mean Squared Error) pour les données de test
# La RMSE est la racine carrée de la moyenne des carrés des erreurs.
# Elle donne une idée de la magnitude des erreurs.
# Une RMSE plus faible indique un meilleur ajustement du modèle aux données.
print("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest, test_predict)))

# Calcule et affiche la MSE (Mean Squared Error) pour les données de test
# La MSE est la moyenne des carrés des erreurs.
# Elle pénalise plus les grandes erreurs que les petites.
# Une MSE plus faible indique un meilleur ajustement du modèle aux données.
print("Test data MSE: ", mean_squared_error(original_ytest, test_predict))

# Calcule et affiche la MAE (Mean Absolute Error) pour les données de test
# La MAE est la moyenne des valeurs absolues des erreurs.
# Elle est moins sensible aux grandes erreurs que la MSE.
# Une MAE plus faible indique un meilleur ajustement du modèle aux données.
print("Test data MAE: ", mean_absolute_error(original_ytest, test_predict))







""" ******************** Variance Regression Score ******************** """
print(" ******************** Variance Regression Score ******************** ")

# Calcule et affiche le score de régression de la variance expliquée pour les données d'entraînement
# Le score de variance expliquée (explained variance score) mesure la proportion
# de la variance dans les valeurs cibles qui est expliquée par le modèle :
# - Un score proche de 1 indique que le modèle explique bien la variance des données.
# - Un score proche de 0 indique que le modèle n'explique pas bien la variance des données.
print("Train data explained variance regression score:",
explained_variance_score(original_ytrain, train_predict))


# Calcule et affiche le score de régression de la variance expliquée pour les données de test
# Le score de variance expliquée (explained variance score) mesure la proportion
# de la variance dans les valeurs cibles qui est expliquée par le modèle :
# - Un score proche de 1 indique que le modèle explique bien la variance des données.
# - Un score proche de 0 indique que le modèle n'explique pas bien la variance des données.
print("Test data explained variance regression score:",
explained_variance_score(original_ytest, test_predict))







""" ******************** R square score for regression ******************** """
print(" ******************** R square score for regression ******************** ")

# Calcule et affiche le score R² pour les données d'entraînement
# Le score R² (coefficient de détermination) mesure la proportion de la variance 
# dans les valeurs cibles qui est expliquée par le modèle :
# - Un score R² proche de 1 indique que le modèle explique bien la variance des données.
# - Un score R² proche de 0 indique que le modèle n'explique pas bien la variance des données.
# - Un score R² négatif indique que le modèle est pire que la moyenne des valeurs cibles.
print("Train data R2 score:", r2_score(original_ytrain, train_predict))


# Calcule et affiche le score R² pour les données de test
# Le score R² (coefficient de détermination) mesure la proportion de la variance 
# dans les valeurs cibles qui est expliquée par le modèle :
# - Un score R² proche de 1 indique que le modèle explique bien la variance des données.
# - Un score R² proche de 0 indique que le modèle n'explique pas bien la variance des données.
# - Un score R² négatif indique que le modèle est pire que la moyenne des valeurs cibles.
print("Test data R2 score:", r2_score(original_ytest, test_predict))

"""
-------------------------------------------------------------------------------
COMMENT INTERPRETER LES RESULTATS ?

Plus les résultats du R2 seront proches entre les données de test et d'entrainement :
Moins il y aura de surajustement. Une grande disparité (R2 fort pour les données 
d'entrainement et faible pour les données de test) indique que le modèle réagit bien aux données 
d'entrainement et moins aux données de test.
-------------------------------------------------------------------------------
"""







""" ********** Perte de Régression Moyenne Gamma déviance de perte de régression (MGD) et Moyenne Poisson déviance de perte de régression (MPD)  ************** """
print(" ******** Perte de Régression Moyenne Gamma déviance de perte de régression (MGD) et Moyenne Poisson déviance de perte de régression (MPD) ************ ")


# Afficher la perte de régression moyenne Gamma déviance pour les données d'entraînement et de test
print("Données d'entraînement MGD :", mean_gamma_deviance(original_ytrain, train_predict))
print("Données de test MGD :", mean_gamma_deviance(original_ytest, test_predict))
print("----------------------------------------------------------------------")


# Afficher la perte de régression moyenne Poisson déviance pour les données d'entraînement et de test
print("Données d'entraînement MPD :", mean_poisson_deviance(original_ytrain, train_predict))
print("Données de test MPD :", mean_poisson_deviance(original_ytest, test_predict))


# Perte de déviance Gamma :
# La perte de déviance Gamma évalue à quel point votre modèle prédit bien les temps de défaillance observés.
# La perte de déviance Gamma mesure la différence entre la déviance du modèle ajusté et la déviance d'un modèle nul :
# - Modèle ajusté : C'est le modèle que vous avez entraîné sur vos données. Il utilise les variables indépendantes pour prédire la variable dépendante.
# - Déviance du modèle ajusté : C'est la déviance calculée pour ce modèle ajusté, qui mesure à quel point les prédictions du modèle diffèrent des valeurs observées.(un modèle qui prédit simplement la moyenne de la variable dépendante).
# - Modèle nul : C'est un modèle de référence très simple qui ne prend en compte aucune variable indépendante. Il prédit simplement la moyenne de la variable dépendante pour toutes les observations.
# - Déviance d'un modèle nul : C'est la déviance calculée pour ce modèle nul, qui mesure à quel point la moyenne des observations diffère des valeurs observées.

















""" ******************** Comparaison entre le prix original et les prédictions ******************** """
print(" ******************** Comparaison entre le prix original et les prédictions ******************** ")

"""
I- PREDICTION D'ENTRAINEMENT :
# Décaler les prédictions d'entraînement pour le tracé :
look_back = time_step
# Créer un tableau vide de la même forme que closedf pour stocker les prédictions d'entraînement :
trainPredictPlot = np.empty_like(closedf)
# Initialiser toutes les valeurs de trainPredictPlot à NaN :
trainPredictPlot[:, :] = np.nan
# Remplir trainPredictPlot avec les prédictions d'entraînement, en les décalant de 'look_back' positions :
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
# Afficher la forme des données prédites d'entraînement :
print("Données prédites d'entraînement :", trainPredictPlot.shape)



II- PREDICTION DE TEST :
# Créer un tableau vide de la même forme que closedf pour stocker les prédictions de test :
testPredictPlot = np.empty_like(closedf)
# Initialiser toutes les valeurs de testPredictPlot à NaN :
testPredictPlot[:, :] = np.nan
# Remplir testPredictPlot avec les prédictions de test, en les décalant de 'look_back * 2 + 1' positions :
testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(closedf) - 1, :] = test_predict
# Les prédictions d'entraînement (train_predict) sont insérées dans trainPredictPlot, mais décalées de look_back positions. Cela signifie que les prédictions commencent à l'index look_back et se poursuivent jusqu'à la fin des prédictions disponibles.
# Afficher la forme des données prédites de test :
print("Données prédites de test :", testPredictPlot.shape)



III- DEFINIR LE GRAPHIQUE :
# Noms des traces pour le graphique
# Créer un itérateur cyclique pour les noms des traces : Un itérateur cyclique est un outil pratique pour répéter une séquence d'éléments de manière infinie :
names = cycle(['Prix de clôture original', 'Prix de clôture prédit (entraînement)', 'Prix de clôture prédit (test)'])



IV- CREER LE DATAFRAME POUR LE TRACE :
# Créer un DataFrame avec les dates, les prix de clôture originaux, et les prix de clôture prédits (entraînement et test)
plotdf = pd.DataFrame({
    'date': close_stock['Date'],
    'original_close': close_stock['Dernier'],
    'train_predicted_close': trainPredictPlot.reshape(1, -1)[0].tolist(),
    'test_predicted_close': testPredictPlot.reshape(1, -1)[0].tolist()
})
"""
"""
NOTIONS DE DATAFRAME :
Un DataFrame est une structure de données bidimensionnelle, similaire à une table ou un 
tableau dans une base de données relationnelle, ou encore à une feuille de calcul Excel.
Les DataFrames sont couramment utilisés dans l'analyse de données et la science des 
données pour stocker, manipuler et analyser des données structurées.
"""
"""



V- CREER LE GRAPHIQUE :
# Utiliser Plotly Express pour créer un graphique en ligne avec les données du DataFrame
fig = px.line(
    plotdf,
    x=plotdf['date'],
    y=[plotdf['original_close'], plotdf['train_predicted_close'], plotdf['test_predicted_close']],
    labels={'value': 'Prix de l\'action', 'date': 'Date'}
)



VI- METTRE A JOUR LA DISPOSITION DU GRAPHIQUE :
# Mettre à jour le titre, la couleur de fond, la taille et la couleur de la police, et le titre de la légende
fig.update_layout(
    title_text='Comparaison entre le prix de clôture original et le prix de clôture prédit',
    plot_bgcolor='white',
    font_size=15,
    font_color='black',
    legend_title_text='Prix de clôture'
)



VII- METTRE A JOUR LES NOMS DES TRACES :
# Mettre à jour les noms des traces en utilisant l'itérateur cyclique 'names'
fig.for_each_trace(lambda t: t.update(name=next(names)))



VIII- METTRE A JOUR LES AXES :
# Désactiver la grille sur les axes x et y
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
"""






""" ******************** Prédictions des 30 prochains jours ******************** """
print(" ******************** Prédictions des 30 prochains jours ******************** ")

"""
#I- PREPARER LES DONNEES D'ENTREE POUR LA PREDICTION :

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
"""






""" ******************** Prix des 15 derniers jours du jeu de données et des 30 prochains jours prédits ******************** """
print(" ******************** Prix des 15 derniers jours du jeu de données et des 30 prochains jours prédits ******************** ")

"""
# I- CREATION DE TABLEAUX DES JOURS UTILISES ET DES JOURS PREDITS :
# Créer un tableau pour les derniers jours utilisés pour la prédiction :
last_days = np.arange(1, time_step + 1)
# Créer un tableau pour les jours prédits :
day_pred = np.arange(time_step + 1, time_step + pred_days + 1)
# Afficher les tableaux :
print("Derniers jours utilisés pour la prédiction :", last_days)
print("Jours prédits :", day_pred)



# II- CREATION UNE MATRICE TEMPORAIRE REMPLIE DE Nan :
temp_mat = np.empty((len(last_days) + pred_days + 1, 1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1, -1).tolist()[0]



# III- INITIALISER LES VALEURS POUR LES DERNIERS JOURS ORIGINAUX ET PREDITS :
last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

# Remplir les valeurs pour les derniers jours originaux :
last_original_days_value[0:time_step + 1] = scaler.inverse_transform(closedf[len(closedf) - time_step:]).reshape(1, -1).tolist()[0]
Sélectionne les time_step dernières valeurs de closedf, Inverse la transformation appliquée précédemment à ces valeurs, Redimensionne le tableau résultant en une seule ligne, Convertit le tableau redimensionné en une liste Python, Accède au premier élément de cette liste, qui est une liste contenant tous les éléments de la ligne, Assigne cette liste à la tranche 0:time_step + 1 de last_original_days_value.

# Remplir les valeurs pour les jours prédits :
next_predicted_days_value[time_step + 1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).reshape(1, -1).tolist()[0]
# Sélectionne les time_step dernières valeurs de closedf, Inverse la transformation appliquée précédemment à ces valeurs, Redimensionne le tableau résultant en une seule ligne, Convertit le tableau redimensionné en une liste Python, Accède au premier élément de cette liste, qui est une liste contenant tous les éléments de la ligne, Assigne cette liste à la tranche 0:time_step + 1 de last_original_days_value.



# IV- CREATION DU DATAFRAME POUR LE TRACE :
new_pred_plot = pd.DataFrame({
    'last_original_days_value': last_original_days_value,
    'next_predicted_days_value': next_predicted_days_value
})



# V- CREER UN ITERATEUR CYCLIQUE POUR LE NOM DES TRACES :
# Noms des traces pour le graphique :
# Un itérateur cyclique est un outil pratique pour répéter une séquence d'éléments de manière infinie :
names = cycle(['Prix de clôture des 15 derniers jours', 'Prix de clôture prédit pour les 30 prochains jours'])



# VI- CREER LE GRAPHIQUE :
fig = px.line(
    new_pred_plot,
    x=new_pred_plot.index,
    y=[new_pred_plot['last_original_days_value'], new_pred_plot['next_predicted_days_value']],
    labels={'value': 'Prix de l\'action', 'index': 'Timestamp'}
)



# VII- METTRE A JOUR LA DISPOSITION DU GRAPHIQUE :
fig.update_layout(
    title_text='Comparaison des 15 derniers jours vs les 30 prochains jours',
    plot_bgcolor='white',
    font_size=15,
    font_color='black',
    legend_title_text='Prix de clôture'
)
# Mettre à jour les noms des traces :
fig.for_each_trace(lambda t: t.update(name=next(names)))
# Mettre à jour les axes :
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
# Afficher le graphique :
fig.show()
"""






""" ******************** Ensemble des prix de clôture avec la période de prédiction des 30 prochains jours ******************** """
print(" ******************** Ensemble des prix de clôture avec la période de prédiction des 30 prochains jours ******************** ")

"""
# Convertir les données de clôture en une liste :
lstmdf = closedf.tolist()

# Ajouter les prédictions à la liste des données de clôture :
lstmdf.extend((np.array(lst_output).reshape(-1, 1)).tolist())
# Convertit lst_output en un tableau NumPy, Redimensionne ce tableau en une colonne, Convertit le tableau redimensionné en une liste de listes, Ajoute chaque sous-liste de cette liste de listes à lstmdf.

# Appliquer la transformation inverse du scaler pour obtenir les valeurs originales :
lstmdf = scaler.inverse_transform(lstmdf).reshape(1, -1).tolist()[0]
# Inverse la transformation appliquée précédemment aux données dans lstmdf, Redimensionne le tableau résultant en une seule ligne. , Convertit le tableau redimensionné en une liste Python,  Accède au premier élément de cette liste, qui est une liste contenant tous les éléments de la ligne.

# Créer un itérateur cyclique pour les noms des traces :
names = cycle(['Close price'])

# Créer le graphique en ligne avec Plotly Express :
fig = px.line(lstmdf, labels={'value': 'Stock price', 'index': 'Timestamp'})

# Mettre à jour la disposition du graphique :
fig.update_layout(
    title_text='Plotting whole closing stock price with prediction',
    plot_bgcolor='white',
    font_size=15,
    font_color='black',
    legend_title_text='Stock'
)

# Mettre à jour les noms des traces en utilisant l'itérateur cyclique :
fig.for_each_trace(lambda t: t.update(name=next(names)))

# Désactiver la grille sur les axes x et y :
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

# Afficher le graphique :
fig.show()
"""

""" ************************************ Créer une classe qui évalue le modèle (Réseau de neurones) ************************************ """

































print(" ************ Fin du test !!! ************ ")







































































































































































""" ************************** VERSION POUR 2014 ************************** """

"""

# Analyse du prix depuis le début :
maindf['Date'] = pd.to_datetime(maindf['Date'], format='%d/%m/%Y')
y_2014 = maindf.loc[(maindf['Date'] >= '01/01/2014') & (maindf['Date'] <= '31/12/2014')]

# Convertir la colonne "Date" au format datetime :
y_2014['Date'] = pd.to_datetime(y_2014['Date'], format='%d/%m/%Y', errors='coerce')

# Remplacer les points par un espace, puis suppression de l'espace, enfin remplacement de la virgule par un point :
numeric_columns = ["Dernier", "Ouv.", " Plus Haut", "Plus Bas", "Variation %"]
for col in numeric_columns:
    y_2014.loc[:, col] = y_2014[col].str.replace('.', ' ').str.replace(' ', '').str.replace(',', '.')

# Conversion des colonnes numériques en float :
for col in numeric_columns:
    y_2014[col] = pd.to_numeric(y_2014[col], errors='coerce')

# Suppression des colonnes "Vol." et "Variation %" :
y_2014 = y_2014.drop(columns=['Vol.', 'Variation %'])
print('y_2014 sans colonnes en moins :' , y_2014)

# Afficher le DataFrame
print("************* Affichage de l'année 2025 *************")
print(y_2014)
y_2014.to_csv(PATH+FILE_CLEAN_DATASET, index=False)

"""













