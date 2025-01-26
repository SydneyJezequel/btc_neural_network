print(" ************ Trading Neural Network ************ ")

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













""" ************ Paramètres ************ """
PATH = '../btc_neural_network/dataset/'
PATH_TRAINING_DATASET = PATH + 'training_dataset/'
DATASET_FILE = 'btc_historic_cotations.csv'
FILE_CLEAN_DATASET = 'clean_dataset.csv'
DATASET_FOR_MODEL = 'dataset_for_model.csv'
DATASET_FOR_MODEL_WITH_TECHNICAL_INDICATORS = 'dataset_for_model_with_technical_indicators.csv'














""" **************************** Loading dataset **************************** """
print(" ************ Loading dataset ************ ")
maindf = pd.read_csv(PATH+DATASET_FILE)
print('Total number of days present in the dataset: ', maindf.shape[0])
print('Total number of fields present in the dataset: ', maindf.shape[1])
print('Nombre de lignes et colonnes: ', maindf.shape)
print('En-têtes du dataset: ', maindf.head())
print('maindf.tail(): ', maindf.tail())
print('maindf.info(): ', maindf.info())
print('maindf.describe()', maindf.describe())
print(" ************ Vérification des données manquantes ************ ")
print('Null Values:',maindf.isnull().values.sum())
print('NA values:',maindf.isnull().values.any())











""" **************************** EDA(Exploratory Data Analysis) **************************** """
print(" ************ EDA(Exploratory Data Analysis) ************ ")

# Affichage des dates de début et fin du dataset :
ed=maindf.iloc[0][0]
sd=maindf.iloc[-1][0]
print('Starting Date : ',sd)
print('Ending Date : ',ed)

# Convertir la colonne "Date" au format datetime :
maindf['Date'] = pd.to_datetime(maindf['Date'], format='%d/%m/%Y', errors='coerce')
# errors='coerce' --> Les valeurs non convertie sont remplacées par Nan.

# Remplacer les points par un espace, puis suppression de l'espace, enfin remplacement de la virgule par un point :
numeric_columns = ["Dernier", "Ouv.", " Plus Haut", "Plus Bas", "Variation %"]
for col in numeric_columns:
    maindf.loc[:, col] = maindf[col].str.replace('.', ' ').str.replace(' ', '').str.replace(',', '.')
    # .loc[:, col] est utilisé pour sélectionner toutes les lignes (:) de la colonne spécifiée par col.

# Conversion des colonnes numériques en float :
for col in numeric_columns:
    maindf[col] = pd.to_numeric(maindf[col], errors='coerce')

# Suppression des colonnes "Vol." et "Variation %" :
maindf =maindf.drop(columns=['Vol.', 'Variation %'])
print('Dataset transformé :' , maindf)

# Afficher le DataFrame :
print('Dataset transformé :' , maindf)
print('Ecriture du dataset :')
maindf.to_csv(PATH+FILE_CLEAN_DATASET, index=False)
# index=False : Indique que l'index du DataFrame ne doit pas être inclus dans le fichier CSV généré.











""" **************************** PREPARATION DU DATASET **************************** """
print(" ******************** PREPARATION DU DATASET ******************** ")



""" 1- Méthodes qui calculent les indicateurs techniques """


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




""" 2- Modification du dataset """


# Forme du dataset :
datas_for_model = pd.read_csv(PATH+FILE_CLEAN_DATASET)
print("Shape of close dataframe:", datas_for_model.shape)


# Affichage des données :
fig = px.line(datas_for_model, x=datas_for_model.Date, y=datas_for_model.Dernier,labels={'Date':'date','Dernier':'Close Stock'})
fig.update_traces(marker_line_width=2, opacity=0.8, marker_line_color='orange')
fig.update_layout(title_text='Whole period of timeframe of Bitcoin close price 2014-2025', plot_bgcolor='white',
                  font_size=15, font_color='black')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# Ajout des indicateurs dans les colonnes :
datas_for_model['MA_150'] = ma(datas_for_model, 150)
datas_for_model['MA_100'] = ma(datas_for_model, 100)
datas_for_model['MA_50'] = ma(datas_for_model, 50)
datas_for_model['RSI'] = rsi(datas_for_model, 14)


# Ajout des signaux générés par les indicateurs :
calculate_signal(datas_for_model, 50, 150)
calculate_signal(datas_for_model, 100, 150)
calculate_signal(datas_for_model, 50, 100)


# Suppression des colonnes de départ :
model_dataset_with_technicals_indicators =datas_for_model.drop(columns=['Ouv.', ' Plus Haut', 'Plus Bas'])


# Contrôle des modifications :
print("En-tête du dataset d'entrainement : ", model_dataset_with_technicals_indicators.head())
print("dataset d'entrainement modifié (dernières lignes) pour vérifier si mes indicateurs sont bien calculés : ", model_dataset_with_technicals_indicators.tail())
print("Enregistrement du dataset dans un fichier csv : ")
model_dataset_with_technicals_indicators.to_csv(PATH_TRAINING_DATASET+DATASET_FOR_MODEL_WITH_TECHNICAL_INDICATORS, index=False)


# Ignorer les valeurs NAN (Remplir les valeurs NaN avec la moyenne des colonnes) :
del model_dataset_with_technicals_indicators['Date']
imputer = SimpleImputer(strategy='mean')
model_dataset_with_technicals_indicators = imputer.fit_transform(model_dataset_with_technicals_indicators)
print("dataset d'entrainement modifié (premières lignes) pour vérifier si j'ai bien la moyenne : ", model_dataset_with_technicals_indicators)


# Normalisation des données par MinMaxScaler :
scaler = MinMaxScaler(feature_range=(0,1))
model_dataset_with_technicals_indicators = scaler.fit_transform(np.array(model_dataset_with_technicals_indicators).reshape(-1,1))
print("forme du dataset modifié : ", model_dataset_with_technicals_indicators.shape)
print("dataset d'entrainement normalisé : ", model_dataset_with_technicals_indicators)
"""
La méthode MinMaxScaler de la bibliothèque scikit-learn est utilisée pour normaliser
les caractéristiques (features) d'un jeu de données.
Elle met à l'échelle chaque caractéristique dans une plage spécifiée, généralement entre 0 et 1.
On peut faire la même chose avec un StandardScaler().
"""



""" 3- Création des datasets d'entrainement et de test """
training_size = int(len(model_dataset_with_technicals_indicators)*0.60)
test_size = len(model_dataset_with_technicals_indicators)-training_size
train_data,test_data = model_dataset_with_technicals_indicators[0:training_size,:], model_dataset_with_technicals_indicators[training_size:len(model_dataset_with_technicals_indicators),:1]
# datas_for_model[0:training_size,:] : Sélectionne les training_size premières lignes de l'array.
# datas_for_model[training_size:len(datas_for_model),:1] : Sélectionne toutes les lignes à partir de training_size jusqu'à la fin de l'array.
print("dataset d'entrainement : ", train_data.shape)
print("dataset de test : ", test_data.shape)



""" 4- Conversion des arrays en matrice """
# Définition de la fonction create_dataset pour créer des ensembles de données pour l'entraînement et les tests
# Cette fonction prend un dataset et un time_step comme arguments
def create_dataset(dataset, time_step=1):
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

# Définition du time_step (longueur des séquences ex : Si 15, chaque séquence d'entrée contiendra 3 valeurs du dataset) :
time_step = 15
# Création des ensembles de données d'entraînement et de test en utilisant la fonction create_dataset :
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)
# Affichage des dimensions des ensembles de données d'entraînement et de test :
print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)

# NOTION DE VALEUR CIBLE :
"""
Une valeur cible, également appelée étiquette ou label, est la valeur que l'on souhaite prédire ou prévoir
dans un problème de machine learning. Elle représente l'output ou la sortie attendue pour une entrée donnée.
Les valeurs cibles sont utilisées
pour entraîner les modèles de machine learning en leur fournissant des exemples de ce qu'ils doivent
apprendre à prédire.
"""



# VI- Mise en forme de la matrice pour obtenir les éléments nécessaires aux traitements
# par les couches LSTM [échantillons, time steps, caractéristiques] :
# On redimension les arrays en ajoutant une dimension supplémenaire.
# Ex : L'array X_train de forme (100, 20, 1) aura la forme suivante après transformation : (100, 20, 1).

# Remodelage de X_train pour obtenir la forme [échantillons, time steps, caractéristiques]
# Cela est nécessaire pour que les données soient compatibles avec les couches LSTM :
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Remodelage de X_test pour obtenir la forme [échantillons, time steps, caractéristiques]
# Cela est nécessaire pour que les données soient compatibles avec les couches LSTM :
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Affichage des dimensions des ensembles de données d'entraînement et de test après remodelage :
print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)







""" ******************** Création et entrainement du modèle ******************** """
print(" ******************** Création et entrainement du modèle ******************** ")

""" 
# Création du modèle :
# Initialisation d'un modèle séquentiel :
model=Sequential()


# Ajout d'une couche LSTM (Long Short-Term Memory) au modèle :
# - 10 unités LSTM
# - input_shape=(None, 1) : La couche LSTM a une sortie de forme (None, 10), ce qui signifie qu'elle produit 10 valeurs pour chaque séquence d'entrée.
# - activation="relu" : Fonction d'activation ReLU (Rectified Linear Unit)
# :  La couche dense a une sortie de forme (None, 1), ce qui signifie qu'elle produit une seule valeur de sortie pour chaque séquence d'entrée.
model.add(LSTM(10,input_shape=(None,1),activation="relu")).
model.add(Dense(1))



# Exécution du modèle :
# Compilation du modèle :
# - loss="mean_squared_error" : Utilisation de la moyenne des erreurs quadratiques comme fonction de perte (écart prévision/résultat).
# - optimizer="adam" : Utilisation de l'optimiseur Adam pour gérer la descente de gradient (algorithme d'optimisation utilisé pour minimiser la fonction de perte. Elle ajuste les paramètres du modèle (comme les poids et les biais dans un réseau de neurones) de manière itérative pour réduire la valeur de la fonction de perte.).
model.compile(loss="mean_squared_error",optimizer="adam")



# Entrainement du modèle :
# - X_train, y_train : données et cibles d'entraînement.
# - validation_data=(X_test, y_test) : données et cibles de validation.
# - epochs=200 : nombre d'époques (passages complets sur l'ensemble des données d'entraînement).
# - batch_size=32 : taille des lots (batch size) pour l'entraînement.
# - verbose=1 : mode verbeux pour afficher les informations de progression pendant l'entraînement.
history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=200,batch_size=32,verbose=1)
"""






""" ******************** Génération de prédiction par le modèle ******************** """
print(" ******************** Génération de prédiction par le modèle ******************** ")

"""
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
train_predict.shape, test_predict.shape
"""






""" ******************** Evaluation du modèle ******************** """
print(" ******************** Evaluation du modèle ******************** ")

"""
# Les datasets d'entrainements et de tests sont ramenées à leur échelle d'origine :
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Inverse la transformation des valeurs cibles d'entraînement pour les ramener à leur échelle d'origine
# Note : y_train et t_train sont remodelées en matrices colonnes pour correspondre à la forme attendue par inverse_transform :
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 
"""






""" ******************** Evaluation des metrics RMSE and MAE******************** """
print(" ******************** Evaluation des metrics RMSE and MAE ******************** ")

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
"""






""" ******************** Variance Regression Score ******************** """
print(" ******************** Variance Regression Score ******************** ")

"""
# Calcule et affiche le score de régression de la variance expliquée pour les données d'entraînement
# Le score de variance expliquée (explained variance score) mesure la proportion
de la variance dans les valeurs cibles qui est expliquée par le modèle :
- Un score proche de 1 indique que le modèle explique bien la variance des données.
- Un score proche de 0 indique que le modèle n'explique pas bien la variance des données.
print("Train data explained variance regression score:",
explained_variance_score(original_ytrain, train_predict))


# Calcule et affiche le score de régression de la variance expliquée pour les données de test
# Le score de variance expliquée (explained variance score) mesure la proportion
de la variance dans les valeurs cibles qui est expliquée par le modèle :
- Un score proche de 1 indique que le modèle explique bien la variance des données.
- Un score proche de 0 indique que le modèle n'explique pas bien la variance des données.
print("Test data explained variance regression score:",
explained_variance_score(original_ytest, test_predict))
"""






""" ******************** R square score for regression ******************** """
print(" ******************** R square score for regression ******************** ")

"""
# Calcule et affiche le score R² pour les données d'entraînement
# Le score R² (coefficient de détermination) mesure la proportion de la variance 
dans les valeurs cibles qui est expliquée par le modèle :
# - Un score R² proche de 1 indique que le modèle explique bien la variance des données.
# - Un score R² proche de 0 indique que le modèle n'explique pas bien la variance des données.
# - Un score R² négatif indique que le modèle est pire que la moyenne des valeurs cibles.
print("Train data R2 score:", r2_score(original_ytrain, train_predict))


# Calcule et affiche le score R² pour les données de test
# Le score R² (coefficient de détermination) mesure la proportion de la variance 
dans les valeurs cibles qui est expliquée par le modèle :
# - Un score R² proche de 1 indique que le modèle explique bien la variance des données.
# - Un score R² proche de 0 indique que le modèle n'explique pas bien la variance des données.
# - Un score R² négatif indique que le modèle est pire que la moyenne des valeurs cibles.
print("Test data R2 score:", r2_score(original_ytest, test_predict))
"""






""" ********** Perte de Régression Moyenne Gamma déviance de perte de régression (MGD) et Moyenne Poisson déviance de perte de régression (MPD)  ************** """
print(" ******** Perte de Régression Moyenne Gamma déviance de perte de régression (MGD) et Moyenne Poisson déviance de perte de régression (MPD) ************ ")
"""
# Afficher la perte de régression moyenne Gamma déviance pour les données d'entraînement et de test
print("Données d'entraînement MGD :", mean_gamma_deviance(original_ytrain, train_predict))
print("Données de test MGD :", mean_gamma_deviance(original_ytest, test_predict))
print("----------------------------------------------------------------------")

# Afficher la perte de régression moyenne Poisson déviance pour les données d'entraînement et de test
print("Données d'entraînement MPD :", mean_poisson_deviance(original_ytrain, train_predict))
print("Données de test MPD :", mean_poisson_deviance(original_ytest, test_predict))
"""

"""
Perte de déviance Gamma :
La perte de déviance Gamma évalue à quel point votre modèle prédit bien les temps de défaillance observés.
La perte de déviance Gamma mesure la différence entre la déviance du modèle ajusté et la déviance d'un modèle nul :
- Modèle ajusté : C'est le modèle que vous avez entraîné sur vos données. Il utilise les variables indépendantes pour prédire la variable dépendante.
- Déviance du modèle ajusté : C'est la déviance calculée pour ce modèle ajusté, qui mesure à quel point les prédictions du modèle diffèrent des valeurs observées.(un modèle qui prédit simplement la moyenne de la variable dépendante).
- Modèle nul : C'est un modèle de référence très simple qui ne prend en compte aucune variable indépendante. Il prédit simplement la moyenne de la variable dépendante pour toutes les observations.
- Déviance d'un modèle nul : C'est la déviance calculée pour ce modèle nul, qui mesure à quel point la moyenne des observations diffère des valeurs observées.
"""






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












