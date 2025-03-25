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
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score, mean_gamma_deviance, mean_poisson_deviance
""" For Plotting we will use these library """
import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import parameters
from sklearn.model_selection import TimeSeriesSplit
from service.technical_indicators import technical_indicators
from service.prepare_dataset_service import PrepareDataset
from service.neural_network import NeuralNetwork




""" ****************************** Paramètres ****************************** """
DATASET_PATH = parameters.DATASET_PATH
PATH_TRAINING_DATASET = parameters.PATH_TRAINING_DATASET
DATASET_FILE = parameters.DATASET_FILE
DATASET_FOR_MODEL = parameters.DATASET_FOR_MODEL
SAVE_MODEL_PATH = parameters.SAVE_MODEL_PATH





""" **************************** Exécution du script principal **************************** """





print(" ************ Etape 1 : Loading dataset ************ ")
initial_dataset = pd.read_csv(DATASET_PATH+DATASET_FILE)






print(" ************ Etape 2 : Preparation of the Dataset ************ ")

prepare_dataset = PrepareDataset()


# Formatage des colonnes :
tmp_dataset = prepare_dataset.format_dataset(initial_dataset)


# Suppression des colonnes :
tmp_dataset = prepare_dataset.delete_columns(tmp_dataset)


# Enregistrement du dataset au format csv :
tmp_dataset.to_csv(PATH_TRAINING_DATASET+DATASET_FOR_MODEL, index=False)


# Contrôle des modifications :
print("En-tête du dataset d'entrainement : ", tmp_dataset.head())
print("dataset d'entrainement modifié (dernières lignes) pour vérifier si mes indicateurs sont bien calculés : ", tmp_dataset.tail())


# Ajout des indicateurs techniques :
tmp_dataset = prepare_dataset.add_technicals_indicators(tmp_dataset)


# Obtenir le scaler ajusté :
scaler = prepare_dataset.get_fitted_scaler(tmp_dataset)
print("tmp_dataset : ", tmp_dataset.shape)


# Normalise dataset :
model_dataset = prepare_dataset.normalize_datas(tmp_dataset, scaler)
print("dataset d'entrainement normalisé :", model_dataset)
print("model_dataset shape : ", model_dataset.shape)


# Créer les matrices de données pour l'entraînement et le test
x_train, y_train = prepare_dataset.create_data_matrix(model_dataset)
x_test, y_test = prepare_dataset.create_data_matrix(model_dataset)
# NOTION DE VALEUR CIBLE :
"""
Une valeur cible, également appelée étiquette ou label, est la valeur que l'on souhaite prédire ou prévoir
dans un problème de machine learning. Elle représente l'output ou la sortie attendue pour une entrée donnée.
Les valeurs cibles sont utilisées
pour entraîner les modèles de machine learning en leur fournissant des exemples de ce qu'ils doivent
apprendre à prédire.
"""









""" ************************************ Créer une classe qui encapsule le modèle (Réseau de neurones) ************************************ """

print(" ************ Etape 3 : Create and train model ************ ")

""" 
----------------------------------------------------------------
# Ajout d'une couche LSTM (Long Short-Term Memory) au modèle :
# - 10 unités LSTM
# - input_shape=(None, 1) : La couche LSTM a une sortie de forme (None, 10), ce qui signifie qu'elle produit 10 valeurs pour chaque séquence d'entrée.
# - activation="relu" : Fonction d'activation ReLU (Rectified Linear Unit)
# :  La couche dense a une sortie de forme (None, 1), ce qui signifie qu'elle produit une seule valeur de sortie pour chaque séquence d'entrée.
----------------------------------------------------------------
"""

# Instanciation du modèle :
neural_network = NeuralNetwork(x_train, y_train)

# Exécution de l'entrainement :
neural_network.train()










""" ************************************ Analyser les résultats avec un Llm ************************************ """

print(" ************ Etape 4 : Analyser les résultats avec un Llm ************ ")



















""" ************************************ Créer une classe qui évalue le modèle (Réseau de neurones) ************************************ """
""" NOM DE LA CLASSE : evaluate_neural_network_training """

print(" ******************** Etape 5 : Overfitting Evaluation ******************** ")






print("Training loss 1 : ")
"""
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





print("Training loss 2 : Training and Validation loss comparison : ")
"""
==> COMMENT ANALYSER CE GRAPHE ?
Plus les 2 courbes se suivent : Moins il y a d'overfitting.
Si la perte d'entrainement continue à diminuer alors que la perte de validation augmente, cela 
signifie que le modèle commence à surajuster les données d'entrainement).
Si les 2 courbes suivent une tendance similaires, cela signifie que le modèle est moins susceptible de
surajuster.
"""

"""
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()
plt.show()
"""













print(" ******************** Génération de prédiction par le modèle ******************** ")
"""
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)
print(train_predict.shape, test_predict.shape)
"""













print(" ******************** Evaluation du modèle ******************** ")

"""
# Les datasets d'entrainements et de tests sont ramenées à leur échelle d'origine :
scaler = MinMaxScaler(feature_range=(0, 1))
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Inverse la transformation des valeurs cibles d'entraînement pour les ramener à leur échelle d'origine
# Note : y_train et t_train sont remodelées en matrices colonnes pour correspondre à la forme attendue par inverse_transform :
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 
"""

""" ************************************ Créer une classe qui évalue le modèle (Réseau de neurones) ************************************ """







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
scaler = MinMaxScaler(feature_range=(0, 1))
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
scaler = MinMaxScaler(feature_range=(0, 1))
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




