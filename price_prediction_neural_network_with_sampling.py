import pandas as pd
import numpy as np
import math
from keras.src.utils.audio_dataset_utils import prepare_dataset
from service.display_results_service import DisplayResultsService
from service.prepare_dataset_service import PrepareDatasetService
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import parameters
from tensorflow.keras.callbacks import Callback




# ******************* TEST ***************** #
import plotly.express as px
from itertools import cycle
# ******************* TEST ***************** #











""" ************************* Paramètres ************************* """

DATASET_PATH = parameters.DATASET_PATH
PATH_TRAINING_DATASET = parameters.PATH_TRAINING_DATASET
TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE
DATASET_FOR_MODEL = parameters.DATASET_FOR_MODEL
DATASET_FOR_PREDICTIONS = parameters.DATASET_FOR_PREDICTIONS
FORMATED_BTC_COTATIONS = parameters.FORMATED_BTC_COTATIONS









""" ************************* Préparation du dataset ************************* """

prepare_dataset = PrepareDatasetService()


# Loading dataset :
initial_dataset = pd.read_csv(PATH_TRAINING_DATASET + TRAINING_DATASET_FILE)


# Préparation du dataset pré-entrainement :
cutoff_date = '2020-01-01'
x_train, y_train, x_test, y_test, scaler = prepare_dataset.prepare_dataset(initial_dataset, cutoff_date)


# Affichage du dataset :
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


















""" ************************* Définition du modèle ************************* """

# Création du modèle :
model = Sequential()
model.add(LSTM(10, input_shape=(None, 1), activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
"""
# Création du modèle amélioré
model = Sequential()
model.add(LSTM(20, input_shape=(None, 1), activation="tanh", return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(20, activation="tanh"))
model.add(Dropout(0.2))
model.add(Dense(1))
# Compilation du modèle
optimizer = Adam(learning_rate=0.001)
model.compile(loss="mean_squared_error", optimizer=optimizer)
"""


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


# Callback pour stocker les métriques toutes les 50 epochs :
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















""" ************************* Entrainement du modèle ************************* """

#early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)


# Entraînement du modèle :
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=200,
    batch_size=32,
    verbose=1,
    callbacks=[MetricsCallback()] # [MetricsCallback(), early_stopping]
)


# Sauvegarde du modèle :
model.save_weights(parameters.SAVE_MODEL_PATH + f'model.weights.h5')














""" ************************* Affichage des résultats ************************* """

# Affichage des métriques stockées :
print("Metrics History:")
for metric, values in metrics_history.items():
    print(f"{metric}: {values}")


display_results = DisplayResultsService()


# Affichage des courbes de pertes :
display_results.plot_loss(history)


# Affichage des courbes de pertes zoomées :
display_results.zoom_plot_loss(history)


# Affichage des sur et sous apprentissage :
loss = history.history['loss']
val_loss = history.history['val_loss']
loss_array = np.array(loss)
val_loss_array = np.array(val_loss)
# Affichage des tableaux :
print("Loss Array:", loss_array)
print("Validation Loss Array:", val_loss_array)
































""" ***************** Charger le modèle sauvegardé ***************** """
"""
model.save_weights(...)
model = create_model()  # fonction qui redéfinit la même architecture
model.load_weights('model.weights.h5')
"""



































""" ***************** Faire des prédictions sur un dataset indépendant ***************** """
"""
def predict_on_new_data(dataset_for_predictions, model, time_step=15):
    # Faire des prédictions sur un dataset indépendant

    # Préparation du dataset :
    dataset_for_predictions, scaler = prepare_dataset.prepare_dataset_to_predict(dataset_for_predictions, time_step)

    print("dataset for predictions : ", dataset_for_predictions)

    # Faire des prédictions :
    new_predictions = model.predict(dataset_for_predictions)
    print("new_predictions normalisées : ", new_predictions)
    new_predictions = scaler.inverse_transform(new_predictions)
    print("new_predictions finales : ", new_predictions)

    return new_predictions
"""


""" Nouvelle version de predict """
def predict_on_new_data(dataset_for_predictions, model, time_step=15):
    # Faire des prédictions sur un dataset indépendant

    # Préparation du dataset :
    dataset_for_predictions, scaler = prepare_dataset.prepare_dataset_to_predict(dataset_for_predictions, time_step)
    print("dataset for predictions : ", dataset_for_predictions)

    # Vérification de la forme actuelle
    print("Shape of dataset_for_predictions: ", dataset_for_predictions.shape)

    # Si les données ont une forme (samples, time_steps, features), aplatissez-les pour avoir (samples, time_steps * features)
    if len(dataset_for_predictions.shape) == 3:
        dataset_for_predictions = dataset_for_predictions.reshape(dataset_for_predictions.shape[0],
                                                                  -1)  # -1 permet de calculer automatiquement la bonne taille
        print("Shape après aplatissement : ", dataset_for_predictions.shape)

    # Convertir en DataFrame pour nettoyage, si nécessaire
    dataset_for_predictions = pd.DataFrame(dataset_for_predictions)

    # Nettoyage des données (remplacer les virgules par des points)
    dataset_for_predictions = dataset_for_predictions.applymap(lambda x: str(x).replace(',', '.'))
    print("dataset for predictions après nettoyage : ", dataset_for_predictions)

    # Convertir les valeurs en float après nettoyage
    dataset_for_predictions = dataset_for_predictions.astype(float)
    print("dataset_for_predictions converti : ", dataset_for_predictions)

    # Faire des prédictions :
    new_predictions = model.predict(dataset_for_predictions)
    print("new_predictions normalisées : ", new_predictions)

    # Inverse transform des prédictions :
    new_predictions = scaler.inverse_transform(new_predictions)
    print("new_predictions finales : ", new_predictions)

    return new_predictions






""" Prédictions """

# Chargement du dataset :
dataset_for_predictions = DATASET_FOR_PREDICTIONS
print("CHEMIN : ", dataset_for_predictions)
dataset_for_predictions = pd.read_csv(dataset_for_predictions)
print("DATASET : ", dataset_for_predictions)

# Conserver les dates avant de les supprimer :
dates = dataset_for_predictions['Date'].values

# Prédictions :
time_step=15
predictions = predict_on_new_data(dataset_for_predictions, model, time_step)
print("predictions datas : ", predictions)









""" Affichage des prédictions """
display_results.plot_predictions(dates, predictions, time_step)







""" Affichage des prédictions à la suite du dataset d'origine """
formated_dataset = pd.read_csv(FORMATED_BTC_COTATIONS)
display_results.display_all_dataset(formated_dataset)

# Conversion de la date :
formated_dataset['Date'] = pd.to_datetime(formated_dataset['Date'])

# Récupération de la dernière date :
formatted_dataset_last_date = formated_dataset['Date'].max()
print("last date : ", formatted_dataset_last_date)

# Numéro de la nouvelle date :
num_new_dates = len(formated_dataset)
num_days = len(predictions)
print("nombre de dates générées : ", num_new_dates)
print("nombre de predictions : ", len(predictions))

# Créer de nouvelles dates :
new_dates = [formatted_dataset_last_date + pd.Timedelta(days=i) for i in range(1, num_days + 1)]
print("new_dates : ", new_dates)

# Affichage pour vérification
for d in new_dates:
    print(d.date())

# Création du dataset de prédictions (Date + Dernier) :
predictions_dataset = pd.DataFrame({
    'Date': new_dates,
    'Dernier': predictions.flatten()
})
print("predictions_dataset : ", predictions_dataset)

# Affichage des prédictions :
display_results.display_all_dataset(predictions_dataset)

# Ajout des prédictions au dataset :
display_results.display_dataset_and_predictions(formated_dataset, predictions_dataset)










""" ********************** TEST DE PLUSIEURS DATASET ********************** """




""" Calcul des prédictions """
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)




""" Dénormalisation """

"""
# Nouveau formatage du dataset :
initial_dataset['Dernier'] = pd.to_numeric(
    initial_dataset['Dernier']
    .astype(str)
    .str.replace('.', '', regex=False)
    .str.replace(',', '.', regex=False),
    errors='coerce'
)
"""

# Taille du dataset complet
real_values = scaler.inverse_transform(initial_dataset['Dernier'].values.reshape(-1, 1)).flatten()
dataset_length = len(real_values)
print("initial_dataset : ", real_values)
time_step = 15

# Dénormalisation des prédictions :
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_true = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1))

# Affichage pour vérification
print("train_predict length : ", len(train_predict))
print("train_predict length : ", train_predict)
print("test_predict length : ", len(test_predict))
print("test_predict length : ", test_predict)




""" Préparation des datasets à afficher """
# Initialisation des tableaux vides alignés avec le dataset initial
train_predict_plot = np.empty(dataset_length)
train_predict_plot[:] = np.nan
test_predict_plot = np.empty(dataset_length)
test_predict_plot[:] = np.nan

# Positionnement des prédictions
training_size = int(dataset_length * 0.60)

train_start = time_step
train_end = train_start + len(train_predict)
train_predict_plot[train_start:train_end] = train_predict.flatten()

test_start = training_size + time_step
test_end = test_start + len(test_predict)
if test_end > dataset_length:
    test_end = dataset_length
    test_predict = test_predict[:(test_end - test_start)]

test_predict_plot[test_start:test_end] = test_predict.flatten()



""" Tracer la courbe """
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
plt.plot(initial_dataset['Date'], real_values, label='Données réelles', color='black')
plt.plot(initial_dataset['Date'], train_predict_plot, label='Prédictions entraînement', color='green')
plt.plot(initial_dataset['Date'], test_predict_plot, label='Prédictions test', color='red')
plt.title("Courbe réelle vs prédictions LSTM")
plt.xlabel("Date")
plt.ylabel("Prix (dénormalisé)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



""" Tracer la courbe (V1) """
"""
datasets_list = [train_predict, test_predict]

display_results.display_dataset_and_predictions(formated_dataset, datasets_list)
"""

















""" Comparaison du prix de clôture original VS les prix prédits """
"""
print(" **************** Comparaison du prix de clôture original VS les prix prédits ****************** ")

initial_dataset = pd.read_csv(PATH_TRAINING_DATASET + TRAINING_DATASET_FILE)

# Prix initiaux :
compare_prices = initial_dataset[['Date', 'Dernier']]
print("compare_prices : ", compare_prices)

# Prix prédits lors de l'entrainement :
# train_predict = scaler.inverse_transform(train_predict)
print("train_predict : ", train_predict)

# Prix prédits lors des tests :
#  test_predict = scaler.inverse_transform(test_predict)
print("test_predict : ", test_predict)

print("original_ytrain : ", original_ytrain)
print("original_ytest : ", original_ytest )
"""

""" Tenter d'implémenter le schéma (1) """








































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































""" ************************* Controle du surapprentissage ************************* """
"""
# Calcul des prédictions :
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

train_predict = train_predict.reshape(-1, 1)
test_predict = test_predict.reshape(-1, 1)

# scaler = prepare_dataset.get_fitted_scaler(train_predict)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

# Affichage des résidus :
display_results.plot_residuals(original_ytrain, train_predict, 'Training Residuals')
display_results.plot_residuals(original_ytest, test_predict, 'Test Residuals')




# Données brutes :
print(" **************** Contrôle résultat ****************** ")
print(" train_predict : ", train_predict)
print(" test_predict : ", test_predict)
print(" original_ytrain  : ", original_ytrain)
print(" original_ytest : ", original_ytest)
print(" **************** Contrôle résultat ****************** ")
"""








# Données avec le scaler :
print(" **************** Contrôle résultat 2 ****************** ")
"""
# Appliquer inverse_transform pour obtenir les prix réels
train_predict_real = scaler.inverse_transform(train_predict)
test_predict_real = scaler.inverse_transform(test_predict)
original_ytrain_real = scaler.inverse_transform(original_ytrain)
original_ytest_real = scaler.inverse_transform(original_ytest)
# Affichage des résultats
print("Train Predictions (real prices):", train_predict_real)
print("Test Predictions (real prices):", test_predict_real)
print("Original Train Prices (real prices):", original_ytrain_real)
print("Original Test Prices (real prices):", original_ytest_real)
"""
print(" **************** Contrôle résultat 2 ****************** ")












































































""" ********************* TEST ******************** """
"""
# Affichage du dataset :
display_results.display_all_dataset(initial_dataset)
# Test :

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
display_results.display_dataset_and_predictions(initial_dataset, additional_datasets=[df_ethereum, df_litecoin])
"""














""" Comparaison du prix de clôture original VS les prix prédits (tentative 2) """
"""
names = cycle(['Original close price'])

plotdf = pd.DataFrame({'date': compare_prices['Date'],
                       'original_close': compare_prices['Dernier']})

fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close']],
              labels={'value':'Stock price','date': 'Date'})
fig.update_layout(title_text='Compare price',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()
"""














""" Test projet Kaggle dans : Dataset initial """
"""
pd.read_csv(PATH_TRAINING_DATASET + TRAINING_DATASET_FILE)
cours = initial_dataset[['Date', 'Dernier']]
print("Shape of course : ", cours.shape)

fig = px.line(cours, x=cours.Date, y=cours.Dernier, labels={'date': 'Date', 'close': 'Close Stock'})

fig.update_traces(marker_line_width=2, opacity=0.8, marker_line_color='orange')
fig.update_layout(title_text='Whole period of timeframe of BTC close price',
                  plot_bgcolor='white', font_size=15, font_color='black')

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()
"""













""" dataset initial """
"""
# Supposons que compare_prices soit déjà défini et contienne les colonnes 'Date' et 'Dernier'
# Convertir la colonne 'Date' en type datetime
initial_dataset['Date'] = pd.to_datetime(initial_dataset['Date'], format='%d/%m/%Y')

start_date = pd.to_datetime('2024-01-01')
end_date = pd.to_datetime('2024-02-15')
# Filtrer les données pour les mois de janvier / février 2024 :
initial_dataset_filtered = initial_dataset[(initial_dataset['Date'] >= start_date) & (initial_dataset['Date'] <= end_date)] [['Date', 'Dernier']]

# Préparer les données pour le traçage
plotdf = pd.DataFrame({
    'date': initial_dataset_filtered['Date'],
    'original_close': initial_dataset_filtered['Dernier']
})

# Tracer les prix de clôture originaux pour 2024
names = cycle(['Original close price'])

fig = px.line(plotdf, x=plotdf['date'], y=[plotdf['original_close']],
              labels={'value': 'Stock price', 'date': 'Date'})
fig.update_layout(title_text='Dataset initial',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.for_each_trace(lambda t: t.update(name=next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()
"""















""" Comparaison du prix de clôture original VS les prix prédits (janvier / février 2024) """
"""
# Supposons que compare_prices soit déjà défini et contienne les colonnes 'Date' et 'Dernier'
# Convertir la colonne 'Date' en type datetime
compare_prices['Date'] = pd.to_datetime(compare_prices['Date'], format='%d/%m/%Y')

start_date = pd.to_datetime('2024-01-01')
end_date = pd.to_datetime('2024-02-15')
# Filtrer les données pour les mois de janvier / février 2024 :
compare_prices_filtered = compare_prices[(compare_prices['Date'] >= start_date) & (compare_prices['Date'] <= end_date)] [['Date', 'Dernier']]

# Préparer les données pour le traçage
plotdf = pd.DataFrame({
    'date': compare_prices_filtered['Date'],
    'original_close': compare_prices_filtered['Dernier']
})

# Tracer les prix de clôture originaux pour 2024
names = cycle(['Original close price'])

fig = px.line(plotdf, x=plotdf['date'], y=[plotdf['original_close']],
              labels={'value': 'Stock price', 'date': 'Date'})
fig.update_layout(title_text='Janvier - Février 2024',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.for_each_trace(lambda t: t.update(name=next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()
"""















""" Comparaison du prix de clôture original VS les prix prédits (2024) """
"""
# Supposons que compare_prices soit déjà défini et contienne les colonnes 'Date' et 'Dernier'
# Convertir la colonne 'Date' en type datetime
compare_prices['Date'] = pd.to_datetime(compare_prices['Date'], format='%d/%m/%Y')

# Filtrer les données pour l'année 2024
compare_prices_2024 = compare_prices[compare_prices['Date'].dt.year == 2024][['Date', 'Dernier']]

# Préparer les données pour le traçage
plotdf = pd.DataFrame({
    'date': compare_prices_2024['Date'],
    'original_close': compare_prices_2024['Dernier']
})

# Tracer les prix de clôture originaux pour 2024
names = cycle(['Original close price'])

fig = px.line(plotdf, x=plotdf['date'], y=[plotdf['original_close']],
              labels={'value': 'Stock price', 'date': 'Date'})
fig.update_layout(title_text='Comparaison entre le prix de clôture original et le prix de clôture prédit en 2024',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.for_each_trace(lambda t: t.update(name=next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()
"""















""" Comparaison du prix de clôture original VS les prix prédits (original) """
"""
# shift train predictions for plotting
look_back=time_step
trainPredictPlot = np.empty_like(compare_prices)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
print("Train predicted data: ", trainPredictPlot.shape)

# Correction mise en forme :
test_predict = test_predict.reshape(-1, 2)
print("test_predict shape: ", test_predict.shape)

# shift test predictions for plotting
testPredictPlot = np.empty_like(compare_prices)
testPredictPlot[:, :] = np.nan
# testPredictPlot[len(train_predict)+(look_back*2)+1:len(compare_prices)-1, :] = test_predict
testPredictPlot[len(train_predict)+(look_back*2)+1:len(train_predict)+(look_back*2)+1+len(test_predict), :] = test_predict

print("Test predicted data: ", testPredictPlot.shape)

names = cycle(['Original close price','Train predicted close price','Test predicted close price'])
"""











"""
plotdf = pd.DataFrame({'date': compare_prices['Date'],
                       'original_close': compare_prices['Dernier'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})
"""









"""
plotdf = pd.DataFrame({'date': compare_prices['Date'],
                       'original_close': compare_prices['Dernier'],
                      'train_predicted_close': trainPredictPlot[:, 0],
                      'test_predicted_close': testPredictPlot[:, 0] })


fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                          plotdf['test_predicted_close']],
              labels={'value':'Stock price','date': 'Date'})
fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()
"""













"""
# Plotting last 15 days of dataset and next predicted 30 days

last_days=np.arange(1,time_step+1)
day_pred=np.arange(time_step+1,time_step+pred_days+1)
print(last_days)
print(day_pred)

temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

new_pred_plot = pd.DataFrame({
    'last_original_days_value':last_original_days_value,
    'next_predicted_days_value':next_predicted_days_value
})

names = cycle(['Last 15 days close price','Predicted next 30 days close price'])

fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                      new_pred_plot['next_predicted_days_value']],
              labels={'value': 'Stock price','index': 'Timestamp'})
fig.update_layout(title_text='Compare last 15 days vs next 30 days',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')

fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()
"""












"""
# Plotting entire Closing Stock Price with next 30 days period of prediction


lstmdf=closedf.tolist()
lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
lstmdf=scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]

names = cycle(['Close price'])

fig = px.line(lstmdf,labels={'value': 'Stock price','index': 'Timestamp'})
fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')

fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()
"""


