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











""" ************************* Paramètres ************************* """

DATASET_PATH = parameters.DATASET_PATH
PATH_TRAINING_DATASET = parameters.PATH_TRAINING_DATASET
TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE
DATASET_FOR_MODEL = parameters.DATASET_FOR_MODEL
DATASET_FOR_PREDICTIONS = parameters.DATASET_FOR_PREDICTIONS










""" ************************* Préparation du dataset ************************* """

prepare_dataset = PrepareDatasetService()


# Loading dataset :
initial_dataset = pd.read_csv(PATH_TRAINING_DATASET + TRAINING_DATASET_FILE)


# Préparation du dataset pré-entrainement :
cutoff_date = '2020-01-01'
x_train, y_train, x_test, y_test = prepare_dataset.prepare_dataset(initial_dataset, cutoff_date)


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















""" ************************* Controle du surapprentissage ************************* """

# Calcul des prédictions :
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

train_predict = train_predict.reshape(-1, 1)
test_predict = test_predict.reshape(-1, 1)

scaler = prepare_dataset.get_fitted_scaler(train_predict)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

# Affichage des résidus :
display_results.plot_residuals(original_ytrain, train_predict, 'Training Residuals')
display_results.plot_residuals(original_ytest, test_predict, 'Test Residuals')
















""" ***************** Faire des prédictions sur un dataset indépendant ***************** """

def predict_on_new_data(dataset_for_predictions, model, time_step=15):
    """Faire des prédictions sur un dataset indépendant"""

    # Préparation du dataset :
    dataset_for_predictions, scaler = prepare_dataset.prepare_dataset_to_predict(dataset_for_predictions, time_step)

    print("dataset for predictions : ", dataset_for_predictions)

    # Faire des prédictions :
    new_predictions = model.predict(dataset_for_predictions)
    print("new_predictions normalisées : ", new_predictions)
    new_predictions = scaler.inverse_transform(new_predictions)
    print("new_predictions finales : ", new_predictions)

    return new_predictions











""" Prédictions """
# Lien vers le projet Kaggle : https://www.kaggle.com/code/meetnagadia/bitcoin-price-prediction-using-lstm
# Lien vers le dataset : https://fr.investing.com/crypto/bitcoin/historical-data


# Chargement du dataset :
dataset_for_predictions = DATASET_FOR_PREDICTIONS
print("CHEMIN : ", dataset_for_predictions)
dataset_for_predictions = pd.read_csv(dataset_for_predictions)
print("DATASET : ", dataset_for_predictions)


# Conserver les dates avant de les supprimer
dates = dataset_for_predictions['Date'].values


# Prédictions :
time_step=15
predictions = predict_on_new_data(dataset_for_predictions, model, time_step)
print("predictions datas : ", predictions)










""" Tracer des prédictions """

import matplotlib.pyplot as plt
import seaborn as sns

# Aligner les prédictions avec les dates
predictions_with_dates = pd.DataFrame({
    'Date': dates[time_step:],  # Assurez-vous que les dates correspondent aux prédictions
    'Prediction': predictions.flatten()
})

# Mise en place de l'index :
predictions_with_dates.set_index('Date', inplace=True)

# Tracer le schéma :
plt.figure(figsize=(12, 6))
sns.lineplot(data=predictions_with_dates, x=predictions_with_dates.index, y='Prediction', marker='o')

plt.title('Prédictions')
plt.xlabel('Date')
plt.ylabel('Valeur')
plt.grid()
plt.show()



























""" 
# shift train predictions for plotting


look_back=time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
print("Train predicted data: ", trainPredictPlot.shape)

# shift test predictions for plotting
testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
print("Test predicted data: ", testPredictPlot.shape)

names = cycle(['Original close price','Train predicted close price','Test predicted close price'])


plotdf = pd.DataFrame({'date': close_stock['Date'],
                       'original_close': close_stock['Close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                          plotdf['test_predicted_close']],
              labels={'value':'Stock price','date': 'Date'})
fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()



















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










