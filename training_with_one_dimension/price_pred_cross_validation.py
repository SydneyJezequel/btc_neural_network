import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score, mean_gamma_deviance, mean_poisson_deviance
import plotly.express as px
import parameters
from sklearn.model_selection import TimeSeriesSplit
from service.prepare_dataset_service import PrepareDatasetService
import joblib




""" ************* Parameters ************* """

DATASET_PATH = parameters.DATASET_PATH
TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE
SAVED_MODEL = parameters.SAVED_MODEL
MODEL_PATH = parameters.MODEL_PATH




""" ************* Méthods ************* """

def ma(df, n):
    """ Calculation of the moving average """
    return pd.Series(df['Dernier'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))


def rsi(df, period):
    """ Calculation of the RSI """
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
    """ Calculation of the SMA crossover signal """
    sma1_col = 'MA_' + str(taille_sma1)
    sma2_col = 'MA_' + str(taille_sma2)
    signal_col = 'signal_' + sma1_col + '_' + sma2_col
    dataset[sma1_col] = ma(dataset, taille_sma1)
    dataset[sma2_col] = ma(dataset, taille_sma2)
    dataset[signal_col] = np.where(dataset[sma1_col] > dataset[sma2_col], 1.0, 0.0)
    return dataset


def create_train_and_test_dataset(model_dataset):
    """ Creation of training and test datasets """
    training_size = int(len(model_dataset) * 0.60)
    test_size = len(model_dataset) - training_size
    train_data, test_data = model_dataset[0:training_size, :], model_dataset[training_size:len(model_dataset), :1]
    return train_data, test_data


def create_dataset(dataset, time_step=1):
    """ Generates multi-dimensional training and test datasets """
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset.iloc[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset.iloc[i + time_step, 0])
    return np.array(dataX), np.array(dataY)




""" ************* Dataset Preparation ************* """

prepare_dataset = PrepareDatasetService()

# Loading dataset :
initial_dataset = pd.read_csv(TRAINING_DATASET_FILE)

# Format dataset :
tmp_dataset = prepare_dataset.format_dataset(initial_dataset)

# Remove dataset :
tmp_dataset = prepare_dataset.delete_columns(tmp_dataset)

# Display datas :
fig = px.line(tmp_dataset, x=tmp_dataset.Date, y=tmp_dataset.Dernier,labels={'Date':'date','Dernier':'Close Stock'})
fig.update_traces(marker_line_width=2, opacity=0.8, marker_line_color='orange')
fig.update_layout(title_text='Whole period of timeframe of Bitcoin close price 2014-2025', plot_bgcolor='white',
                  font_size=15, font_color='black')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# Normalization :
tmp_dataset_copy = tmp_dataset.copy()
columns_to_normalize = ['Dernier']
scaler = prepare_dataset.get_fitted_scaler(tmp_dataset_copy[columns_to_normalize])
joblib.dump(scaler, '../scaler.save')
model_dataset = tmp_dataset
normalized_datas = prepare_dataset.normalize_datas(tmp_dataset_copy[columns_to_normalize], scaler)
model_dataset[columns_to_normalize] = normalized_datas
print("training normalized dataset :", model_dataset)

# Remove 'Date' column :
date_column = model_dataset['Date']
del tmp_dataset['Date']

# Save dataset :
prepare_dataset.save_tmp_dataset(model_dataset)

# Create train and test matrix :
x_train, y_train = prepare_dataset.create_data_matrix(model_dataset)
x_test, y_test = prepare_dataset.create_data_matrix(model_dataset)




""" ************* Model Definition ************* """

# Initialization of TimeSeriesSplit with number of splits :
tscv = TimeSeriesSplit(n_splits=5)

# Initialize lists to store results and metrics :
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

# Initialization :
cpt = 1

for train_index, val_index in tscv.split(x_train):

    """ Model training """
    print("loop iteration : ", cpt)

    x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

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

    # Save model weights :
    model.save_weights(MODEL_PATH+f'best_model_weights{cpt}.weights.h5')

    # Evaluation of the model on the validation dataset :
    val_loss = model.evaluate(x_val_fold, y_val_fold, verbose=0)
    results.append(val_loss)

    # Generates predictions :
    val_predict = model.predict(x_val_fold)

    # Reshape dataset (2 dimensions) :
    y_val_fold_reshaped = y_val_fold.reshape(-1, 1)
    val_predict_reshaped = val_predict.reshape(-1, 1)

    # Fitting the scaler :
    scaler.fit(y_val_fold_reshaped)

    # Inverse transformation :
    original_yval = scaler.inverse_transform(y_val_fold_reshaped)
    val_predict_inversed = scaler.inverse_transform(val_predict_reshaped)

    # Metrics calculation:
    rmse = np.sqrt(mean_squared_error(original_yval, val_predict_inversed))
    mse = mean_squared_error(original_yval, val_predict_inversed)
    mae = mean_absolute_error(original_yval, val_predict_inversed)
    evs = explained_variance_score(original_yval, val_predict_inversed)
    r2 = r2_score(original_yval, val_predict_inversed)

    # Check if values are positive before calculation:
    if np.all(original_yval > 0) and np.all(val_predict_inversed > 0):
        mgd = mean_gamma_deviance(original_yval, val_predict_inversed)
        mpd = mean_poisson_deviance(original_yval, val_predict_inversed)
    else:
        mgd, mpd = np.nan, np.nan

    # Adding results to the lists:
    rmse_results.append(rmse)
    mse_results.append(mse)
    mae_results.append(mae)
    evs_results.append(evs)
    r2_results.append(r2)
    mgd_results.append(mgd)
    mpd_results.append(mpd)

    cpt += 1




""" ************* Display Results ************* """

# Transform lists into arrays numpy :
rmse_results = np.array(rmse_results)
mse_results = np.array(mse_results)
mae_results = np.array(mae_results)
evs_results = np.array(evs_results)
r2_results = np.array(r2_results)
mgd_results = np.array(mgd_results)
mpd_results = np.array(mpd_results)
training_loss_results = np.array(training_loss_results)
validation_loss_results = np.array(validation_loss_results)

# Display results :
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

# Loading weights :
weights_path = os.path.join(SAVED_MODEL)
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"The weight file does not exist at the specified location : {weights_path}")
model.load_weights(weights_path)

# Evaluate model on test dataset :
test_loss = model.evaluate(x_test, y_test, verbose=0)

# Generates predictions :
test_predict = model.predict(x_test)

# Reshape dataset (2 dimensions) :
y_test_reshaped = y_test.reshape(-1, 1)
test_predict_reshaped = test_predict.reshape(-1, 1)

# Fitting the scaler :
scaler.fit(y_test_reshaped)

# Inverse transformation :
original_ytest = scaler.inverse_transform(y_test_reshaped)
test_predict_inversed = scaler.inverse_transform(test_predict_reshaped)

# Metrics calculations :
rmse_test = np.sqrt(mean_squared_error(original_ytest, test_predict_inversed))
mse_test = mean_squared_error(original_ytest, test_predict_inversed)
mae_test = mean_absolute_error(original_ytest, test_predict_inversed)
evs_test = explained_variance_score(original_ytest, test_predict_inversed)
r2_test = r2_score(original_ytest, test_predict_inversed)

# Check if values are positive before calculation :
if np.all(original_ytest > 0) and np.all(test_predict_inversed > 0):
    mgd_test = mean_gamma_deviance(original_ytest, test_predict_inversed)
    mpd_test = mean_poisson_deviance(original_ytest, test_predict_inversed)
else:
    mgd_test, mpd_test = np.nan, np.nan

# Display results :
print("Test RMSE: ", rmse_test)
print("Test MSE: ", mse_test)
print("Test MAE: ", mae_test)
print("Test Explained Variance Score: ", evs_test)
print("Test R2 Score: ", r2_test)
print("Test MGD: ", mgd_test)
print("Test MPD: ", mpd_test)
