import pprint
import pandas as pd
import numpy as np
from keras.src.optimizers import Adam
from BO.metrics_callback import MetricsCallback
from service.display_results_service import DisplayResultsService
from service.prepare_dataset_service import PrepareDatasetService
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import parameters




""" ************* Parameters ************* """

TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE
SAVED_MODEL = parameters.SAVED_MODEL




""" ************* Dataset Preparation ************* """

prepare_dataset = PrepareDatasetService()

# Loading dataset :
initial_dataset = pd.read_csv(TRAINING_DATASET_FILE)

# Prepare dataset :
cutoff_date = '2020-01-01'
x_train, y_train, x_test, y_test, test_data, dates, scaler = prepare_dataset.prepare_one_dimension_dataset(initial_dataset, cutoff_date)




""" ************* Model Definition ************* """

# Create neural network :
model = Sequential()
model.add(LSTM(20, input_shape=(None, 1), activation="relu", return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(20, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1))
optimizer = Adam(learning_rate=0.001)
model.compile(loss="mean_squared_error", optimizer=optimizer)

"""
model = Sequential()
model.add(LSTM(20, input_shape=(None, 1), activation="relu", return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(20, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1))
# Compilation du modèle
optimizer = Adam(learning_rate=0.0001)
model.compile(loss="mean_squared_error", optimizer=optimizer)
"""

"""
model = Sequential()
model.add(LSTM(20, input_shape=(None, 1), activation="relu", return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(20, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1))
# Compilation du modèle
optimizer = Adam(learning_rate=0.0005)
model.compile(loss="mean_squared_error", optimizer=optimizer)
"""




""" ************* Metrics Initialization ************* """

# Metric's storage :
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

# Callback to store metrics every 50 epochs :
metrics_callback = MetricsCallback(x_train, y_train, x_test, y_test, metrics_history)




""" ************* Model Training ************* """

#early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# Model training :
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=200,
    batch_size=32,
    verbose=1,
    callbacks=[metrics_callback] # [metrics_callback, early_stopping]
)

# Save model :
model.save_weights(SAVED_MODEL)




""" ************* Display Metrics ************* """

# Display metrics :
pprint.pprint(metrics_history)

# Display metrics at epochs :
display_results = DisplayResultsService()
display_results.plot_metrics_history(metrics_history, metrics_to_plot=["rmse", "mse", "mae", "explained_variance", "r2", "mgd", "mpd"])




""" ************* Display Results ************* """

display_results = DisplayResultsService()

# Display loss curves :
display_results.plot_loss(history)

# Display loss curves (zoom) :
display_results.zoom_plot_loss(history)

# Display over and under fitting :
loss = history.history['loss']
val_loss = history.history['val_loss']
loss_array = np.array(loss)
val_loss_array = np.array(val_loss)




""" ************* Generation of Training Residuals ************* """

# Generate predictions :
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

# Reshape dataset :
train_predict = train_predict.reshape(-1, 1)
test_predict = test_predict.reshape(-1, 1)

scaler = prepare_dataset.get_fitted_scaler(train_predict)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

# Display residus :
display_results.plot_residuals(original_ytrain, train_predict, 'Training Residuals')
display_results.plot_residuals(original_ytest, test_predict, 'Test Residuals')

