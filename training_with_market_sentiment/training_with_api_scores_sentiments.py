import pprint
import pandas as pd
from BO.metrics_callback import MetricsCallback
from service.display_results_service import DisplayResultsService
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import parameters
from service.prepare_dataset_service import PrepareDatasetService
from keras.optimizers import Adam




""" ************* Parameters ************* """

API_TOKEN = parameters.API_TOKEN
MARKET_SCORES_API_URL = parameters.MARKET_SCORES_API_URL
TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE

""" A SUPPRIMER """
OUTPUT_CSV_FILE = parameters.OUTPUT_CSV_FILE
DATASET_FOR_PREDICTIONS = parameters.DATASET_FOR_PREDICTIONS
FORMATED_BTC_COTATIONS_FILE = parameters.FORMATED_BTC_COTATIONS_FILE
SAVED_MODEL = parameters.SAVED_MODEL
TIME_STEP = parameters.TIME_STEP
TRAIN_PREDICT_START_INDEX = 2000
TEST_PREDICT_START_INDEX = 3200
""" A SUPPRIMER """




""" ************* Merge Btc cotations and api sentiment scores ************* """

prepare_dataset = PrepareDatasetService()

# btc dataset loading :
initial_df = pd.read_csv(TRAINING_DATASET_FILE, parse_dates=['Date'], dayfirst=True)

# api scores loading and sorting :
api_response_data = prepare_dataset.get_api_market_sentiment_scores(MARKET_SCORES_API_URL)
api_scores_map = prepare_dataset.sort_scores_api_data(api_response_data)

# api scores and btc dataset merging :
initial_df = prepare_dataset.merge_data(initial_df, api_scores_map)
print("Merged dataset : ", initial_df)




""" ************* Dataset Preparation ************* """

""" TEST : A SUPPRIMER """
initial_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
""" TEST : A SUPPRIMER """

# Prepare dataset :
cutoff_date = '2020-01-01'
x_train, y_train, x_test, y_test, test_data, dates, scaler = prepare_dataset.prepare_one_dimension_dataset(initial_df, cutoff_date)

# Display dataset :
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)




""" ************* Model Definition ************* """

# Define timesteps and features number :
nb_timesteps = x_train.shape[1]
nb_features = x_train.shape[2]
print("nb_timesteps : ", nb_timesteps)
print("nb_features : ", nb_features)


# Original neural network :
model = Sequential()
model.add(LSTM(10, input_shape=(nb_timesteps, nb_features), activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")


# Model (other version) :
"""
model = Sequential()
model.add(LSTM(10, input_shape=(None, 1), activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
"""


# Enhanced model :
"""
model = Sequential()
model.add(LSTM(20, input_shape=(None, 1), activation="tanh", return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(20, activation="tanh"))
model.add(Dropout(0.2))
model.add(Dense(1))
optimizer = Adam(learning_rate=0.001)
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

# early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# Model training :
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=400,
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

# Display loss curves :
display_results.plot_loss(history)

# Display loss curves (zoom) :
display_results.zoom_plot_loss(history)




""" ************* Generation of Training Residuals ************* """

# Generate predictions :
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

# Denormalization :
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Reshape dataset :
original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

# Display residus :
display_results.plot_residuals(original_ytrain, train_predict, 'Training Residuals')
display_results.plot_residuals(original_ytest, test_predict, 'Test Residuals')

