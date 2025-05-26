import pprint
import pandas as pd
from BO.metrics_callback import MetricsCallback
from service.display_results_service import DisplayResultsService
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import parameters
from service.generate_prediction_service import GeneratePredictionService
from service.prepare_dataset_service import PrepareDatasetService




""" ************* Parameters ************* """

TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE
DATASET_FOR_PREDICTIONS = parameters.DATASET_FOR_PREDICTIONS
FORMATED_BTC_COTATIONS_FILE = parameters.FORMATED_BTC_COTATIONS_FILE
SAVED_MODEL = parameters.SAVED_MODEL
TIME_STEP = parameters.TIME_STEP
TRAIN_PREDICT_START_INDEX = 2000
TEST_PREDICT_START_INDEX = 3200




""" ************* Dataset Preparation ************* """

prepare_dataset = PrepareDatasetService()

# Loading dataset :
initial_dataset = pd.read_csv(TRAINING_DATASET_FILE)

# Prepare dataset :
cutoff_date = '2020-01-01'
x_train, y_train, x_test, y_test, test_data, dates, scaler = prepare_dataset.prepare_one_dimension_dataset(initial_dataset, cutoff_date)

# Display dataset :
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)




""" ************* Model Definition ************* """

# Create neural network :
model = Sequential()
model.add(LSTM(10, input_shape=(None, 1), activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
"""
model = Sequential()
model.add(LSTM(10, input_shape=(None, 1), activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
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
    epochs=300,
    batch_size=50,
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




""" ************* Load save model ************* """
"""
model.save_weights(...)
model = create_model()
model.load_weights('model.weights.h5')
"""




""" ************* Predictions on an Independent Dataset ************* """

# Loading dataset :
dataset_for_predictions = DATASET_FOR_PREDICTIONS
dataset_for_predictions = pd.read_csv(dataset_for_predictions)

# Generate predictions :
generate_prediction_service = GeneratePredictionService()
time_step = TIME_STEP
predictions = generate_prediction_service.predict_on_new_data(test_data, model, scaler, time_step)
print("predictions : ", predictions)

# Display predictions :
display_results.plot_predictions(dates, predictions, time_step)




""" ************* Display the Original Dataset and Predictions ************* """

# Loading initial dataset :
formated_dataset = pd.read_csv(TRAINING_DATASET_FILE)

# Preparing dataset for display :
formated_dataset['Date'] = pd.to_datetime(formated_dataset['Date'], format="%d/%m/%Y")
formated_dataset['Dernier'] = formated_dataset['Dernier'].str.replace('.', '').str.replace(',', '.').astype(float)
print("formated_dataset : ", formated_dataset)
formatted_dataset_last_date = formated_dataset['Date'].max()
num_days = len(predictions)
new_dates = [formatted_dataset_last_date + pd.Timedelta(days=i) for i in range(1, num_days + 1)]
print("last date : ", formatted_dataset_last_date)
print("new_dates : ", new_dates)
predictions_dataset = pd.DataFrame({
    'Date': new_dates,
    'Dernier': predictions.flatten()
})


# Display initial dataset and predictions :
display_results.display_dataset_and_predictions(formated_dataset, predictions_dataset)




""" ************* Display Training and Test Predictions VS the Original Dataset ************* """

# Datasets alignment :
dataset_length = len(initial_dataset)
formated_dataset = formated_dataset.iloc[::-1].reset_index(drop=True)
train_predict_plot = generate_prediction_service.insert_with_padding(train_predict, TRAIN_PREDICT_START_INDEX, dataset_length)
test_predict_plot = generate_prediction_service.insert_with_padding(test_predict, TEST_PREDICT_START_INDEX, dataset_length)

#  Display training and test predictions VS dataset :
display_results.plot_initial_dataset_and_predictions(initial_dataset, formated_dataset, train_predict_plot, test_predict_plot)

