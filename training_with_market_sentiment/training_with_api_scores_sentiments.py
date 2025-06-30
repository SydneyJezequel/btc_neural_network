import json
from datetime import datetime
import csv
import requests
# from training_with_market_sentiment.TRASH.Test_4_integration_sentiments import api_response_data
# from training_with_market_sentiment.training_with_market_sentiment import prepare_dataset_service
import pprint
import pandas as pd
from BO.metrics_callback import MetricsCallback
from service.display_results_service import DisplayResultsService
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import parameters
from service.generate_prediction_service import GeneratePredictionService
from service.prepare_dataset_service import PrepareDatasetService
from keras.optimizers import Adam






""" *********************** Configuration ********************** """

API_TOKEN = parameters.API_TOKEN
MARKET_SENTIMENT_API_URL = parameters.MARKET_SENTIMENT_API_URL
TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE
# Cette ligne sera peut être à supprimer :
OUTPUT_CSV_FILE = parameters.OUTPUT_CSV_FILE






""" *********************** Méthodes ********************** """

def loading_dataset():
    """ Loading btc dataset """
    try:
        # Data loading and parsing of 'Date' column :
        initial_df = pd.read_csv(TRAINING_DATASET_FILE, parse_dates=['Date'], dayfirst=True)
        return initial_df
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{TRAINING_DATASET_FILE}' n'a pas été trouvé.")
        exit()
    except Exception as e:
        print(f"Une erreur est survenue lors du chargement du dataset initial: {e}")
        exit()



def get_market_sentiment_scores() :
    """ Sentiment market scores loading """

    api_response_data = {}

    try:
        response = requests.get(MARKET_SENTIMENT_API_URL)
        response.raise_for_status()
        api_response_data = response.json()
        print("\nDonnées récupérées de l'API.")
        return api_response_data
    except requests.exceptions.HTTPError as errh:
        print(f"Erreur HTTP lors de l'appel API: {errh}")
        exit()
    except requests.exceptions.ConnectionError as errc:
        print(f"Erreur de connexion lors de l'appel API: {errc}")
        exit()
    except requests.exceptions.Timeout as errt:
        print(f"Délai d'attente dépassé lors de l'appel API: {errt}")
        exit()
    except requests.exceptions.RequestException as err:
        print(f"Une erreur inattendue est survenue lors de l'appel API: {err}")
        exit()



def sort_api_data():
    """ Sorting of sentiment market scores by date """

    # Data retrieving :
    btc_articles = []
    api_response_data = get_market_sentiment_scores()

    # Checking :
    if "btc-usd.cc" in api_response_data:  # Cette ligne sera probablement à modifier : 'btc-usd.cc'.
        btc_articles = api_response_data["btc-usd.cc"] # Cette ligne sera probablement à modifier : 'btc-usd.cc'.
    elif "BTC-USD.CC" in api_response_data:
        btc_articles = api_response_data["BTC-USD.CC"]
    else:
        print("La clé 'btc-usd.cc' ou 'BTC-USD.CC' n'a pas été trouvée dans la réponse de l'API.") # Cette ligne sera probablement à modifier : 'btc-usd.cc'.

    # Sorting by date :
    api_scores_map = {}
    if btc_articles:
        btc_articles_sorted = sorted(btc_articles, key=lambda x: x["date"])
        print("\nClés de api_scores_map (dates au format JJ/MM/AAAA) :")
        # Article retrieving :
        api_scores_map = {entry["date"]: entry["normalized"] for entry in btc_articles_sorted}
        # Convert data to "DD/MM/YYYY" :
        api_scores_map = {datetime.strptime(date, "%Y-%m-%d").strftime("%d/%m/%Y"): score for date, score in api_scores_map.items()}
        print(list(api_scores_map.keys())[:5]) # Modifier la liste pour ne pas prendre que les 5 dernières lignes ?
        print(f"Nombre total de dates dans api_scores_map: {len(api_scores_map)}")
        return api_scores_map
    else:
        print("Aucun article BTC-USD.CC à traiter pour les scores API.")



def merge_data(initial_df, api_scores_map):
    """ Api scores et dataset merging """
    # data merging :
    initial_df['Date_for_join'] = initial_df['Date'].dt.strftime('%d/%m/%Y')
    initial_df['Score API'] = initial_df['Date_for_join'].map(api_scores_map)
    initial_df = initial_df.drop(columns=['Date_for_join'])
    # Convert NaN values to 0 :
    initial_df['Score API'] = initial_df['Score API'].fillna(0)
    # Replace values equal to 0 with 0 :
    initial_df['Score API'] = initial_df['Score API'].apply(lambda x: 0 if x == 0 else x)
    return initial_df






""" ********************** Exécution du traitement ********************** """

# btc dataset loading :
initial_df = loading_dataset()

# api scores loading :
api_scores_map = sort_api_data()

# api scores and btc dataset merging :
# merged_datas = merge_data(initial_df, api_scores_map)
initial_df = merge_data(initial_df, api_scores_map)
print(initial_df)





""" *********** Save the dataset ********** """     # CETTE SECTION SERA PROBABLEMENT A SUPPRIMER :

try:
    initial_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
    print(f"\nLe dataset fusionné a été enregistré dans '{OUTPUT_CSV_FILE}' avec succès.")
except Exception as e:
    print(f"Une erreur est survenue lors de l'enregistrement du fichier CSV: {e}")




""" *********** Entrainement ********** """

DATASET_FOR_PREDICTIONS = parameters.DATASET_FOR_PREDICTIONS # Certaines lignes seront peut être à supprimer.
FORMATED_BTC_COTATIONS_FILE = parameters.FORMATED_BTC_COTATIONS_FILE
SAVED_MODEL = parameters.SAVED_MODEL
TIME_STEP = parameters.TIME_STEP
TRAIN_PREDICT_START_INDEX = 2000
TEST_PREDICT_START_INDEX = 3200




# ************* Dataset Preparation *************

prepare_dataset = PrepareDatasetService()

# Loading dataset :
# initial_dataset = pd.read_csv(TRAINING_DATASET_FILE)

# Prepare dataset :
cutoff_date = '2020-01-01'
x_train, y_train, x_test, y_test, test_data, dates, scaler = prepare_dataset.prepare_one_dimension_dataset(initial_df, cutoff_date)

# Display dataset :
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)




# ************* Model Definition *************

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


# Autre version du modèle :
"""
model = Sequential()
model.add(LSTM(10, input_shape=(None, 1), activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
"""


# Création du modèle amélioré :
"""
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




# ************* Metrics Initialization *************

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




# ************* Model Training *************

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




# ************* Display Metrics *************

# Display metrics :
pprint.pprint(metrics_history)

# Display metrics at epochs :
display_results = DisplayResultsService()
display_results.plot_metrics_history(metrics_history, metrics_to_plot=["rmse", "mse", "mae", "explained_variance", "r2", "mgd", "mpd"])




# ************* Display Results *************

# Display loss curves :
display_results.plot_loss(history)

# Display loss curves (zoom) :
display_results.zoom_plot_loss(history)




# ************* Generation of Training Residuals ************* 

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





































































































































































































































""" 
# *********** Suppression des colonnes **********

# prepare_dataset_service = PrepareDatasetService()
# initial_df = prepare_dataset_service.delete_columns(initial_df)
initial_df = initial_df.drop(columns=['Vol.', 'Variation %', 'Ouv.', ' Plus Haut', 'Plus Bas'])






# *********** Afficher les lignes où 'Score API' n'est ni None ni NaN ********** 

print("\nLignes avec un 'Score API' valide :")
df_with_valid_scores = initial_df[initial_df['Score API'].notna()]
print(df_with_valid_scores)

mapped_count = initial_df['Score API'].notna().sum()
print(f"\nNombre de scores API ajoutés au dataset: {mapped_count}")






# *********** Enregistrer le dataset **********

try:
    initial_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
    print(f"\nLe dataset fusionné a été enregistré dans '{OUTPUT_CSV_FILE}' avec succès.")
except Exception as e:
    print(f"Une erreur est survenue lors de l'enregistrement du fichier CSV: {e}")
"""

