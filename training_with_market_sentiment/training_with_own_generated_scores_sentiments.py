import pprint
import requests
import json
import pandas as pd
from datetime import datetime
from openai import OpenAI
import parameters
from service.display_results_service import DisplayResultsService
from BO.metrics_callback import MetricsCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from service.prepare_dataset_service import PrepareDatasetService






""" Définir les constantes """
API_MONSTER_KEY = parameters.API_MONSTER_KEY
BASE_URL = parameters.BASE_URL
MODEL_NAME = parameters.MODEL_NAME
TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE
MARKET_SENTIMENT_API_URL = parameters.MARKET_SENTIMENT_API_URL
OUTPUT_CSV_FILE = 'output.csv'  # Définir le nom du fichier de sortie






class SentimentAnalyzer:
    """ Classe qui analyse les sentiments """



    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = BASE_URL
        self.monster_client = OpenAI(api_key=self.api_key, base_url=self.base_url)



    def analyze_sentiment(self, text):
        """ Analyse le sentiment sur un seul texte. """
        response = self.monster_client.chat.completions.create(
            # model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",
                 "content": f"Évalue le sentiment du texte suivant sur une échelle de 1 à 10, où 1 est très négatif et 10 est très positif : '{text}'. Donne uniquement le score numérique."}
            ]
        )
        sentiment_score = response.choices[0].message.content.strip()
        try:
            sentiment_score = int(sentiment_score)
        except ValueError:
            sentiment_score = 0
        return {"text": text, "score": sentiment_score, "explanation": "Score de sentiment numérique"}



    def analyze_sentiments(self, texts):
        """ Analyse le sentiment sur plusieurs textes. """
        results = []
        for text in texts:
            print(f"Analyse avec {self.model_name}: '{text}'...")
            sentiment = self.analyze_sentiment(text)
            results.append(sentiment)
        return results



    def parse_article_date(self, date_string):
        """ Tri des articles par date """
        try:
            return datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S%z")
        except ValueError:
            try:
                return datetime.strptime(date_string, "%Y-%m-%d")
            except ValueError as ve:
                print(f"Erreur: Date '{date_string}' non parsable. {ve}")
                return datetime(1900, 1, 1)






""" Autres méthodes """

"""  Les méthodes suivantes seront éventuellement à intégrer dans PrepareDatasetService(). """


def loading_dataset():
    """ Charger le dataset initial """
    try:
        initial_df = pd.read_csv(TRAINING_DATASET_FILE, parse_dates=['Date'], dayfirst=True)
        print(f"Dataset initial '{initial_df}' chargé avec succès.")
        return initial_df
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{TRAINING_DATASET_FILE}' n'a pas été trouvé.")
        exit()
    except Exception as e:
        print(f"Une erreur est survenue lors du chargement du dataset initial: {e}")
        exit()



def sort_api_data(btc_articles, analyzer):
    """ Tri des scores de sentiment du marché par date """
    api_scores_map = {}
    if btc_articles:
        btc_articles_sorted = sorted(btc_articles, key=lambda x: analyzer.parse_article_date(x["date"]))
        print("\nClés de api_scores_map (dates au format JJ/MM/AAAA) :")
        api_scores_map = {entry["date"]: entry["sentiment"]["polarity"] for entry in btc_articles_sorted}
        api_scores_map = {datetime.strptime(date, "%Y-%m-%dT%H:%M:%S%z").strftime("%d/%m/%Y"): score for date, score in api_scores_map.items()}
        print(list(api_scores_map.keys())[:5])
        print(f"Nombre total de dates dans api_scores_map: {len(api_scores_map)}")
        return api_scores_map
    else:
        print("Aucun article BTC-USD.CC à traiter pour les scores API.")
        return {}



def merge_data(initial_df, api_scores_map):
    """ Fusion des scores API et du dataset """
    initial_df['Date_for_join'] = initial_df['Date'].dt.strftime('%d/%m/%Y')
    initial_df['Score API'] = initial_df['Date_for_join'].map(api_scores_map)
    initial_df = initial_df.drop(columns=['Date_for_join'])
    initial_df['Score API'] = initial_df['Score API'].fillna(0)
    initial_df['Score API'] = initial_df['Score API'].apply(lambda x: 0 if x == 0 else x)
    return initial_df






""" *********** Génération et intégration des scores dans le dataset *********** """

initial_df = loading_dataset()
btc_articles = []

try:
    # Appel de l'API
    response = requests.get(MARKET_SENTIMENT_API_URL)
    response.raise_for_status()
    data = response.json()

    print("Données brutes récupérées de l'API (première entrée) :")
    if isinstance(data, list) and data:
        print(json.dumps(data[0], indent=4))
        btc_articles = data
    else:
        print("La réponse de l'API n'est pas une liste d'articles comme prévu.")
        btc_articles = []

    if btc_articles:
        # Initialisation de l'analysateur de sentiment
        analyzer = SentimentAnalyzer(API_MONSTER_KEY, MODEL_NAME)

        # Tri des articles
        btc_articles_sorted = sorted(btc_articles, key=lambda x: analyzer.parse_article_date(x["date"]))
        print(f"\nArticles BTC-USD.CC triés par date (du plus ancien au plus récent). Total: {len(btc_articles_sorted)} articles.")
        for i, article in enumerate(btc_articles_sorted[:3]):  # Limité à 3 pour l'affichage initial
            print(f"Date: {article['date']}, Titre: {article['title']}")
        if len(btc_articles_sorted) > 3:
            print("...")

        # Analyse des sentiments avec MonsterAPI
        print("\n" + "="*50)
        print("Analyse des sentiments avec MonsterAPI")
        print("="*50)

        # Extrait les titres des articles (limité aux 5 premiers pour ne pas épuiser le quota API rapidement)
        article_titles_for_analysis = [article["title"] for article in btc_articles_sorted[:5]]

        # Analyse des sentiments
        monster_sentiment_results = analyzer.analyze_sentiments(article_titles_for_analysis)

        for result in monster_sentiment_results:
            print(f"Titre: {result['text']}")
            print(f"  Score (MonsterAPI): {result['score']:.4f}")
            print(f"  Explication: {result['explanation']}\n")

        # Tri et fusion des données
        api_scores_map = sort_api_data(btc_articles, analyzer)
        initial_df = loading_dataset()
        print("INITIAL DF : ", initial_df)
        initial_df = merge_data(initial_df, api_scores_map)
        print("MERGE DF : ", initial_df)

        # Sauvegarde des données
        try:
            initial_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
            print(f"\nLe dataset fusionné a été enregistré dans '{OUTPUT_CSV_FILE}' avec succès.")
        except Exception as e:
            print(f"Une erreur est survenue lors de l'enregistrement du fichier CSV: {e}")

    else:
        print("Aucun article à trier ou traiter pour l'analyse des sentiments.")

except requests.exceptions.HTTPError as errh:
    print(f"Erreur HTTP: {errh}")
except requests.exceptions.ConnectionError as errc:
    print(f"Erreur de connexion: {errc}")
except requests.exceptions.Timeout as errt:
    print(f"Délai d'attente dépassé: {errt}")
except requests.exceptions.RequestException as err:
    print(f"Une erreur inattendue est survenue: {err}")
except KeyError as ke:
    print(f"Erreur de clé: {ke}. Vérifiez si la clé 'date' existe dans la réponse JSON des articles.")
except Exception as e:
    print(f"Une erreur générale est survenue: {e}")






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



