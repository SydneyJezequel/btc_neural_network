""" ************ Paramètres du dataset et du fichier de poids ************ """
DATASET_PATH = '../btc_neural_network/dataset/'
MODEL_PATH = '../btc_neural_network/model/'
TRAINING_DATASET_FILE = DATASET_PATH + 'btc_historic_cotations.csv'
FORMATED_BTC_COTATIONS_FILE = DATASET_PATH + 'formated_daily_btc_cotations.csv'
DATASET_FOR_PREDICTIONS = DATASET_PATH + 'btc_cotations_for_predictions.csv'
# DATASET_FOR_PREDICTIONS = DATASET_PATH + 'btc_cotations_for_predictions.csv'
SAVED_MODEL = MODEL_PATH + f'model.weights.h5'
GRADIENT_BOOSTING_SAVED_MODEL = MODEL_PATH + 'gradient_boosting_model.pkl'



""" ************ Paramètres de prévisions ************ """
TIME_STEP = 15



""" ************ Paramètres du Llm ************ """
API_KEY = ''
API_URL = ''

