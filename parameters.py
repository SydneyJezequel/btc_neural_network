from datetime import date




""" ************ Dataset and weights file parameters ************ """
DATASET_PATH = '../dataset/'
MODEL_PATH = '../model/'
TRAINING_DATASET_FILE = DATASET_PATH + 'btc_historic_cotations.csv'
FORMATED_BTC_COTATIONS_FILE = DATASET_PATH + 'formated_daily_btc_cotations.csv'
DATASET_FOR_PREDICTIONS = DATASET_PATH + 'btc_cotations_for_predictions.csv'
SAVED_MODEL = MODEL_PATH + f'model.weights.h5'
GRADIENT_BOOSTING_SAVED_MODEL = MODEL_PATH + 'gradient_boosting_model.pkl'




""" ************ Forecasting parameters ************ """
TIME_STEP = 15




""" ************ LLM parameters ************ """
LLM_API_URL = ''
LLM_API_KEY = ''
FINANCIAL_NEWS_API_URL = 'To be provided'
FINANCIAL_NEWS_API_KEY = 'https://yfapi.net/v8/finance/some_endpoint' #
ANALYSIS_SENTIMENT_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'




""" ************ API EODHD ************ """
START_DATE = date(2014, 1, 1)
END_DATE = date(2025, 1, 8)
START_DATE  = START_DATE.isoformat()
END_DATE = END_DATE.isoformat()
API_TOKEN = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXX'
MARKET_NEWS_API_URL = f"https://eodhd.com/api/news?s=btc-usd.cc&from={START_DATE}&to={END_DATE}&api_token={API_TOKEN}&fmt=json"
MARKET_SCORES_API_URL = f"https://eodhd.com/api/sentiments?s=btc-usd.cc&from={START_DATE}&to={END_DATE}&api_token={API_TOKEN}&fmt=json"




""" ************ LLM parameters (API monster) ************ """
API_MONSTER_KEY = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXX'
BASE_URL = "https://llm.monsterapi.ai/v1/"
MODEL_NAME = 'monsterapi/Llama3.3_70b'




""" ************ Transformer model parameters ************ """
FEATURE_SIZE =  9
NUM_LAYERS = 2
D_MODEL = 64
NHEAD = 8
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
SEQ_LENGTH = 30
PREDICTION_LENGTH = 1

