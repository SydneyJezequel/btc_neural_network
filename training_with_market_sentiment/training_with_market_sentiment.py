import requests
import parameters
from parameters import FINANCIAL_NEWS_API
from transformers import pipeline # Hugging Face Transformers library, which provides access to pre-trained models like BERT.
import pandas as pd
from service.prepare_dataset_service import PrepareDatasetService






""" ***************** Parameters ***************** """

FINANCIAL_NEWS_API_KEY = parameters.FINANCIAL_NEWS_API_KEY
FINANCIAL_NEWS_API_URL = parameters.FINANCIAL_NEWS_API_URL
FORMATED_BTC_COTATIONS_FILE = parameters.FORMATED_BTC_COTATIONS_FILE






""" ****************** Récupération des données *************** """

# Chargement du dataset :
dataset = pd.read_csv(FORMATED_BTC_COTATIONS_FILE)

# Chargement des données de l'Api :
api_key = 'YAHOO_FINANCE_API_KEY'
prepare_dataset_service = PrepareDatasetService()
news_articles = prepare_dataset_service.fetch_bitcoin_news(FINANCIAL_NEWS_API_KEY)






""" ****************** Extraction du sentiment de marché *************** """

# Jeu  de test :
sample_text = "Bitcoin price surges as institutional investors show increased interest."
sentiment, score = prepare_dataset_service.analyze_sentiment(sample_text)
print(f"Sentiment: {sentiment}, Score: {score}")


# En réel :
sentiment, score = prepare_dataset_service.analyze_sentiment(news_articles)
print(f"Sentiment: {sentiment}, Score: {score}")






""" ****************** Intégration du sentiment de marché dans les prédictions *************** """

prepare_dataset_service.integrate_market_sentiment(dataset, news_articles)










"""
Notes:
API Key: Replace 'YOUR_API_KEY' with your actual NewsAPI key.
Dataset: The sample dataset is minimal. In a real-world scenario, you would have a more comprehensive dataset with historical Bitcoin prices.
Sentiment Analysis: The sentiment analysis is performed on the titles of the news articles. You can extend this to analyze the full content of the articles if needed.
Libraries: Ensure you have the required libraries installed. You can install them using pip:
Copier
pip install requests pandas transformers
This code provides a basic framework to get you started. You may need to adjust and expand it based on your specific requirements and the complexity of your project.
"""

