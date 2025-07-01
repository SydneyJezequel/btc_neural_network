from openai import OpenAI
from datetime import datetime






class SentimentAnalyzer:
    """ Sentiment market analyzer"""



    def __init__(self, api_key, model_name, base_url):
        """ Constructor """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.monster_client = OpenAI(api_key=self.api_key, base_url=self.base_url)



    def analyze_sentiment(self, text):
        """ Sentiment scoring on multiple articles """
        response = self.monster_client.chat.completions.create(
            # model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            model=self.model_name,
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



    def analyze_sentiments(self, articles):
        """ Sentiment scoring on an article """
        results = []
        for article in articles:
            sentiment = self.analyze_sentiment(article['title'])
            sentiment['date'] = article['date']
            results.append(sentiment)
        return results



    def parse_article_date(self, date_string):
        """ News sorting by date """
        try:
            return datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S%z")
        except ValueError:
            try:
                return datetime.strptime(date_string, "%Y-%m-%d")
            except ValueError as ve:
                print(f"Error: Date '{date_string}' non parsable. {ve}")
                return datetime(1900, 1, 1)

