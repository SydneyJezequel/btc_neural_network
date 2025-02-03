import requests





class Llm:
    """ Classe qui instancie et entraîne le modèle """



    def __init__(self, all_predictions, all_true_values, api_url, api_key):
        """ Constructeur """
        self.all_predictions = all_predictions
        self.all_true_values = all_true_values
        self.api_url = api_url
        self.api_key = api_key



    def analyze_predictions(self):
        """ Analyser les prédictions avec un LLM via une API """

        # Convertir les prédictions et les valeurs réelles en chaînes de caractères :
        predictions_str = ", ".join(map(str, self.all_predictions))
        true_values_str = ", ".join(map(str, self.all_true_values))

        # Créer le prompt pour l'analyse :
        analysis_prompt = f"""
        Voici les prédictions du modèle : {predictions_str}
        Et voici les valeurs réelles correspondantes : {true_values_str}
        Peux-tu analyser ces prédictions et fournir des insights ?
        """
        self.call_llm(analysis_prompt)



    def call_llm(self, analysis_prompt):
        """ Appel du LLM via Api """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "prompt": analysis_prompt,
            "max_tokens": 150  # Ajustez selon vos besoins
        }

        response = requests.post(self.api_url, headers=headers, json=data)

        if response.status_code == 200:
            analysis_results = response.json().get("choices", [{}])[0].get("text", "")
            # Afficher les résultats de l'analyse
            print("Analyse des prédictions par le LLM :")
            print(analysis_results)
        else:
            print(f"Erreur lors de l'appel à l'API LLM : {response.status_code}")
            print(response.text)