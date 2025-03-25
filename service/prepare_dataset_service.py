import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from service.technical_indicators import technical_indicators



class PrepareDataset:
    """ Classe qui re-traite le dataset avant l'entrainement du modèle """



    def __init__(self):
        pass



    def format_dataset(self, initial_dataset):
        """ Préparation des données """
        tmp_dataset = initial_dataset.copy()
        tmp_dataset['Date'] = pd.to_datetime(initial_dataset['Date'], format='%d/%m/%Y', errors='coerce')
        tmp_dataset = tmp_dataset.sort_values(by='Date')
        numeric_columns = ["Dernier", "Ouv.", " Plus Haut", "Plus Bas", "Variation %"]
        for col in numeric_columns:
            tmp_dataset.loc[:, col] = tmp_dataset[col].str.replace('.', ' ').str.replace(' ', '').str.replace(',', '.')
        for col in numeric_columns:
            tmp_dataset[col] = pd.to_numeric(tmp_dataset[col], errors='coerce')
        return tmp_dataset


    def delete_columns(self, tmp_dataset):
        """ Suppression des colones du dataset d'origine """
        tmp_dataset = tmp_dataset.drop(columns=['Vol.', 'Variation %', 'Ouv.', ' Plus Haut', 'Plus Bas'])
        return tmp_dataset


    def add_technicals_indicators(self, tmp_dataset):
        """ Méthode add_technicals_indicators() """
        # Ajout des indicateurs dans les colonnes :
        tmp_dataset['MA_150'] = technical_indicators.ma(tmp_dataset, 150)
        tmp_dataset['MA_100'] = technical_indicators.ma(tmp_dataset, 100)
        tmp_dataset['MA_50'] = technical_indicators.ma(tmp_dataset, 50)
        tmp_dataset['RSI'] = technical_indicators.rsi(tmp_dataset, 14)
        # Ajout des signaux générés par les indicateurs :
        technical_indicators.calculate_signal(tmp_dataset, 50, 150)
        technical_indicators.calculate_signal(tmp_dataset, 100, 150)
        technical_indicators.calculate_signal(tmp_dataset, 50, 100)
        # Supprimer les lignes où MA_150 est NaN
        tmp_dataset = tmp_dataset.dropna(subset=['MA_150'])
        tmp_dataset.to_csv('../btc_neural_network/dataset/training_dataset/dataset_for_model.csv', index=False)
        return tmp_dataset


    def get_fitted_scaler(self, tmp_dataset):
        """ Méthode pour obtenir le scaler ajusté """
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(tmp_dataset)
        return scaler


    def normalize_datas(self, tmp_dataset, scaler):
        """ Méthode normalize_data() """
        return scaler.transform(tmp_dataset)
    """
    def normalize_datas2(self, tmp_dataset, scaler, columns_to_normalize):
        #  Méthode normalize_datas() 
        # Normalisation des colonnes spécifiées
        scaler.transform(tmp_dataset_copy[columns_to_normalize])
        return tmp_dataset_copy
    """

    def create_train_and_test_dataset(self, model_dataset):
        """ Création des datasets d'entrainement et tests """
        training_size = int(len(model_dataset) * 0.60)
        train_data, test_data = model_dataset.iloc[0:training_size, :], model_dataset.iloc[training_size:len(model_dataset), :]
        return train_data, test_data
    """
        def create_train_and_test_dataset(self, model_dataset):
        # Création des datasets d'entrainement et tests
        training_size = int(len(model_dataset) * 0.60)
        train_data, test_data = model_dataset[0:training_size, :], model_dataset[training_size:len(model_dataset), :1]
        return train_data, test_data
    """

    def create_dataset(self, dataset, time_step=1):
        """ Méthode qui génère les datasets d'entrainement et de test """
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset.iloc[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset.iloc[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    """
    def create_dataset(self, dataset, time_step=1):
        # Méthode create_dataset()
        dataX, dataY = [], []
        # Boucle sur le dataset pour créer des séquences de longueur time_step :
        for i in range(len(dataset) - time_step - 1):
            # Extrait une séquence de longueur time_step à partir de l'index i
            a = dataset[i:(i + time_step), 0]
            # Ajoute la séquence à dataX :
            dataX.append(a)
            # Ajoute la valeur cible correspondante à dataY :
            dataY.append(dataset[i + time_step, 0])
        # Convertit les listes dataX et dataY en arrays numpy pour faciliter leur utilisation dans les modèles de machine learning :
        return np.array(dataX), np.array(dataY)
    """


    def create_data_matrix(self, dataset, time_step=15):
        """ Méthode create_data_matrix() """
        # Création des ensembles de données en utilisant la fonction create_dataset :
        x, y = self.create_dataset(dataset, time_step)
        # Remodelage de X pour obtenir la forme [échantillons, time steps, caractéristiques]
        # Cela est nécessaire pour que les données soient compatibles avec les couches LSTM :
        x = x.reshape(x.shape[0], x.shape[1], 1)
        # Affichage des dimensions des ensembles de données après remodelage :
        print("dataset x: ", x.shape)
        # On ne modifie pas la forme du dataset y car elle sert de valeur cible à comparer avec le dataset x :
        print("dataset y: ", y.shape)
        return x, y



    def subsample_old_data(self, tmp_dataset, cutoff_date, fraction=0.1):
        """ Sous-échantillonnage des anciennes données """
        old_data = tmp_dataset[tmp_dataset['Date'] < cutoff_date]
        recent_data = tmp_dataset[tmp_dataset['Date'] >= cutoff_date]
        old_data_sampled = old_data.sample(frac=fraction, random_state=42)
        combined_data = pd.concat([old_data_sampled, recent_data])
        combined_data = combined_data.sort_values(by='Date').reset_index(drop=True)
        return combined_data



    def add_lag_features(self, dataset, lags):
        """ Ajout des caractéristiques de lag """
        for lag in lags:
            dataset[f'Lag_{lag}'] = dataset['Dernier'].shift(lag)
        return dataset



    def calculate_historical_volatility(self, dataset, window=252):
        """ Calcul de la volatilité historique """
        dataset['Returns'] = np.log(dataset['Dernier'] / dataset['Dernier'].shift(1))
        volatility = dataset['Returns'].rolling(window=window).std() * np.sqrt(252)
        return volatility