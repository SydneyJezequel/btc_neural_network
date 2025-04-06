import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from service.technical_indicators_service import TechnicalIndicatorsService
from service.display_results_service import DisplayResultsService
import parameters






class PrepareDatasetService:
    """ Traite le dataset avant l'entrainement du modèle """


    def __init__(self):
        pass
        parameters.FORMATED_BTC_COTATIONS


    def format_dataset(self, initial_dataset):
        """ Préparation des données """
        tmp_dataset = initial_dataset.copy()
        tmp_dataset['Date'] = pd.to_datetime(initial_dataset['Date'], format='%d/%m/%Y', errors='coerce')
        tmp_dataset = tmp_dataset.sort_values(by='Date')
        numeric_columns = ["Dernier", "Ouv.", " Plus Haut", "Plus Bas", "Variation %"]
        for col in numeric_columns:
            tmp_dataset[col] = tmp_dataset[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
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
        tmp_dataset['MA_150'] = TechnicalIndicatorsService.ma(tmp_dataset, 150)
        tmp_dataset['MA_100'] = TechnicalIndicatorsService.ma(tmp_dataset, 100)
        tmp_dataset['MA_50'] = TechnicalIndicatorsService.ma(tmp_dataset, 50)
        tmp_dataset['RSI'] = TechnicalIndicatorsService.rsi(tmp_dataset, 14)
        # Ajout des signaux générés par les indicateurs :
        TechnicalIndicatorsService.calculate_signal(tmp_dataset, 50, 150)
        TechnicalIndicatorsService.calculate_signal(tmp_dataset, 100, 150)
        TechnicalIndicatorsService.calculate_signal(tmp_dataset, 50, 100)
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


    def create_train_and_test_dataset(self, model_dataset):
        """ Création des datasets d'entrainement et tests """
        training_size = int(len(model_dataset) * 0.60)
        train_data, test_data = model_dataset.iloc[0:training_size, :], model_dataset.iloc[training_size:len(model_dataset), :]
        return train_data, test_data


    def create_dataset(self, dataset, time_step=1):
        """ Méthode qui génère les datasets d'entrainement et de test """
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset.iloc[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset.iloc[i + time_step, 0])
        return np.array(dataX), np.array(dataY)


    def create_dataset_for_predictions(self, dataset, time_step=1):
        """ Méthode qui génère le dataset utilisé pour les prédictions """
        dataX = []
        print("dataset before transforme : ", dataset)
        for i in range(len(dataset) - time_step):
            a = dataset.iloc[i:(i + time_step), 0].values
            dataX.append(a)
        return np.array(dataX)


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


    def save_tmp_dataset(self, dataset):
        """ Sauvegarde du dataset dans un fichier .csv """
        saved_dataset = pd.DataFrame(dataset)
        saved_dataset.to_csv(parameters.FORMATED_BTC_COTATIONS, index=False, encoding='utf-8')


    def prepare_dataset(self, dataset, cutoff_date = '2020-01-01'):
        """ Préparation du dataset """

        # Préparation du dataset :
        tmp_dataset = self.format_dataset(dataset)
        tmp_dataset = self.delete_columns(tmp_dataset)

        # Sauvegarde du dataset formaté :
        self.save_tmp_dataset(tmp_dataset)

        # Affichage de l'intégralité du dataset avant la transformation des prix :
        display_results = DisplayResultsService()
        display_results.display_all_dataset(tmp_dataset)

        # Sous-échantillonnage :
        tmp_dataset = self.subsample_old_data(tmp_dataset, cutoff_date, fraction=0.1)

        # Normalisation :
        tmp_dataset_copy = tmp_dataset.copy()
        columns_to_normalize = ['Dernier']
        scaler = self.get_fitted_scaler(tmp_dataset_copy[columns_to_normalize])
        # joblib.dump(scaler, 'scaler.save')
        model_dataset = tmp_dataset
        normalized_datas = self.normalize_datas(tmp_dataset_copy[columns_to_normalize], scaler)
        model_dataset[columns_to_normalize] = normalized_datas
        print("dataset d'entrainement normalisé :", model_dataset)
        print("model_dataset shape : ", model_dataset.shape)

        # Suppression de la colonne date :
        del model_dataset['Date']

        # Création des datasets d'entrainement et test :
        train_data, test_data = self.create_train_and_test_dataset(model_dataset)
        time_step = 15
        x_train, y_train =  self.create_dataset(train_data, time_step)
        x_test, y_test =  self.create_dataset(test_data, time_step)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        return x_train, y_train, x_test, y_test, scaler


    def prepare_dataset_to_predict(self, dataset, time_step):
        """ Préparation du dataset """

        # Préparation du dataset :
        tmp_dataset = self.format_dataset(dataset)
        tmp_dataset = self.delete_columns(tmp_dataset)

        # Normalisation :
        tmp_dataset_copy = tmp_dataset.copy()
        columns_to_normalize = ['Dernier']
        scaler = self.get_fitted_scaler(tmp_dataset_copy[columns_to_normalize])
        dataset = tmp_dataset
        print("dataset avant normalisation : ", dataset)
        normalized_datas = self.normalize_datas(tmp_dataset_copy[columns_to_normalize], scaler)
        dataset[columns_to_normalize] = normalized_datas
        print("dataset d'entrainement normalisé :", dataset)
        print("dataset shape : ", dataset.shape)

        # Suppression de la colonne date :
        del dataset['Date']

        # Création du dataset utilisé pour les prédictions :
        dataset = self.create_dataset_for_predictions(dataset, time_step)
        print("create_dataset_for_predictions : ", dataset)
        dataset = dataset.reshape(dataset.shape[0], dataset.shape[1], 1)
        print("after reshape : ", dataset)

        return dataset, scaler

