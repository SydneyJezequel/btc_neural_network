import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from technical_indicators import technical_indicators



class prepare_dataset:
    """ Classe qui re-traite le dataset avant l'entrainement du modèle """


    def __init__(self, initial_dataset):
        self.initial_dataset = initial_dataset



    def format_dataset(self):
        """ Méthode format_dataset() """
        tmp_dataset = self.initial_dataset.copy()
        # Convertir la colonne "Date" au format datetime :
        tmp_dataset['Date'] = pd.to_datetime(self.initial_dataset['Date'], format='%d/%m/%Y', errors='coerce')
        # errors='coerce' --> Les valeurs non converties sont remplacées par NaN.
        # Remplacer les points par un espace, puis suppression de l'espace, enfin remplacement de la virgule par un point :
        numeric_columns = ["Dernier", "Ouv.", " Plus Haut", "Plus Bas", "Variation %"]
        for col in numeric_columns:
            tmp_dataset.loc[:, col] = tmp_dataset[col].str.replace('.', ' ').str.replace(' ', '').str.replace(',', '.')
            # .loc[:, col] est utilisé pour sélectionner toutes les lignes (:) de la colonne spécifiée par col.
        # Conversion des colonnes numériques en float :
        for col in numeric_columns:
            tmp_dataset[col] = pd.to_numeric(tmp_dataset[col], errors='coerce')
        return tmp_dataset



    def delete_columns(self, tmp_dataset):
        """ Méthode delete_columns() """
        # Suppression des colonnes de départ :
        tmp_dataset = tmp_dataset.drop(columns=['Vol.', 'Variation %', 'Ouv.', ' Plus Haut', 'Plus Bas'])
        print('Dataset transformé :', tmp_dataset)
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
        # Ignorer les valeurs NAN (Remplir les valeurs NaN avec la moyenne des colonnes) :
        del tmp_dataset['Date']
        imputer = SimpleImputer(strategy='mean')
        tmp_dataset = imputer.fit_transform(tmp_dataset)
        return tmp_dataset



    def normalize_data(self, tmp_dataset):
        """ Méthode normalize_data() """
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(np.array(tmp_dataset).reshape(-1, 1))



    def create_train_and_test_dataset(self, model_dataset):
        """ Méthode create_train_and_test_dataset() """
        # Création des datasets d'entrainement et de test
        training_size = int(len(model_dataset) * 0.60)
        test_size = len(model_dataset) - training_size
        train_data, test_data = model_dataset[0:training_size, :], model_dataset[training_size:len(model_dataset), :1]
        # datas_for_model[0:training_size,:] : Sélectionne les training_size premières lignes de l'array.
        # datas_for_model[training_size:len(datas_for_model),:1] : Sélectionne toutes les lignes à partir de training_size jusqu'à la fin de l'array.
        print("dataset d'entrainement :", train_data.shape)
        print("dataset de test :", test_data.shape)
        return train_data, test_data



    def create_dataset(self, dataset, time_step=1):
        """ Méthode create_dataset() """
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



    def create_data_matrix(self, train_data, test_data):
        """ Méthode create_data_matrix() """
        # Définition du time_step (longueur des séquences ex : Si 15, chaque séquence d'entrée contiendra 3 valeurs du dataset) :
        time_step = 15
        # Création des ensembles de données d'entraînement et de test en utilisant la fonction create_dataset :
        x_train, y_train = self.create_dataset(train_data, time_step)
        x_test, y_test = self.create_dataset(test_data, time_step)
        # Affichage des dimensions des ensembles de données d'entraînement et de test :
        print("X_train: ", x_train.shape)
        print("y_train: ", y_train.shape)
        print("X_test: ", x_test.shape)
        print("y_test: ", y_test.shape)
        # Remodelage de X_train pour obtenir la forme [échantillons, time steps, caractéristiques]
        # Cela est nécessaire pour que les données soient compatibles avec les couches LSTM :
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        # Remodelage de X_test pour obtenir la forme [échantillons, time steps, caractéristiques]
        # Cela est nécessaire pour que les données soient compatibles avec les couches LSTM :
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        # Affichage des dimensions des ensembles de données d'entraînement et de test après remodelage :
        print("X_train: ", x_train.shape)
        print("X_test: ", x_test.shape)
        return x_train, y_train, x_test, y_test


