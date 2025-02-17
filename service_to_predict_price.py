import os
import parameters
from BO.predictions_generator import PredictionsGenerator
from BO.prepare_dataset import PrepareDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt  # Correction de l'importation

""" ****************************** Paramètres ****************************** """
DATASET_PATH = parameters.DATASET_PATH
PATH_TRAINING_DATASET = parameters.PATH_TRAINING_DATASET
DATASET_FILE = parameters.DATASET_FILE
DATASET_FOR_MODEL = parameters.DATASET_FOR_MODEL
MODEL_FOR_PREDICTIONS_PATH = parameters.MODEL_FOR_PREDICTIONS_PATH

""" **************************** Méthodes **************************** """

def create_train_and_test_dataset(model_dataset):
    """ Méthode create_train_and_test_dataset() """
    training_size = int(len(model_dataset) * 0.60)
    test_size = len(model_dataset) - training_size
    train_data, test_data = model_dataset[0:training_size, :], model_dataset[training_size:len(model_dataset), :1]
    print("dataset d'entraînement :", train_data.shape)
    print("dataset de test :", test_data.shape)
    return train_data, test_data

""" **************************** Exécution du script principal **************************** """

print(" ************ Étape 1 : Loading dataset ************ ")
initial_dataset = pd.read_csv(DATASET_PATH + DATASET_FILE)

print(" ************ Étape 2 : Préparation of the Dataset ************ ")

prepare_dataset = PrepareDataset()

# Formatage des colonnes
dataset = prepare_dataset.format_dataset(initial_dataset)

# Suppression des colonnes
dataset = prepare_dataset.delete_columns(dataset)

# Enregistrement du dataset au format csv
dataset.to_csv(PATH_TRAINING_DATASET + DATASET_FOR_MODEL, index=False)

# Vérification et gestion des valeurs manquantes
dataset = dataset.apply(pd.to_numeric, errors='coerce')
dataset = dataset.fillna(dataset.mean())

# Normalisation du dataset
scaler = prepare_dataset.get_fitted_scaler(dataset)


print("forme du dataset : ", dataset.shape)

scaler.fit(dataset)
dataset = scaler.transform(dataset)

# Création des datasets d'entraînement et de test
train_data, test_data = create_train_and_test_dataset(dataset)

# Utilisation
weights_path = os.path.join(MODEL_FOR_PREDICTIONS_PATH, 'model.weights.h5')

# Vérifiez si le fichier de poids existe
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Le fichier de poids n'existe pas à l'emplacement spécifié : {weights_path}")

# Créez le modèle avec la même architecture
model = Sequential()
model.add(LSTM(10, input_shape=(train_data.shape[1], 1), activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")

# Chargement des poids
model.load_weights(weights_path)

# Génération des prédictions
generator = PredictionsGenerator(model, test_data, scaler)
predictions = generator.generate_predictions()
generator.plot_predictions(predictions)
