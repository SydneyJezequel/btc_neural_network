import os
import parameters
from TRASH.predictions_generator import PredictionsGenerator
from service.prepare_dataset_service import PrepareDataset
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import joblib














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
prepare_dataset = PrepareDataset()
initial_dataset = pd.read_csv(DATASET_PATH + DATASET_FILE)



print(" ************ Étape 2 : Préparation of the Dataset ************ ")


# Formatage des colonnes
dataset = prepare_dataset.format_dataset(initial_dataset)


# Suppression des colonnes :
dataset = prepare_dataset.delete_columns(dataset)


# dataset = dataset.apply(pd.to_numeric, errors='coerce')
# dataset = dataset.fillna(dataset.mean())

dataset = prepare_dataset.add_technicals_indicators(dataset)
print(" forme dataset 0 : ", dataset.shape)


# Initialisation du scaler :
print("forme du dataset 1 : ", dataset.shape)
# scaler = prepare_dataset.get_fitted_scaler(dataset)
scaler = joblib.load('../scaler.save')
print("Valeurs minimales (min_) :", scaler.min_)
print("Échelle (scale_) :", scaler.scale_)
print("forme du dataset 2 : ", dataset.shape)
print("dataset 2 : ", dataset)


# Normalise dataset :
dataset = prepare_dataset.normalize_datas(dataset, scaler)
print("dataset normalisé : ", dataset)
print("dataset normalisé shape : ", dataset.shape)


# Création des datasets d'entraînement et de test :
train_data, test_data = create_train_and_test_dataset(dataset)
test_data = dataset

"""
# Chargement du dataset de test :
test_data = joblib.load('test_data.save')
"""
print("forme du dataset 3 : ", test_data.shape)
print("dataset 3 : ", test_data)


# CONVERSION DES ARRAY EN MATRICE ?
""" 
# Conversion des arrays en matrice
time_step = 15
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
"""


# Création du modèle avec la même architecture :
model = Sequential()
model.add(LSTM(10, input_shape=(None, 1), activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")


# Vérification de l'existence du fichier :
weights_path = os.path.join(MODEL_FOR_PREDICTIONS_PATH, 'model.weights.h5')
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Le fichier de poids n'existe pas à l'emplacement spécifié : {weights_path}")


# Chargement des poids :
model.load_weights(weights_path)


# Génération des prédictions :
generator = PredictionsGenerator(model, test_data, scaler)
predictions = generator.generate_predictions()
generator.plot_predictions(predictions)
