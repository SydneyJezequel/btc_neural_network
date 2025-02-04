""" ************ Paramètres du dataset et du fichier de poids ************ """
DATASET_PATH = '../btc_neural_network/dataset/'
PATH_TRAINING_DATASET = DATASET_PATH + 'training_dataset/'
DATASET_FILE = 'btc_historic_cotations.csv'
DATASET_FOR_MODEL = 'dataset_for_model.csv'
SAVE_MODEL_PATH = '../btc_neural_network/model/'




""" ************ Paramètres du modèle ************ """
LSTM_LAYERS_NUMBER = 50
ACTIVATION_FUNCTION = "relu"
DROPOUT = 0.2
LOSS_FUNCTION = "mean_squared_error"
OPTIMIZER = "adam"
EPOCHS_NUMBER = 200
BATCH_SIZE = 32




""" ************ Paramètres pour faire des prédictions ************ """
LOAD_MODEL_PATH = SAVE_MODEL_PATH
PATH_PREDICTIONS_DATASET = DATASET_PATH + 'dataset_for_predictions/'
DATASET_FOR_PREDICTIONS = PATH_PREDICTIONS_DATASET + 'btc_cotations_for_predictions.csv'




""" ************ Paramètres du Llm ************ """
API_KEY = ''
API_URL = ''