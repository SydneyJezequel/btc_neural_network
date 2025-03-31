""" ************ Paramètres du dataset et du fichier de poids ************ """
DATASET_PATH = '../btc_neural_network/dataset/'
# PATH_TRAINING_DATASET = DATASET_PATH + 'training_dataset/'
PATH_TRAINING_DATASET = DATASET_PATH + 'dataset_for_predictions/'
TRAINING_DATASET_FILE = 'btc_cotations_for_training.csv'
DATASET_FILE_FOR_TEST_PREDICTIONS = 'btc_cotations_for_test_predictions.csv'
DATASET_FOR_MODEL = 'dataset_for_model.csv'
SAVE_MODEL_PATH = '../btc_neural_network/model/trained_model/'
MODEL_FOR_PREDICTIONS_PATH = '../btc_neural_network/model/model_for_predictions/'







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
DATASET_FOR_PREDICTIONS_PATH= '../btc_neural_network/dataset/dataset_for_predictions/'
DATASET_FOR_PREDICTIONS = DATASET_FOR_PREDICTIONS_PATH + 'btc_cotations_for_test_predictions.csv'





""" ************ Paramètres du Llm ************ """
API_KEY = ''
API_URL = ''