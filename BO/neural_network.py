from keras.models import Sequential
from keras.layers import LSTM, Dense



class neural_network:
    """ Classe qui définit le réseau de neurones """


    def __init__(self):
        self.model = None



    def create_model(self):
        """ Création du modèle """
        print(" ******************** Création et entrainement du modèle ******************** ")
        # Initialisation d'un modèle séquentiel :
        self.model = Sequential()
        # Ajout d'une couche LSTM (Long Short-Term Memory) au modèle :
        # - 10 unités LSTM
        # - input_shape=(None, 1) : La couche LSTM a une sortie de forme (None, 10), ce qui signifie qu'elle produit 10 valeurs pour chaque séquence d'entrée.
        # - activation="relu" : Fonction d'activation ReLU (Rectified Linear Unit)
        self.model.add(LSTM(10, input_shape=(None, 1), activation="relu"))
        self.model.add(Dense(1))
        # Compilation du modèle :
        # - loss="mean_squared_error" : Utilisation de la moyenne des erreurs quadratiques comme fonction de perte (écart prévision/résultat).
        # - optimizer="adam" : Utilisation de l'optimiseur Adam pour gérer la descente de gradient (algorithme d'optimisation utilisé pour minimiser la fonction de perte. Elle ajuste les paramètres du modèle (comme les poids et les biais dans un réseau de neurones) de manière itérative pour réduire la valeur de la fonction de perte.).
        self.model.compile(loss="mean_squared_error", optimizer="adam")



    def train_model(self, X_train, y_train, X_test, y_test, epochs=200, batch_size=32, verbose=1):
        """ Entrainement du modèle """
        # Entrainement du modèle :
        # - X_train, y_train : données et cibles d'entraînement.
        # - validation_data=(X_test, y_test) : données et cibles de validation.
        # - epochs=200 : nombre d'époques (passages complets sur l'ensemble des données d'entraînement).
        # - batch_size=32 : taille des lots (batch size) pour l'entraînement.
        # - verbose=1 : mode verbeux pour afficher les informations de progression pendant l'entraînement.
        history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=verbose)
        return history



    def generate_predictions(self, X_train, X_test):
        """ Génération de prédiction par le modèle """
        print(" ******************** Génération de prédiction par le modèle ******************** ")
        train_predict = self.model.predict(X_train)
        test_predict = self.model.predict(X_test)
        print("train_predict shape:", train_predict.shape)
        print("test_predict shape:", test_predict.shape)
        return train_predict, test_predict


