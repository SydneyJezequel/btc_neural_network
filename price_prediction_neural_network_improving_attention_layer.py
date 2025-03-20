import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Attention, LayerNormalization
import numpy as np
import parameters
import matplotlib.pyplot as plt





# Fonction pour construire la fenêtre de données
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)




# Paramètres
WINDOW_SIZE = 50  # Longueur de la séquence d'entrée
FEATURES = 1  # Nombre de variables (ex: uniquement le prix ici)

# Exemple de dataset (remplace par tes vraies données)
data = np.sin(np.linspace(0, 100, 4000))  # Ex: une série temporelle simulée
X, y = create_sequences(data, WINDOW_SIZE)

# Reshape pour correspondre aux attentes de la GRU
X = X.reshape((X.shape[0], X.shape[1], FEATURES))




# Définition du modèle GRU avec Attention
class AttentionLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.attention = Attention()

    def call(self, inputs):
        query = inputs  # La dernière sortie de la GRU est la "query"
        value = inputs  # L'ensemble des sorties de la GRU est la "value"
        attention_output = self.attention([query, value])
        return attention_output




# Construction du modèle
input_layer = Input(shape=(WINDOW_SIZE, FEATURES))
gru_output, state_h = GRU(64, return_sequences=True, return_state=True)(input_layer)
attention_output = AttentionLayer()(gru_output)
norm_output = LayerNormalization()(attention_output)
output_layer = Dense(1)(norm_output)
model = Model(inputs=input_layer, outputs=output_layer)


# Compilation avec Huber Loss
model.compile(optimizer="adam", loss=tf.keras.losses.Huber(delta=1.0))


# Affichage du modèle
model.summary()


# Entraînement
model.fit(X,
          y,
          epochs=50,
          batch_size=32,
          validation_split=0.2
          )






""" ************************* A ajouter ************************* """


# AJOUTER LES METRIQUES.


"""
history = model.fit(X,
          y, 
          epochs=50,
          batch_size=32,
          validation_split=0.2
          )

# Sauvegarde du modèle :
model.save_weights(parameters.SAVE_MODEL_PATH + f'model.weights.h5')


# Affichage des métriques stockées :
print("Metrics History:")
for metric, values in metrics_history.items():
    print(f"{metric}: {values}")


# Évaluation du sur-apprentissage :
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()
plt.show()

"""




