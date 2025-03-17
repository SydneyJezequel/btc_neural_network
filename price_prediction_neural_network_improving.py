import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Attention, LayerNormalization
import numpy as np





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
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)


