import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Attention, LayerNormalization
from tensorflow.keras.callbacks import Callback
import numpy as np
import parameters
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score, mean_gamma_deviance, mean_poisson_deviance
import math

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

# Séparation des données en ensembles d'entraînement et de test
split_index = int(0.8 * len(X))
x_train, x_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

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

# Initialisation des tableaux pour stocker les métriques
metrics_history = {
    "epoch": [],
    "train_rmse": [],
    "train_mse": [],
    "train_mae": [],
    "test_rmse": [],
    "test_mse": [],
    "test_mae": [],
    "train_explained_variance": [],
    "test_explained_variance": [],
    "train_r2": [],
    "test_r2": [],
    "train_mgd": [],
    "test_mgd": [],
    "train_mpd": [],
    "test_mpd": [],
}

# Callback pour stocker les métriques toutes les 50 epochs
class MetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 50 == 0:
            train_predict = self.model.predict(x_train)
            test_predict = self.model.predict(x_test)

            train_predict = train_predict.reshape(-1, 1)
            test_predict = test_predict.reshape(-1, 1)

            # Supposons que les données sont déjà normalisées, donc pas besoin de scaler ici
            original_ytrain = y_train.reshape(-1, 1)
            original_ytest = y_test.reshape(-1, 1)

            metrics_history["epoch"].append(epoch + 1)
            metrics_history["train_rmse"].append(math.sqrt(mean_squared_error(original_ytrain, train_predict)))
            metrics_history["train_mse"].append(mean_squared_error(original_ytrain, train_predict))
            metrics_history["train_mae"].append(mean_absolute_error(original_ytrain, train_predict))
            metrics_history["test_rmse"].append(math.sqrt(mean_squared_error(original_ytest, test_predict)))
            metrics_history["test_mse"].append(mean_squared_error(original_ytest, test_predict))
            metrics_history["test_mae"].append(mean_absolute_error(original_ytest, test_predict))
            metrics_history["train_explained_variance"].append(explained_variance_score(original_ytrain, train_predict))
            metrics_history["test_explained_variance"].append(explained_variance_score(original_ytest, test_predict))
            metrics_history["train_r2"].append(r2_score(original_ytrain, train_predict))
            metrics_history["test_r2"].append(r2_score(original_ytest, test_predict))

            # Vérification des valeurs strictement positives avant le calcul de la déviance gamma
            if np.all(original_ytrain > 0) and np.all(train_predict > 0):
                mgd = mean_gamma_deviance(original_ytrain, train_predict)
                metrics_history["train_mgd"].append(mgd)
            else:
                metrics_history["train_mgd"].append(np.nan)

            if np.all(original_ytest > 0) and np.all(test_predict > 0):
                mgd = mean_gamma_deviance(original_ytest, test_predict)
                metrics_history["test_mgd"].append(mgd)
            else:
                metrics_history["test_mgd"].append(np.nan)

            # Vérification des valeurs strictement positives avant le calcul de la déviance poisson
            if np.all(original_ytrain > 0) and np.all(train_predict > 0):
                mpd = mean_poisson_deviance(original_ytrain, train_predict)
                metrics_history["train_mpd"].append(mpd)
            else:
                metrics_history["train_mpd"].append(np.nan)

            if np.all(original_ytest > 0) and np.all(test_predict > 0):
                mpd = mean_poisson_deviance(original_ytest, test_predict)
                metrics_history["test_mpd"].append(mpd)
            else:
                metrics_history["test_mpd"].append(np.nan)

# Entraînement du modèle
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=200,
    batch_size=32,
    verbose=1,
    callbacks=[MetricsCallback()]
)

# Sauvegarde du modèle
model.save_weights(parameters.SAVE_MODEL_PATH + 'model.weights.h5')

# Affichage des métriques stockées
print("Metrics History:")
for metric, values in metrics_history.items():
    print(f"{metric}: {values}")

# Évaluation du sur-apprentissage
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()
plt.show()

# Évaluation du sur-apprentissage avec agrandissement des zones où se trouvent les courbes
def plot_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    loss_array = np.array(loss)
    val_loss_array = np.array(val_loss)
    plt.figure(figsize=(12, 6))
    plt.plot(loss_array, label='Training Loss', color='red')
    plt.plot(val_loss_array, label='Validation Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Zoomed)')
    plt.legend()
    plt.ylim(0, 0.003)  # Zoom sur la zone des pertes basses
    plt.grid(True)
    plt.show()

plot_loss(history)

# Affichage des sur et sous apprentissage
loss_array = np.array(loss)
val_loss_array = np.array(val_loss)
print("Loss Array:", loss_array)
print("Validation Loss Array:", val_loss_array)

# Affichage des résidus
def plot_residuals(y_true, y_pred, title):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.plot(residuals, label='Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(title)
    plt.xlabel('Observations')
    plt.ylabel('Residuals')
    plt.legend()
    plt.show()

# Tracer les résidus
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

train_predict = train_predict.reshape(-1, 1)
test_predict = test_predict.reshape(-1, 1)

original_ytrain = y_train.reshape(-1, 1)
original_ytest = y_test.reshape(-1, 1)

plot_residuals(original_ytrain, train_predict, 'Training Residuals')
plot_residuals(original_ytest, test_predict, 'Test Residuals')
