
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2


# Créez une nouvelle instance du modèle avec la même architecture
model = Sequential()
model.add(LSTM(50, input_shape=(x_train_fold.shape[1], 1), activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, kernel_regularizer=l2(0.01)))
model.compile(loss="mean_squared_error", optimizer="adam")

# Chargez les poids du modèle
model.load_weights('model_weights.h5')