import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score, mean_gamma_deviance, mean_poisson_deviance
import matplotlib.pyplot as plt






class NeuralNetwork:
    """ Classe qui instancie et entraîne le modèle """



    def __init__(self, x_train, y_train, n_splits=5, save_model_path=''):
        """ Constructeur """
        self.model = None
        self.x_train = x_train
        self.y_train = y_train
        self.n_splits = n_splits
        self.save_model_path = save_model_path
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.results = []
        self.rmse_results = []
        self.mse_results = []
        self.mae_results = []
        self.evs_results = []
        self.r2_results = []
        self.mgd_results = []
        self.mpd_results = []
        self.training_loss_results = []
        self.validation_loss_results = []
        self.cpt = 1



    def generate_model(self, x_train_fold):
        """ Méthode qui génère le modèle """
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(x_train_fold.shape[1], 1), activation="relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, kernel_regularizer=l2(0.01)))
        self.model.compile(loss="mean_squared_error", optimizer="adam")
        return self.model



    def train(self):
        """ Méthode d'entraînement du modèle """
        for train_index, val_index in self.tscv.split(self.x_train):
            print("tour de boucle : ", self.cpt)

            x_train_fold, x_val_fold = self.x_train[train_index], self.x_train[val_index]
            y_train_fold, y_val_fold = self.y_train[train_index], self.y_train[val_index]

            print("shape of datasets : ")
            print("x_train_fold : ", x_train_fold.shape)
            print("y_train_fold : ", y_train_fold.shape)
            print("x_val_fold : ", x_val_fold.shape)
            print("y_val_fold : ", y_val_fold.shape)

            model = self.generate_model(x_train_fold)
            model.add(LSTM(50, input_shape=(x_train_fold.shape[1], 1), activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(1, kernel_regularizer=l2(0.01)))
            model.compile(loss="mean_squared_error", optimizer="adam")

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            history = model.fit(
                x_train_fold, y_train_fold,
                validation_data=(x_val_fold, y_val_fold),
                epochs=200,
                batch_size=32,
                verbose=1,
                callbacks=[early_stopping]
            )

            print("Enregistrement du modèle.")
            model.save_weights(self.save_model_path + f'best_model_weights{self.cpt}.weights.h5')

            val_loss = model.evaluate(x_val_fold, y_val_fold, verbose=0)
            self.results.append(val_loss)

            self.predict(model, x_val_fold, y_val_fold)

            self.cpt += 1

        self.convert_results_to_numpy()

        self.loss_drawing(self.training_loss_results, self.validation_loss_results)



    def predict(self, model, x_val_fold, y_val_fold):
        """ Méthode de prédiction """
        val_predict = model.predict(x_val_fold)

        y_val_fold_reshaped = y_val_fold.reshape(-1, 1)
        print("y_val_fold_reshaped : ", y_val_fold_reshaped.shape)
        val_predict_reshaped = val_predict.reshape(-1, 1)
        print("val_predict_reshaped : ", val_predict_reshaped.shape)

        self.scaler.fit(y_val_fold_reshaped)

        original_yval = self.scaler.inverse_transform(y_val_fold_reshaped)
        val_predict_inversed = self.scaler.inverse_transform(val_predict_reshaped)

        self.evaluate_fold(original_yval, val_predict_inversed)




    def evaluate_fold(self, original_yval, val_predict_inversed):
        """ Méthode d'évaluation pour un pli """
        rmse = np.sqrt(mean_squared_error(original_yval, val_predict_inversed))
        mse = mean_squared_error(original_yval, val_predict_inversed)
        mae = mean_absolute_error(original_yval, val_predict_inversed)
        evs = explained_variance_score(original_yval, val_predict_inversed)
        r2 = r2_score(original_yval, val_predict_inversed)

        if np.all(original_yval > 0) and np.all(val_predict_inversed > 0):
            mgd = mean_gamma_deviance(original_yval, val_predict_inversed)
            mpd = mean_poisson_deviance(original_yval, val_predict_inversed)
        else:
            mgd, mpd = np.nan, np.nan

        self.rmse_results.append(rmse)
        self.mse_results.append(mse)
        self.mae_results.append(mae)
        self.evs_results.append(evs)
        self.r2_results.append(r2)
        self.mgd_results.append(mgd)
        self.mpd_results.append(mpd)



    def convert_results_to_numpy(self):
        """ Convertir les résultats en arrays numpy """
        self.rmse_results = np.array(self.rmse_results)
        self.mse_results = np.array(self.mse_results)
        self.mae_results = np.array(self.mae_results)
        self.evs_results = np.array(self.evs_results)
        self.r2_results = np.array(self.r2_results)
        self.mgd_results = np.array(self.mgd_results)
        self.mpd_results = np.array(self.mpd_results)
        self.training_loss_results = np.array(self.training_loss_results)
        self.validation_loss_results = np.array(self.validation_loss_results)



    def evaluate(self):
        """ Calcul des métriques """
        mean_rmse = np.mean(self.rmse_results)
        mean_mse = np.mean(self.mse_results)
        mean_mae = np.mean(self.mae_results)
        mean_evs = np.mean(self.evs_results)
        mean_r2 = np.mean(self.r2_results)
        mean_mgd = np.nanmean(self.mgd_results)  # Utiliser nanmean pour gérer les NaN
        mean_mpd = np.nanmean(self.mpd_results)  # Utiliser nanmean pour gérer les NaN
        mean_validation_loss = np.mean(self.validation_loss_results)
        mean_training_loss = np.mean(self.training_loss_results)

        # Afficher les résultats
        print("Mean Validation RMSE: ", mean_rmse)
        print("Mean Validation MSE: ", mean_mse)
        print("Mean Validation MAE: ", mean_mae)
        print("Mean Validation Explained Variance Score: ", mean_evs)
        print("Mean Validation R2 Score: ", mean_r2)
        print("Mean Validation MGD: ", mean_mgd)
        print("Mean Validation MPD: ", mean_mpd)
        print("Mean Validation Loss: ", mean_validation_loss)
        print("Mean Training Loss: ", mean_training_loss)

        # Retourner les métriques moyennes
        return {
            "mean_rmse": mean_rmse,
            "mean_mse": mean_mse,
            "mean_mae": mean_mae,
            "mean_evs": mean_evs,
            "mean_r2": mean_r2,
            "mean_mgd": mean_mgd,
            "mean_mpd": mean_mpd,
            "mean_validation_loss": mean_validation_loss,
            "mean_training_loss": mean_training_loss
        }



    def loss_drawing(self, training_loss_results, validation_loss_results):
        """ Tracer les pertes """
        plt.plot(training_loss_results, label='Training Loss')
        plt.plot(validation_loss_results, label='Validation Loss')
        plt.xlabel('Fold')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Comparison')
        plt.legend()
        plt.show()

