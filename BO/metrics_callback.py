import math
import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import ( mean_squared_error, mean_absolute_error, explained_variance_score, r2_score, mean_poisson_deviance, mean_gamma_deviance)
from service.prepare_dataset_service import PrepareDatasetService






class MetricsCallback(Callback):
    """ Stockage des métriques """



    def __init__(self, x_train, y_train, x_test, y_test, metrics_history):
        """ Constructeur """
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.metrics_history = metrics_history
        self.prepare_dataset = PrepareDatasetService()



    def on_epoch_end(self, epoch, logs=None):
        print("hello")
        """ Callback pour stocker les métriques toutes les 50 epochs """
        if (epoch + 1) % 50 == 0:
            train_predict = self.model.predict(self.x_train)
            test_predict = self.model.predict(self.x_test)

            train_predict = train_predict.reshape(-1, 1)
            test_predict = test_predict.reshape(-1, 1)

            scaler = self.prepare_dataset.get_fitted_scaler(train_predict)
            train_predict = scaler.inverse_transform(train_predict)
            test_predict = scaler.inverse_transform(test_predict)

            original_ytrain = scaler.inverse_transform(self.y_train.reshape(-1, 1))
            original_ytest = scaler.inverse_transform(self.y_test.reshape(-1, 1))

            self.metrics_history["epoch"].append(epoch + 1)
            self.metrics_history["train_rmse"].append(math.sqrt(mean_squared_error(original_ytrain, train_predict)))
            self.metrics_history["train_mse"].append(mean_squared_error(original_ytrain, train_predict))
            self.metrics_history["train_mae"].append(mean_absolute_error(original_ytrain, train_predict))
            self.metrics_history["test_rmse"].append(math.sqrt(mean_squared_error(original_ytest, test_predict)))
            self.metrics_history["test_mse"].append(mean_squared_error(original_ytest, test_predict))
            self.metrics_history["test_mae"].append(mean_absolute_error(original_ytest, test_predict))
            self.metrics_history["train_explained_variance"].append(explained_variance_score(original_ytrain, train_predict))
            self.metrics_history["test_explained_variance"].append(explained_variance_score(original_ytest, test_predict))
            self.metrics_history["train_r2"].append(r2_score(original_ytrain, train_predict))
            self.metrics_history["test_r2"].append(r2_score(original_ytest, test_predict))

            # Vérification des valeurs strictement positives avant le calcul de la déviance gamma :
            if np.all(original_ytrain > 0) and np.all(train_predict > 0):
                mgd = mean_gamma_deviance(original_ytrain, train_predict)
                self.metrics_history["train_mgd"].append(mgd)
            else:
                self.metrics_history["train_mgd"].append(np.nan)

            if np.all(original_ytest > 0) and np.all(test_predict > 0):
                mgd = mean_gamma_deviance(original_ytest, test_predict)
                self.metrics_history["test_mgd"].append(mgd)
            else:
                self.metrics_history["test_mgd"].append(np.nan)

            # Vérification des valeurs strictement positives avant le calcul de la déviance poisson :
            if np.all(original_ytrain > 0) and np.all(train_predict > 0):
                mpd = mean_poisson_deviance(original_ytrain, train_predict)
                self.metrics_history["train_mpd"].append(mpd)
            else:
                self.metrics_history["train_mpd"].append(np.nan)

            if np.all(original_ytest > 0) and np.all(test_predict > 0):
                mpd = mean_poisson_deviance(original_ytest, test_predict)
                self.metrics_history["test_mpd"].append(mpd)
            else:
                self.metrics_history["test_mpd"].append(np.nan)

