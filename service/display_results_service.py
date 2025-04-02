import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd





class DisplayResultsService:
    """ Affiche les résultats du modèle """


    def __init__(self):
        pass


    def plot_loss(self, history):
        """ Affichage des courbes de pertes """
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend(loc=0)
        plt.figure()
        plt.show()


    def zoom_plot_loss(self, history):
        """ Affichage des courbes de pertes avec agrandissement des zones ou se trouvent les courbes """
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


    def plot_residuals(self, y_true, y_pred, title):
        """ Affichage des résidus """
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.plot(residuals, label='Residuals')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(title)
        plt.xlabel('Observations')
        plt.ylabel('Residuals')
        plt.legend()
        plt.show()

    def plot_predictions(self, dates, predictions, time_step):
        """ Affichage des prédictions avec une date sur deux affichée """
        # Aligner les prédictions et les dates :
        predictions_with_dates = pd.DataFrame({
            'Date': dates[time_step:],
            'Prediction': predictions.flatten()
        })
        # Index :
        predictions_with_dates.set_index('Date', inplace=True)
        # Schéma :
        plt.figure(figsize=(12, 6))
        ax = sns.lineplot(data=predictions_with_dates, x=predictions_with_dates.index, y='Prediction', marker='o')
        # Sélection d'une date sur deux pour l'affichage :
        tick_indices = range(0, len(predictions_with_dates), 2)
        ax.set_xticks(predictions_with_dates.index[tick_indices])
        ax.set_xticklabels(predictions_with_dates.index[tick_indices], rotation=45)
        # Titres et légendes :
        plt.title('Prédictions')
        plt.xlabel('Date')
        plt.ylabel('Valeur')
        plt.grid()
        plt.show()

