import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go






class DisplayResultsService:
    """ Affiche les résultats du modèle """


    def __init__(self):
        pass


    def display_all_dataset(self, dataset):
        """  Affichage de l'intégralité du dataset """
        fig = px.line(dataset, x=dataset.Date, y=dataset.Dernier,
                      labels={'Date': 'date', 'Dernier': 'Close Stock'})
        fig.update_traces(marker_line_width=2, opacity=0.8, marker_line_color='orange')
        fig.update_layout(title_text='Whole period of timeframe of Bitcoin close price 2014-2025', plot_bgcolor='white',
                          font_size=15, font_color='black')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.show()


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













    """ **************************** TEST ***************************** """

    def display_dataset_and_predictions(self, dataset, additional_datasets=None):
        """
        Affichage de l'intégralité du dataset principal avec la possibilité d'ajouter d'autres courbes de prix.

        :param dataset: DataFrame principal contenant les données de prix.
        :param additional_datasets: Liste de DataFrames supplémentaires à ajouter au graphique.
        """
        # Création de la figure avec la première courbe de prix
        fig = go.Figure()

        # Ajout de la première courbe de prix
        fig.add_trace(go.Scatter(
            x=dataset['Date'],
            y=dataset['Dernier'],
            mode='lines',
            name='Bitcoin',
            line=dict(color='orange', width=2)
        ))

        # Ajout des courbes supplémentaires si fournies
        if additional_datasets:
            for i, additional_dataset in enumerate(additional_datasets, start=2):
                fig.add_trace(go.Scatter(
                    x=additional_dataset['Date'],
                    y=additional_dataset['Dernier'],
                    mode='lines',
                    name='prédictions',
                    line=dict(width=2)
                ))

        # Mise à jour de la mise en page
        fig.update_layout(
            title_text='Whole period of timeframe of Bitcoin close price 2014-2025',
            plot_bgcolor='white',
            font_size=15,
            font_color='black',
            xaxis_title='Date',
            yaxis_title='Close Price'
        )

        # Suppression des grilles
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        # Affichage du graphique
        fig.show()


