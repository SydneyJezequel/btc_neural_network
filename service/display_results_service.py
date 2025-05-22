from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go






class DisplayResultsService:
    """ Displays the model results """



    def __init__(self):
        pass



    def display_all_dataset(self, dataset):
        """ Display the entire dataset """
        fig = px.line(dataset, x=dataset.Date, y=dataset.Dernier,
                      labels={'Date': 'date', 'Last': 'Close Stock'})
        fig.update_traces(marker_line_width=2, opacity=0.8, marker_line_color='orange')
        fig.update_layout(title_text='Whole period of timeframe of Bitcoin close price 2014-2025', plot_bgcolor='white',
                          font_size=15, font_color='black')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.show()



    def plot_loss(self, history):
        """ Display loss curves """
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
        """ Display loss curves with zoom """
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
        # Zoom on loss area (graphic bottom) :
        plt.ylim(0, 0.003)
        plt.grid(True)
        plt.show()



    def plot_residuals(self, y_true, y_pred, title):
        """ Display residuals """
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
        """ Display predictions with every other date shown """
        # Transform predictions in numpy array :
        predictions = np.array(predictions)
        # Create new dates for predictions :
        derniere_date = dates.max()
        nouvelles_dates = [derniere_date + timedelta(days=i) for i in range(1, len(predictions) + 1)]
        nouvelles_dates = [date.strftime("%Y-%m-%d") for date in nouvelles_dates]
        # Align predictions and new dates :
        predictions_with_dates = pd.DataFrame({
            'Date': nouvelles_dates,
            'Prediction': predictions.flatten()
        })
        # Graphic :
        predictions_with_dates.set_index('Date', inplace=True)
        plt.figure(figsize=(12, 6))
        ax = sns.lineplot(data=predictions_with_dates, x=predictions_with_dates.index, y='Prediction', marker='o')
        # Display 1/2 date :
        tick_indices = range(0, len(predictions_with_dates), 2)
        ax.set_xticks(predictions_with_dates.index[tick_indices])
        ax.set_xticklabels(predictions_with_dates.index[tick_indices], rotation=45)
        plt.title('Predictions')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.grid()
        plt.show()



    def display_dataset_and_predictions(self, dataset, predictions_dataset):
        """ Display the main dataset with the possibility to add other price curves """
        fig = go.Figure()
        # Add first price curve :
        fig.add_trace(go.Scatter(
            x=dataset['Date'],
            y=dataset['Dernier'],
            mode='lines',
            name='Bitcoin',
            line=dict(color='orange', width=2)
        ))
        # Additionnal price curves :
        fig.add_trace(go.Scatter(
            x=predictions_dataset['Date'],
            y=predictions_dataset['Dernier'],
            mode='lines',
            name='predictions',
            line=dict(width=2)
        ))
        # Graphic :
        fig.update_layout(
            title_text='Whole period of timeframe of Bitcoin close price 2014-2025',
            plot_bgcolor='white',
            font_size=15,
            font_color='black',
            xaxis_title='Date',
            yaxis_title='Close Price'
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.show()



    def plot_initial_dataset_and_predictions(self, initial_dataset, formated_dataset, train_predict_plot, test_predict_plot):
        """ Display all datasets (initial, training predictions, test predictions) """
        plt.figure(figsize=(15, 6))
        plt.plot(initial_dataset['Date'], formated_dataset['Dernier'], label='Datas', color='black')
        plt.plot(initial_dataset['Date'], train_predict_plot, label='Training predictions', color='green')
        plt.plot(initial_dataset['Date'], test_predict_plot, label='Test predictions', color='red')
        plt.title("Real price vs LSTM predictions")
        plt.xlabel("Date")
        plt.ylabel("Price (denormalized)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    def plot_metrics_history(self, metrics_history, metrics_to_plot=None, title_prefix="Evolution of"):
        """ Displays the evolution of training/test metrics over the epochs. """
        available_metrics = set()
        for key in metrics_history.keys():
            if "_" in key:
                available_metrics.add(key.split("_")[1])

        if metrics_to_plot is None:
            metrics_to_plot = sorted(available_metrics)

        epochs = metrics_history["epoch"]

        for metric in metrics_to_plot:
            train_key = f"train_{metric}"
            test_key = f"test_{metric}"

            if train_key in metrics_history or test_key in metrics_history:
                plt.figure(figsize=(10, 5))
                if train_key in metrics_history:
                    plt.plot(epochs, metrics_history[train_key], label="Train")
                if test_key in metrics_history:
                    plt.plot(epochs, metrics_history[test_key], label="Test")
                plt.xlabel("Epoch")
                plt.ylabel(metric.upper())
                plt.title(f"{title_prefix} {metric.upper()}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            else:
                print(f"No available data for metric : {metric}")

