import torch.nn as nn
import numpy as np
import torch
""" ********* TEST ********* """
from torch.optim.lr_scheduler import ReduceLROnPlateau
""" ********* TEST ********* """





class TrainTransformerModelService:
    """ Processing the transformer model training """



    def __init__(self):
        """ Constructor """
        pass



    def train_transformer_model(self,
            # Hyperparameters :
            model,
            train_loader,
            val_loader=None,
            lr=1e-3,
            epochs=20,
            device='cpu',
            metrics_logger=None,
    ):
        """ Trains a Transformer model for time series tasks. """
        # Losses, optimizer and model initialisation :
        criterion = nn.MSELoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        """ ********* TEST ********* """
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        """ ********* TEST ********* """
        """ ********* TEST ********* """
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        """ ********* TEST ********* """
        model.to(device)
        train_epoch_losses = []
        val_epoch_losses = []
        """ ********* TEST ********* """
        # Early Stopping parameters :
        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = 5
        val_loss= float('inf')
        """ ********* TEST ********* """

        # Training :
        for epoch in range(epochs):
            epoch_train_losses_temp = []
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                output = model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_train_losses_temp.append(loss.item())
            # Adding train losses in numpy array for plot losses :
            mean_train_loss = np.mean(epoch_train_losses_temp)
            train_epoch_losses.append(mean_train_loss)

            # Valuation :
            if val_loader is not None:
                epoch_val_losses_temp = []
                model.eval()
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val = x_val.to(device)
                        y_val = y_val.to(device)
                        output_val = model(x_val)
                        loss_val = criterion(output_val, y_val)
                        epoch_val_losses_temp.append(loss_val.item())

                # Adding val losses in numpy array for plot losses :
                mean_val_loss = np.mean(epoch_val_losses_temp)
                val_epoch_losses.append(mean_val_loss)
                """ ********* TEST ********* """
                scheduler.step(mean_val_loss)
                """ ********* TEST ********* """
                """ ********* TEST ********* """
                # Early Stopping check :
                should_stop, best_val_loss, epochs_no_improve = self.check_early_stopping(
                    mean_val_loss, best_val_loss, epochs_no_improve, patience
                )
                if should_stop:
                    break
                """ ********* TEST ********* """

                # Print losses :
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {mean_train_loss:.6f}, Val Loss: {mean_val_loss:.6f}")
            else:
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {mean_train_loss:.6f}")

            # Metrics calculation and Storage :
            if metrics_logger:
                metrics_logger.metrics_history["epoch"].append(epoch + 1)
                # Training metrics :
                train_predictions, train_targets = metrics_logger.get_predictions_and_targets(model, train_loader, device)
                metrics_logger.calculate_and_store_metrics(train_predictions, train_targets, "train_")
                # Valuation metrics :
                if val_loader is not None:
                    val_predictions, val_targets = metrics_logger.get_predictions_and_targets(model, val_loader, device)
                    metrics_logger.calculate_and_store_metrics(val_predictions, val_targets, "val_")

        return model, train_epoch_losses, val_epoch_losses



    def check_early_stopping(self, mean_val_loss, best_val_loss, epochs_no_improve, patience):
        """ Checks if the training should be stopped early based on validation loss. """
        print("mean_val_loss : ", mean_val_loss, " best_val_loss : ", best_val_loss)
        should_stop = False
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping triggered!")
                should_stop = True
        return should_stop, best_val_loss, epochs_no_improve

