import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error,explained_variance_score, r2_score, mean_poisson_deviance, mean_gamma_deviance
import math






class TransformerMetricsLogger:
    """ Calculating and displaying metrics for Transformer model """



    def __init__(self, scaler, target_col_idx):
        """ Constructor """
        self.scaler = scaler
        self.target_col_idx = target_col_idx
        self.metrics_history = {
            "epoch": [],
            "train_rmse": [], "train_mse": [], "train_mae": [], "train_explained_variance": [], "train_r2": [],
            "train_mgd": [], "train_mpd": [],
            "val_rmse": [], "val_mse": [], "val_mae": [], "val_explained_variance": [], "val_r2": [], "val_mgd": [],
            "val_mpd": [],
        }



    def get_predictions_and_targets(self, model, data_loader, device):
        """ Get all predictions and original targets from a DataLoader to calculate the metrics """
        # Initialization :
        model.eval()
        all_predictions = []
        all_targets = []

        # Predictions calculation :
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(device)
                output = model(x_batch).squeeze().cpu().numpy()
                target = y_batch.squeeze().cpu().numpy()
                all_predictions.extend(output)
                all_targets.extend(target)
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # Predictions retrieving :
        temp_preds = np.zeros((len(all_predictions), self.scaler.n_features_in_))
        temp_preds[:, self.target_col_idx] = all_predictions
        temp_targets = np.zeros((len(all_targets), self.scaler.n_features_in_))
        temp_targets[:, self.target_col_idx] = all_targets
        original_predictions = self.scaler.inverse_transform(temp_preds)[:, self.target_col_idx]
        original_targets = self.scaler.inverse_transform(temp_targets)[:, self.target_col_idx]

        return original_predictions, original_targets



    def calculate_and_store_metrics(self, predictions, targets, prefix):
        """ Calculates and stores metrics """

        # Metrics calculation :
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        evs = explained_variance_score(targets, predictions)

        # Adding metrics into metrics_history :
        self.metrics_history[f"{prefix}rmse"].append(math.sqrt(mse))
        self.metrics_history[f"{prefix}mse"].append(mse)
        self.metrics_history[f"{prefix}mae"].append(mae)
        self.metrics_history[f"{prefix}r2"].append(r2)
        self.metrics_history[f"{prefix}explained_variance"].append(evs)
        if np.all(targets > 0) and np.all(predictions > 0):
            mgd = mean_gamma_deviance(targets, predictions)
            mpd = mean_poisson_deviance(targets, predictions)
            self.metrics_history[f"{prefix}mgd"].append(mgd)
            self.metrics_history[f"{prefix}mpd"].append(mpd)
        else:
            self.metrics_history[f"{prefix}mgd"].append(np.nan)
            self.metrics_history[f"{prefix}mpd"].append(np.nan)



    def print_metrics(self):
        """ Prints metrics for each epoch. """
        print("****** Metrics for each epoch ******")
        for i, epoch in enumerate(self.metrics_history["epoch"]):
            print(f"\nEpoch {epoch}:")

            # Print RMSE, MSE, MAE, EVS, R2 :
            print(
                f"  RMSE Train: {self.metrics_history['train_rmse'][i]:.4f} | RMSE Val: {self.metrics_history['val_rmse'][i]:.4f}")
            print(
                f"  MSE Train: {self.metrics_history['train_mse'][i]:.4f} | MSE Val: {self.metrics_history['val_mse'][i]:.4f}")
            print(
                f"  MAE Train: {self.metrics_history['train_mae'][i]:.4f} | MAE Val: {self.metrics_history['val_mae'][i]:.4f}")
            print(
                f"  EVS Train: {self.metrics_history['train_explained_variance'][i]:.4f} | EVS Val: {self.metrics_history['val_explained_variance'][i]:.4f}")
            print(
                f"  R2 Train: {self.metrics_history['train_r2'][i]:.4f} | R2 Val: {self.metrics_history['val_r2'][i]:.4f}")

            # Print MGD :
            train_mgd = self.metrics_history['train_mgd'][i]
            val_mgd = self.metrics_history['val_mgd'][i]
            if not np.isnan(train_mgd) and not np.isnan(val_mgd):
                print(f"  MGD Train: {train_mgd:.4f} | MGD Val: {val_mgd:.4f}")

            # Print MPD :
            train_mpd = self.metrics_history['train_mpd'][i]
            val_mpd = self.metrics_history['val_mpd'][i]
            if not np.isnan(train_mpd) and not np.isnan(val_mpd):
                print(f"  MPD Train: {train_mpd:.4f} | MPD Val: {val_mpd:.4f}")

