import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from BO.time_series_transformer import TimeSeriesTransformer
from BO.transformer_dataset import TransformerDataset
import parameters
from BO.transformer_metrics_logger import TransformerMetricsLogger
from service.display_results_service import DisplayResultsService
from service.prepare_dataset_service import PrepareDatasetService
from service.train_transformer_model_service import TrainTransformerModelService






""" ************* Parameters ************* """

TRAINING_DATASET_FILE = parameters.TRAINING_DATASET_FILE




""" ************* Dataset Preparation ************* """

# Loading initial dataset :
df = pd.read_csv(TRAINING_DATASET_FILE)

# Data formatting and indicators adding into data :
prepare_dataset = PrepareDatasetService()
df = prepare_dataset.data_formatting_for_transformer_model(df)
df = prepare_dataset.add_technical_indicators(df)

# Select data features and scale :
data_scaled, scaler, feature_cols = prepare_dataset.select_and_scale_features(df)

# Create dataset for model
target_col_idx = feature_cols.index('Dernier')
seq_length = 30
pred_length = 1
dataset = TransformerDataset(data_scaled, seq_length, pred_length, len(feature_cols), target_col_idx)

# Train/Validation/Test Split (80% train, 10% val, 10% test) :
train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)
test_size = len(dataset) - train_size - val_size

# Perform sequential splitting :
train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, len(dataset)))
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




""" ************* Model Training ************* """

# Transformer model and device initialisation :
model = TimeSeriesTransformer(
    feature_size=len(feature_cols),
    num_layers=2,
    d_model=64,
    nhead=8,
    dim_feedforward=256,
    dropout=0.1,
    seq_length=seq_length,
    prediction_length=pred_length
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Logs initialisation :
metrics_logger = TransformerMetricsLogger(scaler, target_col_idx)

# Training execution :
train_transformer_model_service = TrainTransformerModelService()
trained_model, train_losses, val_losses = train_transformer_model_service.train_transformer_model(
    model,
    train_loader,
    val_loader,
    lr=1e-3,
    epochs=20,
    device=device,
    metrics_logger=metrics_logger
)




""" ***************** Evaluate and Plot Results ***************** """

display_results = DisplayResultsService()

# Comparison btc price vs model price predictions :
display_results.evaluate_model(trained_model, test_loader, scaler, feature_cols, target_col_idx, window_width=45, start_index=70, pred_length=1, device=device)

# Display loss curves (with and without zoom) :
display_results.transformer_plot_loss(train_losses, val_losses)
display_results.transformer_zoom_plot_loss(train_losses, val_losses)




""" ************* Display Metrics ************* """

# Display metrics :
metrics_logger.print_metrics()

# Display metrics plots :
display_results = DisplayResultsService()
metrics_history = metrics_logger.metrics_history
display_results.transformer_plot_metrics(metrics_history, metrics_to_plot=["rmse", "mse", "mae", "explained_variance", "r2", "mgd", "mpd"])

