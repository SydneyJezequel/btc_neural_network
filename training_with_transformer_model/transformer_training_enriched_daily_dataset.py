import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from BO.transformer_dataset import TransformerDataset
from BO.time_series_transformer import TimeSeriesTransformer
from BO.transformer_metrics_logger import TransformerMetricsLogger
from service.display_results_service import DisplayResultsService
from service.prepare_dataset_service import PrepareDatasetService
import parameters
from service.train_transformer_model_service import TrainTransformerModelService






""" ************* Parameters ************* """

API_TOKEN = parameters.API_TOKEN
MARKET_SCORES_API_URL = parameters.MARKET_SCORES_API_URL
TRAINING_DATASET_FILE = parameters.TRAINING_DAILY_DATASET_FILE

# ************ A supprimer ???? ************ #
FEATURE_SIZE =  parameters.FEATURE_SIZE
NUM_LAYERS = parameters.NUM_LAYERS
D_MODEL = parameters.D_MODEL
NHEAD = parameters.NHEAD
DIM_FEEDFORWARD = parameters.DIM_FEEDFORWARD
DROPOUT = parameters.DROPOUT
SEQ_LENGTH = parameters.SEQ_LENGTH
PREDICTION_LENGTH = parameters.PREDICTION_LENGTH
# ************ A supprimer ???? ************ #




""" ************* Merge Btc cotations and api sentiment scores ************* """

# Loading initial dataset :
dataset = pd.read_csv(TRAINING_DATASET_FILE)

# Adding market sentiment scores :
"""
# api scores loading and sorting :
api_response_data = prepare_dataset.get_api_market_sentiment_scores(MARKET_SCORES_API_URL)
api_scores_map = prepare_dataset.sort_scores_api_data(api_response_data)

# api scores and btc dataset merging :
dataset = prepare_dataset.merge_data(dataset, api_scores_map)
print("Merged dataset : ", dataset)
"""




""" ************* Dataset Preparation ************* """

# Data formatting and indicators adding into data :
prepare_dataset = PrepareDatasetService()
cutoff_date = '2020-01-01'
dataset, feature_cols, scaler = prepare_dataset.prepare_many_dimensions_dataset_for_transformer_model(dataset, cutoff_date)

# Create dataset for model :
target_col_idx = feature_cols.index('Dernier')
seq_length = 5
pred_length = 1
dataset = dataset.values
dataset =  TransformerDataset(dataset, seq_length, pred_length, len(feature_cols), target_col_idx)

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
    dropout=0.3,
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
    lr=1e-4,
    epochs=50,
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

