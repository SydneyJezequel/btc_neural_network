import torch
import torch.nn as nn
import parameters






class TimeSeriesTransformer(nn.Module):
    """ Transformer Model """



    def __init__(
        # Transformer model features :
        self,
        feature_size=parameters.FEATURE_SIZE,
        num_layers=parameters.NUM_LAYERS,
        d_model=parameters.D_MODEL,
        nhead=parameters.NHEAD,
        dim_feedforward=parameters.DIM_FEEDFORWARD,
        dropout=parameters.DROPOUT,
        seq_length=parameters.SEQ_LENGTH,
        prediction_length=parameters.PREDICTION_LENGTH
    ):
        """ Constructor """
        """
        # feature_size: Number of features in each time step of the input data.
        # num_layers: Number of encoder layers in the Transformer model.
        # d_model: The embedding dimension of the Transformer model.
        # nhead: Number of attention heads in the multi-head attention mechanism.
        # dim_feedforward: Dimension of the hidden layer in the feedforward network.
        # dropout: Dropout rate for regularization to prevent overfitting.
        # seq_length: Number of time steps in the input sequence.
        : prediction_length: Number of future time steps to predict.
        """
        super(TimeSeriesTransformer, self).__init__()
        # Each input vector (feature_size) is embeded into a d_model-sized vector :
        self.input_fc = nn.Linear(feature_size, d_model)
        # Positional Encoding (simple learnable or sinusoidal). We'll do a learnable here :
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, d_model))
        # Transformer Encoder :
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Forecasts `prediction_length` steps is done for 1 step and 1 dimension (Close price).
        # To get multi-step and multi-dimensional forecasts, adjust accordingly.
        self.fc_out = nn.Linear(d_model, prediction_length)



    def forward(self, src):
        """ Propagate the input data through the Transformer model to generate a prediction. """
        """ src shape: [batch_size, seq_length, feature_size] """
        batch_size, seq_len, _ = src.shape
        # First project features into d_model :
        src = self.input_fc(src)  # -> [batch_size, seq_length, d_model]
        # Add positional embedding
        # pos_embedding -> [1, seq_length, d_model], so broadcast along batch dimension
        src = src + self.pos_embedding[:, :seq_len, :]
        # Transformer expects shape: [sequence_length, batch_size, d_model]
        src = src.permute(1, 0, 2)  # -> [seq_length, batch_size, d_model]
        # Pass through the transformer :
        encoded = self.transformer_encoder(src)  # [seq_length, batch_size, d_model]
        # Retrieve the output at the last time step for forecasting the future :
        last_step = encoded[-1, :, :]  # [batch_size, d_model]
        out = self.fc_out(last_step)  # [batch_size, prediction_length]
        return out

