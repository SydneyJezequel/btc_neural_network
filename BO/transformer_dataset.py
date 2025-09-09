from torch.utils.data import Dataset
import torch






class TransformerDataset(Dataset):
    """ Dataset for a Transformer model handling time series data """



    def __init__(self, data, seq_length=60, prediction_length=1, feature_dim=4, target_column_idx=3):
        """
        Constructor
            dataset: numpy array of shape [num_samples, num_features]
            seq_length: how many timesteps in the input sequence
            prediction_length: how many future steps we want to predict
            feature_dim: total number of features in dataset (for dimension checking)
            target_column_idx: which column to use as the target (e.g., close=3)
        """
        self.data = data
        self.seq_length = seq_length
        self.pred_length = prediction_length
        self.feature_dim = feature_dim
        self.target_column_idx = target_column_idx



    def __len__(self):
        # The maximum starting index is total_length - seq_length - prediction_length
        return len(self.data) - self.seq_length - self.pred_length + 1



    def __getitem__(self, idx):
        # Input sequence :
        x = self.data[idx : idx + self.seq_length]
        # Future price(s) :
        y = self.data[idx + self.seq_length : idx + self.seq_length + self.pred_length, self.target_column_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

