import os
import torch

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class CryptoDataset(Dataset):
    def __init__(self, root_path, feature_cols, target_col, past_period = 30, future_period = 2,
                 transform=None):
        super(CryptoDataset, self).__init__()

        self.data = pd.read_csv(os.path.join(root_path, "values.csv"))
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.past_period = past_period
        self.future_period = future_period
        self.transform = transform

        # [num_time, num_features]
        self.x = torch.tensor(self.data[feature_cols].values.astype(np.float32))
        self.y = torch.tensor(self.data[target_col].values.astype(np.float32))

        # Calculate returns for all possible windows
        numerator = self.y[self.past_period+self.future_period-1:]
        denominator = self.y[self.past_period-1:-self.future_period]
        self.y = numerator / denominator - 1
        self.y = self.y.unsqueeze(-1)  # [num_time - past_period - future_period + 1, 1]

        assert len(self.x)-self.past_period-self.future_period+1 == len(self.y), "Features and target length mismatch"

    def __len__(self):
        return len(self.data) - self.past_period - self.future_period + 1
    
    @property
    def feature_dim(self):
        return len(self.feature_cols)

    def __getitem__(self, idx):
        x = self.x[idx : idx+self.past_period]
        # future return
        y = self.y[idx]
        sample = {'x': x, 'y': y}

        if self.transform:
            sample = self.transform(sample)
        return sample
    

if __name__ == "__main__":
    dataset = CryptoDataset(
        root_path='data/BTC/raw/',
        feature_cols=['open', 'high', 'low', 'close', 'volume', 'ema_20', 'ema_60',
       'ema_100', 'ema_200', 'macd_20_60', 'macd_10_20', 'adx_10', 'adx_20',
       'adx_60', 'rsi_7', 'rsi_20', 'rsi_60', 'stoch_k_10', 'stoch_k_30',
       'stoch_k_100', 'roc_rsi_10', 'roc_rsi_20', 'roc_rsi_50', 'atr_20',
       'atr_60', 'atr_100', 'bb_width', 'bb_percent', 'obv', 'vwap', 'mom_5',
       'mom_20', 'mom_50', 'proc_5', 'proc_20', 'vol_ma_20', 'vol_ma_60'],
        target_col='close',
        past_period=30,
        future_period=2
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Feature dimension: {dataset.feature_dim}")
    sample = dataset[0]
    print(f"Sample x shape: {sample['x'].shape}, Sample y: {sample['y']}")