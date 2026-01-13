# src/Moksha_1/models/deep_quant/dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Tuple
from Moksha_1.data.storage.timescaledb import TimescaleStorage

class MokshaDataset(Dataset):
    def __init__(self, symbols: list = None):
        """
        Pytorch Dataset for Financial ML.
        Aligns Features (T) with Target Returns (T+1).
        """
        self.db = TimescaleStorage()
        print("🧠 Agent 2: Loading training data from TimescaleDB...")
        
        # 1. Load Data
        self.features_df = self.db.get_features_df(symbols=symbols)
        self.bars_df = self.db.get_bars_df(symbols=symbols)
        
        if self.features_df.empty or self.bars_df.empty:
            print("❌ Error: Raw data not found in DB.")
            self.X, self.y = torch.tensor([]), torch.tensor([])
            return

        # 2. Preprocess & Align
        self.X, self.y, self.indices = self._prepare_tensors()
        
        print(f"✅ Dataset Ready. Samples: {len(self.X)}, Features: {self.X.shape[1] if len(self.X)>0 else 0}")

    def _prepare_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, list]:
        # Identify feature columns (exclude metadata)
        feature_cols = [c for c in self.features_df.columns if c not in ['time', 'symbol']]
        
        # --- FIX 1: Drop "Poison Pill" Columns (All-NaN features) ---
        # If a feature was never calculated, drop the column, don't kill the rows.
        self.features_df = self.features_df.dropna(axis=1, how='all')
        
        # Re-identify valid feature columns
        valid_feature_cols = [c for c in self.features_df.columns if c not in ['time', 'symbol']]
        if not valid_feature_cols:
            print("❌ Error: No valid feature columns remained after cleaning.")
            return torch.tensor([]), torch.tensor([]), []

        print(f"   > Using {len(valid_feature_cols)} valid features (Dropped {len(feature_cols) - len(valid_feature_cols)} empty ones).")

        # Prepare Targets
        returns = self.bars_df.sort_values(['symbol', 'time']).copy()
        returns['target_ret'] = returns.groupby('symbol')['close'].pct_change().shift(-1) # T+1
        
        # Merge
        data = pd.merge(
            self.features_df, 
            returns[['time', 'symbol', 'target_ret']], 
            on=['time', 'symbol'], 
            how='inner'
        )
        
        # --- FIX 2: Gentle Cleaning ---
        # Instead of strict dropna(), we fill missing features with 0.0 (Median Rank)
        # We only drop rows where the TARGET is missing (e.g., the very last day)
        data = data.dropna(subset=['target_ret'])
        data[valid_feature_cols] = data[valid_feature_cols].fillna(0.0)
        
        if data.empty:
            return torch.tensor([]), torch.tensor([]), []

        # Extract Tensors
        X_data = data[valid_feature_cols].values.astype(np.float32)
        y_data = data['target_ret'].values.astype(np.float32)
        indices = data[['time', 'symbol']].values
        
        return torch.tensor(X_data), torch.tensor(y_data).unsqueeze(1), indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]