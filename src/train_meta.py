import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from Moksha_1.data.storage.timescaledb import TimescaleStorage
from Moksha_1.data.processing.feature_engineering import FeatureEngine
from Moksha_1.config import settings
from Moksha_1.utils.logger import logger

class MetaNet(nn.Module):
    def __init__(self, input_size):
        super(MetaNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )
    def forward(self, x): return self.network(x)

def train_meta_brain():
    logger.info("🧠 Training META-LABELER (V6.0)...")
    db = TimescaleStorage()
    fe = FeatureEngine()
    
    raw_df = db.get_bars_df(settings.UNIVERSE, limit=100000)
    if raw_df.empty: return
    
    df = fe.create_features(raw_df, training_mode=True)
    
    if 'adx' not in df.columns:
        logger.error("❌ 'adx' missing. Feature Engine update failed.")
        return

    signal_mask = (df['dist_sma50'] > 0) & (df['adx'] > 20)
    trend_df = df[signal_mask].copy()
    logger.info(f"   📉 Filtered to {len(trend_df)} trend candidates.")
    
    if len(trend_df) < 50:
        logger.error("❌ Not enough data points.")
        return

    trend_df['fwd_max'] = trend_df.groupby('symbol')['close'].transform(lambda x: x.rolling(10).max().shift(-10))
    trend_df['fwd_min'] = trend_df.groupby('symbol')['close'].transform(lambda x: x.rolling(10).min().shift(-10))
    trend_df['target'] = np.where((trend_df['fwd_max'] > trend_df['close']*1.05) & (trend_df['fwd_min'] > trend_df['close']*0.98), 1.0, 0.0)
    trend_df = trend_df.dropna()

    X_df = trend_df[fe.feature_columns]
    y = trend_df['target']
    
    input_size = len(X_df.columns)
    logger.info(f"📊 Training Input Size: {input_size} (Should be 5)")
    
    X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    
    model = MetaNet(input_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    for epoch in range(100):
        optimizer.zero_grad()
        loss = criterion(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1)%20==0: logger.info(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "/app/moksha_meta_v1.pth")
    logger.info("✅ Saved /app/moksha_meta_v1.pth")

if __name__ == "__main__":
    train_meta_brain()
