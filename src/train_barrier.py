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

class BarrierNet(nn.Module):
    def __init__(self, input_size):
        super(BarrierNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.network(x)

def get_volatility(close_series, span=20):
    """Calculates daily volatility (standard deviation of returns)"""
    returns = close_series.pct_change()
    vol = returns.ewm(span=span).std()
    return vol

def apply_triple_barrier(df, barrier_width=1.0, vertical_barrier=5):
    """
    Marcos Lopez de Prado's Triple Barrier Method.
    Label 1: Hits Upper Barrier first.
    Label 0: Hits Lower Barrier first OR Vertical Barrier (Time expiry).
    """
    labels = []
    
    # Pre-calculate volatility for dynamic barriers
    df['volatility'] = get_volatility(df['close'])
    
    # We iterate (slow but accurate) to simulate path
    # Optimally this should be vectorized, but loop is safer for logic
    closes = df['close'].values
    vols = df['volatility'].values
    times = df.index
    
    for i in range(len(df) - vertical_barrier):
        current_price = closes[i]
        current_vol = vols[i]
        
        # Dynamic Barriers based on volatility (e.g., 2% if vol is high, 0.5% if low)
        # Avoid zero vol
        if np.isnan(current_vol) or current_vol < 0.001: current_vol = 0.01
            
        upper = current_price * (1 + (barrier_width * current_vol))
        lower = current_price * (1 - (barrier_width * current_vol))
        
        outcome = 0 # Default (Time Expiry / Stop Loss)
        
        # Look forward 'vertical_barrier' days
        for future_i in range(i + 1, i + vertical_barrier + 1):
            future_price = closes[future_i]
            
            if future_price >= upper:
                outcome = 1 # Profit Take Hit
                break
            elif future_price <= lower:
                outcome = 0 # Stop Loss Hit
                break
        
        labels.append(outcome)
    
    # Pad the end
    labels.extend([0] * vertical_barrier)
    df['target'] = labels
    return df

def train_barrier_brain():
    logger.info("🧠 Initializing TRIPLE BARRIER Training (De Prado Method)...")
    
    db = TimescaleStorage()
    feature_engine = FeatureEngine()
    
    # 1. GET DATA
    logger.info("⏳ Fetching deep history (100k rows)...")
    raw_df = db.get_bars_df(settings.UNIVERSE, limit=100000) 
    if raw_df.empty: return

    # 2. GENERATE FEATURES
    logger.info("🧬 Generating Factors...")
    full_features = feature_engine.create_features(raw_df, training_mode=True)
    
    # 3. LAGS
    cols_to_lag = [c for c in full_features.columns if c not in ['time', 'symbol', 'target']]
    for lag in [1, 2]:
        lagged_df = full_features.groupby('symbol')[cols_to_lag].shift(lag)
        lagged_df.columns = [f"{c}_lag{lag}" for c in cols_to_lag]
        full_features = pd.concat([full_features, lagged_df], axis=1)

    # 4. TRIPLE BARRIER LABELING (The Alpha Fix)
    logger.info("🚧 Applying Dynamic Triple Barriers...")
    labeled_dfs = []
    for sym, group in full_features.groupby('symbol'):
        group = group.sort_values('time').reset_index(drop=True)
        # Barrier Width 1.5x Volatility, Hold Max 5 Days
        labeled_group = apply_triple_barrier(group, barrier_width=1.5, vertical_barrier=5)
        labeled_dfs.append(labeled_group)
    
    full_features = pd.concat(labeled_dfs).dropna()
    
    # Balance Classes (Optional but good)
    positives = full_features[full_features['target'] == 1]
    negatives = full_features[full_features['target'] == 0]
    logger.info(f"   📊 Class Balance: Wins {len(positives)} | Losses/Timeouts {len(negatives)}")

    # 5. PREPARE TENSORS
    y = full_features['target']
    drop_cols = ['time', 'symbol', 'target', 'typical_price', 'next_ret', 'volatility']
    X_df = full_features.drop(columns=[c for c in drop_cols if c in full_features.columns])
    
    input_size = len(X_df.columns)
    X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    # 6. TRAIN
    model = BarrierNet(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    epochs = 200
    logger.info(f"🏋️ Starting Training ({epochs} epochs)...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            predicted = (outputs > 0.5).float()
            accuracy = (predicted.eq(y_tensor).sum() / y_tensor.shape[0]).item()
            logger.info(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f} | Acc: {accuracy*100:.2f}%")

    # 7. SAVE (Overwrite the old alpha model for compatibility)
    save_path = "/app/moksha_alpha_v1.pth"
    torch.save(model.state_dict(), save_path)
    logger.info(f"✅ Triple Barrier Brain Saved: {save_path}")

if __name__ == "__main__":
    train_barrier_brain()