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

# --- DEFINE MODEL ARCHITECTURE ---
class MokshaSuperNet(nn.Module):
    def __init__(self, input_size):
        super(MokshaSuperNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)

def train_new_brain():
    logger.info("🧠 Initializing Super-Brain Training Sequence...")
    
    db = TimescaleStorage()
    feature_engine = FeatureEngine()
    
    # 1. GET DATA
    logger.info("⏳ Fetching historical data (Limit increased to 100k)...")
    raw_df = db.get_bars_df(settings.UNIVERSE, limit=100000) 
    
    if len(raw_df) < 2000:
        logger.error("❌ Not enough data! Please run 'python src/backfill.py' first.")
        return

    # 2. GENERATE FEATURES
    logger.info("🧬 Generating Base Factors...")
    full_features = feature_engine.create_features(raw_df, training_mode=True)
    
    # 3. LAGS
    logger.info("🐇 Generating Lag Features (Memory)...")
    cols_to_lag = [c for c in full_features.columns if c not in ['time', 'symbol', 'target']]
    for lag in [1, 2]:
        lagged_df = full_features.groupby('symbol')[cols_to_lag].shift(lag)
        lagged_df.columns = [f"{c}_lag{lag}" for c in cols_to_lag]
        full_features = pd.concat([full_features, lagged_df], axis=1)

    # 4. CLEANUP
    full_features['target'] = full_features.groupby('symbol')['log_returns'].shift(-1)
    full_features = full_features.dropna()

    # 5. PREPARE TENSORS
    y = full_features['target'] * 100
    y = y.clip(-1.0, 1.0)
    
    drop_cols = ['time', 'symbol', 'target', 'typical_price']
    X_df = full_features.drop(columns=[c for c in drop_cols if c in full_features.columns])
    
    input_size = len(X_df.columns)
    logger.info(f"📊 Training on {len(X_df)} rows with {input_size} features (Base + Lags).")
    
    X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    # 6. TRAIN
    model = MokshaSuperNet(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 100
    logger.info(f"🏋️ Starting Training ({epochs} epochs)...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    # 7. SAVE (ABSOLUTE PATH FIX)
    save_path = "/app/moksha_nn4.pth"  # <--- FIXED: Absolute path
    torch.save(model.state_dict(), save_path)
    
    logger.info(f"✅ Super-Brain Saved: {save_path}")
    logger.info(f"ℹ️ UPDATE decision_engine.py -> set input_size={input_size}")

if __name__ == "__main__":
    train_new_brain()