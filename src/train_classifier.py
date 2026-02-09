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

# --- NEW ARCHITECTURE: CLASSIFIER ---
class MokshaClassifier(nn.Module):
    def __init__(self, input_size):
        super(MokshaClassifier, self).__init__()
        self.network = nn.Sequential(
            # Layer 1: Expansion & Feature Extraction
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4),

            # Layer 2: Compression & Pattern Recognition
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4),
            
            # Layer 3: Decision Boundary
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),

            # Output: Single Probability (0.0 to 1.0)
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.network(x)

def train_classifier_brain():
    logger.info("🧠 Initializing ALPHA CLASSIFIER Training...")
    
    db = TimescaleStorage()
    feature_engine = FeatureEngine()
    
    # 1. GET DATA
    logger.info("⏳ Fetching deep history (100k rows)...")
    raw_df = db.get_bars_df(settings.UNIVERSE, limit=100000) 
    
    if len(raw_df) < 5000:
        logger.error("❌ Not enough data! Run 'python src/backfill.py' first.")
        return

    # 2. GENERATE FEATURES
    logger.info("🧬 Generating Factors...")
    full_features = feature_engine.create_features(raw_df, training_mode=True)
    
    # 3. LAGS
    logger.info("🐇 Generating Memory (Lags)...")
    cols_to_lag = [c for c in full_features.columns if c not in ['time', 'symbol', 'target']]
    for lag in [1, 2]:
        lagged_df = full_features.groupby('symbol')[cols_to_lag].shift(lag)
        lagged_df.columns = [f"{c}_lag{lag}" for c in cols_to_lag]
        full_features = pd.concat([full_features, lagged_df], axis=1)

    # 4. CREATE BINARY TARGETS (The Pivot)
    # Target = 1 if Tomorrow's Return > 0.05%. Else 0.
    full_features['next_ret'] = full_features.groupby('symbol')['log_returns'].shift(-1)
    full_features['target'] = (full_features['next_ret'] > 0.0005).astype(float)
    
    full_features = full_features.dropna()

    # 5. PREPARE TENSORS
    y = full_features['target']
    
    drop_cols = ['time', 'symbol', 'target', 'typical_price', 'next_ret']
    X_df = full_features.drop(columns=[c for c in drop_cols if c in full_features.columns])
    
    input_size = len(X_df.columns)
    logger.info(f"📊 Training on {len(X_df)} rows. Input Size: {input_size}")
    
    X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    # 6. TRAIN
    model = MokshaClassifier(input_size)
    # BCELoss is specific for Binary Classification (Probability)
    criterion = nn.BCELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    epochs = 150
    logger.info(f"🏋️ Starting Training ({epochs} epochs)...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            # Calc Accuracy
            predicted = (outputs > 0.5).float()
            accuracy = (predicted.eq(y_tensor).sum() / y_tensor.shape[0]).item()
            logger.info(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f} | Acc: {accuracy*100:.2f}%")

    # 7. SAVE TO ABSOLUTE PATH
    save_path = "/app/moksha_alpha_v1.pth"
    torch.save(model.state_dict(), save_path)
    
    logger.info(f"✅ Alpha Classifier Saved: {save_path}")
    logger.info(f"ℹ️ REMINDER: Decision Engine will auto-detect input size {input_size}")

if __name__ == "__main__":
    train_classifier_brain()