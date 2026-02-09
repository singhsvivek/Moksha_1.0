import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from Moksha_1.data.storage.timescaledb import TimescaleStorage
from Moksha_1.data.processing.feature_engineering import FeatureEngine
from Moksha_1.config import settings
from Moksha_1.utils.logger import logger

class ReversionNet(nn.Module):
    def __init__(self, input_size):
        super(ReversionNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.network(x)

def train_reversion_brain():
    logger.info("🧠 Initializing MEAN REVERSION Brain...")
    
    db = TimescaleStorage()
    fe = FeatureEngine()
    
    # 1. Fetch Data
    logger.info("⏳ Fetching deep history (100k rows)...")
    raw_df = db.get_bars_df(settings.UNIVERSE, limit=100000)
    
    # 2. Indicators
    logger.info("🧬 Calculating RSI & Bollinger Bands...")
    df = fe.create_features(raw_df, training_mode=True)
    
    # 3. FILTER: ONLY TRAIN ON DIPS (RSI < 40)
    # We want the AI to be a specialist in "buying the dip".
    # We don't care what happens when RSI is 70.
    dip_df = df[df['rsi'] < 40].copy()
    
    logger.info(f"   📉 Training on {len(dip_df)} 'Dip' scenarios (out of {len(df)} total rows).")
    
    # 4. Target: Does it bounce?
    # Success = Price is > 2% higher in 5 days
    dip_df['future_close'] = dip_df.groupby('symbol')['close'].shift(-5)
    dip_df['return_5d'] = (dip_df['future_close'] / dip_df['close']) - 1
    dip_df['target'] = (dip_df['return_5d'] > 0.02).astype(float) # 2% Bounce Target
    
    dip_df = dip_df.dropna()
    
    # 5. Prepare Training
    y = dip_df['target']
    drop_cols = ['time', 'symbol', 'target', 'future_close', 'return_5d', 'typical_price']
    X_df = dip_df.drop(columns=[c for c in drop_cols if c in dip_df.columns])
    
    input_size = len(X_df.columns)
    X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    
    # 6. Train
    model = ReversionNet(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 200
    logger.info(f"🏋️ Training... (Input Size: {input_size})")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            pred = (outputs > 0.5).float()
            acc = (pred.eq(y_tensor).sum() / y_tensor.shape[0]).item()
            logger.info(f"   Epoch {epoch+1}: Loss {loss.item():.4f} | Acc {acc*100:.1f}%")

    # 7. Save
    save_path = "/app/moksha_reversion_v1.pth"
    torch.save(model.state_dict(), save_path)
    logger.info(f"✅ Reversion Brain Saved: {save_path}")

if __name__ == "__main__":
    train_reversion_brain()