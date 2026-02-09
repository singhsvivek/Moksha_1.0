import os

# --- 1. FEATURE ENGINE (V6.0) ---
feature_engine_code = """import pandas as pd
import numpy as np
from Moksha_1.utils.logger import logger

class FeatureEngine:
    "Moksha Trend Engine (v6.0 - Synced)"
    def __init__(self):
        # The EXACT 5 features used by the Meta-Brain
        self.feature_columns = ['adx', 'trend_slope', 'rel_vol', 'dist_sma50', 'rsi']

    def _compute_adx(self, df, period=14):
        df = df.sort_values('time')
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        sum_di = plus_di + minus_di
        dx = (abs(plus_di - minus_di) / sum_di.replace(0, 1)) * 100
        return dx.rolling(period).mean().fillna(0)

    def create_features(self, df: pd.DataFrame, training_mode: bool = False) -> pd.DataFrame:
        if df.empty: return pd.DataFrame()
        df = df.sort_values(['symbol', 'time']).copy()
        
        try:
            results = []
            for symbol, group in df.groupby('symbol'):
                group = group.sort_values('time').copy()
                
                # 1. ADX
                group['adx'] = self._compute_adx(group)
                # 2. Trend Slope
                group['trend_slope'] = group['close'].pct_change(20) * 100
                # 3. Relative Volatility
                group['vol'] = group['close'].pct_change().rolling(14).std()
                group['rel_vol'] = group['vol'] / group['vol'].rolling(50).mean()
                # 4. Distance from SMA 50
                sma50 = group['close'].rolling(50).mean()
                group['dist_sma50'] = (group['close'] / sma50) - 1
                # 5. RSI
                delta = group['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                group['rsi'] = 100 - (100 / (1 + rs))
                
                results.append(group)
            
            full_df = pd.concat(results).dropna()
            
            cols = ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume'] + self.feature_columns
            valid_cols = [c for c in cols if c in full_df.columns]
            return full_df[valid_cols]

        except Exception as e:
            logger.error(f"❌ Feature Calc Error: {e}")
            return pd.DataFrame()
"""

# --- 2. TRAIN META (V6.0) ---
train_meta_code = """import torch
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
"""

# --- WRITE FILES ---
paths = {
    "/app/src/Moksha_1/data/processing/feature_engineering.py": feature_engine_code,
    "/app/src/train_meta.py": train_meta_code
}

for path, content in paths.items():
    with open(path, "w") as f:
        f.write(content)
    print(f"✅ Patched {path}")
