import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
from Moksha_1.data.storage.timescaledb import TimescaleStorage
from Moksha_1.data.processing.feature_engineering import FeatureEngine
from Moksha_1.utils.logger import logger

# --- META ARCHITECTURE ---
class MetaNet(nn.Module):
    def __init__(self, input_size=110):
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

    def forward(self, x):
        return self.network(x)

class DecisionEngine:
    def __init__(self):
        self.db = TimescaleStorage()
        self.fe = FeatureEngine()
        self.device = torch.device("cpu")
        self.EXPECTED_INPUT_SIZE = 5 # Default
        self.model = self._load_model()

    def _load_model(self):
        path = "/app/moksha_meta_v1.pth"
        
        # 1. Check File Existence
        if not os.path.exists(path):
            logger.error(f"❌ CRITICAL: Model file not found at {path}")
            logger.error("   👉 ACTION REQUIRED: Run 'python src/train_meta.py' to generate the model.")
            return None
            
        try:
            # 2. Load Weights
            state = torch.load(path, map_location=self.device)
            
            # 3. Auto-Detect Input Size
            if 'network.0.weight' in state:
                self.EXPECTED_INPUT_SIZE = state['network.0.weight'].shape[1]
                logger.info(f"   🧠 Model Architecture detected input size: {self.EXPECTED_INPUT_SIZE}")
            
            # 4. Initialize & Load
            model = MetaNet(self.EXPECTED_INPUT_SIZE)
            model.load_state_dict(state)
            model.eval()
            
            logger.info(f"✅ Meta-Brain Loaded Successfully: {path}")
            return model
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to load model weights: {e}")
            return None

    def analyze_market(self, symbols, override_data=None):
        if self.model is None: 
            return pd.DataFrame()
        
        # 1. Data & Features
        raw = override_data if override_data is not None else self.db.get_bars_df(symbols, limit=500)
        if raw.empty: return pd.DataFrame()
        
        df = self.fe.create_features(raw, training_mode=True)
        latest = df.groupby('symbol').tail(1).copy()
        
        decisions = []
        with torch.no_grad():
            for _, row in latest.iterrows():
                symbol = row['symbol']
                
                # --- STEP 1: PRIMARY SIGNAL (The Rule) ---
                # "Is this stock trending?"
                # Price > SMA50 and ADX > 20 (Strong Trend)
                is_trending = (row['dist_sma50'] > 0) and (row['adx'] > 20)
                
                if not is_trending:
                    continue 

                # --- STEP 2: META-LABELING (The Filter) ---
                # Check for required features
                feats = ['adx', 'trend_slope', 'rel_vol', 'dist_sma50', 'rsi']
                missing = [f for f in feats if f not in row]
                
                if missing:
                    logger.warning(f"⚠️ Missing features for {symbol}: {missing}")
                    continue

                # Prepare Input
                inp_vals = row[feats].values.astype(np.float32)
                
                # Padding
                if len(inp_vals) != self.EXPECTED_INPUT_SIZE:
                    inp_vals = np.pad(inp_vals, (0, max(0, self.EXPECTED_INPUT_SIZE - len(inp_vals))), 'constant')[:self.EXPECTED_INPUT_SIZE]

                prob = self.model(torch.tensor(inp_vals).unsqueeze(0)).item()
                
                # Decision: Threshold 0.30 (Aggressive)
                if prob > 0.30:
                    decisions.append({
                        "symbol": symbol,
                        "regime_label": "TREND_META",
                        "raw_signal": prob,
                        "final_signal": 1.0 # Buy
                    })

        return pd.DataFrame(decisions)
