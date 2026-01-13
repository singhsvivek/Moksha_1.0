
import torch
import pandas as pd
import numpy as np
from typing import Dict, List
from Moksha_1.data.storage.timescaledb import TimescaleStorage
from Moksha_1.models.deep_quant.model import FinancialNN3
from Moksha_1.data.processing.regime_detector import RegimeDetector
from Moksha_1.config import settings

class DecisionEngine:
    def __init__(self):
        self.db = TimescaleStorage()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # 1. Load the Brain (Agent 2)
        # Note: We assume input_dim=8 based on your successful training run. 
        # In production, we'd save this metadata in a config file.
        self.model = FinancialNN3(input_dim=8).to(self.device)
        try:
            self.model.load_state_dict(torch.load("moksha_nn3.pth", map_location=self.device))
            self.model.eval() # Set to Inference Mode (No Dropout)
            print("✅ Decision Engine: Neural Network Loaded.")
        except FileNotFoundError:
            print("⚠️ Warning: Model file not found. Run train_agent2.py first.")

        # 2. Load the Guardian (Agent 3)
        self.regime_detector = RegimeDetector()

    def analyze_market(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        Runs the full Daily Cycle: Data -> Features -> Alpha -> Risk -> Decision.
        """
        print("\n--- 🧠 STARTING DAILY ANALYSIS CYCLE ---")
        
        # A. Fetch Latest Data (Features)
        df_features = self.db.get_features_df(symbols=symbols)
        if df_features.empty:
            print("❌ No features found.")
            return pd.DataFrame()

        # Get the absolute latest timestamp per symbol
        latest_time = df_features['time'].max()
        current_features = df_features[df_features['time'] == latest_time].copy()
        
        # --- FIX: Align Feature Columns with Training Data ---
        # The model was trained on 8 features. The DB has 9 (including the empty rank_vol_1m).
        # We must explicitly exclude the "Poison Pill" column.
        exclude_cols = ['time', 'symbol', 'rank_vol_1m'] 
        
        valid_cols = [c for c in current_features.columns if c not in exclude_cols]
        
        # Verification to prevent crashing
        if len(valid_cols) != 8:
            # Fallback: If strict name matching fails, just take the first 8 columns
            # (assuming relative order is preserved from DB creation)
            print(f"⚠️ Warning: Found {len(valid_cols)} features, expected 8. Truncating extra columns.")
            valid_cols = valid_cols[:8]

        # Fill NaNs with 0.0 (Median) just like in training
        current_features[valid_cols] = current_features[valid_cols].fillna(0.0)

        # B. Generate Alpha (Prediction)
        try:
            X_pred = torch.tensor(current_features[valid_cols].values.astype(np.float32)).to(self.device)
            
            with torch.no_grad():
                alpha_raw = self.model(X_pred).cpu().numpy().flatten()
                
            current_features['predicted_return'] = alpha_raw
            
        except RuntimeError as e:
            print(f"❌ Model Shape Error: {e}")
            print(f"Debug: Input Shape: {X_pred.shape}")
            return pd.DataFrame()
        
        # C. Assess Risk (Regime)
        df_bars = self.db.get_bars_df(symbols=symbols)
        regime_df = self.regime_detector.detect_regime(df_bars)
        
        # Merge Regime info
        latest_regime = regime_df[regime_df['time'] == regime_df['time'].max()][['symbol', 'regime_label', 'regime_score']]
        
        decision_df = pd.merge(current_features, latest_regime, on='symbol', how='left')
        
        # D. The Council Logic
        decision_df['final_signal'] = decision_df.apply(self._apply_council_logic, axis=1)
        
        return decision_df[['time', 'symbol', 'predicted_return', 'regime_label', 'final_signal']]

    def _apply_council_logic(self, row) -> float:
        """
        Synthesizes Alpha and Risk into a Target Weight.
        """
        raw_signal = row['predicted_return'] # e.g., 0.05 (5% predicted return)
        regime = row['regime_label']
        
        # 1. Base Confidence (Scale the raw return to a -1 to 1 conviction score)
        # Using a tanh function to clamp extreme predictions
        confidence = np.tanh(raw_signal * 10) 
        
        # 2. Risk Multiplier
        if regime == 'CRITICAL_TURBULENCE':
            risk_mult = 0.0  # HARD STOP: Do not trade
        elif regime == 'HIGH_VOLATILITY':
            risk_mult = 0.5  # CAUTION: Halve the position size
        else:
            risk_mult = 1.0  # NORMAL: Full size
            
        return confidence * risk_mult
