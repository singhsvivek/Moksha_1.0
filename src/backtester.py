import pandas as pd
import numpy as np
import torch
import os
import sys
from Moksha_1.data.storage.timescaledb import TimescaleStorage
from Moksha_1.data.processing.feature_engineering import FeatureEngine
from Moksha_1.core.decision_engine import DecisionEngine
from Moksha_1.config import settings
from Moksha_1.utils.logger import logger

class TimeMachine:
    def __init__(self):
        self.db = TimescaleStorage()
        self.fe = FeatureEngine()
        self.engine = DecisionEngine()
        
    def run_simulation(self, start_date='2020-01-01'):
        logger.info(f"⏳ Warming up The Time Machine (Start: {start_date})...")
        
        if self.engine.model is None:
            logger.error("🛑 STOP: The Neural Network model is missing.")
            return

        # 1. Fetch History
        universe = settings.UNIVERSE
        # IMPORTANT: Ensure SPY is in the universe for the Regime Filter
        if 'SPY' not in universe: universe.append('SPY')
        
        raw_df = self.db.get_bars_df(universe, limit=100000)
        
        if raw_df.empty:
            logger.error("❌ No history found!")
            return

        raw_df['time'] = pd.to_datetime(raw_df['time'])
        raw_df = raw_df[raw_df['time'] >= start_date].copy()
        
        # 2. Generate Features
        logger.info("   🧬 Generating factors...")
        features_df = self.fe.create_features(raw_df, training_mode=True)
        
        # 3. Create Lags
        processed_dfs = []
        for sym, group in features_df.groupby('symbol'):
            group = group.sort_values('time')
            cols_to_lag = [c for c in group.columns if c not in ['time', 'symbol', 'target', 'typical_price']]
            for lag in [1, 2]:
                lagged = group[cols_to_lag].shift(lag)
                lagged.columns = [f"{c}_lag{lag}" for c in cols_to_lag]
                group = pd.concat([group, lagged], axis=1)
            processed_dfs.append(group)
            
        full_df = pd.concat(processed_dfs).dropna()
        
        # --- PADDING ---
        drop_cols = ['time', 'symbol', 'target', 'typical_price', 'next_ret']
        feature_cols = [c for c in full_df.columns if c not in drop_cols]
        
        X_numpy = full_df[feature_cols].values.astype(np.float32)
        current_dim = X_numpy.shape[1]
        expected_dim = self.engine.EXPECTED_INPUT_SIZE
        
        if current_dim < expected_dim:
            X_numpy = np.pad(X_numpy, ((0,0), (0, expected_dim - current_dim)), 'constant')
        elif current_dim > expected_dim:
            X_numpy = X_numpy[:, :expected_dim]
            
        X_tensor = torch.tensor(X_numpy)

        # 4. Neural Net Prediction
        logger.info("   🧠 Running Gen 4 Brain on History...")
        full_df['ai_signal'] = 0.0
        try:
            with torch.no_grad():
                probs = self.engine.model(X_tensor).numpy().flatten()
                # Convert Probability (0-1) to Signal (-1 to 1)
                full_df['ai_signal'] = (probs - 0.5) * 2.0
        except Exception as e:
            logger.critical(f"❌ Prediction Failed: {e}")
            return

        # --- NEW: REGIME FILTER (The Shield) ---
        logger.info("   🛡️ Applying Macro Regime Filter (SPY 200 SMA)...")
        
        # Calculate SPY 200 SMA
        spy_df = raw_df[raw_df['symbol'] == 'SPY'].set_index('time').sort_index()
        spy_df['sma_200'] = spy_df['close'].rolling(window=200).mean()
        
        # Define Bull Market: Price > SMA 200
        spy_df['is_bull'] = spy_df['close'] > spy_df['sma_200']
        
        # Map Regime to the main dataframe
        regime_map = spy_df['is_bull'].to_dict()
        
        # Create a boolean mask for the whole dataframe
        full_df['market_is_bull'] = full_df['time'].map(regime_map).fillna(True) # Assume bull if missing
        
        # LOGIC: If Market is Bear, kill all Buy signals.
        # (Optional: Allow Shorting in Bear, but for now let's go to Cash)
        original_signal = full_df['ai_signal'].copy()
        
        full_df.loc[full_df['market_is_bull'] == False, 'ai_signal'] = 0.0
        
        filtered_count = (original_signal != full_df['ai_signal']).sum()
        logger.info(f"   📉 Regime Filter Blocked {filtered_count} trades during Bear Markets.")
        # ---------------------------------------

        # 5. Simulate Trading
        logger.info("   💰 Calculating Profits...")
        
        full_df['next_return'] = full_df.groupby('symbol')['close'].pct_change().shift(-1)
        full_df = full_df.dropna(subset=['next_return'])
        
        full_df['strategy_return'] = full_df['ai_signal'] * full_df['next_return']
        
        daily_perf = full_df.groupby('time')['strategy_return'].mean()
        cumulative_returns = (1 + daily_perf).cumprod()
        
        if not cumulative_returns.empty:
            total_return = (cumulative_returns.iloc[-1] - 1) * 100
            sharpe = (daily_perf.mean() / daily_perf.std()) * (252 ** 0.5) if daily_perf.std() != 0 else 0
            
            rolling_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_dd = drawdown.min() * 100
            
            logger.info("\n" + "="*40)
            logger.info(f"📊 BACKTEST RESULTS ({start_date} to Now)")
            logger.info(f"   Strategy Return: {total_return:.2f}%")
            logger.info(f"   Sharpe Ratio:    {sharpe:.2f}")
            logger.info(f"   Max Drawdown:    {max_dd:.2f}%")
            logger.info("="*40)
        else:
            logger.error("❌ No returns calculated.")

if __name__ == "__main__":
    sim = TimeMachine()
    sim.run_simulation()