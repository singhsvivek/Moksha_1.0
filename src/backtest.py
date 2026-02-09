import pandas as pd
import numpy as np
import torch
import os
import sys
import argparse
from tqdm import tqdm
from Moksha_1.data.storage.timescaledb import TimescaleStorage
from Moksha_1.data.processing.feature_engineering import FeatureEngine
from Moksha_1.core.decision_engine import DecisionEngine
from Moksha_1.core.optimizer import PortfolioOptimizer
from Moksha_1.config import settings
from Moksha_1.utils.logger import logger

class TimeMachine:
    def __init__(self):
        self.db = TimescaleStorage()
        self.fe = FeatureEngine()
        self.engine = DecisionEngine()
        self.optimizer = PortfolioOptimizer()
        
    def run_simulation(self, start_date):
        logger.info(f"⏳ Warming up The Time Machine (Target: {start_date})...")
        if self.engine.model is None: return

        universe = settings.UNIVERSE
        if 'SPY' not in universe: universe.append('SPY')
        
        raw_df = self.db.get_bars_df(universe, limit=100000)
        if raw_df.empty: return
        
        raw_df['time'] = pd.to_datetime(raw_df['time'])
        price_history = raw_df.pivot(index='time', columns='symbol', values='close').sort_index()
        
        logger.info("   🧬 Generating Factors...")
        features_df = self.fe.create_features(raw_df, training_mode=True)
        test_df = features_df[features_df['time'] >= start_date].copy()
        
        logger.info("   🛡️ Calculating Macro Regime...")
        spy_df = raw_df[raw_df['symbol'] == 'SPY'].set_index('time').sort_index()
        spy_df['sma_200'] = spy_df['close'].rolling(200).mean()
        regime_map = (spy_df['close'] > spy_df['sma_200']).to_dict()
        
        logger.info("   🧠 Generating Sniper Signals...")
        test_df['is_trending'] = (test_df['dist_sma50'] > 0) & (test_df['adx'] > 20)
        
        test_df['ai_prob'] = 0.0
        trend_mask = test_df['is_trending']
        
        if trend_mask.any():
            feats = ['adx', 'trend_slope', 'rel_vol', 'dist_sma50', 'rsi']
            valid_cols = [c for c in feats if c in test_df.columns]
            input_data = test_df.loc[trend_mask, valid_cols].values.astype(np.float32)
            
            curr = input_data.shape[1]
            exp = self.engine.EXPECTED_INPUT_SIZE
            if curr < exp: input_data = np.pad(input_data, ((0,0),(0,exp-curr)), 'constant')
            
            with torch.no_grad():
                probs = self.engine.model(torch.tensor(input_data)).numpy().flatten()
            test_df.loc[trend_mask, 'ai_prob'] = probs

        threshold = 0.30
        logger.info(f"   🔓 Threshold: {threshold}")
        
        test_df['raw_signal'] = np.where(test_df['is_trending'] & (test_df['ai_prob'] > threshold), 1.0, 0.0)
        test_df['is_bull'] = test_df['time'].map(regime_map).fillna(True)
        test_df['final_signal'] = np.where(test_df['is_bull'], test_df['raw_signal'], 0.0)

        # 5. SIMULATION
        logger.info("   💰 Simulating Top-3 Sniping + Volatility Target...")
        unique_dates = np.sort(test_df['time'].unique())
        portfolio_curve = []
        test_df['next_return'] = test_df.groupby('symbol')['close'].pct_change().shift(-1)
        
        current_equity = 10000.0
        
        # PARAMETERS
        TOP_K = 3  # Only buy top 3 highest conviction stocks
        TARGET_VOL = 0.20 # Annualized Volatility Target (20%)
        
        # Volatility Tracking Window
        returns_window = []

        for date in tqdm(unique_dates, desc="Trading"):
            day_data = test_df[test_df['time'] == date]
            
            # --- SNIPER LOGIC ---
            # Filter for buys, sort by AI confidence, take top K
            candidates = day_data[day_data['final_signal'] > 0]
            if not candidates.empty:
                active = candidates.sort_values('ai_prob', ascending=False).head(TOP_K)
            else:
                active = pd.DataFrame()
            
            daily_total_ret = 0.0
            
            if not active.empty:
                decisions = active[['symbol', 'final_signal']].copy()
                lookback = pd.Timestamp(date) - pd.Timedelta(days=90)
                hist = price_history[(price_history.index >= lookback) & (price_history.index < date)]
                
                # HRP Optimization (on just the Top 3)
                allocations = self.optimizer.optimize_weights(decisions, hist)
                
                # Calculate Base Portfolio Return
                merged = allocations.merge(day_data[['symbol', 'next_return']], on='symbol')
                raw_ret = (merged['final_signal'] * merged['next_return'].fillna(0)).sum()
                
                # --- VOLATILITY TARGETING ---
                # Estimate recent volatility (20-day rolling)
                if len(returns_window) > 20:
                    recent_vol = np.std(returns_window[-20:]) * (252 ** 0.5)
                    if recent_vol == 0: recent_vol = 0.10 # Prevent div by zero
                    
                    # Vol Scalar = Target / Current
                    # If market is calm (10% vol), we leverage 2x.
                    # If market is crazy (40% vol), we de-leverage 0.5x.
                    leverage = TARGET_VOL / recent_vol
                    leverage = min(leverage, 1.5) # Cap leverage at 1.5x
                else:
                    leverage = 1.0
                
                daily_total_ret = raw_ret * leverage
            
            current_equity *= (1 + daily_total_ret)
            portfolio_curve.append({'time': date, 'equity': current_equity})
            
            # Update history for vol calculation
            returns_window.append(daily_total_ret)

        results = pd.DataFrame(portfolio_curve).set_index('time')
        results['return'] = results['equity'].pct_change().fillna(0)
        
        total = ((results['equity'].iloc[-1]/10000)-1)*100
        sharpe = (results['return'].mean()/results['return'].std()) * (252**0.5) if results['return'].std() != 0 else 0
        
        roll_max = results['equity'].cummax()
        dd = (results['equity'] - roll_max) / roll_max
        max_dd = dd.min() * 100
        
        logger.info("="*40)
        logger.info(f"📊 MOKSHA 9.0 (Sniper): {total:.2f}% | Sharpe: {sharpe:.2f} | DD: {max_dd:.2f}%")
        logger.info(f"   Final Equity: ${results['equity'].iloc[-1]:.2f}")
        logger.info("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", type=str, default="2020-01-01")
    args = parser.parse_args()
    sim = TimeMachine()
    sim.run_simulation(start_date=args.start_date)
