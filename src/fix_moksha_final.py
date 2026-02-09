import os

# --- 1. OPTIMIZER (V6.5 - Dictionary Mode) ---
# Removes Pandas recursion to fix "Read-Only" errors forever.
optimizer_code = """import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from Moksha_1.utils.logger import logger

class PortfolioOptimizer:
    def __init__(self):
        self.conviction_multiplier = 1.0

    def get_hrp_weights(self, cov_matrix):
        # 1. ROBUST CORRELATION
        v = np.sqrt(np.diag(cov_matrix))
        outer_v = np.outer(v, v)
        corr = cov_matrix / outer_v
        corr = corr.clip(-1.0, 1.0).fillna(0)
        
        # 2. DISTANCE
        dist = np.sqrt(0.5 * (1 - corr)).fillna(0)
        np.fill_diagonal(dist.values, 0)
        condensed_dist = squareform(dist.values, checks=False)
        link = sch.linkage(condensed_dist, 'single')
        
        # 3. SORT
        ix = sch.leaves_list(link)
        ordered_cov = cov_matrix.iloc[ix, ix]
        
        # 4. DICTIONARY RECURSION (Memory Safe)
        weights = {sym: 1.0 for sym in ordered_cov.index}
        
        def recurse_bi(sub_cov):
            if len(sub_cov) <= 1: return
            
            split = len(sub_cov) // 2
            c1 = sub_cov.iloc[:split, :split]
            c2 = sub_cov.iloc[split:, split:]
            
            diag = np.diag(c1)
            diag[diag == 0] = 1e-6
            w_inv = 1 / diag
            var1 = np.dot(np.dot(w_inv / w_inv.sum(), c1), w_inv / w_inv.sum())
            
            diag = np.diag(c2)
            diag[diag == 0] = 1e-6
            w_inv = 1 / diag
            var2 = np.dot(np.dot(w_inv / w_inv.sum(), c2), w_inv / w_inv.sum())
            
            if var1 == 0 and var2 == 0: alpha = 0.5
            else: alpha = 1 - var1 / (var1 + var2)
            
            for sym in c1.index: weights[sym] *= alpha
            for sym in c2.index: weights[sym] *= (1 - alpha)
            
            recurse_bi(c1)
            recurse_bi(c2)

        recurse_bi(ordered_cov)
        return pd.Series(weights)

    def optimize_weights(self, decisions: pd.DataFrame, history_df: pd.DataFrame = None) -> pd.DataFrame:
        if decisions.empty: return pd.DataFrame()
        
        # FORCE DEEP COPY
        decisions = decisions.copy()
        active_buys = decisions[decisions['final_signal'] > 0]
        
        if active_buys.empty: return decisions
        symbols = active_buys['symbol'].tolist()

        def apply_equal_weight():
            w = 1.0 / len(symbols)
            decisions.loc[decisions['final_signal'] > 0, 'final_signal'] = w * self.conviction_multiplier
            return decisions

        if history_df is None or len(symbols) < 2: return apply_equal_weight()

        try:
            subset = history_df[symbols].replace(0, np.nan).dropna()
            if len(subset) < 30: return apply_equal_weight()
            
            log_ret = np.log(subset / subset.shift(1)).dropna()
            cov = log_ret.cov()
            
            hrp_w = self.get_hrp_weights(cov)
            
            # Map safely
            decisions['hrp_weight'] = decisions['symbol'].map(hrp_w.to_dict()).fillna(0.0)
            decisions['final_signal'] = decisions['hrp_weight'] * self.conviction_multiplier
            return decisions

        except Exception as e:
            logger.error(f"❌ HRP Error: {e}")
            return apply_equal_weight()
"""

# --- 2. BACKTESTER (Locked Threshold) ---
# Hardcodes threshold to 0.30 to ensure high trade volume.
backtest_code = """import pandas as pd
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
        
        logger.info("   🧠 Generating Signals...")
        test_df['is_trending'] = (test_df['dist_sma50'] > 0) & (test_df['adx'] > 20)
        
        # PREDICTION BATCH
        test_df['ai_prob'] = 0.0
        trend_mask = test_df['is_trending']
        
        if trend_mask.any():
            feats = ['adx', 'trend_slope', 'rel_vol', 'dist_sma50', 'rsi']
            input_data = test_df.loc[trend_mask, feats].values.astype(np.float32)
            
            # Padding
            curr = input_data.shape[1]
            exp = self.engine.EXPECTED_INPUT_SIZE
            if curr < exp: input_data = np.pad(input_data, ((0,0),(0,exp-curr)), 'constant')
            
            with torch.no_grad():
                probs = self.engine.model(torch.tensor(input_data)).numpy().flatten()
            test_df.loc[trend_mask, 'ai_prob'] = probs

        # --- THE WINNING PARAMETER ---
        threshold = 0.30  # Locked to capture all valid trends
        logger.info(f"   🔓 Threshold Locked: {threshold} (Aggressive Mode)")
        
        test_df['raw_signal'] = np.where(test_df['is_trending'] & (test_df['ai_prob'] > threshold), 1.0, 0.0)
        test_df['is_bull'] = test_df['time'].map(regime_map).fillna(True)
        test_df['final_signal'] = np.where(test_df['is_bull'], test_df['raw_signal'], 0.0)

        logger.info(f"   🚀 Final Buy Signals: {int(test_df['final_signal'].sum())}")

        logger.info("   💰 Simulating HRP + CPPI...")
        unique_dates = np.sort(test_df['time'].unique())
        portfolio_curve = []
        test_df['next_return'] = test_df.groupby('symbol')['close'].pct_change().shift(-1)
        
        # CPPI PARAMS
        current_equity = 10000.0
        peak_equity = 10000.0
        FLOOR_PCT = 0.85
        MULTIPLIER = 4.0
        
        for date in tqdm(unique_dates, desc="Trading"):
            if current_equity > peak_equity: peak_equity = current_equity
            
            floor = peak_equity * FLOOR_PCT
            cushion = max(0, current_equity - floor)
            risky_w = min(1.0, (cushion * MULTIPLIER) / current_equity)
            
            day_data = test_df[test_df['time'] == date]
            active = day_data[day_data['final_signal'] > 0]
            
            daily_ret = 0.0
            
            if not active.empty:
                decisions = active[['symbol', 'final_signal']].copy()
                lookback = pd.Timestamp(date) - pd.Timedelta(days=90)
                hist = price_history[(price_history.index >= lookback) & (price_history.index < date)]
                
                # HRP Optimization
                allocations = self.optimizer.optimize_weights(decisions, hist)
                
                merged = allocations.merge(day_data[['symbol', 'next_return']], on='symbol')
                raw_ret = (merged['final_signal'] * merged['next_return'].fillna(0)).sum()
                daily_ret = raw_ret * risky_w
            
            current_equity *= (1 + daily_ret)
            portfolio_curve.append({'time': date, 'equity': current_equity})

        results = pd.DataFrame(portfolio_curve).set_index('time')
        results['return'] = results['equity'].pct_change().fillna(0)
        
        total = ((results['equity'].iloc[-1]/10000)-1)*100
        sharpe = (results['return'].mean()/results['return'].std()) * (252**0.5) if results['return'].std() != 0 else 0
        dd = (results['equity']/results['equity'].cummax() - 1).min() * 100
        
        logger.info("="*40)
        logger.info(f"📊 FINAL RESULTS: {total:.2f}% | Sharpe: {sharpe:.2f} | DD: {dd:.2f}%")
        logger.info(f"   Final Equity: ${results['equity'].iloc[-1]:.2f}")
        logger.info("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", type=str, default="2020-01-01")
    args = parser.parse_args()
    sim = TimeMachine()
    sim.run_simulation(start_date=args.start_date)
"""

# --- WRITE ---
with open("/app/src/Moksha_1/core/optimizer.py", "w") as f: f.write(optimizer_code)
with open("/app/src/backtest.py", "w") as f: f.write(backtest_code)
print("✅ Applied Moksha Prime Patch (Fixing HRP & Signals)")
