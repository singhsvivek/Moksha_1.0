# src/Moksha_1/data/processing/regime_detector.py
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from typing import Dict, Tuple

class RegimeDetector:
    """
    Detects market regimes using Wasserstein Distance (Earth Mover's Distance).
    Compares 'Recent' market behavior vs. 'Historical' baseline.
    """
    
    def __init__(self, window_short: int = 21, window_long: int = 252):
        self.window_short = window_short  # ~1 Month (Current Regime)
        self.window_long = window_long    # ~1 Year (Baseline)

    def detect_regime(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Wasserstein Anomaly Score for each day.
        Input: DataFrame with 'symbol' and 'close' (or pre-calculated returns).
        Output: DataFrame with [time, symbol, regime_score, regime_label]
        """
        # 1. Calculate Returns if not present
        if 'return' not in returns_df.columns:
            df = returns_df.sort_values(['symbol', 'time']).copy()
            df['return'] = df.groupby('symbol')['close'].pct_change()
        else:
            df = returns_df.copy()

        df = df.dropna()
        results = []

        print(f"🌪️ Detecting Market Regimes ({len(df['symbol'].unique())} symbols)...")

        # 2. Rolling Wasserstein Calculation
        # We group by symbol to calculate regime per stock (or you could do Market Wide)
        for symbol, group in df.groupby('symbol'):
            # We need numpy arrays for speed
            rets = group['return'].values
            times = group['time'].values
            
            # We need at least (Long Window) data points
            if len(rets) < self.window_long:
                continue

            # Iterate through time (Rolling Window)
            # This is O(N) but Wasserstein is computationally intensive, so we might optimize
            # to only run on the last X days for production.
            # For this MVP, we calculate the last 100 days to visualize.
            
            start_idx = len(rets) - 100 
            if start_idx < self.window_long: 
                start_idx = self.window_long

            for i in range(start_idx, len(rets)):
                # Recent Distribution (Short Window)
                dist_p = rets[i-self.window_short : i]
                
                # Baseline Distribution (Long Window)
                dist_q = rets[i-self.window_long : i]
                
                # Earth Mover's Distance
                wd_score = wasserstein_distance(dist_p, dist_q)
                
                results.append({
                    'time': times[i],
                    'symbol': symbol,
                    'regime_score': wd_score
                })

        # 3. Classify Regimes
        if not results:
            return pd.DataFrame()

        res_df = pd.DataFrame(results)
        
        # Define Thresholds dynamically based on the scores we found
        # High Score = High Anomaly (Crash/Turbulence)
        threshold_high = res_df['regime_score'].quantile(0.90) # Top 10% outlier
        threshold_med = res_df['regime_score'].quantile(0.75)
        
        def classify(score):
            if score > threshold_high: return 'CRITICAL_TURBULENCE'
            if score > threshold_med: return 'HIGH_VOLATILITY'
            return 'NORMAL'

        res_df['regime_label'] = res_df['regime_score'].apply(classify)
        
        return res_df

    def get_market_health(self, regime_df: pd.DataFrame) -> str:
        """
        Aggregates individual stock regimes into a global market signal.
        """
        # Take the most recent date
        latest_date = regime_df['time'].max()
        current = regime_df[regime_df['time'] == latest_date]
        
        avg_score = current['regime_score'].mean()
        # Simple voting mechanism
        turbulence_count = len(current[current['regime_label'] == 'CRITICAL_TURBULENCE'])
        total_count = len(current)
        
        if turbulence_count / total_count > 0.4:
            return "MARKET_CRASH_WARNING"
        elif avg_score > 0.02: # Heuristic value, requires tuning
            return "UNSTABLE"
        else:
            return "STABLE_BULL/BEAR"