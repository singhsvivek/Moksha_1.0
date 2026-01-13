# src/Moksha_1/data/processing/feature_engineering.py
import pandas as pd
import numpy as np
from typing import List
from Moksha_1.core.interfaces import IFeatureCalculator, StockCharacteristics

class FeatureEngine(IFeatureCalculator):
    """
    Calculates the 'Moksha 94' factor set using Vectorized Pandas operations.
    Focuses on Price/Volume first (Momentum, Volatility, Liquidity).
    """
    
    def calculate_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Input: DataFrame with columns [time, symbol, open, high, low, close, volume]
        Output: DataFrame with [time, symbol, rank_mom_1m, rank_vol_20d, ...]
        """
        if bars.empty:
            return pd.DataFrame()

        # Ensure sorted and indexed
        df = bars.sort_values(['symbol', 'time']).copy()
        df.set_index('time', inplace=True)
        
        # --- 1. RAW FACTORS CALCULATION (Per Stock) ---
        # We use groupby('symbol') to ensure calculations don't bleed between stocks
        grouped = df.groupby('symbol')

        # A. Momentum (Return-based)
        # 1 Month Momentum (~21 trading days)
        df['raw_mom_1m'] = grouped['close'].pct_change(21)
        # 12 Month Momentum (Standard academic definition: skip most recent month)
        df['raw_mom_12m'] = grouped['close'].pct_change(252).shift(21)
        # Short-term Reversal (5 days)
        df['raw_ret_1w'] = grouped['close'].pct_change(5)

        # B. Volatility (Risk)
        # 20-day Volatility
        df['raw_vol_20d'] = grouped['close'].transform(lambda x: x.pct_change().rolling(20).std())
        # 60-day Volatility
        df['raw_vol_60d'] = grouped['close'].transform(lambda x: x.pct_change().rolling(60).std())

        # C. Volume/Liquidity
        # Log Dollar Volume = Log(Price * Volume)
        df['raw_log_dol_vol'] = np.log(df['close'] * df['volume'])
        # Volume Trend (Current vol / 20-day avg)
        df['raw_vol_trend'] = df['volume'] / grouped['volume'].transform(lambda x: x.rolling(20).mean())

        # D. Technicals (RSI)
        df['raw_rsi_14'] = grouped['close'].transform(self._calculate_rsi)

        # --- 2. CROSS-SECTIONAL RANK STANDARDIZATION ---
        # The NN needs stationary inputs. We rank stocks against each other at EACH timestamp.
        # Range: [-1, 1]
        
        feature_cols = [
            'raw_mom_1m', 'raw_mom_12m', 'raw_ret_1w', 
            'raw_vol_20d', 'raw_vol_60d', 
            'raw_log_dol_vol', 'raw_vol_trend', 'raw_rsi_14'
        ]
        
        # Reset index to make 'time' a column again for grouping
        df.reset_index(inplace=True)
        
        final_cols = ['time', 'symbol']
        
        for col in feature_cols:
            rank_col = col.replace('raw_', 'rank_')
            # Group by TIME (Cross-sectional) -> Rank -> Normalize
            df[rank_col] = df.groupby('time')[col].transform(self._rank_standardize)
            final_cols.append(rank_col)

        # Drop NaNs (first 252 days will have NaNs due to lookback)
        result = df[final_cols].dropna()
        
        return result

    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _rank_standardize(self, series):
        """
        Maps values to [-1, 1] based on rank.
        Formula: (Rank - 0.5 * Count) / (0.5 * Count) ?? 
        Simpler: (Rank - Median) / (Count/2)
        """
        # Handle Cleanliness
        series = series.replace([np.inf, -np.inf], np.nan)
        if series.isnull().all():
            return series
            
        # pct=True gives percentile [0, 1]
        # We shift it to [-1, 1] -> (x - 0.5) * 2
        return (series.rank(pct=True, method='dense') - 0.5) * 2

    # Interface compliance (Optional, if strict)
    def get_feature_names(self) -> List[str]:
        return ['rank_mom_1m', 'rank_vol_20d', 'rank_rsi_14'] # etc...