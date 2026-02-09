import pandas as pd
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
