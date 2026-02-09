import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi 
from Moksha_1.core.decision_engine import DecisionEngine
from Moksha_1.utils.logger import logger
from Moksha_1.config import settings

class CommodityArbTimeMachine:
    def __init__(self):
        self.api = None
        
    def generate_synthetic_commodities(self, length=10000):
        """
        Generates 2x Leveraged Commodity Data (Gold vs Silver)
        Silver (AGQ) is much more volatile than Gold (UGL).
        """
        logger.warning(f"⚠️ Generating PLATINUM MIDAS DATA (UGL vs AGQ)...")
        
        # 1. Base Asset: Gold Spot
        # Random Walk with drift
        returns = np.random.normal(0.0001, 0.0008, length) 
        gold_spot = 2000 * np.cumprod(1 + returns)
        
        # 2. Gold/Silver Ratio - Volatile Reversion
        ratio_mean = 75.0
        ratio = np.zeros(length)
        ratio[0] = ratio_mean
        
        for i in range(1, length):
            force = 0.01 * (ratio_mean - ratio[i-1])
            shock = np.random.normal(0, 0.6) # High vol
            ratio[i] = ratio[i-1] + force + shock
            
        silver_spot = gold_spot / ratio
        
        # 3. Create 2x Leveraged ETFs
        gold_ret = pd.Series(gold_spot).pct_change().fillna(0)
        silver_ret = pd.Series(silver_spot).pct_change().fillna(0)
        
        # 2x Leverage
        ugl_price = 60 * (1 + (gold_ret * 2.0)).cumprod()
        agq_price = 40 * (1 + (silver_ret * 2.0)).cumprod()
        
        start_time = pd.Timestamp.now(tz='America/Chicago') - pd.Timedelta(minutes=length*5)
        
        data_ugl = []
        data_agq = []
        
        for i in range(length):
            t = start_time + pd.Timedelta(minutes=i*5)
            data_ugl.append({'time': t, 'symbol': 'UGL', 'close': ugl_price[i]})
            data_agq.append({'time': t, 'symbol': 'AGQ', 'close': agq_price[i]})
            
        df_y = pd.DataFrame(data_ugl).set_index('time') # Gold
        df_x = pd.DataFrame(data_agq).set_index('time') # Silver
        
        return df_y, df_x

    def calculate_signals(self, df_y, df_x):
        df = pd.merge(df_y, df_x, on='time', suffixes=('_y', '_x'))
        
        # Ratio: UGL / AGQ
        df['ratio'] = df['close_y'] / df['close_x']
        
        # Z-Score (Window 40)
        window = 40
        df['ratio_mean'] = df['ratio'].rolling(window).mean()
        df['ratio_std'] = df['ratio'].rolling(window).std()
        df['z_score'] = (df['ratio'] - df['ratio_mean']) / df['ratio_std']
        
        # RSI of Ratio
        delta = df['ratio'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df.dropna()

    def run(self):
        logger.info("⏳ Warming up Moksha 36.0 (Platinum Midas)...")
        
        capital = 25000.0
        trades = 0
        wins = 0
        
        df_y, df_x = self.generate_synthetic_commodities(length=5000)
        df = self.calculate_signals(df_y, df_x)
        
        position_state = 0 
        entry_val_y = 0
        entry_val_x = 0
        qty_y = 0
        qty_x = 0
        
        for i in tqdm(range(len(df)), desc="Simulating Platinum Arb"):
            row = df.iloc[i]
            z = row['z_score']
            rsi = row['rsi']
            
            # --- STRATEGY ---
            
            # ENTRY
            if position_state == 0:
                # *** LEVERAGE ***
                # Using 3.0x Leverage (Safe Production Level)
                investable = capital * 3.0
                leg_value = investable / 2
                
                qty_y = int(leg_value / row['close_y'])
                qty_x = int(leg_value / row['close_x'])
                
                if qty_y < 1 or qty_x < 1: continue

                # LONG RATIO: Z < -2.2 & RSI < 35
                if z < -2.2 and rsi < 35:
                    position_state = 1
                    entry_val_y = row['close_y']
                    entry_val_x = row['close_x']
                    trades += 1
                    
                # SHORT RATIO: Z > 2.2 & RSI > 65
                elif z > 2.2 and rsi > 65:
                    position_state = -1
                    entry_val_y = row['close_y']
                    entry_val_x = row['close_x']
                    trades += 1

            # EXIT (ASYMMETRIC TARGETS)
            elif position_state != 0:
                exit_trade = False
                
                # OVERSHOOT Mean (+/- 0.5 Z)
                if position_state == 1 and z >= 0.5: exit_trade = True
                elif position_state == -1 and z <= -0.5: exit_trade = True
                
                # Stop: Blowout
                if abs(z) > 4.5: exit_trade = True
                
                if exit_trade:
                    pnl = 0
                    if position_state == 1:
                        pnl += (row['close_y'] - entry_val_y) * qty_y
                        pnl += (entry_val_x - row['close_x']) * qty_x
                    else:
                        pnl += (entry_val_y - row['close_y']) * qty_y
                        pnl += (row['close_x'] - entry_val_x) * qty_x
                        
                    # Fees
                    pnl -= (qty_y * 0.01 + qty_x * 0.01 + 2.0)
                    
                    capital += pnl
                    if pnl > 0: wins += 1
                    position_state = 0

        total_return = ((capital - 25000) / 25000) * 100
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        logger.info("="*40)
        logger.info(f"📊 MOKSHA 36.0 PLATINUM MIDAS RESULTS")
        logger.info(f"   Return:   {total_return:.2f}%")
        logger.info(f"   Trades:   {trades}")
        logger.info(f"   Win Rate: {win_rate:.1f}%")
        logger.info(f"   Final Equity: ${capital:.2f}")
        logger.info("="*40)

if __name__ == "__main__":
    sim = CommodityArbTimeMachine()
    sim.run()
