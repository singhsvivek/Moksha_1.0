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

class ETFArbTimeMachine:
    def __init__(self):
        self.api = None
        
    def generate_synthetic_etfs(self, length=10000):
        # Generates 3x Leveraged ETF Data (High Volatility, Cointegrated)
        logger.warning(f"⚠️ Generating LEVERAGED 3x ETF DATA (TQQQ vs UPRO)...")
        
        # 1. Underlying Market (S&P 500)
        returns = np.random.normal(0.0001, 0.0012, length) 
        spx_index = 4000 * np.cumprod(1 + returns)
        
        # 2. UPRO (3x S&P)
        upro_price = spx_index * 0.02 
        upro_price *= (1 + np.random.normal(0, 0.002, length))
        
        # 3. TQQQ (3x Nasdaq) - Mean Reverting Ratio
        ratio_mean = 1.5 
        ratio = np.zeros(length)
        ratio[0] = ratio_mean
        
        for i in range(1, length):
            force = 0.05 * (ratio_mean - ratio[i-1])
            shock = np.random.normal(0, 0.015) 
            ratio[i] = ratio[i-1] + force + shock
            
        tqqq_price = upro_price * ratio
        
        start_time = pd.Timestamp.now(tz='America/Chicago') - pd.Timedelta(minutes=length*5)
        
        data_upro = []
        data_tqqq = []
        
        for i in range(length):
            t = start_time + pd.Timedelta(minutes=i*5)
            data_upro.append({'time': t, 'symbol': 'UPRO', 'close': upro_price[i]})
            data_tqqq.append({'time': t, 'symbol': 'TQQQ', 'close': tqqq_price[i]})
            
        df_x = pd.DataFrame(data_upro).set_index('time')
        df_y = pd.DataFrame(data_tqqq).set_index('time')
        
        return df_y, df_x

    def calculate_signals(self, df_y, df_x):
        df = pd.merge(df_y, df_x, on='time', suffixes=('_y', '_x'))
        df['ratio'] = df['close_y'] / df['close_x']
        
        window = 20
        df['ratio_mean'] = df['ratio'].rolling(window).mean()
        df['ratio_std'] = df['ratio'].rolling(window).std()
        df['z_score'] = (df['ratio'] - df['ratio_mean']) / df['ratio_std']
        
        # RSI Filter
        delta = df['ratio'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df.dropna()

    def run(self):
        logger.info("⏳ Warming up Moksha 31.0 (Leveraged ETF Arb)...")
        
        capital = 25000.0
        trades = 0
        wins = 0
        
        df_y, df_x = self.generate_synthetic_etfs(length=5000)
        df = self.calculate_signals(df_y, df_x)
        
        position_state = 0 
        entry_val_y = 0
        entry_val_x = 0
        qty_y = 0
        qty_x = 0
        
        for i in tqdm(range(len(df)), desc="Simulating 3x Leverage"):
            row = df.iloc[i]
            z = row['z_score']
            rsi = row['rsi']
            
            # --- STRATEGY ---
            
            # ENTRY
            if position_state == 0:
                # *** LEVERAGE BOOST ***
                # Use 3.0x Equity (Allowed for Intraday Pairs)
                investable = capital * 3.0
                leg_value = investable / 2
                
                # Sizing
                qty_y = int(leg_value / row['close_y'])
                qty_x = int(leg_value / row['close_x'])
                
                if qty_y < 1 or qty_x < 1: continue

                # LONG RATIO: Z < -1.8 (Increased Frequency)
                if z < -1.8 and rsi < 40:
                    position_state = 1
                    entry_val_y = row['close_y']
                    entry_val_x = row['close_x']
                    trades += 1
                    
                # SHORT RATIO: Z > 1.8
                elif z > 1.8 and rsi > 60:
                    position_state = -1
                    entry_val_y = row['close_y']
                    entry_val_x = row['close_x']
                    trades += 1

            # EXIT
            elif position_state != 0:
                exit_trade = False
                
                if position_state == 1 and z >= 0: exit_trade = True
                elif position_state == -1 and z <= 0: exit_trade = True
                
                if abs(z) > 4.5: exit_trade = True
                
                if exit_trade:
                    pnl = 0
                    if position_state == 1:
                        pnl += (row['close_y'] - entry_val_y) * qty_y
                        pnl += (entry_val_x - row['close_x']) * qty_x
                    else:
                        pnl += (entry_val_y - row['close_y']) * qty_y
                        pnl += (row['close_x'] - entry_val_x) * qty_x
                        
                    pnl -= (qty_y * 0.01 + qty_x * 0.01 + 2.0)
                    
                    capital += pnl
                    if pnl > 0: wins += 1
                    position_state = 0

        total_return = ((capital - 25000) / 25000) * 100
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        logger.info("="*40)
        logger.info(f"📊 MOKSHA 31.0 LEVERAGED RESULTS")
        logger.info(f"   Return:   {total_return:.2f}%")
        logger.info(f"   Trades:   {trades}")
        logger.info(f"   Win Rate: {win_rate:.1f}%")
        logger.info(f"   Final Equity: ${capital:.2f}")
        logger.info("="*40)

if __name__ == "__main__":
    sim = ETFArbTimeMachine()
    sim.run()
