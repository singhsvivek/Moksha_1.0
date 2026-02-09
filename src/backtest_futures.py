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

class FuturesTimeMachine:
    def __init__(self):
        self.api = None # Not needed for pure backtest logic
        self.MULT_Y = 2.0  # MNQ
        self.MULT_X = 5.0  # MES
        
    def generate_cointegrated_futures(self, length=10000):
        logger.warning(f"⚠️ Generating FUTURES DATA (Ratio Mean Reversion)...")
        
        # 1. Market Driver (S&P 500)
        # Random Walk with drift
        returns = np.random.normal(0.00005, 0.0015, length) 
        mes_index = 4000 * np.cumprod(1 + returns)
        
        # 2. The Ratio (MNQ / MES)
        # We model the RATIO as a Mean Reverting Process (Ornstein-Uhlenbeck)
        # Mean Ratio ~ 3.5 (Nasdaq is ~14000, SPX ~4000)
        ratio_mean = 3.5
        ratio = np.zeros(length)
        ratio[0] = ratio_mean
        
        for i in range(1, length):
            # Speed of reversion = 0.05
            # Volatility of ratio = 0.02
            force = 0.05 * (ratio_mean - ratio[i-1])
            shock = np.random.normal(0, 0.01)
            ratio[i] = ratio[i-1] + force + shock
            
            # Occasional "Blowout" (Fat Tail) to test Stop Loss
            if np.random.random() > 0.995:
                ratio[i] += np.random.choice([0.1, -0.1])
        
        # 3. Derive MNQ Price
        mnq_index = mes_index * ratio
        
        start_time = pd.Timestamp.now(tz='America/Chicago') - pd.Timedelta(minutes=length*15)
        
        data_mes = []
        data_mnq = []
        
        for i in range(length):
            t = start_time + pd.Timedelta(minutes=i*15)
            data_mes.append({'time': t, 'symbol': 'MES', 'close': mes_index[i]})
            data_mnq.append({'time': t, 'symbol': 'MNQ', 'close': mnq_index[i]})
            
        df_x = pd.DataFrame(data_mes).set_index('time')
        df_y = pd.DataFrame(data_mnq).set_index('time')
        
        return df_x, df_y

    def calculate_ratio_stats(self, df_x, df_y):
        df = pd.merge(df_y, df_x, on='time', suffixes=('_y', '_x'))
        
        # 1. The Ratio
        df['ratio'] = df['close_y'] / df['close_x']
        
        # 2. Rolling Z-Score of Ratio (Window 60 = 15 hours of 15m bars)
        window = 60
        df['ratio_mean'] = df['ratio'].rolling(window).mean()
        df['ratio_std'] = df['ratio'].rolling(window).std()
        df['z_score'] = (df['ratio'] - df['ratio_mean']) / df['ratio_std']
        
        # 3. RSI of the Ratio (Filter)
        # We confirm the turn using RSI(14) of the ratio itself
        delta = df['ratio'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['ratio_rsi'] = 100 - (100 / (1 + rs))
        
        return df.dropna()

    def run(self):
        logger.info("⏳ Warming up Moksha 29.0 (Dollar Neutral Ratio Arb)...")
        
        capital = 50000.0 # Realistic Futures Account
        trades = 0
        wins = 0
        
        # Generate Data
        df_x, df_y = self.generate_cointegrated_futures(length=5000)
        df = self.calculate_ratio_stats(df_x, df_y)
        
        position_state = 0 
        entry_ratio = 0
        qty_y = 0
        qty_x = 0
        
        # Tracking PnL
        entry_px_y = 0
        entry_px_x = 0
        
        for i in tqdm(range(len(df)), desc="Simulating Ratio Arb"):
            row = df.iloc[i]
            z = row['z_score']
            rsi = row['ratio_rsi']
            
            # --- STRATEGY ---
            
            # ENTRY
            if position_state == 0:
                # 1. Sizing: Use 30% of Capital for Margin (Conservative Leverage)
                target_notional = capital * 3.0 # 3x Leverage total exposure
                
                # Split Notional 50/50
                notional_leg = target_notional / 2
                
                # Contracts Y (MNQ)
                # Notional = Price * Multiplier
                val_y_Contract = row['close_y'] * self.MULT_Y
                contracts_y = int(notional_leg / val_y_Contract)
                if contracts_y < 1: contracts_y = 1
                
                # Contracts X (MES) - Dollar Match
                val_x_Contract = row['close_x'] * self.MULT_X
                # We want Notional_X ~= Notional_Y
                # Notional_Y_Total = contracts_y * val_y_Contract
                target_notional_x = contracts_y * val_y_Contract
                contracts_x = int(round(target_notional_x / val_x_Contract))
                
                # LONG RATIO (MNQ Cheap): Z < -2 + RSI Oversold (< 40)
                if z < -2.0 and rsi < 40:
                    position_state = 1
                    qty_y = contracts_y # Buy MNQ
                    qty_x = contracts_x # Sell MES
                    entry_px_y = row['close_y']
                    entry_px_x = row['close_x']
                    trades += 1
                    
                # SHORT RATIO (MNQ Expensive): Z > 2 + RSI Overbought (> 60)
                elif z > 2.0 and rsi > 60:
                    position_state = -1
                    qty_y = contracts_y # Sell MNQ
                    qty_x = contracts_x # Buy MES
                    entry_px_y = row['close_y']
                    entry_px_x = row['close_x']
                    trades += 1
                    
            # EXIT
            elif position_state != 0:
                exit_trade = False
                
                # Target: Mean Reversion (Z crosses 0)
                if position_state == 1 and z >= 0: exit_trade = True
                elif position_state == -1 and z <= 0: exit_trade = True
                
                # Stop Loss: 4.5 Sigma Blowout
                if abs(z) > 4.5: exit_trade = True
                
                if exit_trade:
                    pnl = 0
                    
                    if position_state == 1:
                        # Long MNQ, Short MES
                        pnl_y = (row['close_y'] - entry_px_y) * self.MULT_Y * qty_y
                        pnl_x = (entry_px_x - row['close_x']) * self.MULT_X * qty_x
                    else:
                        # Short MNQ, Long MES
                        pnl_y = (entry_px_y - row['close_y']) * self.MULT_Y * qty_y
                        pnl_x = (row['close_x'] - entry_px_x) * self.MULT_X * qty_x
                        
                    pnl = pnl_y + pnl_x
                    
                    # Commission ($1.50 per side * 2 sides * total contracts)
                    comm = (qty_y + qty_x) * 3.0 
                    pnl -= comm
                    
                    capital += pnl
                    if pnl > 0: wins += 1
                    position_state = 0

        total_return = ((capital - 50000) / 50000) * 100
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        logger.info("="*40)
        logger.info(f"📊 MOKSHA 29.0 FINAL ARB RESULTS")
        logger.info(f"   Return:   {total_return:.2f}%")
        logger.info(f"   Trades:   {trades}")
        logger.info(f"   Win Rate: {win_rate:.1f}%")
        logger.info(f"   Final Equity: ${capital:.2f}")
        logger.info("="*40)

if __name__ == "__main__":
    sim = FuturesTimeMachine()
    sim.run()
