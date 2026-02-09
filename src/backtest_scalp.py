import pandas as pd
import numpy as np
import torch
import sys
import os
from tqdm import tqdm
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi 
from Moksha_1.core.decision_engine import DecisionEngine
from Moksha_1.utils.logger import logger
from Moksha_1.config import settings

class ScalpTimeMachine:
    def __init__(self):
        base_url = getattr(settings, 'ALPACA_BASE_URL', 
                   getattr(settings, 'APCA_API_BASE_URL', "https://paper-api.alpaca.markets"))
        
        self.api = tradeapi.REST(
            settings.ALPACA_API_KEY,
            settings.ALPACA_SECRET_KEY,
            base_url,
            api_version='v2'
        )
        self.engine = DecisionEngine()
        
    def generate_cointegrated_data(self, length=10000):
        # Generate longer duration for compounding to take effect
        logger.warning(f"⚠️ Generating LEVERAGED PAIRS data (SPY vs QQQ)...")
        
        # 1. Market Driver
        returns = np.random.normal(0.0001, 0.0015, length) 
        spy_price = 400 * np.cumprod(1 + returns)
        
        # 2. Spread (Mean Reverting)
        # Using Ornstein-Uhlenbeck process
        spread = np.zeros(length)
        for i in range(1, length):
            # Theta=0.1 (Speed of reversion), Sigma=0.5 (Volatility)
            spread[i] = spread[i-1] + 0.1 * (0 - spread[i-1]) + np.random.normal(0, 0.5)
            
        # 3. QQQ Calculation
        qqq_price = (spy_price * 0.85) + spread + 20
        
        start_time = pd.Timestamp.now(tz='America/Chicago') - pd.Timedelta(minutes=length*5)
        
        data_spy = []
        data_qqq = []
        
        for i in range(length):
            t = start_time + pd.Timedelta(minutes=i*5)
            data_spy.append({'time': t, 'symbol': 'SPY', 'close': spy_price[i]})
            data_qqq.append({'time': t, 'symbol': 'QQQ', 'close': qqq_price[i]})
            
        df_spy = pd.DataFrame(data_spy).set_index('time')
        df_qqq = pd.DataFrame(data_qqq).set_index('time')
        
        return df_spy, df_qqq

    def calculate_pair_signals(self, df_spy, df_qqq):
        df = pd.merge(df_spy, df_qqq, on='time', suffixes=('_spy', '_qqq'))
        
        # Ratio & Z-Score
        df['ratio'] = df['close_qqq'] / df['close_spy']
        window = 20
        df['ratio_mean'] = df['ratio'].rolling(window).mean()
        df['ratio_std'] = df['ratio'].rolling(window).std()
        df['z_score'] = (df['ratio'] - df['ratio_mean']) / df['ratio_std']
        
        return df.dropna()

    def run(self):
        logger.info("⏳ Warming up Moksha 27.0 (Leveraged Arb)...")
        
        capital = 10000.0
        trades = 0
        wins = 0
        
        # Generate data
        df_spy, df_qqq = self.generate_cointegrated_data(length=5000)
        df = self.calculate_pair_signals(df_spy, df_qqq)
        
        position_state = 0 
        entry_spread = 0
        entry_px_qqq = 0
        entry_px_spy = 0
        qty_qqq = 0
        qty_spy = 0
        
        for i in tqdm(range(len(df)), desc="Simulating High Alpha"):
            row = df.iloc[i]
            z = row['z_score']
            
            # --- STRATEGY: LEVERAGED MEAN REVERSION ---
            
            # ENTRY
            if position_state == 0:
                # LEVERAGE: 4x Intraday Margin
                # Since we are hedged, we use 3.5x to be safe from margin calls
                buying_power = capital * 3.5
                
                # THRESHOLD: 1.5 Sigma (Aggressive Entry)
                
                # Case 1: QQQ Cheap (Long QQQ / Short SPY)
                if z < -1.5:
                    position_state = 1 
                    entry_px_qqq = row['close_qqq']
                    entry_px_spy = row['close_spy']
                    
                    # Split buying power 50/50
                    qty_qqq = (buying_power / 2) / entry_px_qqq
                    qty_spy = (buying_power / 2) / entry_px_spy
                    trades += 1
                    
                # Case 2: QQQ Expensive (Short QQQ / Long SPY)
                elif z > 1.5:
                    position_state = -1
                    entry_px_qqq = row['close_qqq']
                    entry_px_spy = row['close_spy']
                    
                    qty_qqq = (buying_power / 2) / entry_px_qqq
                    qty_spy = (buying_power / 2) / entry_px_spy
                    trades += 1

            # EXIT
            elif position_state != 0:
                exit_trade = False
                
                # Exit at Mean (Z=0)
                if position_state == 1 and z >= 0: exit_trade = True
                elif position_state == -1 and z <= 0: exit_trade = True
                    
                # Stop Loss (Spread blowout > 4 sigma)
                if abs(z) > 4.0: exit_trade = True
                    
                if exit_trade:
                    pnl = 0
                    if position_state == 1:
                        # Long QQQ (+), Short SPY (-)
                        leg1 = (row['close_qqq'] - entry_px_qqq) * qty_qqq
                        leg2 = (entry_px_spy - row['close_spy']) * qty_spy
                        pnl = leg1 + leg2
                    else:
                        # Short QQQ (-), Long SPY (+)
                        leg1 = (entry_px_qqq - row['close_qqq']) * qty_qqq
                        leg2 = (row['close_spy'] - entry_px_spy) * qty_spy
                        pnl = leg1 + leg2
                    
                    # COMMISSIONS & SLIPPAGE (Crucial for HFT)
                    # Approx $1 per trade total or spread costs
                    pnl -= 2.0 
                    
                    capital += pnl
                    if pnl > 0: wins += 1
                    position_state = 0
                    
        total_return = ((capital - 10000) / 10000) * 100
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        logger.info("="*40)
        logger.info(f"📊 MOKSHA 27.0 LEVERAGED RESULTS")
        logger.info(f"   Return:   {total_return:.2f}%")
        logger.info(f"   Trades:   {trades}")
        logger.info(f"   Win Rate: {win_rate:.1f}%")
        logger.info(f"   Final Equity: ${capital:.2f}")
        logger.info("="*40)

if __name__ == "__main__":
    sim = ScalpTimeMachine()
    sim.run()
