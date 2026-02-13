import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from tqdm import tqdm
import sys
from Moksha_1.config import settings
from Moksha_1.utils.logger import logger

# --- CONFIGURATION ---
LONG_TECH, SHORT_TECH = 'TQQQ', 'SQQQ'
LONG_SPY, SHORT_SPY = 'UPRO', 'SPXU'
CAPITAL = 25000.0

# --- THE KINETIC SCALAR SETTINGS ---
WINDOW = 25             
Z_ENTRY = 0.65          
Z_EXIT_BASE = 0.1       # Minimum Z to consider profit taking
Z_EXIT_HARD = 0.8       # Immediate take-profit (extreme move)
TREND_WINDOW = 2000     
LEVERAGE_MAX = 1.98     

class EquityArbBacktest:
    def __init__(self):
        try:
            base_url = getattr(settings, 'ALPACA_BASE_URL', "https://paper-api.alpaca.markets")
            self.api = tradeapi.REST(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY, base_url=base_url, api_version='v2')
            logger.info("✅ Alpaca Connected.")
        except: sys.exit(1)

    def fetch_data(self):
        logger.info(f"⏳ Fetching Quad-Ticker Data...")
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=365)
        try:
            tickers = [LONG_TECH, SHORT_TECH, LONG_SPY, SHORT_SPY]
            data_map = {sym: self.api.get_bars(sym, '5Min', start=start_dt.strftime('%Y-%m-%d'), end=end_dt.strftime('%Y-%m-%d'), feed='iex').df['close'] for sym in tickers}
            df = pd.DataFrame(data_map).dropna()
            if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('America/Chicago')
            return df
        except Exception as e:
            logger.error(f"❌ Data Error: {e}"); return pd.DataFrame()

    def run(self):
        df = self.fetch_data()
        if df.empty: return

        logger.info(f"🧮 Calculating Kinetic Indicators...")
        df['ratio'] = df[LONG_TECH] / df[LONG_SPY]
        df['mean'] = df['ratio'].rolling(WINDOW).mean()
        df['std'] = df['ratio'].rolling(WINDOW).std()
        df['z_score'] = (df['ratio'] - df['mean']) / df['std']
        
        # RSI & RSI Momentum (Slope)
        delta = df['ratio'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss)))
        df['rsi_sma'] = df['rsi'].rolling(3).mean() # Momentum Lead
        
        df['trend_sma'] = df['ratio'].rolling(TREND_WINDOW).mean()
        df = df.dropna()

        equity, trades, wins, position = CAPITAL, 0, 0, 0
        entry_prices, qty = {}, {}
        high_water, max_dd = CAPITAL, 0.0
        
        logger.info("⚔️ Running Strategy: 'The Kinetic Scalar'...")

        for i in range(len(df)):
            row = df.iloc[i]
            z, rsi, rsi_sma = row['z_score'], row['rsi'], row['rsi_sma']
            ratio, trend = row['ratio'], row['trend_sma']
            
            if position == 0:
                # Position Sizing: Scale down if volatility (std) is 2x the average
                vol_scalar = 1.0 if row['std'] < df['std'].mean() * 1.5 else 0.7
                investable = equity * LEVERAGE_MAX * vol_scalar
                
                if z < -Z_ENTRY and rsi < 50 and ratio > trend:
                    position = 1
                    qty = {LONG_TECH: int((investable/2)/row[LONG_TECH]), SHORT_SPY: int((investable/2)/row[SHORT_SPY])}
                    entry_prices = {LONG_TECH: row[LONG_TECH], SHORT_SPY: row[SHORT_SPY]}
                    trades += 1
                elif z > Z_ENTRY and rsi > 50 and ratio < trend:
                    position = -1
                    qty = {SHORT_TECH: int((investable/2)/row[SHORT_TECH]), LONG_SPY: int((investable/2)/row[LONG_SPY])}
                    entry_prices = {SHORT_TECH: row[SHORT_TECH], LONG_SPY: row[LONG_SPY]}
                    trades += 1

            elif position != 0:
                exit_signal = False
                if position == 1:
                    # EXIT LONG: If Z is profitable AND RSI momentum is rolling over
                    if z >= Z_EXIT_HARD: exit_signal = True
                    elif z >= Z_EXIT_BASE and rsi < rsi_sma: exit_signal = True
                elif position == -1:
                    # EXIT SHORT: If Z is profitable AND RSI momentum is bottoming
                    if z <= -Z_EXIT_HARD: exit_signal = True
                    elif z <= -Z_EXIT_BASE and rsi > rsi_sma: exit_signal = True
                
                if abs(z) > 4.5: exit_signal = True 

                if exit_signal:
                    pnl = 0
                    if position == 1:
                        pnl += (row[LONG_TECH] - entry_prices[LONG_TECH]) * qty[LONG_TECH]
                        pnl += (row[SHORT_SPY] - entry_prices[SHORT_SPY]) * qty[SHORT_SPY]
                    else:
                        pnl += (row[SHORT_TECH] - entry_prices[SHORT_TECH]) * qty[SHORT_TECH]
                        pnl += (row[LONG_SPY] - entry_prices[LONG_SPY]) * qty[LONG_SPY]
                    
                    pnl -= (2.0 + (sum(qty.values())*0.01))
                    equity += pnl
                    if pnl > 0: wins += 1
                    high_water = max(high_water, equity)
                    max_dd = max(max_dd, (high_water - equity) / high_water)
                    position, qty = 0, {}

        logger.info(f"\n{'='*40}\n📊 KINETIC SCALAR RESULTS\n   Trades: {trades}\n   Win Rate: {(wins/trades*100):.1f}%\n   Return: {((equity-CAPITAL)/CAPITAL*100):.2f}%\n   Max DD: {(max_dd*100):.2f}%\n   Final Eq: ${equity:,.2f}\n{'='*40}")

if __name__ == "__main__":
    bt = EquityArbBacktest(); bt.run()