import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import time
import sys
from datetime import datetime, timedelta
# Adjust import based on your Dockerfile PYTHONPATH
from Moksha_1.config import settings
from Moksha_1.utils.logger import logger

# --- CONFIGURATION ---
# Hardcoded to the parameters that yielded 504% return
PAIRS = {
    'LONG_TECH': 'TQQQ', 'SHORT_TECH': 'SQQQ',
    'LONG_SPY': 'UPRO', 'SHORT_SPY': 'SPXU'
}
LEVERAGE_MAX = 1.98
WINDOW = 40          # Rolling window for Mean/Std
Z_ENTRY = 2.0        # High Conviction Entry
TREND_WINDOW = 1000  # Trend Filter

class KineticScalarProduction:
    def __init__(self):
        try:
            self.api = tradeapi.REST(
                settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY,
                base_url=settings.ALPACA_BASE_URL, api_version='v2'
            )
            self.symbol_map = list(PAIRS.values())
            logger.info(f"✅ [EQUITY] Engine Initialized. Pairs: {PAIRS}")
        except Exception as e:
            logger.critical(f"❌ [EQUITY] API Connection Failed: {e}")
            sys.exit(1)

    def get_market_data(self):
        """Fetches aligned 5m bars."""
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=20) 
        
        try:
            # Fetch data for all 4 tickers
            bars = self.api.get_bars(
                self.symbol_map, '5Min',
                start=start_dt.strftime('%Y-%m-%d'),
                end=end_dt.strftime('%Y-%m-%d'),
                feed='iex'
            ).df
            
            if bars.empty: return pd.DataFrame()
            
            # Pivot to align timestamps
            df = bars.pivot_table(index='timestamp', columns='symbol', values='close')
            df = df.dropna()
            return df
        except Exception as e:
            logger.error(f"❌ [EQUITY] Data Fetch Error: {e}")
            return pd.DataFrame()

    def calculate_signals(self, df):
        """Calculates Structural Z-Score."""
        # Ratio: Tech / Market
        df['ratio'] = df[PAIRS['LONG_TECH']] / df[PAIRS['LONG_SPY']]
        
        # Z-Score Calculation
        df['mean'] = df['ratio'].rolling(WINDOW).mean()
        df['std'] = df['ratio'].rolling(WINDOW).std()
        df['z_score'] = (df['ratio'] - df['mean']) / df['std']
        
        # Trend Filter
        df['trend_sma'] = df['ratio'].rolling(TREND_WINDOW).mean()
        
        return df.iloc[-1] 

    def reconcile_positions(self, signal_row):
        """Executes trades based on Z-Score extremes."""
        try:
            current_pos = {p.symbol: float(p.qty) for p in self.api.list_positions()}
            equity = float(self.api.get_account().equity)
            
            z = signal_row['z_score']
            ratio = signal_row['ratio']
            trend = signal_row['trend_sma']
            
            # Check if we are currently holding positions
            is_long = current_pos.get(PAIRS['LONG_TECH'], 0) > 0
            is_short = current_pos.get(PAIRS['SHORT_TECH'], 0) > 0
            
            # --- ENTRY LOGIC (High Conviction Z > 2.0) ---
            if not is_long and not is_short:
                investable = equity * LEVERAGE_MAX
                
                # LONG ENTRY: Ratio is -2.0 Sigma (Undervalued) AND Uptrending
                if z < -Z_ENTRY and ratio > trend:
                    logger.info(f"🚀 [EQUITY] LONG ENTRY (Z: {z:.2f})")
                    qty_tech = int((investable/2)/signal_row[PAIRS['LONG_TECH']])
                    qty_spy = int((investable/2)/signal_row[PAIRS['SHORT_SPY']])
                    self.api.submit_order(PAIRS['LONG_TECH'], qty_tech, 'buy', 'market', 'day')
                    self.api.submit_order(PAIRS['SHORT_SPY'], qty_spy, 'buy', 'market', 'day')

                # SHORT ENTRY: Ratio is +2.0 Sigma (Overvalued) AND Downtrending
                elif z > Z_ENTRY and ratio < trend:
                    logger.info(f"🚀 [EQUITY] SHORT ENTRY (Z: {z:.2f})")
                    qty_tech = int((investable/2)/signal_row[PAIRS['SHORT_TECH']])
                    qty_spy = int((investable/2)/signal_row[PAIRS['LONG_SPY']])
                    self.api.submit_order(PAIRS['SHORT_TECH'], qty_tech, 'buy', 'market', 'day')
                    self.api.submit_order(PAIRS['LONG_SPY'], qty_spy, 'buy', 'market', 'day')

            # --- EXIT LOGIC (Mean Reversion to 0.0) ---
            elif is_long and z >= 0:
                logger.info(f"💰 [EQUITY] EXIT LONG (Mean Reversion)")
                self.api.close_all_positions()

            elif is_short and z <= 0:
                logger.info(f"💰 [EQUITY] EXIT SHORT (Mean Reversion)")
                self.api.close_all_positions()
                
        except Exception as e:
            logger.error(f"❌ [EQUITY] Execution Error: {e}")

    def run_loop(self):
        logger.info("🟢 [EQUITY] Strategy Loop Started.")
        while True:
            # Run at the start of every 5-minute candle
            now = datetime.now()
            if now.minute % 5 == 0 and now.second < 10:
                logger.info(f"⏳ [EQUITY] Analyzing Candle: {now}")
                df = self.get_market_data()
                if not df.empty and len(df) > TREND_WINDOW:
                    latest = self.calculate_signals(df)
                    self.reconcile_positions(latest)
                time.sleep(60) # Prevent double execution
            time.sleep(1)

if __name__ == "__main__":
    bot = KineticScalarProduction()
    bot.run_loop()