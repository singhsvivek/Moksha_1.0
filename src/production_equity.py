import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import time
import sys
import pytz
from datetime import datetime, timedelta
from Moksha_1.config import settings
from Moksha_1.utils.logger import logger

# --- CONFIGURATION ---
PAIRS = {
    'LONG_TECH': 'TQQQ', 'SHORT_TECH': 'SQQQ',
    'LONG_SPY': 'UPRO', 'SHORT_SPY': 'SPXU'
}
LEVERAGE_MAX = 1.98
WINDOW = 40          
Z_ENTRY = 2.0        
TREND_WINDOW = 1000  
TZ = pytz.timezone('America/Chicago')

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

    def is_market_open(self):
        now = datetime.now(TZ)
        return 8 <= now.hour < 15 and now.weekday() < 5

    def get_market_data(self):
        end_dt = datetime.now(TZ)
        start_dt = end_dt - timedelta(days=20) 
        try:
            bars = self.api.get_bars(
                self.symbol_map, '5Min',
                start=start_dt.strftime('%Y-%m-%d'),
                end=end_dt.strftime('%Y-%m-%d'),
                feed='iex'
            ).df
            if bars.empty: return pd.DataFrame()
            df = bars.pivot_table(index='timestamp', columns='symbol', values='close')
            return df.dropna()
        except Exception as e:
            logger.error(f"❌ [EQUITY] Data Fetch Error: {e}")
            return pd.DataFrame()

    def calculate_signals(self, df):
        df['ratio'] = df[PAIRS['LONG_TECH']] / df[PAIRS['LONG_SPY']]
        df['mean'] = df['ratio'].rolling(WINDOW).mean()
        df['std'] = df['ratio'].rolling(WINDOW).std()
        df['z_score'] = (df['ratio'] - df['mean']) / df['std']
        df['trend_sma'] = df['ratio'].rolling(TREND_WINDOW).mean()
        return df.iloc[-1] 

    def reconcile_positions(self, signal_row):
        try:
            current_pos = {p.symbol: float(p.qty) for p in self.api.list_positions()}
            equity = float(self.api.get_account().equity)
            z = signal_row['z_score']
            ratio = signal_row['ratio']
            trend = signal_row['trend_sma']
            
            # TELEMETRY LOG
            if abs(z) > 1.0:
                logger.info(f"👀 [EQUITY] WATCHING | Z-Score: {z:.2f} | Ratio: {ratio:.4f}")

            is_long = current_pos.get(PAIRS['LONG_TECH'], 0) > 0
            is_short = current_pos.get(PAIRS['SHORT_TECH'], 0) > 0
            
            if not is_long and not is_short:
                investable = equity * LEVERAGE_MAX
                if z < -Z_ENTRY and ratio > trend:
                    logger.info(f"🚀 [EQUITY] LONG ENTRY (Z: {z:.2f})")
                    q1 = int((investable/2)/signal_row[PAIRS['LONG_TECH']])
                    q2 = int((investable/2)/signal_row[PAIRS['SHORT_SPY']])
                    self.api.submit_order(PAIRS['LONG_TECH'], q1, 'buy', 'market', 'day')
                    self.api.submit_order(PAIRS['SHORT_SPY'], q2, 'buy', 'market', 'day')
                elif z > Z_ENTRY and ratio < trend:
                    logger.info(f"🚀 [EQUITY] SHORT ENTRY (Z: {z:.2f})")
                    q1 = int((investable/2)/signal_row[PAIRS['SHORT_TECH']])
                    q2 = int((investable/2)/signal_row[PAIRS['LONG_SPY']])
                    self.api.submit_order(PAIRS['SHORT_TECH'], q1, 'buy', 'market', 'day')
                    self.api.submit_order(PAIRS['LONG_SPY'], q2, 'buy', 'market', 'day')
            elif is_long and z >= 0:
                self.api.close_all_positions()
            elif is_short and z <= 0:
                self.api.close_all_positions()
        except Exception as e:
            logger.error(f"❌ [EQUITY] Execution Error: {e}")

    def run_loop(self):
        logger.info("🟢 [EQUITY] Strategy Loop Started.")
        while True:
            try:
                if not self.is_market_open():
                    time.sleep(900)
                    continue
                now = datetime.now()
                if now.minute % 5 == 0 and now.second < 10:
                    df = self.get_market_data()
                    if not df.empty and len(df) > TREND_WINDOW:
                        latest = self.calculate_signals(df)
                        self.reconcile_positions(latest)
                    time.sleep(60) 
                time.sleep(1)
            except Exception as e:
                logger.error(f"⚠️ [EQUITY] Loop Error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    KineticScalarProduction().run_loop()