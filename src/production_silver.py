import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import time
import sys
import pytz
from datetime import datetime
from Moksha_1.config import settings
from Moksha_1.utils.logger import logger

SYMBOL = 'AGQ'
CAPITAL_ALLOCATION = 5000.0
RISK_PER_TRADE = 0.03
Q1_END = "10:00"
Q2_END = "11:30"
HARD_STOP_TIME = "14:30"
TZ = pytz.timezone('America/Chicago')

class SilverStructureProduction:
    def __init__(self):
        try:
            self.api = tradeapi.REST(
                settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY,
                base_url=settings.ALPACA_BASE_URL, api_version='v2'
            )
            self.q1_levels = {}
            logger.info("✅ [SILVER] Intraday Engine Initialized.")
        except Exception as e:
            logger.critical(f"❌ [SILVER] Init Failed: {e}")
            sys.exit(1)

    def is_market_open(self):
        now = datetime.now(TZ)
        return 8 <= now.hour < 15 and now.weekday() < 5

    def get_atr(self, df):
        high_low = df['high'] - df['low']
        return high_low.rolling(14).mean().iloc[-1]

    def check_signals(self):
        if not self.q1_levels: return
        try:
            bars = self.api.get_bars(SYMBOL, '5Min', limit=20).df
            if bars.empty: return
        except: return

        current = bars.iloc[-1]
        atr = self.get_atr(bars)
        vol_sma = bars['volume'].rolling(4).mean().iloc[-1]
        
        signal, entry, stop = 0, 0, 0
        
        if current['low'] < self.q1_levels['low'] and current['close'] > self.q1_levels['low'] and current['volume'] > vol_sma:
             signal = 1; entry = current['close']; stop = entry - (1.2 * atr)
        elif current['high'] > self.q1_levels['high'] and current['close'] < self.q1_levels['high'] and current['volume'] > vol_sma:
             signal = -1; entry = current['close']; stop = entry + (1.2 * atr)

        if signal != 0: self.execute_bracket(signal, entry, stop)

    def execute_bracket(self, direction, entry, stop):
        shares = int((CAPITAL_ALLOCATION * RISK_PER_TRADE) / abs(entry - stop))
        if shares < 1: return
        side = 'buy' if direction == 1 else 'sell'
        try:
            self.api.submit_order(
                symbol=SYMBOL, qty=shares, side=side, type='market', time_in_force='gtc',
                order_class='bracket', stop_loss={'stop_price': round(stop, 2)},
                take_profit={'limit_price': round(entry + (abs(entry-stop)*2*direction), 2)}
            )
            logger.info(f"🚀 [SILVER] Trade Sent: {side} {shares}")
        except Exception as e: logger.error(f"❌ [SILVER] Error: {e}")

    def run_loop(self):
        logger.info("🟢 [SILVER] Strategy Loop Started.")
        while True:
            try:
                now = datetime.now(TZ)
                
                # Market Closed Check
                if not self.is_market_open():
                    if now.hour < 8 or now.hour >= 15:
                        logger.info("💤 [SILVER] Market Closed. Sleeping...")
                        time.sleep(3600)
                        self.q1_levels = {}
                    else:
                         time.sleep(60)
                    continue
                
                curr_time = now.time()
                
                # Phase 1: Wait for Q1
                if curr_time < pd.Timestamp(Q1_END).time():
                    if now.minute % 30 == 0 and now.second < 10:
                        logger.info("⏳ [SILVER] Waiting for Q1 Structure...")
                    time.sleep(60)
                
                # Phase 2: Q2 Trading
                elif curr_time < pd.Timestamp(Q2_END).time():
                    if not self.q1_levels:
                        bars = self.api.get_bars(SYMBOL, '5Min', limit=100).df
                        self.q1_levels = {'low': bars['low'].min(), 'high': bars['high'].max()}
                        logger.info(f"📊 [SILVER] Q1 Range: {self.q1_levels['low']} - {self.q1_levels['high']}")
                    
                    if now.minute % 5 == 0 and now.second < 10:
                        self.check_signals()
                        time.sleep(60)

                # Phase 3: Monitoring (Dead Zone Fix)
                elif curr_time <= pd.Timestamp(HARD_STOP_TIME).time():
                    if now.minute % 30 == 0 and now.second < 10:
                        logger.info("🔒 [SILVER] Q2 Closed. Monitoring Positions...")
                    time.sleep(60)

                # Phase 4: EOD
                elif curr_time > pd.Timestamp(HARD_STOP_TIME).time():
                    self.api.close_all_positions()
                    self.q1_levels = {}
                    time.sleep(3600)
                time.sleep(1)
            except Exception as e:
                logger.error(f"⚠️ [SILVER] Loop Error: {e}"); time.sleep(60)

if __name__ == "__main__":
    SilverStructureProduction().run_loop()