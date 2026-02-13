import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import time
from datetime import datetime
from Moksha_1.config import settings
from Moksha_1.utils.logger import logger

# --- CONFIGURATION ---
SYMBOL = 'AGQ'
CAPITAL_ALLOCATION = 5000.0 # Safety Allocation
RISK_PER_TRADE = 0.03       # 3% Risk per trade
Q1_END = "10:00"
Q2_END = "11:30"
HARD_STOP_TIME = "14:30"

class SilverStructureProduction:
    def __init__(self):
        self.api = tradeapi.REST(
            settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY,
            base_url=settings.ALPACA_BASE_URL, api_version='v2'
        )
        self.q1_levels = {}
        logger.info("✅ [SILVER] Intraday Engine Initialized.")

    def get_atr(self, df):
        """Calculates 14-period ATR for stop placement."""
        high_low = df['high'] - df['low']
        return high_low.rolling(14).mean().iloc[-1]

    def analyze_q1(self, df):
        """Captures the High/Low of the 8:30-10:00 Accumulation Phase."""
        today = datetime.now().date()
        # Filter for today's Q1
        q1_df = df[df.index.time < pd.Timestamp(Q1_END).time()]
        q1_df = q1_df[q1_df.index.date == today]
        
        if q1_df.empty: return
        
        self.q1_levels = {
            'high': q1_df['high'].max(),
            'low': q1_df['low'].min()
        }
        logger.info(f"📊 [SILVER] Q1 Range Set: {self.q1_levels['low']} - {self.q1_levels['high']}")

    def check_signals(self):
        """Checks for Reclaim + Volume Absorption."""
        if not self.q1_levels: return

        bars = self.api.get_bars(SYMBOL, '5Min', limit=20).df
        current = bars.iloc[-1]
        atr = self.get_atr(bars)
        
        # Calculate Volume SMA
        vol_sma = bars['volume'].rolling(4).mean().iloc[-1]
        
        signal = 0
        entry = 0
        stop = 0
        
        # 1. LONG RECLAIM (Swept Low, Closed Back Inside, High Volume)
        if current['low'] < self.q1_levels['low'] and \
           current['close'] > self.q1_levels['low'] and \
           current['volume'] > vol_sma:
             signal = 1
             entry = current['close']
             stop = entry - (1.2 * atr) # ATR Buffer Stop

        # 2. SHORT RECLAIM (Swept High, Closed Back Inside, High Volume)
        elif current['high'] > self.q1_levels['high'] and \
             current['close'] < self.q1_levels['high'] and \
             current['volume'] > vol_sma:
             signal = -1
             entry = current['close']
             stop = entry + (1.2 * atr)

        if signal != 0:
            self.execute_bracket(signal, entry, stop)

    def execute_bracket(self, direction, entry, stop):
        """Submits an OCO (One-Cancels-Other) Bracket Order."""
        risk_amt = CAPITAL_ALLOCATION * RISK_PER_TRADE
        dist = abs(entry - stop)
        if dist == 0: return
        
        shares = int(risk_amt / dist)
        if shares < 1: return
        
        side = 'buy' if direction == 1 else 'sell'
        # Target is 2x Risk
        take_profit = entry + (dist * 2 * direction)
        
        try:
            self.api.submit_order(
                symbol=SYMBOL,
                qty=shares,
                side=side,
                type='market',
                time_in_force='gtc',
                order_class='bracket',
                stop_loss={'stop_price': round(stop, 2)},
                take_profit={'limit_price': round(take_profit, 2)}
            )
            logger.info(f"🚀 [SILVER] Executed {side.upper()} {shares} shares @ {entry}")
        except Exception as e:
            logger.error(f"❌ [SILVER] Order Error: {e}")

    def run_loop(self):
        logger.info("🟢 [SILVER] Strategy Loop Started.")
        while True:
            now = datetime.now()
            curr_time = now.time()
            
            # Phase 1: Wait for Q1
            if curr_time < pd.Timestamp(Q1_END).time():
                time.sleep(60)
                continue
            
            # Phase 2: Set Q1 Levels (Once)
            if not self.q1_levels:
                df = self.api.get_bars(SYMBOL, '5Min', limit=100).df
                self.analyze_q1(df)
            
            # Phase 3: Trade Q2 (10:00 - 11:30)
            if curr_time < pd.Timestamp(Q2_END).time():
                if now.minute % 5 == 0 and now.second < 10:
                    self.check_signals()
                    time.sleep(60)
            
            # Phase 4: Hard Stop at end of day
            elif curr_time > pd.Timestamp(HARD_STOP_TIME).time():
                self.api.close_all_positions()
                self.q1_levels = {} # Reset for next day
                logger.info("💤 [SILVER] Market Closed. Sleeping...")
                time.sleep(3600)
                
            time.sleep(1)

if __name__ == "__main__":
    SilverStructureProduction().run_loop()