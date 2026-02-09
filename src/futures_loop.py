import time
import schedule
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import pytz
import alpaca_trade_api as tradeapi

# --- Imports ---
from Moksha_1.utils.logger import logger
from Moksha_1.config import settings
from Moksha_1.utils.messenger import messenger

# --- Constants ---
# CURRENT ACTIVE CONTRACTS (MARCH 2026)
# UPDATE THESE QUARTERLY: H (Mar), M (Jun), U (Sep), Z (Dec)
SYMBOL_Y = 'MNQH6'  # Micro Nasdaq (Driver)
SYMBOL_X = 'MESH6'  # Micro S&P (Hedge)

MULT_Y = 2.0  # $2 per point
MULT_X = 5.0  # $5 per point

TIMEFRAME = '15Min' # 15-Minute Bars
WINDOW = 60         # 60-Bar Lookback (15 Hours)
Z_ENTRY = 2.0       # Entry Threshold
Z_EXIT = 0.0        # Mean Reversion Target
LEVERAGE = 3.0      # 3x Leverage (Safe for hedged)

# --- State ---
CURRENT_STATE = 0 

try:
    logger.info("⚙️ Initializing Moksha 29.0 (Live Futures Arb)...")
    
    # Direct API Connection
    base_url = getattr(settings, 'ALPACA_BASE_URL', 
               getattr(settings, 'APCA_API_BASE_URL', "https://paper-api.alpaca.markets"))
    
    api = tradeapi.REST(
        settings.ALPACA_API_KEY,
        settings.ALPACA_SECRET_KEY,
        base_url,
        api_version='v2'
    )
    logger.info(f"✅ Connected to Alpaca. Trading {SYMBOL_Y} vs {SYMBOL_X}")

except Exception as e:
    logger.critical(f"❌ Init Failed: {e}")
    sys.exit(1)

def get_market_data():
    """Fetches data and calculates Ratio Z-Score + RSI"""
    try:
        limit = WINDOW + 20 # Buffer for RSI
        
        # Fetch Data
        bars_y = api.get_bars(SYMBOL_Y, TIMEFRAME, limit=limit).df
        bars_x = api.get_bars(SYMBOL_X, TIMEFRAME, limit=limit).df
        
        # Merge on Time
        df = pd.merge(bars_y['close'], bars_x['close'], left_index=True, right_index=True, suffixes=('_y', '_x'))
        
        if len(df) < WINDOW:
            logger.warning("⚠️ Insufficient Data.")
            return None, None
            
        # 1. Ratio
        df['ratio'] = df['close_y'] / df['close_x']
        
        # 2. Z-Score
        mean = df['ratio'].rolling(WINDOW).mean()
        std = df['ratio'].rolling(WINDOW).std()
        df['z_score'] = (df['ratio'] - mean) / std
        
        # 3. RSI of Ratio
        delta = df['ratio'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        last = df.iloc[-1]
        return last, df.iloc[-2] # Return last two for crossover checks if needed

    except Exception as e:
        logger.error(f"❌ Data Error: {e}")
        return None, None

def calculate_position_size(px_y, px_x):
    """Calculates Dollar Neutral Contract Sizing"""
    try:
        acct = api.get_account()
        equity = float(acct.equity)
        
        # Target Notional per Leg = (Equity * Leverage) / 2
        target_leg_notional = (equity * LEVERAGE) / 2
        
        # Contracts Y (MNQ)
        notional_y_one = px_y * MULT_Y
        qty_y = int(target_leg_notional / notional_y_one)
        if qty_y < 1: qty_y = 1
        
        # Contracts X (MES) - Match Dollar Value
        total_notional_y = qty_y * notional_y_one
        notional_x_one = px_x * MULT_X
        
        qty_x = int(round(total_notional_y / notional_x_one))
        if qty_x < 1: qty_x = 1
        
        return qty_y, qty_x
        
    except Exception as e:
        logger.error(f"Sizing Error: {e}")
        return 1, 1

def execute_cycle():
    global CURRENT_STATE
    
    clock = api.get_clock()
    if not clock.is_open:
        logger.info("💤 Market Closed.")
        return

    data, prev_data = get_market_data()
    if data is None: return
    
    z = data['z_score']
    rsi = data['rsi']
    
    logger.info(f"📊 Z-Score: {z:.2f} | RSI: {rsi:.1f} | Ratio: {data['ratio']:.4f}")
    
    # --- LOGIC ---
    
    # ENTRY: LONG RATIO (MNQ Cheap)
    # Z < -2.0 AND RSI Oversold (< 40)
    if CURRENT_STATE == 0 and z < -Z_ENTRY and rsi < 40:
        qty_y, qty_x = calculate_position_size(data['close_y'], data['close_x'])
        
        logger.info(f"🚀 ENTRY LONG RATIO: +{qty_y} {SYMBOL_Y} / -{qty_x} {SYMBOL_X}")
        
        # Execute Legs
        api.submit_order(SYMBOL_Y, qty_y, 'buy', 'market', 'day')
        api.submit_order(SYMBOL_X, qty_x, 'sell', 'market', 'day')
        
        CURRENT_STATE = 1
        messenger.send_message(f"Long {SYMBOL_Y} / Short {SYMBOL_X}", title="Arb Entry")

    # ENTRY: SHORT RATIO (MNQ Expensive)
    # Z > 2.0 AND RSI Overbought (> 60)
    elif CURRENT_STATE == 0 and z > Z_ENTRY and rsi > 60:
        qty_y, qty_x = calculate_position_size(data['close_y'], data['close_x'])
        
        logger.info(f"🚀 ENTRY SHORT RATIO: -{qty_y} {SYMBOL_Y} / +{qty_x} {SYMBOL_X}")
        
        # Execute Legs
        api.submit_order(SYMBOL_Y, qty_y, 'sell', 'market', 'day')
        api.submit_order(SYMBOL_X, qty_x, 'buy', 'market', 'day')
        
        CURRENT_STATE = -1
        messenger.send_message(f"Short {SYMBOL_Y} / Long {SYMBOL_X}", title="Arb Entry")

    # EXIT: MEAN REVERSION (Z Crosses 0)
    elif CURRENT_STATE == 1 and z >= Z_EXIT:
        logger.info("💰 TARGET HIT: Closing Long Ratio.")
        api.close_all_positions()
        CURRENT_STATE = 0
        messenger.send_message("Target Hit (Mean Reversion)", title="Arb Exit")
        
    elif CURRENT_STATE == -1 and z <= -Z_EXIT:
        logger.info("💰 TARGET HIT: Closing Short Ratio.")
        api.close_all_positions()
        CURRENT_STATE = 0
        messenger.send_message("Target Hit (Mean Reversion)", title="Arb Exit")
        
    # STOP LOSS: BLOWOUT (> 4.5 Sigma)
    elif abs(z) > 4.5 and CURRENT_STATE != 0:
        logger.warning("🛑 STOP LOSS: Ratio Blowout.")
        api.close_all_positions()
        CURRENT_STATE = 0
        messenger.send_message("STOP LOSS HIT", title="Arb EMERGENCY")

def start():
    logger.info(f"🚀 Moksha 29.0 Futures Daemon Started.")
    logger.info(f"   Pairs: {SYMBOL_Y} (Long/Short) vs {SYMBOL_X} (Hedge)")
    
    # Run every 1 minute to check (data is 15m, but we check frequently)
    schedule.every(1).minutes.do(execute_cycle)
    execute_cycle()
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    start()
