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
TZ_CENTRAL = pytz.timezone("America/Chicago")
SYMBOL_A = 'SPY'
SYMBOL_B = 'QQQ'
TIMEFRAME = '5Min'
Z_ENTRY = 1.5
Z_EXIT = 0.0
LEVERAGE = 3.5 # Safe buffer below 4x limit
WINDOW = 20

# --- State ---
# 0 = Cash, 1 = Long Ratio (Long QQQ/Short SPY), -1 = Short Ratio (Short QQQ/Long SPY)
CURRENT_STATE = 0 

try:
    logger.info("⚙️ Initializing Moksha 27.0 (Live Pairs Trader)...")
    
    # Direct API Connection
    base_url = getattr(settings, 'ALPACA_BASE_URL', 
               getattr(settings, 'APCA_API_BASE_URL', "https://paper-api.alpaca.markets"))
    
    api = tradeapi.REST(
        settings.ALPACA_API_KEY,
        settings.ALPACA_SECRET_KEY,
        base_url,
        api_version='v2'
    )
    
    logger.info("✅ Connected to Alpaca.")

except Exception as e:
    logger.critical(f"❌ Init Failed: {e}")
    sys.exit(1)

def get_z_score():
    """Fetches data and calculates the current Z-Score of QQQ/SPY ratio"""
    try:
        # Fetch last 30 bars (need 20 for window + buffer)
        # We fetch separately to ensure we get aligned timestamps
        limit = 30
        
        bars_a = api.get_bars(SYMBOL_A, TIMEFRAME, limit=limit).df
        bars_b = api.get_bars(SYMBOL_B, TIMEFRAME, limit=limit).df
        
        # Merge to align timestamps (Crucial)
        df = pd.merge(bars_a['close'], bars_b['close'], left_index=True, right_index=True, suffixes=('_spy', '_qqq'))
        
        if len(df) < 20:
            logger.warning("⚠️ Insufficient data for Z-Score.")
            return None, None
            
        # Calc Ratio: QQQ / SPY
        df['ratio'] = df['close_qqq'] / df['close_spy']
        
        # Calc Z-Score
        mean = df['ratio'].rolling(WINDOW).mean()
        std = df['ratio'].rolling(WINDOW).std()
        df['z_score'] = (df['ratio'] - mean) / std
        
        last_row = df.iloc[-1]
        return last_row['z_score'], last_row
        
    except Exception as e:
        logger.error(f"❌ Data Fetch Error: {e}")
        return None, None

def sync_positions():
    """Checks actual account positions to sync state on restart"""
    global CURRENT_STATE
    try:
        positions = {p.symbol: p for p in api.list_positions()}
        
        has_spy = SYMBOL_A in positions
        has_qqq = SYMBOL_B in positions
        
        if not has_spy and not has_qqq:
            CURRENT_STATE = 0
            return
            
        # Determine State
        qty_qqq = float(positions[SYMBOL_B].qty) if has_qqq else 0
        
        if qty_qqq > 0:
            CURRENT_STATE = 1 # Long QQQ / Short SPY
        elif qty_qqq < 0:
            CURRENT_STATE = -1 # Short QQQ / Long SPY
            
        logger.info(f"🔄 State Synced: {CURRENT_STATE}")
        
    except Exception as e:
        logger.error(f"Sync Error: {e}")

def execute_pairs_cycle():
    global CURRENT_STATE
    
    # 1. Check Market Status
    clock = api.get_clock()
    if not clock.is_open:
        logger.info("💤 Market Closed.")
        return

    # 2. Get Math
    z_score, data = get_z_score()
    if z_score is None: return
    
    logger.info(f"📊 Z-Score: {z_score:.2f} | Ratio: {data['ratio']:.4f}")
    
    # 3. Account Data
    acct = api.get_account()
    equity = float(acct.equity)
    buying_power = equity * LEVERAGE # 3.5x
    
    # Each leg gets half the buying power
    leg_size = buying_power / 2
    
    # --- LOGIC ---
    
    # ENTRY: LONG RATIO (QQQ Cheap)
    if CURRENT_STATE == 0 and z_score < -Z_ENTRY:
        logger.info(f"🚀 ENTRY SIGNAL: Long Ratio (Z={z_score:.2f})")
        
        # Calculate shares
        qty_qqq = int(leg_size / data['close_qqq'])
        qty_spy = int(leg_size / data['close_spy'])
        
        # Submit: Buy QQQ, Sell SPY
        api.submit_order(SYMBOL_B, qty_qqq, 'buy', 'market', 'day')
        api.submit_order(SYMBOL_A, qty_spy, 'sell', 'market', 'day')
        
        messenger.send_message(f"Long {SYMBOL_B} / Short {SYMBOL_A}", title="Pairs Entry")
        CURRENT_STATE = 1
        
    # ENTRY: SHORT RATIO (QQQ Expensive)
    elif CURRENT_STATE == 0 and z_score > Z_ENTRY:
        logger.info(f"🚀 ENTRY SIGNAL: Short Ratio (Z={z_score:.2f})")
        
        qty_qqq = int(leg_size / data['close_qqq'])
        qty_spy = int(leg_size / data['close_spy'])
        
        # Submit: Sell QQQ, Buy SPY
        api.submit_order(SYMBOL_B, qty_qqq, 'sell', 'market', 'day')
        api.submit_order(SYMBOL_A, qty_spy, 'buy', 'market', 'day')
        
        messenger.send_message(f"Short {SYMBOL_B} / Long {SYMBOL_A}", title="Pairs Entry")
        CURRENT_STATE = -1

    # EXIT: MEAN REVERSION (Long Ratio -> 0)
    elif CURRENT_STATE == 1 and z_score >= Z_EXIT:
        logger.info(f"💰 EXIT SIGNAL: Mean Reverted (Z={z_score:.2f})")
        api.close_all_positions()
        messenger.send_message("Positions Closed (Target)", title="Pairs Exit")
        CURRENT_STATE = 0
        
    # EXIT: MEAN REVERSION (Short Ratio -> 0)
    elif CURRENT_STATE == -1 and z_score <= -Z_EXIT:
        logger.info(f"💰 EXIT SIGNAL: Mean Reverted (Z={z_score:.2f})")
        api.close_all_positions()
        messenger.send_message("Positions Closed (Target)", title="Pairs Exit")
        CURRENT_STATE = 0
        
    # STOP LOSS (Blowout Protection > 4 Sigma)
    elif abs(z_score) > 4.0 and CURRENT_STATE != 0:
        logger.warning(f"🛑 STOP LOSS: Z-Score Blowout ({z_score:.2f})")
        api.close_all_positions()
        messenger.send_message("CRITICAL STOP LOSS HIT", title="Pairs EMERGENCY")
        CURRENT_STATE = 0

def start():
    logger.info("🚀 Starting Pairs Trader Loop...")
    sync_positions()
    
    # Run every 5 minutes (aligned with bar close)
    schedule.every(1).minutes.do(execute_pairs_cycle)
    
    # Initial run
    execute_pairs_cycle()
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    start()
