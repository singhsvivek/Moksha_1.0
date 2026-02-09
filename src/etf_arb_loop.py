import time
import schedule
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import alpaca_trade_api as tradeapi

from Moksha_1.utils.logger import logger
from Moksha_1.config import settings
from Moksha_1.utils.messenger import messenger

# --- CONFIG ---
LONG_TECH = 'TQQQ'
SHORT_TECH = 'SQQQ'
LONG_SPY = 'UPRO'
SHORT_SPY = 'SPXU'

TIMEFRAME = '5Min'
WINDOW = 20
Z_ENTRY = 1.8
Z_EXIT = 0.0
LEVERAGE = 3.0

CURRENT_STATE = 0 
LAST_REPORT_HOUR = -1

try:
    logger.info("⚙️ Initializing Moksha 31.0 (Equity Engine)...")
    base_url = getattr(settings, 'ALPACA_BASE_URL', "https://paper-api.alpaca.markets")
    api = tradeapi.REST(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY, base_url, api_version='v2')
except Exception as e:
    sys.exit(1)

def get_market_data():
    try:
        limit = WINDOW + 20
        bars_y = api.get_bars(LONG_TECH, TIMEFRAME, limit=limit).df
        bars_x = api.get_bars(LONG_SPY, TIMEFRAME, limit=limit).df
        
        df = pd.merge(bars_y['close'], bars_x['close'], left_index=True, right_index=True, suffixes=('_y', '_x'))
        if len(df) < WINDOW: return None
        
        df['ratio'] = df['close_y'] / df['close_x']
        mean = df['ratio'].rolling(WINDOW).mean()
        std = df['ratio'].rolling(WINDOW).std()
        df['z_score'] = (df['ratio'] - mean) / std
        
        delta = df['ratio'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df.iloc[-1]
    except:
        return None

def send_heartbeat(z, rsi):
    """Sends hourly status to Discord"""
    global LAST_REPORT_HOUR
    now_hour = datetime.now().hour
    
    # Only send once per hour
    if now_hour != LAST_REPORT_HOUR:
        msg = f"**Status:** Scanning\n**Z-Score:** {z:.2f} (Trig: {Z_ENTRY})\n**RSI:** {rsi:.1f}\n**State:** {CURRENT_STATE}"
        messenger.send_message(msg, title="Equity Engine Heartbeat")
        LAST_REPORT_HOUR = now_hour

def execute_cycle():
    global CURRENT_STATE
    
    if not api.get_clock().is_open: return

    data = get_market_data()
    if data is None: return
    
    z = data['z_score']
    rsi = data['rsi']
    
    # Send Heartbeat
    send_heartbeat(z, rsi)
    
    # Sizing
    acct = api.get_account()
    leg_value = (float(acct.buying_power) * 0.60) / 2 # 60% Allocation
    
    # ENTRY: LONG RATIO
    if CURRENT_STATE == 0 and z < -Z_ENTRY and rsi < 40:
        qty_tech = int(leg_value / api.get_latest_trade(LONG_TECH).price)
        qty_spy = int(leg_value / api.get_latest_trade(SHORT_SPY).price)
        
        api.submit_order(LONG_TECH, qty_tech, 'buy', 'market', 'day')
        api.submit_order(SHORT_SPY, qty_spy, 'buy', 'market', 'day')
        
        CURRENT_STATE = 1
        messenger.send_message(f"Buying {LONG_TECH} & {SHORT_SPY}\nZ: {z:.2f} | RSI: {rsi:.1f}", title="🚀 EQUITY ENTRY")

    # ENTRY: SHORT RATIO
    elif CURRENT_STATE == 0 and z > Z_ENTRY and rsi > 60:
        qty_tech = int(leg_value / api.get_latest_trade(SHORT_TECH).price)
        qty_spy = int(leg_value / api.get_latest_trade(LONG_SPY).price)
        
        api.submit_order(SHORT_TECH, qty_tech, 'buy', 'market', 'day')
        api.submit_order(LONG_SPY, qty_spy, 'buy', 'market', 'day')
        
        CURRENT_STATE = -1
        messenger.send_message(f"Buying {SHORT_TECH} & {LONG_SPY}\nZ: {z:.2f} | RSI: {rsi:.1f}", title="🚀 EQUITY ENTRY")

    # EXIT
    elif CURRENT_STATE != 0:
        if (CURRENT_STATE == 1 and z >= 0) or (CURRENT_STATE == -1 and z <= 0):
            api.close_all_positions()
            CURRENT_STATE = 0
            messenger.send_message(f"Mean Reversion Hit. Z: {z:.2f}", title="💰 EQUITY EXIT")

def start():
    messenger.send_message("Engine Online. Waiting for market data...", title="Moksha 31.0 Equity Boot")
    schedule.every(1).minutes.do(execute_cycle)
    execute_cycle()
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    start()
