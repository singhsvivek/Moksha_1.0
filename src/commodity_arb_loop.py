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

SYMBOL_Y = 'UGL'
SYMBOL_X = 'AGQ'
TIMEFRAME = '5Min'   
WINDOW = 40
Z_ENTRY = 2.2
Z_EXIT_OFFSET = 0.5
LEVERAGE = 3.0

CURRENT_STATE = 0 
LAST_REPORT_HOUR = -1

try:
    logger.info("⚙️ Initializing Moksha 36.0 (Midas)...")
    base_url = getattr(settings, 'ALPACA_BASE_URL', "https://paper-api.alpaca.markets")
    api = tradeapi.REST(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY, base_url, api_version='v2')
except Exception as e:
    sys.exit(1)

def get_market_data():
    try:
        limit = WINDOW + 20
        bars_y = api.get_bars(SYMBOL_Y, TIMEFRAME, limit=limit).df
        bars_x = api.get_bars(SYMBOL_X, TIMEFRAME, limit=limit).df
        
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
    global LAST_REPORT_HOUR
    now_hour = datetime.now().hour
    if now_hour != LAST_REPORT_HOUR:
        msg = f"**Status:** Hunting\n**Z-Score:** {z:.2f} (Trig: {Z_ENTRY})\n**RSI:** {rsi:.1f}"
        messenger.send_message(msg, title="✨ Midas Heartbeat")
        LAST_REPORT_HOUR = now_hour

def execute_cycle():
    global CURRENT_STATE
    
    if not api.get_clock().is_open: return

    data = get_market_data()
    if data is None: return
    
    z = data['z_score']
    rsi = data['rsi']
    
    send_heartbeat(z, rsi)
    
    acct = api.get_account()
    leg_value = (float(acct.buying_power) * 0.40) / 2 # 40% Allocation
    
    # ENTRY: LONG RATIO
    if CURRENT_STATE == 0 and z < -Z_ENTRY and rsi < 35:
        qty_y = int(leg_value / data['close_y'])
        qty_x = int(leg_value / data['close_x'])
        
        api.submit_order(SYMBOL_Y, qty_y, 'buy', 'market', 'day')
        api.submit_order(SYMBOL_X, qty_x, 'sell', 'market', 'day')
        
        CURRENT_STATE = 1
        messenger.send_message(f"Long Gold / Short Silver\nZ: {z:.2f}", title="✨ MIDAS ENTRY")

    # ENTRY: SHORT RATIO
    elif CURRENT_STATE == 0 and z > Z_ENTRY and rsi > 65:
        qty_y = int(leg_value / data['close_y'])
        qty_x = int(leg_value / data['close_x'])
        
        api.submit_order(SYMBOL_Y, qty_y, 'sell', 'market', 'day')
        api.submit_order(SYMBOL_X, qty_x, 'buy', 'market', 'day')
        
        CURRENT_STATE = -1
        messenger.send_message(f"Short Gold / Long Silver\nZ: {z:.2f}", title="✨ MIDAS ENTRY")

    # EXIT
    elif CURRENT_STATE != 0:
        exit_trade = False
        if CURRENT_STATE == 1 and z >= Z_EXIT_OFFSET: exit_trade = True
        elif CURRENT_STATE == -1 and z <= -Z_EXIT_OFFSET: exit_trade = True
        
        if exit_trade:
            api.close_all_positions()
            CURRENT_STATE = 0
            messenger.send_message(f"Overshoot Target Hit. Z: {z:.2f}", title="💰 MIDAS EXIT")

def start():
    messenger.send_message("Engine Online. Tracking Gold/Silver...", title="Moksha 36.0 Midas Boot")
    schedule.every(1).minutes.do(execute_cycle)
    execute_cycle()
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    start()
