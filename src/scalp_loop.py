import time
import schedule
import pandas as pd
import numpy as np
import torch
import sys
import os
from datetime import datetime
import pytz

# --- Imports ---
from Moksha_1.utils.logger import logger
from Moksha_1.config import settings
from Moksha_1.utils.messenger import messenger
from Moksha_1.core.decision_engine import DecisionEngine
from Moksha_1.core.execution import ExecutionHandler
from Moksha_1.data.storage.timescaledb import TimescaleStorage
from Moksha_1.data.ingestion.alpaca_client import AlpacaDataProvider

# --- Global Config ---
TZ_CENTRAL = pytz.timezone("America/Chicago")
SCALP_TARGETS = ['SPY', 'QQQ'] # Proxies for ES/NQ

# State Management
STATE = {
    "position": {sym: 0 for sym in SCALP_TARGETS},
    "entry_price": {sym: 0.0 for sym in SCALP_TARGETS},
    "stop_loss": {sym: 0.0 for sym in SCALP_TARGETS}
}

try:
    logger.info("⚙️ Initializing Moksha 10.0 (Quantum Scalper)...")
    db = TimescaleStorage()
    data_provider = AlpacaDataProvider()
    executor = ExecutionHandler()
    brain = DecisionEngine()
    
    # Check Model
    if brain.model is None:
        logger.critical("❌ Neural Network not loaded. Cannot Scalp.")
        sys.exit(1)
        
    logger.info("✅ Scalping Agents Ready.")

except Exception as e:
    logger.critical(f"❌ Init Failed: {e}")
    sys.exit(1)

def calculate_factors(df):
    """
    Intraday Feature Engineering (Fast)
    """
    # 1. Z-Score (Mean Reversion)
    window = 20
    df['mean'] = df['close'].rolling(window).mean()
    df['std'] = df['close'].rolling(window).std()
    df['z_score'] = (df['close'] - df['mean']) / df['std']
    
    # 2. Volume Imbalance (Microstructure)
    # Approx: (Close - Open) / (High - Low) * Volume
    # Measures buying vs selling pressure within the candle
    range_len = df['high'] - df['low']
    range_len = range_len.replace(0, 0.01) # Avoid div by zero
    df['imbalance'] = ((df['close'] - df['open']) / range_len) * df['volume']
    df['imbalance_smooth'] = df['imbalance'].rolling(3).mean()
    
    # 3. ATR (Volatility)
    df['tr'] = np.maximum(df['high'] - df['low'], 
               np.maximum(abs(df['high'] - df['close'].shift()), 
                          abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(14).mean()
    
    return df.iloc[-1]

def execute_scalp_cycle():
    now = datetime.now(TZ_CENTRAL)
    
    # 0. Market Hours Check (08:30 - 15:00 Central)
    market_open = now.replace(hour=8, minute=30, second=0)
    market_close = now.replace(hour=15, minute=0, second=0)
    
    if not (market_open <= now <= market_close):
        if now.minute % 15 == 0: logger.info("💤 Market Closed/Illiquid. Standing by.")
        return

    for symbol in SCALP_TARGETS:
        try:
            # 1. Fetch Data (Last 60 1-min bars)
            # Try Live first, Fallback to DB
            try:
                # Alpaca SDK specific method for bars
                bars = data_provider.api.get_bars(symbol, '1Min', limit=60).df
                # Fix timezone if needed
                if bars.empty: continue
            except:
                bars = db.get_bars_df([symbol], limit=60)
            
            if len(bars) < 30: continue

            # 2. Calc Signals
            latest = calculate_factors(bars)
            z_score = latest['z_score']
            atr = latest['atr']
            price = latest['close']
            
            # 3. AI Confirmation
            # Input vector: [Imbalance, DistMean, RelVol, ZScore, RSI_Proxy]
            # RSI Proxy: ZScore is mathematically similar to RSI on short timeframes
            features = np.array([
                latest['imbalance_smooth'],
                (price / latest['mean']) - 1,
                atr / price,
                z_score,
                50.0 # Placeholder
            ], dtype=np.float32)
            
            # Pad/Crop
            if len(features) < brain.EXPECTED_INPUT_SIZE:
                features = np.pad(features, (0, brain.EXPECTED_INPUT_SIZE - len(features)), 'constant')
            
            with torch.no_grad():
                ai_prob = brain.model(torch.tensor(features).unsqueeze(0)).item()

            # 4. State Machine
            pos = STATE["position"][symbol]
            
            # --- ENTRY LOGIC ---
            if pos == 0:
                # LONG: Oversold (Z < -2) + AI Bullish (> 0.6)
                if z_score < -2.0 and ai_prob > 0.60:
                    logger.info(f"🚀 SCALP LONG {symbol} | Z:{z_score:.2f} | AI:{ai_prob:.2f}")
                    
                    # Risk Management: Risk 0.5% of Equity
                    acct = executor.client.get_account()
                    risk_amt = float(acct.equity) * 0.005
                    stop_dist = atr * 2.0
                    qty = int(risk_amt / stop_dist)
                    if qty < 1: qty = 1
                    
                    # Execute
                    executor.submit_order(symbol, qty, 'buy')
                    STATE["position"][symbol] = qty
                    STATE["entry_price"][symbol] = price
                    STATE["stop_loss"][symbol] = price - stop_dist
                    messenger.send_message(f"Long {symbol} @ {price}", title="Scalp Entry")

                # SHORT: Overbought (Z > 2) + AI Bearish (< 0.4)
                elif z_score > 2.0 and ai_prob < 0.40:
                    logger.info(f"🔻 SCALP SHORT {symbol} | Z:{z_score:.2f} | AI:{ai_prob:.2f}")
                    
                    acct = executor.client.get_account()
                    risk_amt = float(acct.equity) * 0.005
                    stop_dist = atr * 2.0
                    qty = int(risk_amt / stop_dist)
                    if qty < 1: qty = 1
                    
                    executor.submit_order(symbol, qty, 'sell')
                    STATE["position"][symbol] = -qty
                    STATE["entry_price"][symbol] = price
                    STATE["stop_loss"][symbol] = price + stop_dist
                    messenger.send_message(f"Short {symbol} @ {price}", title="Scalp Entry")

            # --- EXIT LOGIC ---
            else:
                is_long = pos > 0
                stop_hit = (is_long and price < STATE["stop_loss"][symbol]) or \
                           (not is_long and price > STATE["stop_loss"][symbol])
                
                # Target: Mean Reversion (Z returns to 0)
                # Or AI flips against us
                target_hit = (is_long and z_score >= 0) or (not is_long and z_score <= 0)
                ai_exit = (is_long and ai_prob < 0.4) or (not is_long and ai_prob > 0.6)
                
                if stop_hit:
                    logger.warning(f"🛑 STOP LOSS {symbol} @ {price}")
                    executor.close_position(symbol)
                    STATE["position"][symbol] = 0
                elif target_hit:
                    logger.info(f"💰 PROFIT TAKING {symbol} @ {price} (Mean Reverted)")
                    executor.close_position(symbol)
                    STATE["position"][symbol] = 0
                elif ai_exit:
                    logger.info(f"⚠️ AI EXIT {symbol} (Conviction Lost)")
                    executor.close_position(symbol)
                    STATE["position"][symbol] = 0

        except Exception as e:
            logger.error(f"Error in Scalp Loop ({symbol}): {e}")

def run():
    logger.info("⏳ Starting 1-Minute Scalp Loop...")
    schedule.every(1).minutes.do(execute_scalp_cycle)
    
    # Run once immediately to check connections
    if executor.check_market_status():
        execute_scalp_cycle()
        
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    run()
