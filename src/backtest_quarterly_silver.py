import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from tqdm import tqdm
import sys
from Moksha_1.config import settings
from Moksha_1.utils.logger import logger

# --- CONFIGURATION ---
SYMBOL = 'AGQ'       # Silver 2x ETF
START_DATE = '2025-01-01'
CAPITAL = 25000.0
RISK_PCT = 0.03      # 3% Risk

class QuarterlySilverBacktest:
    def __init__(self):
        try:
            base_url = getattr(settings, 'ALPACA_BASE_URL', "https://paper-api.alpaca.markets")
            self.api = tradeapi.REST(
                settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY, 
                base_url=base_url, api_version='v2'
            )
            logger.info("✅ Alpaca Connected.")
        except Exception as e:
            logger.error(f"❌ Connection Failed: {e}")
            sys.exit(1)

    def fetch_real_data(self):
        logger.info(f"⏳ Fetching Data for {SYMBOL}...")
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=365)
        
        try:
            bars = self.api.get_bars(
                SYMBOL, '5Min', 
                start=start_dt.strftime('%Y-%m-%d'), 
                end=end_dt.strftime('%Y-%m-%d'), 
                feed='iex'
            ).df
            
            if bars.empty: raise ValueError("No Data")
            if bars.index.tz is None: bars.index = bars.index.tz_localize('UTC')
            bars.index = bars.index.tz_convert('America/Chicago')
            return bars
        except Exception:
            logger.exception("❌ Fetch Error")
            return pd.DataFrame()

    def identify_cycles(self, df):
        df['time_only'] = df.index.time
        t_open = pd.Timestamp("08:30").time()
        t_q1_end = pd.Timestamp("10:00").time()
        t_q2_end = pd.Timestamp("11:30").time()
        
        signals = []
        grouped = df.groupby(df.index.date)
        
        for date, day_data in tqdm(grouped, desc="Scanning for Traps"):
            if len(day_data) < 30: continue
            
            # Q1 Range
            q1_data = day_data[(day_data['time_only'] >= t_open) & (day_data['time_only'] < t_q1_end)]
            if q1_data.empty: continue
            q1_high, q1_low = q1_data['high'].max(), q1_data['low'].min()
            
            # Q2 Analysis
            q2_data = day_data[(day_data['time_only'] >= t_q1_end) & (day_data['time_only'] < t_q2_end)]
            if q2_data.empty: continue
            
            # Reset index to handle Time correctly
            candles = q2_data.rename_axis('timestamp').reset_index().to_dict('records')
            
            for i in range(len(candles)):
                curr = candles[i]
                prev = candles[i-1] if i > 0 else None
                
                signal = 0
                setup_type = ""
                
                # 1. Wick Sweep (Type A)
                if curr['low'] < q1_low and curr['close'] > q1_low:
                    signal = 1; setup_type = "Wick Sweep"; stop_loss = curr['low'] - 0.05
                elif curr['high'] > q1_high and curr['close'] < q1_high:
                    signal = -1; setup_type = "Wick Sweep"; stop_loss = curr['high'] + 0.05
                
                if signal != 0:
                    signals.append({'time': curr['timestamp'], 'signal': signal, 'entry': curr['close'], 'stop': stop_loss, 'type': setup_type})
                    break # Priority to Wick Sweep
                
                # 2. Fakeout Reclaim (Type B)
                if prev:
                    if prev['close'] < q1_low and curr['close'] > q1_low:
                        signal = 1; setup_type = "Fakeout Reclaim"; stop_loss = min(prev['low'], curr['low']) - 0.05
                    elif prev['close'] > q1_high and curr['close'] < q1_high:
                        signal = -1; setup_type = "Fakeout Reclaim"; stop_loss = max(prev['high'], curr['high']) + 0.05
                
                if signal != 0:
                    signals.append({'time': curr['timestamp'], 'signal': signal, 'entry': curr['close'], 'stop': stop_loss, 'type': setup_type})
                    break

        return pd.DataFrame(signals)

    def run(self):
        logger.info("🚀 Starting A/B Split Backtest...")
        df = self.fetch_real_data()
        if df.empty: return
        
        signals = self.identify_cycles(df)
        if signals.empty: return
        
        equity = CAPITAL
        trades = 0
        
        # --- NEW: A/B TESTING STATS ---
        stats = {
            "Wick Sweep": {"wins": 0, "count": 0, "pnl": 0.0},
            "Fakeout Reclaim": {"wins": 0, "count": 0, "pnl": 0.0}
        }
        
        for i, setup in signals.iterrows():
            entry, stop, direction = setup['entry'], setup['stop'], setup['signal']
            stype = setup['type']
            
            risk_amt = equity * RISK_PCT
            shares = int(risk_amt / abs(entry - stop))
            if shares == 0: continue
            
            take_profit = entry + (abs(entry - stop) * 2 * direction)
            be_trigger = entry + (abs(entry - stop) * 1 * direction)
            
            future_data = df[df.index > setup['time']].head(48)
            outcome = 0
            stop_moved = False
            
            for _, bar in future_data.iterrows():
                curr_stop = entry if stop_moved else stop
                if direction == 1:
                    if bar['low'] <= curr_stop: outcome = -1 if not stop_moved else 0; break
                    if bar['high'] >= take_profit: outcome = 1; break
                    if bar['high'] >= be_trigger: stop_moved = True
                else:
                    if bar['high'] >= curr_stop: outcome = -1 if not stop_moved else 0; break
                    if bar['low'] <= take_profit: outcome = 1; break
                    if bar['low'] <= be_trigger: stop_moved = True
            
            trade_pnl = 0
            if outcome == 1:
                trade_pnl = abs(take_profit - entry) * shares
                stats[stype]['wins'] += 1
            elif outcome == -1:
                trade_pnl = -abs(stop - entry) * shares
            else:
                 # Time-based exit
                last_price = future_data.iloc[-1]['close']
                trade_pnl = (last_price - entry) * shares * direction
                if stop_moved and trade_pnl < 0: trade_pnl = 0
                if trade_pnl > 0: stats[stype]['wins'] += 1

            equity += trade_pnl
            trades += 1
            stats[stype]['count'] += 1
            stats[stype]['pnl'] += trade_pnl

        # Print Split Results
        logger.info("\n" + "="*50)
        logger.info(f"📊 A/B TEST RESULTS (365 Days)")
        logger.info("="*50)
        
        for name, data in stats.items():
            wr = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0
            logger.info(f"🔹 STRATEGY: {name.upper()}")
            logger.info(f"   Trades:   {data['count']}")
            logger.info(f"   Win Rate: {wr:.1f}%")
            logger.info(f"   Net PnL:  ${data['pnl']:,.2f}")
            logger.info("-" * 30)

        logger.info(f"💰 FINAL EQUITY: ${equity:,.2f} (Total Return: {((equity-CAPITAL)/CAPITAL)*100:.1f}%)")

if __name__ == "__main__":
    bt = QuarterlySilverBacktest()
    bt.run()