import pandas as pd
from datetime import datetime, timedelta
from Moksha_1.config import settings
from Moksha_1.data.storage.timescaledb import TimescaleStorage
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from Moksha_1.utils.logger import logger

def backfill_history():
    logger.info("🚜 Starting Massive Data Backfill (5 Years)...")
    
    # 1. Connect
    db = TimescaleStorage()
    client = StockHistoricalDataClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)
    
    # 2. Define Range (5 Years ago to Yesterday)
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=365 * 5) # 5 Years
    
    logger.info(f"📅 Range: {start_date.date()} -> {end_date.date()}")
    
    # 3. Fetch in chunks to avoid Timeouts
    universe = settings.UNIVERSE
    
    for symbol in universe:
        logger.info(f"📥 Fetching {symbol}...")
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            bars = client.get_stock_bars(request_params)
            
            if not bars.data:
                logger.warning(f"⚠️ No data found for {symbol}")
                continue
                
            # Convert to DataFrame
            df = bars.df.reset_index()
            
            # Rename columns to match DB schema
            # Alpaca returns: timestamp, symbol, open, high, low, close, volume, trade_count, vwap
            df = df.rename(columns={'timestamp': 'time'})
            
            # Save to DB
            db.save_bars(df)
            logger.info(f"✅ Saved {len(df)} rows for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Failed to backfill {symbol}: {e}")

    logger.info("🏁 Backfill Complete. Database is hydrated.")

if __name__ == "__main__":
    backfill_history()