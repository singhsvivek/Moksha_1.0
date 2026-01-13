# populate_db.py
import asyncio
from datetime import datetime, timedelta
import pytz

from Moksha_1.data.ingestion.alpaca_client import AlpacaDataProvider
from Moksha_1.data.storage.timescaledb import TimescaleStorage

async def run_ingestion():
    # 1. Initialize
    provider = AlpacaDataProvider()
    storage = TimescaleStorage()

    # 2. CLEANUP: Clear old data to prevent "Unique Violation" errors
    print("🧹 Cleaning up old data...")
    storage.clear_bars_table()

    # 3. Define Scope
    universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY", "QQQ", "DIA", "CRWV"]
    
    # FIX: Fetch 3 Years (approx 750 trading days)
    # This ensures we have enough history for the 252-day momentum lookback
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=365 * 3) 

    # 4. Fetch Data
    print(f"🚀 Starting Ingestion for {len(universe)} tickers (3 Years history)...")
    bars_data = await provider.get_bars(
        symbols=universe,
        start=start_date,
        end=end_date,
        timeframe='1Day'
    )

    # 5. Store Data
    if bars_data:
        storage.save_bars(bars_data)
        print("🎉 Ingestion Complete. Data is now in TimescaleDB.")
    else:
        print("⚠️ No data received.")

if __name__ == "__main__":
    asyncio.run(run_ingestion())