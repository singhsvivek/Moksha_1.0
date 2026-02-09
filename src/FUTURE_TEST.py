# Save as check_futures.py and run it
import alpaca_trade_api as tradeapi
from Moksha_1.config import settings

api = tradeapi.REST(
    settings.ALPACA_API_KEY,
    settings.ALPACA_SECRET_KEY,
    "https://paper-api.alpaca.markets", # Or live URL
    api_version='v2'
)

try:
    # Try to fetch a specific futures asset
    asset = api.get_asset('MNQH6') # March 2026 Contract
    print(f"✅ SUCCESS: Found asset {asset.symbol}. Tradable: {asset.tradable}")
except Exception as e:
    print(f"❌ FAILURE: Account not authorized for Futures or Symbol invalid. Error: {e}")