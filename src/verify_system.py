import sys
import os
sys.path.append(os.getcwd())

from Moksha_1.config import settings
from Moksha_1.utils.messenger import messenger
from Moksha_1.utils.logger import logger
import alpaca_trade_api as tradeapi

print("⚡ MOKSHA SYSTEM DIAGNOSTIC ⚡")
print("==============================")

# 1. Check Alpaca Connection
print(f"[1/3] Testing Alpaca API...", end=" ")
try:
    api = tradeapi.REST(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY, getattr(settings, 'ALPACA_BASE_URL', "https://paper-api.alpaca.markets"))
    acct = api.get_account()
    print(f"✅ SUCCESS")
    print(f"      Connected as: {acct.account_number} | Status: {acct.status}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# 2. Check Database (If applicable, skipping for brevity if not critical right now)
print(f"[2/3] Checking Logs Directory...", end=" ")
if os.path.exists("/app/logs"):
    print("✅ EXISTS")
else:
    print("⚠️ MISSING (Creating now...)")
    os.makedirs("/app/logs", exist_ok=True)

# 3. Test Discord Channels
print(f"[3/3] Testing Discord Channels...")
print("      Sending 'Test' to Alert Channel...", end=" ")
try:
    messenger.send_message("🧪 This is a test from the Verification Script.", title="System Check", channel="alert")
    print("✅ SENT")
except: print("❌ FAILED")

print("      Sending 'Test' to Heartbeat Channel...", end=" ")
try:
    messenger.send_message("💓 System Heartbeat Test.", title="Heartbeat Check", channel="heartbeat")
    print("✅ SENT")
except: print("❌ FAILED")

print("\n✨ DIAGNOSTIC COMPLETE.")
