# src/test_alert.py
from Moksha_1.utils.messenger import messenger

print("🔔 Sending Test Notification...")

# 1. Send a standard status message
messenger.send_message("System is online and listening.", title="✅ Connectivity Test")

# 2. Simulate a Daily Execution Report
# We make up some fake trades to see how the table looks
fake_trades = [
    {"symbol": "AAPL", "side": "BUY", "qty": 15, "price": 185.50},
    {"symbol": "NVDA", "side": "SELL", "qty": 2, "price": 460.00},
    {"symbol": "MSFT", "side": "BUY", "qty": 10, "price": 350.20}
]

start_equity = 100000.0
end_equity = 101250.0 # Simulation of profit

messenger.send_execution_report(fake_trades, start_equity, end_equity)

print("✅ Done! Check your Discord channel.")