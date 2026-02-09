import time
import random
import os

LOG_FILE = "/app/logs/moksha.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

print("💉 Injecting urban telemetry into logs...")

while True:
    # 1. Simulate Equity Engine (Scanning)
    eq_z = round(random.uniform(-1.5, 1.5), 2)
    eq_rsi = round(random.uniform(30, 70), 1)
    
    # 2. Simulate Midas Engine (Near Trigger)
    mid_z = round(random.uniform(1.8, 2.5), 2) # Sometimes hits trigger range
    mid_rsi = round(random.uniform(20, 80), 1)
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    with open(LOG_FILE, "a") as f:
        # Equity Heartbeat
        f.write(f"{timestamp} | INFO | Equity Engine Heartbeat | Z-Score: {eq_z} (Trig: 2.0) | RSI: {eq_rsi}\n")
        
        # Midas Heartbeat
        f.write(f"{timestamp} | INFO | Midas Heartbeat | Z-Score: {mid_z} (Trig: 2.2) | RSI: {mid_rsi}\n")
        
        # Occasional Trade Logic
        if mid_z > 2.2:
            f.write(f"{timestamp} | WARNING | 🚀 MIDAS ENTRY DETECTED | Short Gold / Long Silver\n")
            
    print(f"   Written: Eq Z={eq_z}, Midas Z={mid_z}")
    time.sleep(2) # Update every 2 seconds
