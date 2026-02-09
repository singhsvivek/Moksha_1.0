import pandas as pd
import os
from datetime import datetime

def generate_dummy_scan():
    print("🧪 Generating Dummy Brain Scan Data...")

    # Define the number of symbols to keep lengths consistent
    num_symbols = 8

    # 1. Create Dummy Data (Simulating a Bull Market)
    data = {
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'SPY'],
        'raw_signal': [0.15, 0.05, 0.20, -0.05, 0.30, 0.10, -0.15, 0.12],
        'final_signal': [0.45, 0.15, 0.60, -0.15, 0.90, 0.30, -0.45, 0.36],
        'regime_label': ['AI_GEN4'] * num_symbols,
        
        # --- THE FIX: Multiply single items by num_symbols ---
        'regime': ['BULL (Test)'] * num_symbols,
        'multiplier': [3.0] * num_symbols,
        'timestamp': [datetime.now().isoformat()] * num_symbols
    }

    try:
        df = pd.DataFrame(data)

        # 2. Ensure Logs Directory Exists
        if not os.path.exists("logs"):
            os.makedirs("logs")

        # 3. Save File
        save_path = "logs/latest_brain_scan.csv"
        df.to_csv(save_path, index=False)
        
        print(f"✅ Test Data Saved to {save_path}")
        print("👉 Refresh your Dashboard 'Neural Network' tab now.")
        
    except Exception as e:
        print(f"❌ Error creating DataFrame: {e}")

if __name__ == "__main__":
    generate_dummy_scan()