# test_regime.py
from Moksha_1.data.storage.timescaledb import TimescaleStorage
from Moksha_1.data.processing.regime_detector import RegimeDetector
import pandas as pd

def run_regime_test():
    # 1. Load Data
    db = TimescaleStorage()
    print("⏳ Loading Market Data...")
    bars_df = db.get_bars_df()
    
    if bars_df.empty:
        print("❌ No data found.")
        return

    # 2. Run Detector
    detector = RegimeDetector(window_short=21, window_long=252)
    regime_df = detector.detect_regime(bars_df)
    
    if regime_df.empty:
        print("⚠️ Not enough history to calculate regimes (Need > 252 days).")
        return

    # 3. Report
    print("\n--- 🌪️ MARKET REGIME REPORT (Last Day) ---")
    latest_date = regime_df['time'].max()
    latest_data = regime_df[regime_df['time'] == latest_date]
    
    print(f"Date: {latest_date}")
    print(f"Global Market Health: {detector.get_market_health(regime_df)}")
    
    print("\nSymbol Breakdown:")
    print(latest_data[['symbol', 'regime_score', 'regime_label']].sort_values('regime_score', ascending=False))

    # Optional: Save to DB
    # We would add a 'regimes' table to TimescaleDB in production
    
if __name__ == "__main__":
    run_regime_test()