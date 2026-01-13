# calculate_features.py
import time
from Moksha_1.data.storage.timescaledb import TimescaleStorage
from Moksha_1.data.processing.feature_engineering import FeatureEngine

def run_feature_pipeline():
    
    try:
        # 1. Initialize
        db = TimescaleStorage()
        engine = FeatureEngine()

        print("⏳ Fetching raw bars from TimescaleDB...")
        start_time = time.time()
        
        # Fetch ALL data (in production, you'd fetch only recent data)
        df_bars = db.get_bars_df()
        
        if df_bars.empty:
            print("⚠️ No data found in DB. Run populate_db.py first.")
            return

        print(f"📊 Processing {len(df_bars)} rows...")
        
        # 2. Calculate Features
        df_features = engine.calculate_features(df_bars)
        
        calc_time = time.time()
        print(f"🧠 Calculated features for {len(df_features)} rows in {calc_time - start_time:.2f}s.")

        # 3. Save to DB
        # Note: Ensure your schema has the columns! 
        # For this MVP run, we might just print the head if schema isn't fully synced,
        # but let's try to save.
        
        # IMPORTANT: The schema in step 1 (SQL) only had a few columns.
        # To save ALL these new columns, we might need to alter the table or just save the ones that exist.
        # For now, let's just inspect the output to prove logic works.
        
        print("\n--- Sample Feature Output (Ranked [-1, 1]) ---")
        print(df_features.tail(10)[['time', 'symbol', 'rank_mom_1m', 'rank_rsi_14']])
        
        # Uncomment this when schema matches perfectly
        db.save_features(df_features) 
    except Exception as e:
        print(f"❌ Error during feature pipeline: {e}")


if __name__ == "__main__":
    run_feature_pipeline()