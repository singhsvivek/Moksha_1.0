# test_ipca.py
from Moksha_1.data.storage.timescaledb import TimescaleStorage
from Moksha_1.data.processing.ipca_engine import IPCAEngine
import pandas as pd

def run_backtest():
    # 1. Initialize
    db = TimescaleStorage()
    ipca = IPCAEngine()
    
    print("⏳ Loading Data from TimescaleDB...")
    df_bars = db.get_bars_df()
    df_features = db.get_features_df()
    
    if df_bars.empty:
        print("❌ Error: No Market Bars found in DB.")
        return
    if df_features.empty:
        print("❌ Error: No Features found in DB.")
        return

    # 2. Run IPCA Math
    print("🧠 Running IPCA Engine...")
    factor_returns = ipca.calculate_factor_returns(df_features, df_bars)
    
    # --- CRITICAL FIX: Check if empty before accessing ---
    if factor_returns.empty:
        print("❌ Error: IPCA Engine returned no data. Check feature/return alignment.")
        return

    # 3. Analyze Results
    print(f"\n--- 🏆 FACTOR PERFORMANCE REPORT ({len(factor_returns)} Days) ---")
    
    # Calculate Cumulative Returns safely
    cumulative_returns = (1 + factor_returns).cumprod()
    
    for col in factor_returns.columns:
        # Check if the column has valid data
        if pd.isna(cumulative_returns[col].iloc[-1]):
            print(f"Factor: {col} -> ⚠️ Insufficient data to calculate return.")
            continue
            
        total_ret = (cumulative_returns.iloc[-1][col] - 1) * 100
        # Sharpe calculation (annualized)
        daily_std = factor_returns[col].std()
        if daily_std == 0:
            sharpe = 0.0
        else:
            sharpe = factor_returns[col].mean() / daily_std * (252**0.5)
        
        print(f"\nFactor: {col}")
        print(f"  > Total Return: {total_ret:.2f}%")
        print(f"  > Annualized Sharpe: {sharpe:.2f}")
        
        if total_ret > 0:
            print("  ✅ POSITIVE SIGNAL: Strategy profitable.")
        else:
            print("  📉 NEGATIVE SIGNAL: Strategy unprofitable.")

if __name__ == "__main__":
    run_backtest()