# src/Moksha_1/data/processing/ipca_engine.py
import pandas as pd
import numpy as np

class IPCAEngine:
    def calculate_factor_returns(self, features: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the returns of portfolios sorted by characteristics.
        """
        # 1. Pivot Returns
        r_wide = returns_df.pivot(index='time', columns='symbol', values='close').pct_change()
        
        # 2. Pivot Features
        z_wide_dict = {}
        # Exclude metadata columns
        feature_cols = [c for c in features.columns if c not in ['time', 'symbol']]
        
        for col in feature_cols:
            # Shift(1): Yesterday's rank trades Today's return
            z_wide_dict[col] = features.pivot(index='time', columns='symbol', values=col).shift(1)

        # 3. Align Dates
        if not z_wide_dict:
            print("⚠️ No feature columns found.")
            return pd.DataFrame()

        # Intersection of dates
        valid_dates = r_wide.index
        for z in z_wide_dict.values():
            valid_dates = valid_dates.intersection(z.index)
        
        if len(valid_dates) == 0:
            print("⚠️ No overlapping dates between Returns and Features.")
            return pd.DataFrame()

        r_wide = r_wide.loc[valid_dates]
        factor_returns = pd.DataFrame(index=valid_dates)
        
        print(f"🧮 Calculating Factor Returns for {len(valid_dates)} days...")
        
        for factor_name, z_matrix in z_wide_dict.items():
            z_matrix = z_matrix.loc[valid_dates]
            
            # Check if this specific factor is fully empty/NaN
            if z_matrix.isna().all().all():
                print(f"⚠️ Skipping {factor_name}: Factor data is all NaN.")
                continue

            # Calculate Managed Portfolio Return
            # Mean of (Weight * Return)
            daily_factor_ret = (z_matrix * r_wide).mean(axis=1)
            factor_returns[factor_name] = daily_factor_ret
            
        # 4. Cleanup
        # If a specific day has NaNs (e.g., partial data), we drop it.
        # But if the entire DF is empty, we return it as is to avoid errors downstream.
        clean_df = factor_returns.dropna()
        
        if clean_df.empty and not factor_returns.empty:
            print("⚠️ Warning: 'dropna()' removed all rows. Inspecting raw output...")
            print(factor_returns.head()) # Debug print
            return factor_returns.fillna(0) # Fallback: Fill NaNs with 0 instead of dropping

        return clean_df