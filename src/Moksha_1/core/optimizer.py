import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from Moksha_1.utils.logger import logger

class PortfolioOptimizer:
    """
    Moksha HRP Allocator (v6.6 - Numpy Safe)
    Fixed 'read-only' errors by forcing memory copies on diagonal extraction.
    """
    def __init__(self):
        print("   ✅ HRP Safe-Numpy Optimizer Loaded")
        self.conviction_multiplier = 1.0

    def get_hrp_weights(self, cov_matrix):
        # 1. ROBUST CORRELATION
        v = np.sqrt(np.diag(cov_matrix))
        outer_v = np.outer(v, v)
        corr = cov_matrix / outer_v
        corr = corr.clip(-1.0, 1.0).fillna(0)
        
        # 2. DISTANCE
        dist = np.sqrt(0.5 * (1 - corr)).fillna(0)
        np.fill_diagonal(dist.values, 0)
        condensed_dist = squareform(dist.values, checks=False)
        link = sch.linkage(condensed_dist, 'single')
        
        # 3. SORT
        ix = sch.leaves_list(link)
        ordered_cov = cov_matrix.iloc[ix, ix]
        
        # 4. RECURSION
        weights = pd.Series(1.0, index=ordered_cov.index)
        
        def recurse_bi(sub_cov):
            if len(sub_cov) <= 1: return
            
            split = len(sub_cov) // 2
            c1 = sub_cov.iloc[:split, :split]
            c2 = sub_cov.iloc[split:, split:]
            
            # --- CLUSTER VARIANCE (The Fix) ---
            def get_cluster_var(c):
                # FIX: .copy() ensures we own the memory and can write to it
                diag = np.diag(c).copy() 
                diag[diag == 0] = 1e-6 
                
                w_inv = 1 / diag
                # Manual Dot Product to avoid dimension errors
                # w_inv is 1D array, c is 2D matrix
                # (w_inv * c) applies weighting to rows
                # dot(..., w_inv) applies weighting to cols
                
                # Normalize weights
                w_inv = w_inv / w_inv.sum()
                
                # Variance = w'Cw
                var = np.dot(np.dot(w_inv, c), w_inv)
                return var

            var1 = get_cluster_var(c1)
            var2 = get_cluster_var(c2)
            
            # Allocation Factor
            if var1 == 0 and var2 == 0: alpha = 0.5
            else: alpha = 1 - var1 / (var1 + var2)
            
            # Update Weights (Using index matching)
            weights[c1.index] *= alpha
            weights[c2.index] *= (1 - alpha)
            
            recurse_bi(c1)
            recurse_bi(c2)

        recurse_bi(ordered_cov)
        return weights

    def optimize_weights(self, decisions: pd.DataFrame, history_df: pd.DataFrame = None) -> pd.DataFrame:
        if decisions.empty: return pd.DataFrame()
        
        # Deep Copy to prevent SettingWithCopy warnings in main loop
        decisions = decisions.copy()
        
        active_buys = decisions[decisions['final_signal'] > 0]
        if active_buys.empty: return decisions
        
        symbols = active_buys['symbol'].tolist()

        def apply_equal_weight():
            w = 1.0 / len(symbols)
            # Safe assignment using loc
            decisions.loc[decisions['final_signal'] > 0, 'final_signal'] = w * self.conviction_multiplier
            return decisions

        if history_df is None or len(symbols) < 2: return apply_equal_weight()

        try:
            # Data Prep
            subset = history_df[symbols].replace(0, np.nan).dropna()
            if len(subset) < 30: return apply_equal_weight()
            
            log_ret = np.log(subset / subset.shift(1)).dropna()
            cov = log_ret.cov()
            
            # Run HRP
            hrp_weights = self.get_hrp_weights(cov)
            
            # Map Results
            weight_map = hrp_weights.to_dict()
            decisions['hrp_weight'] = decisions['symbol'].map(weight_map).fillna(0.0)
            
            # Final Calculation
            decisions['final_signal'] = decisions['hrp_weight'] * self.conviction_multiplier
            
            return decisions

        except Exception as e:
            logger.error(f"❌ HRP Error: {e}")
            return apply_equal_weight()
