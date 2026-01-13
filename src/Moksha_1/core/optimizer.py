# src/Moksha_1/core/optimizer.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List
from Moksha_1.core.interfaces import IPortfolioOptimizer, Position, MarketRegime

class PortfolioOptimizer(IPortfolioOptimizer):
    """
    Agent 5: Optimizes weights using Mean-Variance Optimization (MVO).
    Goal: Maximize Sharpe Ratio (Return / Risk).
    """
    
    def __init__(self, risk_free_rate: float = 0.04):
        self.rf = risk_free_rate

    def optimize(self, 
                 signals: Dict[str, float], 
                 regime: MarketRegime = None, # Not used in MVP MVO but kept for interface
                 current_positions: List[Position] = None, 
                 capital: float = 100000.0) -> Dict[str, float]:
        """
        Adapts the interface to the internal logic.
        """
        # This wrapper ensures we comply with the interface but delegation to the core logic below
        pass 

    def optimize_weights(self, signals: pd.DataFrame, cov_matrix: pd.DataFrame, max_weight: float = 0.20) -> pd.DataFrame:
        """
        Input:
            signals: DataFrame with ['symbol', 'final_signal'] (Expected Return proxy)
            cov_matrix: DataFrame (N x N) covariance of returns
        Output:
            DataFrame with optimized 'target_weight'
        """
        # 1. Align Data
        valid_symbols = [s for s in signals['symbol'] if s in cov_matrix.index]
        if not valid_symbols:
            print("⚠️ Agent 5: No overlap between Signals and Covariance data.")
            return signals[['symbol', 'final_signal']].rename(columns={'final_signal': 'optimized_weight'})

        # Extract vectors
        mu = signals.set_index('symbol').loc[valid_symbols]['final_signal'].values
        sigma = cov_matrix.loc[valid_symbols, valid_symbols].values
        n_assets = len(valid_symbols)

        # 2. Define Objective: Minimize Negative Sharpe
        def neg_sharpe(weights):
            p_ret = np.sum(weights * mu)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
            if p_vol == 0: return 0
            return -((p_ret - self.rf) / p_vol)

        # 3. Constraints
        # Bounds: -Max to +Max (Long/Short)
        bounds = tuple((-max_weight, max_weight) for _ in range(n_assets))
        
        # Gross Exposure Constraint: Sum(|weights|) <= 1.0 (No Leverage beyond 1x)
        constraints = (
            {'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(np.abs(x))}
        )

        # 4. Optimization
        # Initial Guess: The naive signals
        init_guess = np.array([0.1 * np.sign(s) if s != 0 else 0.0 for s in mu])
        
        try:
            result = minimize(
                neg_sharpe, 
                init_guess, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )
            
            if not result.success:
                print(f"⚠️ Optimization Failed: {result.message}. Using naive weights.")
                return self._naive_sizing(signals, max_weight)

            return pd.DataFrame({
                'symbol': valid_symbols,
                'optimized_weight': result.x
            })

        except Exception as e:
            print(f"❌ Agent 5 Error: {e}")
            return self._naive_sizing(signals, max_weight)

    def _naive_sizing(self, signals, max_weight):
        df = signals.copy()
        df['optimized_weight'] = df['final_signal'].clip(-max_weight, max_weight)
        return df[['symbol', 'optimized_weight']]