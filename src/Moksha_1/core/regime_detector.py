import pandas as pd
import numpy as np
from Moksha_1.utils.logger import logger
from Moksha_1.data.storage.timescaledb import TimescaleStorage

class RegimeDetector:
    """
    Agent 1: The General.
    Determines the Macro Market State (Risk-On vs Risk-Off).
    """
    def __init__(self):
        self.db = TimescaleStorage()
        self.benchmark_symbol = 'SPY' # The Market Proxy

    def get_market_regime(self) -> dict:
        """
        Returns a regime dict:
        {
            "status": "BULL" | "BEAR" | "SIDEWAYS",
            "risk_score": 0.0 to 1.0 (0=Safe, 1=Danger),
            "suggested_multiplier": float (e.g., 0.5 to 3.0)
        }
        """
        try:
            # 1. Fetch Benchmark Data (Last 200 days for long-term trend)
            df = self.db.get_bars_df([self.benchmark_symbol], limit=250)
            
            if df.empty or len(df) < 200:
                logger.warning("⚠️ Not enough SPY data for Regime Detection. Defaulting to Neutral.")
                return {"status": "SIDEWAYS", "risk_score": 0.5, "suggested_multiplier": 1.0}

            df = df.sort_values('time')
            close = df['close']

            # 2. Calculate Key Metrics
            # A. Trend (Price vs 200 SMA)
            sma_200 = close.rolling(200).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1]
            current_price = close.iloc[-1]
            
            # B. Volatility (ATR / Standard Deviation)
            # Recent volatility vs Historical volatility
            returns = close.pct_change().dropna()
            vol_recent = returns.tail(20).std()
            vol_historical = returns.tail(200).std()
            vol_ratio = vol_recent / vol_historical # >1.0 means volatility is rising

            # 3. Determine Regime
            regime = "SIDEWAYS"
            multiplier = 1.0
            
            # LOGIC TREE
            if current_price > sma_200:
                # We are in an Uptrend
                if current_price > sma_50:
                    regime = "BULL (Strong)"
                    multiplier = 3.0  # Max Aggression
                else:
                    regime = "BULL (Weak)"
                    multiplier = 1.5
            else:
                # We are in a Downtrend
                if current_price < sma_50:
                    regime = "BEAR (Crash)"
                    multiplier = 0.0  # Stop Buying! (Or go short)
                else:
                    regime = "BEAR (Recovery?)"
                    multiplier = 0.5

            # 4. Volatility Filter (The "Panic" Switch)
            # If volatility is spiking (Ratio > 1.5), cut risk in half regardless of trend
            if vol_ratio > 1.5:
                regime += " [VOLATILE]"
                multiplier = multiplier * 0.5
                logger.warning(f"⚠️ Market Volatility Spiking (Ratio: {vol_ratio:.2f}). Reducing Exposure.")

            logger.info(f"🏛️ Agent 1 (General): Market is {regime}. Multiplier: {multiplier}x")
            
            return {
                "status": regime,
                "suggested_multiplier": multiplier,
                "metric_sma_200": sma_200,
                "current_price": current_price
            }

        except Exception as e:
            logger.error(f"❌ Regime Detection Failed: {e}")
            # Fallback to safe mode
            return {"status": "ERROR", "suggested_multiplier": 1.0}