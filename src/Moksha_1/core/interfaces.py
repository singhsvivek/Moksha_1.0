# src/moksha/core/interfaces.py
"""
Core interfaces defining contracts for Moksha components.

This module establishes the foundation for SOLID architecture by defining
abstract interfaces that concrete implementations must follow. This allows
us to swap implementations (e.g., switch from Alpaca to Interactive Brokers)
without changing higher-level code.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Optional, Protocol
from decimal import Decimal

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field


# ============================================================================
# Data Models (Pydantic for validation and serialization)
# ============================================================================

class BarData(BaseModel):
    """Represents a single price bar (OHLCV data)."""
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    vwap: Optional[Decimal] = None
    trade_count: Optional[int] = None

    class Config:
        frozen = True  # Immutable for thread safety


class StockCharacteristics(BaseModel):
    """Collection of calculated features for a stock."""
    symbol: str
    timestamp: datetime
    
    # Price-based features
    returns_1d: float
    returns_5d: float
    returns_20d: float
    returns_60d: float
    volatility_20d: float
    volatility_60d: float
    
    # Volume-based features
    volume_ratio: float  # Current vs average
    vwap_deviation: float
    
    # Momentum indicators
    rsi_14: float
    macd: float
    macd_signal: float
    
    # Will be extended to 94 factors in full implementation
    
    class Config:
        frozen = True


class Order(BaseModel):
    """Represents a trading order."""
    symbol: str
    quantity: Decimal
    side: str = Field(..., pattern="^(buy|sell)$")
    order_type: str = Field(..., pattern="^(market|limit|stop)$")
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "day"
    
    
class Position(BaseModel):
    """Represents a current position in a security."""
    symbol: str
    quantity: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: float
    

class MarketRegime(BaseModel):
    """Wasserstein-based market regime classification."""
    regime_type: str = Field(..., pattern="^(CALM|VOLATILE|CRASH|RECOVERY)$")
    wasserstein_distance: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime
    distribution_params: Dict[str, float]


# ============================================================================
# Core Interfaces (Following Dependency Inversion Principle)
# ============================================================================

class IMarketDataProvider(ABC):
    """
    Interface for market data sources.
    
    This abstraction allows us to swap between Alpaca, Interactive Brokers,
    or any other data provider without changing dependent code. Following
    the Dependency Inversion Principle, high-level modules depend on this
    interface, not on concrete implementations.
    """
    
    @abstractmethod
    async def get_bars(
        self, 
        symbols: List[str], 
        timeframe: str,
        start: datetime, 
        end: datetime
    ) -> Dict[str, List[BarData]]:
        """
        Retrieve historical bar data for given symbols.
        
        Args:
            symbols: List of stock symbols
            timeframe: Bar interval (e.g., '1Min', '1Hour', '1Day')
            start: Start datetime for data range
            end: End datetime for data range
            
        Returns:
            Dictionary mapping symbols to list of BarData objects
        """
        pass
    
    @abstractmethod
    async def get_latest_bars(
        self, 
        symbols: List[str]
    ) -> Dict[str, BarData]:
        """Get the most recent bar for each symbol."""
        pass
    
    @abstractmethod
    async def subscribe_bars(
        self, 
        symbols: List[str], 
        callback: callable
    ) -> None:
        """Subscribe to real-time bar updates."""
        pass


class IOrderExecutor(ABC):
    """
    Interface for order execution.
    
    Separating order execution from data retrieval follows the Interface
    Segregation Principle - clients that only need data don't need to
    depend on execution methods.
    """
    
    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """
        Submit an order and return order ID.
        
        Args:
            order: Order object containing order details
            
        Returns:
            Order ID assigned by the broker
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Dict:
        """Get current status of an order."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get all current positions."""
        pass


class IFeatureCalculator(ABC):
    """
    Interface for feature engineering.
    
    This allows us to have different feature calculation strategies
    (e.g., simple technical indicators vs. complex ML-derived features)
    that can be swapped or combined.
    """
    
    @abstractmethod
    def calculate_features(
        self, 
        bars: pd.DataFrame
    ) -> StockCharacteristics:
        """
        Calculate features from historical bar data.
        
        Args:
            bars: DataFrame with OHLCV data
            
        Returns:
            StockCharacteristics object with calculated features
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return list of feature names this calculator produces."""
        pass


class IDataCache(ABC):
    """
    Interface for caching layer.
    
    Abstracts caching so we can use Redis, Memcached, or even
    in-memory caching without changing dependent code.
    """
    
    @abstractmethod
    async def get(self, key: str) -> Optional[bytes]:
        """Retrieve cached data."""
        pass
    
    @abstractmethod
    async def set(
        self, 
        key: str, 
        value: bytes, 
        expire_seconds: Optional[int] = None
    ) -> bool:
        """Store data in cache with optional expiration."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove data from cache."""
        pass


class IRegimeDetector(ABC):
    """
    Interface for market regime detection.
    
    This abstraction allows different regime detection algorithms
    (Wasserstein, HMM, etc.) to be used interchangeably.
    """
    
    @abstractmethod
    def detect_regime(
        self, 
        returns: np.ndarray,
        window_size: int = 100
    ) -> MarketRegime:
        """
        Detect current market regime from returns data.
        
        Args:
            returns: Array of log returns
            window_size: Lookback window for regime detection
            
        Returns:
            MarketRegime object with classification and confidence
        """
        pass
    
    @abstractmethod
    def calculate_wasserstein_distance(
        self, 
        current_dist: np.ndarray,
        reference_dist: np.ndarray
    ) -> float:
        """Calculate Wasserstein-1 distance between distributions."""
        pass


class ISignalGenerator(ABC):
    """
    Interface for trading signal generation.
    
    This represents any model that generates expected returns or
    trading signals (NN3, IPCA, simple factors, etc.).
    """
    
    @abstractmethod
    def generate_signals(
        self, 
        features: Dict[str, StockCharacteristics]
    ) -> Dict[str, float]:
        """
        Generate trading signals from features.
        
        Args:
            features: Dictionary mapping symbols to their characteristics
            
        Returns:
            Dictionary mapping symbols to expected returns (signals)
        """
        pass
    
    @abstractmethod
    def get_confidence(self, symbol: str) -> float:
        """Get confidence score for a symbol's signal."""
        pass


class IPortfolioOptimizer(ABC):
    """
    Interface for portfolio optimization strategies.
    
    This allows different optimization approaches (MSRR, mean-variance,
    risk parity, etc.) to be used interchangeably.
    """
    
    @abstractmethod
    def optimize(
        self,
        signals: Dict[str, float],
        regime: MarketRegime,
        current_positions: List[Position],
        capital: Decimal
    ) -> Dict[str, Decimal]:
        """
        Determine optimal position sizes.
        
        Args:
            signals: Expected returns for each symbol
            regime: Current market regime
            current_positions: Existing positions
            capital: Available capital
            
        Returns:
            Dictionary mapping symbols to target position sizes
        """
        pass


# ============================================================================
# Configuration Models
# ============================================================================

class DataSourceConfig(BaseModel):
    """Configuration for data source."""
    api_key: str
    secret_key: str
    base_url: str
    rate_limit_per_minute: int = 200
    retry_attempts: int = 3
    timeout_seconds: int = 30


class TradingConfig(BaseModel):
    """Configuration for trading behavior."""
    initial_capital: Decimal
    max_position_size: float = Field(0.10, ge=0.01, le=0.50)
    max_portfolio_leverage: float = Field(1.0, ge=1.0, le=3.0)
    enable_short_selling: bool = False
    trading_enabled: bool = False  # Safety switch
    

class FeatureConfig(BaseModel):
    """Configuration for feature calculation."""
    lookback_periods: Dict[str, int] = {
        "short": 20,
        "medium": 60,
        "long": 252
    }
    volatility_windows: List[int] = [20, 60]
    momentum_periods: List[int] = [1, 5, 20, 60]