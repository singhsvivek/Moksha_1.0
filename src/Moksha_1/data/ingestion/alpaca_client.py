# src/Moksha_1/data/ingestion/alpaca_client.py
from datetime import datetime
from typing import List, Dict
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest # <--- Added Import
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from Moksha_1.config import settings
from Moksha_1.core.interfaces import IMarketDataProvider, BarData

class AlpacaDataProvider(IMarketDataProvider):
    def __init__(self):
        self.client = StockHistoricalDataClient(
            settings.ALPACA_API_KEY,
            settings.ALPACA_SECRET_KEY
        )
        # Dynamic feed selection based on config
        self.feed = DataFeed.SIP if settings.ALPACA_DATA_FEED.lower() == 'sip' else DataFeed.IEX

    async def get_bars(self, symbols: List[str], start: datetime, end: datetime, timeframe: str = '1Day') -> Dict[str, List[BarData]]:
        """
        Fetches historical bars from Alpaca.
        """
        tf_map = {
            '1Min': TimeFrame.Minute,
            '1Hour': TimeFrame.Hour,
            '1Day': TimeFrame.Day
        }
        
        print(f"📡 Fetching {timeframe} data for {len(symbols)} symbols from {self.feed}...")
        
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf_map.get(timeframe, TimeFrame.Day),
            start=start,
            end=end,
            feed=self.feed
        )

        try:
            # Alpaca SDK handles pagination automatically
            bars = self.client.get_stock_bars(request_params)
            
            result = {}
            for symbol, data in bars.data.items():
                result[symbol] = [
                    BarData(
                        symbol=symbol,  # Added explicit symbol field as per your interface
                        timestamp=b.timestamp,
                        open=b.open,
                        high=b.high,
                        low=b.low,
                        close=b.close,
                        volume=b.volume,
                        vwap=b.vwap,
                        trade_count=b.trade_count
                    ) for b in data
                ]
            return result
            
        except Exception as e:
            print(f"❌ Alpaca API Error (get_bars): {e}")
            return {}

    # --- THE MISSING METHOD (FIX) ---
    async def get_latest_bars(self, symbols: List[str]) -> Dict[str, BarData]:
        """
        Get the most recent bar for each symbol.
        Required by IMarketDataProvider interface.
        """
        request_params = StockLatestBarRequest(
            symbol_or_symbols=symbols,
            feed=self.feed
        )
        
        try:
            latest_bars = self.client.get_stock_latest_bar(request_params)
            
            result = {}
            for symbol, bar in latest_bars.items():
                result[symbol] = BarData(
                    symbol=symbol,
                    timestamp=bar.timestamp,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    vwap=bar.vwap,
                    trade_count=bar.trade_count
                )
            return result
        except Exception as e:
            print(f"❌ Alpaca API Error (get_latest_bars): {e}")
            return {}

    async def subscribe_bars(self, symbols: List[str], callback: callable):
        """
        Placeholder for Real-time Websocket implementation (Sprint 3).
        """
        pass