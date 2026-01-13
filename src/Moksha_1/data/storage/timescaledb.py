# src/Moksha_1/data/storage/timescaledb.py
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict
from datetime import datetime
from Moksha_1.config import settings
from Moksha_1.core.interfaces import BarData

class TimescaleStorage:
    def __init__(self):
        self.engine = create_engine(settings.DATABASE_URL)

    def save_bars(self, bars_data: Dict[str, List[BarData]]):
        """Bulk save bars to DB."""
        records = []
        for symbol, bars in bars_data.items():
            for bar in bars:
                records.append({
                    'time': bar.timestamp,
                    'symbol': symbol,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': float(bar.volume),
                    'vwap': float(bar.vwap) if bar.vwap else None,
                    'trade_count': getattr(bar, 'trade_count', 0)
                })
        
        if not records:
            return

        df = pd.DataFrame(records)
        # Ensure UTC
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC')
        
        try:
            with self.engine.begin() as conn:
                df.to_sql('market_bars', conn, if_exists='append', index=False, method='multi', chunksize=1000)
            print(f"✅ Saved {len(df)} bars.")
        except SQLAlchemyError as e:
            print(f"❌ DB Error: {e}")

    # --- THE FIXED METHOD ---
    def get_bars_df(self, symbols: List[str] = None, start_date: datetime = None) -> pd.DataFrame:
        """
        Reads bars directly into a DataFrame for vectorized feature calculation.
        FIX: Uses self.engine.connect() to support SQLAlchemy 2.0
        """
        query_str = "SELECT * FROM market_bars"
        params = {}
        
        conditions = []
        if symbols:
            conditions.append("symbol IN :symbols")
            params['symbols'] = tuple(symbols)
        if start_date:
            conditions.append("time >= :start_date")
            params['start_date'] = start_date
            
        if conditions:
            query_str += " WHERE " + " AND ".join(conditions)
            
        query_str += " ORDER BY time ASC"
        
        try:
            # FIX: Open a connection explicitly
            with self.engine.connect() as conn:
                return pd.read_sql(
                    text(query_str), # Wrap query in text() for SA 2.0 safety
                    conn, 
                    params=params, 
                    parse_dates=['time']
                )
        except Exception as e:
            print(f"❌ Read Error: {e}")
            return pd.DataFrame()
        

    def clear_bars_table(self):
        """Truncates the market_bars table to allow fresh ingestion."""
        try:
            with self.engine.begin() as conn:
                conn.execute(text("TRUNCATE TABLE market_bars CASCADE;"))
            print("🧹 Database cleared: market_bars table is now empty.")
        except SQLAlchemyError as e:
            print(f"❌ Error clearing table: {e}")

    def get_features_df(self, symbols: List[str] = None, start_date: datetime = None) -> pd.DataFrame:
        """
        Reads calculated features from the DB.
        """
        query_str = "SELECT * FROM factor_features"
        params = {}
        
        conditions = []
        if symbols:
            conditions.append("symbol IN :symbols")
            params['symbols'] = tuple(symbols)
        if start_date:
            conditions.append("time >= :start_date")
            params['start_date'] = start_date
            
        if conditions:
            query_str += " WHERE " + " AND ".join(conditions)
            
        query_str += " ORDER BY time ASC"
        
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(
                    text(query_str),
                    conn, 
                    params=params, 
                    parse_dates=['time']
                )
        except Exception as e:
            print(f"❌ Read Features Error: {e}")
            return pd.DataFrame()
        
    def save_features(self, df: pd.DataFrame):
        """
        Saves calculated features to the 'factor_features' table.
        """
        try:
            with self.engine.begin() as conn:
                df.to_sql('factor_features', conn, if_exists='append', index=False, method='multi', chunksize=1000)
            print(f"✅ Saved features for {len(df)} rows.")
        except SQLAlchemyError as e:
            print(f"❌ Feature Save Error: {e}")