import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from Moksha_1.config import settings
from Moksha_1.utils.logger import logger

class TimescaleStorage:
    def __init__(self):
        self.conn = self._connect()
        if self.conn:
            self._initialize_schema()

    def _connect(self):
        try:
            conn = psycopg2.connect(settings.DB_CONNECTION_STRING)
            conn.autocommit = True
            return conn
        except Exception as e:
            logger.error(f"❌ DB Connection Failed: {e}")
            return None

    def _initialize_schema(self):
        """Creates necessary tables if they don't exist."""
        # 1. Market Data Table
        stock_query = """
        CREATE TABLE IF NOT EXISTS stock_bars (
            time TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume DOUBLE PRECISION,
            vwap DOUBLE PRECISION,
            trade_count DOUBLE PRECISION,
            PRIMARY KEY (time, symbol)
        );
        """
        
        # 2. Portfolio Performance Table (NEW)
        portfolio_query = """
        CREATE TABLE IF NOT EXISTS portfolio_metrics (
            time TIMESTAMPTZ NOT NULL,
            equity DOUBLE PRECISION,
            cash DOUBLE PRECISION,
            daily_pl DOUBLE PRECISION,
            PRIMARY KEY (time)
        );
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(stock_query)
                cur.execute(portfolio_query)
            logger.info("✅ Database Schema Verified (stock_bars + portfolio_metrics).")
        except Exception as e:
            logger.error(f"❌ Schema Initialization Failed: {e}")

    # --- MARKET DATA METHODS ---
    def save_bars(self, df: pd.DataFrame):
        if df.empty or self.conn is None: return

        query = """
            INSERT INTO stock_bars (time, symbol, open, high, low, close, volume, vwap, trade_count)
            VALUES %s
            ON CONFLICT (time, symbol) DO NOTHING;
        """
        cols = ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']
        if 'trade_count' not in df.columns: df['trade_count'] = 0
            
        data = [tuple(x) for x in df[cols].to_numpy()]

        try:
            with self.conn.cursor() as cur:
                execute_values(cur, query, data)
            logger.info(f"💾 Saved {len(df)} rows to DB.")
        except Exception as e:
            logger.error(f"❌ DB Save Error: {e}")
            self.conn = self._connect()

    def get_bars_df(self, symbols: list, limit=1000) -> pd.DataFrame:
        if not symbols or self.conn is None: return pd.DataFrame()
        sym_str = ",".join([f"'{s}'" for s in symbols])
        query = f"SELECT * FROM stock_bars WHERE symbol IN ({sym_str}) ORDER BY time ASC LIMIT {limit};"
        try:
            return pd.read_sql(query, self.conn)
        except Exception as e:
            logger.error(f"❌ DB Read Error: {e}")
            return pd.DataFrame()

    # --- PORTFOLIO SNAPSHOT METHODS (NEW) ---
    def save_portfolio_snapshot(self, equity: float, cash: float, daily_pl: float):
        """Records the fund's NAV for the day."""
        if self.conn is None: return
        
        query = """
            INSERT INTO portfolio_metrics (time, equity, cash, daily_pl)
            VALUES (NOW(), %s, %s, %s);
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (equity, cash, daily_pl))
            logger.info(f"📸 Portfolio Snapshot Saved (Equity: ${equity:,.2f})")
        except Exception as e:
            logger.error(f"❌ Failed to save snapshot: {e}")

    def get_portfolio_history(self, limit=30) -> pd.DataFrame:
        """Fetches the equity curve for the dashboard."""
        if self.conn is None: return pd.DataFrame()
        query = f"SELECT * FROM portfolio_metrics ORDER BY time ASC LIMIT {limit};"
        try:
            return pd.read_sql(query, self.conn)
        except Exception:
            return pd.DataFrame()