import os
from dotenv import load_dotenv

# Load .env variables immediately
load_dotenv()

class Settings:
    """
    Optimized Configuration. 
    Removes Pydantic dependency to prevent 'model_config' errors.
    """
    
    # --- 1. ALPACA SETTINGS ---
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")

    # --- 2. TRADING UNIVERSE ---
    # Parses "AAPL,MSFT,SPY" string into a Python List
    _universe_str = os.getenv("UNIVERSE", "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META,SPY")
    UNIVERSE = [s.strip() for s in _universe_str.split(",") if s.strip()]

    # --- 3. DATABASE SETTINGS ---
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "moksha_db")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres") # Default to service name 'postgres'
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

    # --- 4. SYSTEM SETTINGS ---
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
    # --- COMMUNICATIONS GRID (Moksha 3.0) ---
    # Default fallback is provided but should be overridden by .env
    DISCORD_WEBHOOK_ALERTS = os.getenv('DISCORD_WEBHOOK_ALERTS', '')
    DISCORD_WEBHOOK_HEARTBEAT = os.getenv('DISCORD_WEBHOOK_HEARTBEAT', '')
    DISCORD_WEBHOOK_ERROR = os.getenv('DISCORD_WEBHOOK_ERROR', '')

    # --- 5. CRITICAL FIX: CONNECTION STRING ---
    # The DB module looks for 'DB_CONNECTION_STRING', not 'DATABASE_URL'
    @property
    def DB_CONNECTION_STRING(self):
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

# Initialize Singleton
settings = Settings()