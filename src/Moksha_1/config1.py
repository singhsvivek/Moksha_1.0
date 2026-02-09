import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Settings:
    # Alpaca Keys
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    
    # Database Config
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "moksha_db")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres") # Default to service name
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

    # Trading Config
    UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'SPY']
    
    # System Config
    LOG_LEVEL = "INFO"

    # --- THE FIX: Property to construct the connection string ---
    @property
    def DB_CONNECTION_STRING(self):
        return f"postgres://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

settings = Settings()