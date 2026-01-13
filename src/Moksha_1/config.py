# src/Moksha_1/config.py
import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field # <--- Import Field

class Settings(BaseSettings):
    ALPACA_API_KEY: str
    ALPACA_SECRET_KEY: str
    ALPACA_DATA_FEED: str = "iex"
    
    # --- THE FIX ---
    # In Pydantic V2, we use Field(validation_alias=...) instead of Config fields
    UNIVERSE_STRING: str = Field("SPY", validation_alias="UNIVERSE")
    
    @property
    def UNIVERSE(self) -> List[str]:
        return [s.strip() for s in self.UNIVERSE_STRING.split(",") if s.strip()]

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
        # Removed 'fields' dict entirely to stop the warning
    )

settings = Settings()