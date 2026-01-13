# src/Moksha_1/utils/logger.py
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

class MokshaLogger:
    """
    Centralized Logging System.
    Follows SOLID:
    - Single Responsibility: Only handles logging configuration.
    - Open/Closed: extensible via new handlers (Slack/Email) without modifying core logic.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MokshaLogger, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self):
        self.logger = logging.getLogger("MokshaBot")
        self.logger.setLevel(logging.INFO)
        
        # Format: [Time] [Level] [Module] Message
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(module)-15s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 1. Console Handler (Stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 2. File Handler (Rotating)
        # Keeps 5 files of 10MB each (prevents disk fill-up)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = RotatingFileHandler(
            filename=log_dir / "moksha.log",
            maxBytes=10*1024*1024, # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger

# Global accessor
logger = MokshaLogger().get_logger()