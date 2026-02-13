import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import pytz

# --- IMPORT MESSENGER ---
try:
    from Moksha_1.utils.messenger import messenger
except ImportError:
    messenger = None

# --- CUSTOM HANDLERS ---

class CentralTimeFormatter(logging.Formatter):
    """
    Forces logs into US/Central Time.
    """
    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        return dt.astimezone(pytz.timezone('America/Chicago'))

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.isoformat()
        return s

class UnbufferedRotatingFileHandler(RotatingFileHandler):
    """
    Forces the OS to write to disk immediately after every log.
    Fixes "Laggy Logs" on Docker Volumes.
    """
    def emit(self, record):
        super().emit(record)
        self.flush() 

# --- NEW: DISCORD HANDLER ---
class DiscordLoggingHandler(logging.Handler):
    """
    Automatically sends ERROR and CRITICAL logs to Discord.
    """
    def emit(self, record):
        # Only proceed if messenger exists and level is high enough
        if record.levelno >= logging.ERROR and messenger:
            try:
                # Format the log message
                log_entry = self.format(record)
                
                # Send to the 'error' channel defined in messenger.py
                # Wrap in code block ``` for readability in Discord
                messenger.send_message(
                    message=f"```\n{log_entry}\n```", 
                    title=f"🚨 LOG: {record.levelname}", 
                    channel="error"
                )
            except Exception:
                # Never crash the app if Discord fails
                pass

class MokshaLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MokshaLogger, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self):
        self.logger = logging.getLogger("MokshaBot")
        self.logger.setLevel(logging.INFO)
        
        if self.logger.hasHandlers():
            return

        # Format
        log_format = "%(asctime)s | %(levelname)-8s | %(module)-15s | %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = CentralTimeFormatter(fmt=log_format, datefmt=date_format)

        # 1. Console (Stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 2. File (Unbuffered)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = UnbufferedRotatingFileHandler(
            filename=log_dir / "moksha.log",
            maxBytes=10*1024*1024,
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 3. Discord Error Bridge (THE FIX)
        discord_handler = DiscordLoggingHandler()
        discord_handler.setFormatter(formatter)
        discord_handler.setLevel(logging.ERROR) # Only send ERROR and CRITICAL
        self.logger.addHandler(discord_handler)

    def get_logger(self):
        return self.logger

logger = MokshaLogger().get_logger()