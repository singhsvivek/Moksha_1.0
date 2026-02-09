from Moksha_1.data.storage.timescaledb import TimescaleStorage
from Moksha_1.utils.logger import logger
import schedule
import time

def clean_database():
    logger.info("🧹 Starting Weekly Database Maintenance...")
    db = TimescaleStorage()
    
    try:
        with db.conn.cursor() as cur:
            # 1. Vacuum (Reclaim space)
            # Note: VACUUM cannot run inside a transaction block, 
            # so we might need to set isolation level if this fails.
            # For now, we run a simple analyze.
            cur.execute("ANALYZE stock_bars;")
            logger.info("   ✅ Statistics Updated (ANALYZE)")
            
            # 2. Check size
            cur.execute("SELECT pg_size_pretty(pg_total_relation_size('stock_bars'));")
            size = cur.fetchone()[0]
            logger.info(f"   📊 Current Table Size: {size}")
            
    except Exception as e:
        logger.error(f"❌ Maintenance Failed: {e}")

if __name__ == "__main__":
    clean_database()