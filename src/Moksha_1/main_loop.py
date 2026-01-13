# src/Moksha_1/main_loop.py
import time
import schedule
import pandas as pd
from datetime import datetime
import pytz

# Import Logger
from Moksha_1.utils.logger import logger  # <--- NEW IMPORT

from Moksha_1.config import settings
from Moksha_1.core.decision_engine import DecisionEngine
from Moksha_1.core.execution import ExecutionHandler
from Moksha_1.core.optimizer import PortfolioOptimizer
from Moksha_1.data.storage.timescaledb import TimescaleStorage
from Moksha_1.data.ingestion.alpaca_client import AlpacaDataProvider

# ... (Initialize components as before) ...
TZ_CENTRAL = pytz.timezone("America/Chicago")

def run_daily_cycle():
    # Replace print() with logger.info()
    logger.info(f"⏰ MARKET OPEN JOB STARTED: {datetime.now(TZ_CENTRAL)}")
    
    universe = settings.UNIVERSE
    logger.info(f"🌌 Trading Universe ({len(universe)}): {universe}")
    
    decisions = council.analyze_market(symbols=universe)
    if decisions.empty:
        logger.warning("💤 No signals generated. Skipping cycle.")
        return

    logger.info("⚖️  Agent 5: Optimizing Portfolio Weights...")
    try:
        bars = db.get_bars_df(symbols=universe)
        returns_df = bars.pivot(index='time', columns='symbol', values='close').pct_change().tail(60)
        cov_matrix = returns_df.cov()
        
        optimized_plan = optimizer.optimize_weights(decisions, cov_matrix)
        final_plan = pd.merge(decisions, optimized_plan, on='symbol', how='left')
        final_plan['final_signal'] = final_plan['optimized_weight'].fillna(0.0)
        
        # Log the optimization table properly
        table_str = final_plan[['symbol', 'regime_label', 'optimized_weight']].to_string(index=False)
        logger.info(f"\n--- 🏆 OPTIMIZED ALLOCATION ---\n{table_str}")
        
    except Exception as e:
        logger.error(f"⚠️ Optimization Skipped: {e}", exc_info=True)
        final_plan = decisions

    logger.info("⚡ Agent 6: Executing Rebalance...")
    executor.execute_rebalance(final_plan, max_allocation=1.0, live_run=True)

def start_scheduler():
    logger.info("⏳ Moksha 2.0 Production Scheduler Started...")
    logger.info(f"   Target Universe: {len(settings.UNIVERSE)} Symbols")
    
    schedule.every().day.at("09:45").do(run_daily_cycle)
    
    logger.info("🚀 Running Startup Check...")
    run_daily_cycle()
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    start_scheduler()