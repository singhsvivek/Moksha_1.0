# main_paper_trade.py
from Moksha_1.core.decision_engine import DecisionEngine
from Moksha_1.core.execution import ExecutionHandler
import time

def run_moksha_bot():
    print("\n========================================================")
    print("      🕉️  MOKSHA 2.0 - INSTITUTIONAL TRADING BOT      ")
    print("========================================================\n")
    
    # 1. Initialize Agents
    council = DecisionEngine()
    executor = ExecutionHandler()
    
    # 2. Define Universe
    universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY", "QQQ", "DIA", "CRWV"]
    
    # 3. The Council Deliberates
    decisions = council.analyze_market(symbols=universe)
    
    if decisions.empty:
        print("💤 Market is closed or no data available.")
        return

    # 4. Display Briefing
    print("\n--- 🏛️ FINAL DECISIONS ---")
    print(decisions[['symbol', 'regime_label', 'final_signal']].to_string(index=False))
    
    # 5. The Executor Acts
    # We set max_allocation to 10% per stock for safety
    executor.execute_rebalance(decisions, max_allocation=0.10, live_run=True)
    
    print("\n✅ Cycle Complete. Sleeping...")

if __name__ == "__main__":
    run_moksha_bot()