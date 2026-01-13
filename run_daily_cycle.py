# run_daily_cycle.py
from Moksha_1.core.decision_engine import DecisionEngine

def run_simulation():
    # 1. Initialize The Council
    engine = DecisionEngine()
    
    # 2. Define Universe
    universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY"]
    
    # 3. Run Analysis
    decisions = engine.analyze_market(symbols=universe)
    
    if decisions.empty:
        print("❌ No decisions generated.")
        return

    # 4. Print "The Morning Briefing"
    print("\n============== 🏛️ MOKSHA COUNCIL BRIEFING ==============")
    print(decisions.sort_values('final_signal', ascending=False).to_string(index=False))
    print("========================================================")
    
    # Simple Interpretation
    top_pick = decisions.loc[decisions['final_signal'].idxmax()]
    print(f"\n🚀 TOP PICK: {top_pick['symbol']} (Signal: {top_pick['final_signal']:.4f})")
    print(f"   Reason: Predicted {top_pick['predicted_return']*100:.2f}% return in {top_pick['regime_label']} regime.")

if __name__ == "__main__":
    run_simulation()