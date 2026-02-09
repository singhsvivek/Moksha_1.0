import sys
import os
from Moksha_1.core.sentiment import SentimentAgent

# Force unbuffered output for Docker logs
sys.stdout.reconfigure(line_buffering=True)

def test_ears():
    print("👂 Testing Sentiment Agent (The Ears)...")
    
    try:
        agent = SentimentAgent()
        symbol = "AAPL"
        
        print(f"   📰 Fetching news for {symbol}...")
        score = agent.get_sentiment_score(symbol)
        
        print(f"   ✅ Score Received: {score}")
        
        if score == 0.0:
            print("   ⚠️ Score is 0.0 (Neutral). This might mean no news found or API limitation.")
        elif score > 0:
            print("   🙂 Sentiment is POSITIVE.")
        else:
            print("   ☹️ Sentiment is NEGATIVE.")
            
        print("🎉 Sentiment Module is OPERATIONAL.")
        
    except Exception as e:
        print(f"❌ Sentiment Error: {e}")
        print("💡 Tip: Did you install 'textblob' and download corpora?")
        print("   Run: pip install textblob && python -m textblob.download_corpora")

if __name__ == "__main__":
    test_ears()