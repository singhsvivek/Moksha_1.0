from textblob import TextBlob
from datetime import datetime, timedelta
from alpaca.data.requests import NewsRequest
from alpaca.data.historical import NewsClient
from Moksha_1.config import settings
from Moksha_1.utils.logger import logger

class SentimentAgent:
    def __init__(self):
        self.news_client = NewsClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)

    def get_sentiment_score(self, symbol: str) -> float:
        """
        Fetches last 24h news and returns average polarity.
        Range: -1.0 (Very Negative) to 1.0 (Very Positive).
        Returns 0.0 if no news found.
        """
        try:
            # 1. Fetch News (Last 24 Hours)
            request_params = NewsRequest(
                symbols=symbol,
                start=datetime.now() - timedelta(hours=24),
                limit=5
            )
            news_list = self.news_client.get_news(request_params).news
            
            if not news_list:
                return 0.0

            # 2. Analyze Headlines
            total_score = 0
            count = 0
            
            for article in news_list:
                # Combine headline and summary for context
                text = f"{article.headline} {article.summary}"
                blob = TextBlob(text)
                score = blob.sentiment.polarity
                total_score += score
                count += 1
                
            if count == 0: return 0.0
            
            avg_score = total_score / count
            
            # 3. Log Significant Sentiment
            if abs(avg_score) > 0.2:
                logger.info(f"📰 {symbol} News Sentiment: {avg_score:.2f} ({count} articles)")
                
            return avg_score

        except Exception as e:
            # If Sentiment fails, don't crash the bot. Assume Neutral.
            # logger.warning(f"Sentiment check failed for {symbol}: {e}")
            return 0.0