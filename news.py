from newsapi import NewsApiClient
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()
# Fetch and analyze news sentiment
def get_enhanced_news_sentiment(symbol, days_back=3):
    newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
    sia = SentimentIntensityAnalyzer()

    company_name = symbol
    from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')

    news = newsapi.get_everything(q=company_name, from_param=from_date, to=to_date, language='en', sort_by='relevancy', page=1)
    sentiments = []

    for article in news['articles']:
        text = article['title'] + ' ' + (article['description'] or '')
        sentiment = sia.polarity_scores(text)['compound']
        sentiments.append(sentiment)

    if sentiments:
        return sum(sentiments) / len(sentiments)
    else:
        print(f"No news found for {symbol}")
        return 0

# Example usage
symbol = 'GOOG'
sentiment_score = get_enhanced_news_sentiment(symbol)
print(f"Sentiment score for {symbol}: {sentiment_score}")
