from abc import ABC, abstractmethod
import pandas as pd
from newsapi import NewsApiClient
from textblob import TextBlob
import numpy as np

class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, data):
        pass


class MeanReversionStrategy(Strategy):
    def __init__(self, window=20, std_dev=2):
        self.window = window
        self.std_dev = std_dev

    def generate_signal(self, data):
        data['SMA'] = data['Close'].rolling(window=self.window).mean()
        data['Upper'] = data['SMA'] + (data['Close'].rolling(window=self.window).std() * self.std_dev)
        data['Lower'] = data['SMA'] - (data['Close'].rolling(window=self.window).std() * self.std_dev)
        
        data['Signal'] = 0
        data.loc[data['Close'] < data['Lower'], 'Signal'] = 1  # Buy signal
        data.loc[data['Close'] > data['Upper'], 'Signal'] = -1  # Sell signal
        
        return data['Signal'].iloc[-1]
    
class MomentumStrategy(Strategy):
    def __init__(self, short_window=12, long_window=26):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signal(self, data):
        data['Short_MA'] = data['Close'].rolling(window=self.short_window).mean()
        data['Long_MA'] = data['Close'].rolling(window=self.long_window).mean()
        
        data['Signal'] = 0
        data.loc[data['Short_MA'] > data['Long_MA'], 'Signal'] = 1  # Buy signal
        data.loc[data['Short_MA'] < data['Long_MA'], 'Signal'] = -1  # Sell signal
        
        return data['Signal'].iloc[-1]
    
class SentimentAnalysisStrategy(Strategy):
    def __init__(self, api_key):
        self.api_key = api_key
        self.newsapi = NewsApiClient(api_key=self.api_key)

    def get_news_sentiment(self, symbol):
        news = self.newsapi.get_everything(q=symbol, language='en', sort_by='publishedAt', page_size=10)
        sentiments = [TextBlob(article['title']).sentiment.polarity for article in news['articles']]
        return np.mean(sentiments)

    def generate_signal(self, data):
        symbol = data['Symbol'].iloc[0] if 'Symbol' in data.columns else data.name
        sentiment = self.get_news_sentiment(symbol)
        
        if sentiment > 0.2:
            return 1  # Buy signal
        elif sentiment < -0.2:
            return -1  # Sell signal
        else:
            return 0  # Hold