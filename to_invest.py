import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class StockInvestor:
    def __init__(self):
        self.url = 'https://www.investopedia.com/stock-market-news/'
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.clf = MultinomialNB()

    def get_news(self):
        response = requests.get(self.url)
        soup = BeautifulSoup(response.text, 'html.parser')

        articles = []
        for article in soup.find_all('article'):
            title = article.find('h2').text
            text = article.find('div', {'class': 'article-body'}).text
            articles.append({'title': title, 'text': text})

        return articles

    def process_articles(self, articles):
        X = self.vectorizer.fit_transform([article['text'] for article in articles])
        y = [1] * len(articles)  # Assuming all articles are positive (good stocks)

        self.clf.fit(X, y)

    def get_top_performing_stocks():
        # Define the time range for which we want to retrieve data
        end = datetime.today()
        start = end - timedelta(days=30)

        # Retrieve stock prices for the past 30 days
        ticker_data = []
        stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']  # Example list of top-performing stocks
        for stock in stocks:
            ticker = yf.Ticker(stock)
            history = ticker.history(start=start, end=end)
            ticker_data.append({
            'stock': stock,
            'return': (history['Close'].iloc[-1] / history['Close'].iloc[0]) - 1
        })

    # Sort the stocks by their return
        ticker_data.sort(key=lambda x: x['return'], reverse=True)

    # Return the top-performing stocks
        return [stock['stock'] for stock in ticker_data[:5]]

    

    def predict(self):
        vectorized_articles = self.vectorizer.transform(['This is a test'])
        return self.clf.predict(vectorized_articles)

def main():
    investor = StockInvestor()
    articles = investor.get_news()
    investor.process_articles(articles)
    prediction = investor.predict()

    if prediction[0] == 1:
        print('The best stocks to invest in for a short period of time are:')
        # Get the top performing stocks from a reliable source (e.g., Yahoo Finance, Investopedia)
        top_stocks = investor.get_top_performing_stocks()
        print(top_stocks)
    else:
        print('No stocks to invest in at this time.')

if __name__ == '__main__':
    main()