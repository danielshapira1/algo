import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from trader_logic.trader_utils import get_project_root
import os
import logging

logger = logging.getLogger(__name__)

def get_stock_data(symbol, min_price, max_price):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        if 'currentPrice' in info:
            current_price = info['currentPrice']
        elif 'regularMarketPrice' in info:
            current_price = info['regularMarketPrice']
        else:
            hist = stock.history(period="1d")
            if hist.empty:
                return None
            current_price = hist['Close'].iloc[-1]
        
        if min_price <= current_price <= max_price:
            hist = stock.history(period="1y")
            if hist.empty:
                return None
            
            returns = hist['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (returns.mean() * 252) / volatility if volatility != 0 else 0
            
            hist['MA50'] = hist['Close'].rolling(window=50).mean()
            hist['MA200'] = hist['Close'].rolling(window=200).mean()
            hist['MACD'], hist['Signal_Line'] = calculate_macd(hist)
            
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            hist['RSI'] = 100 - (100 / (1 + rs))
            
            last_row = hist.iloc[-1]
            
            return {
                'Symbol': symbol,
                'Price': current_price,
                'Market Cap': info.get('marketCap', 0),
                'P/E Ratio': info.get('trailingPE', 0),
                'Dividend Yield': info.get('dividendYield', 0),
                'Volume': last_row['Volume'],
                'Volatility': volatility,
                'Sharpe Ratio': sharpe_ratio,
                'RSI': last_row['RSI'],
                'Above MA50': last_row['Close'] > last_row['MA50'],
                'Above MA200': last_row['Close'] > last_row['MA200'],
                'MACD': last_row['MACD'],
                'Signal_Line': last_row['Signal_Line'],
            }
    except Exception as e:
        logger.error(f"Error processing {symbol}: {str(e)}")
    return None

def evaluate_stock(stock_data):
    score = 0
    
    # Trend
    if stock_data['Above MA200']:
        score += 2
    elif stock_data['Above MA50']:
        score += 1
    
    # RSI
    if 40 < stock_data['RSI'] < 60:
        score += 1
    elif 50 < stock_data['RSI'] < 70:
        score += 2

    # MACD
    if stock_data['MACD'] > stock_data['Signal_Line']:
        score += 2
    elif stock_data['MACD'] > 0:
        score += 1
    
    # Volatility
    if 0.3 < stock_data['Volatility'] < 0.5:
        score += 2
    elif 0.5 <= stock_data['Volatility'] < 0.7:
        score += 3
    
    # Sharpe Ratio
    if stock_data['Sharpe Ratio'] > 1:
        score += 2
    elif stock_data['Sharpe Ratio'] > 0.5:
        score += 1
    
    # Volume
    if stock_data['Volume'] > 1000000:
        score += 1
    
    # P/E Ratio
    if 0 < stock_data['P/E Ratio'] < 25:
        score += 1
    
    # Dividend Yield
    if stock_data['Dividend Yield'] > 0.02:
        score += 1
    
    return score

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def get_best_stocks(min_price=20, max_price=50, top_n=20):
    try:
        tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        symbols = tickers['Symbol'].tolist()
    except Exception as e:
        logger.error(f"Error fetching S&P 500 companies: {e}")
        return pd.DataFrame()
    
    stocks_data = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(get_stock_data, symbol, min_price, max_price) for symbol in symbols]
        for future in tqdm(as_completed(futures), total=len(symbols), desc="Processing stocks"):
            result = future.result()
            if result:
                stocks_data.append(result)
                time.sleep(0.1)
    
    for stock in stocks_data:
        stock['Score'] = evaluate_stock(stock)
    
    best_stocks = sorted(stocks_data, key=lambda x: x['Score'], reverse=True)[:top_n]
    
    return pd.DataFrame(best_stocks)

def save_stocks_to_csv(stocks, filename=None):
    if filename is None:
        filename = os.path.join(get_project_root(), 'data', 'csv', 'stocks_to_trade.csv')
    try:
        stocks.to_csv(filename, index=False)
        logger.info(f"Saved {len(stocks)} stocks to {filename}")
    except Exception as e:
        logger.error(f"Error saving stocks to CSV: {e}")

def get_stocks_to_trade():
    try:
        stocks = get_best_stocks()
        save_stocks_to_csv(stocks)
        return stocks
    except Exception as e:
        logger.error(f"Error in get_stocks_to_trade: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    stocks = get_stocks_to_trade()
    logger.info(f"Found {len(stocks)} stocks to trade")
    logger.info(stocks)