import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import ta
import csv
import requests
from bs4 import BeautifulSoup
import pytz
import os
from dotenv import load_dotenv
from utils import api, fetch_latest_price, wait_for_order_fill, log_transaction, is_trading_hours, fetch_data, supertrend, moving_average, add_indicators, train_ml_model, get_time_until_ny_930
from p_m import start_monitoring, ensure_monitoring_thread
import traceback
import json
from newsapi import NewsApiClient
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet=True)

load_dotenv()

newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
print(f"Account status: {api.get_account().status}")

# Track settled and unsettled cash
settled_cash = 100000  # Initialize with your starting settled cash amount
unsettled_cash = 0
peak_cash = settled_cash
day_trade_count = 0
day_trade_dates = []

# Cache directory for storing news sentiment
NEWS_CACHE_DIR = "news_cache"
LAST_CLEANUP_FILE = "last_cleanup.txt"
if not os.path.exists(NEWS_CACHE_DIR):
    os.makedirs(NEWS_CACHE_DIR)

# Cleanup cache if it hasn't been cleaned in the last month
def cleanup_news_cache():
    if not os.path.exists(LAST_CLEANUP_FILE):
        with open(LAST_CLEANUP_FILE, "w") as f:
            f.write(datetime.now().isoformat())
        return
    
    with open(LAST_CLEANUP_FILE, "r") as f:
        last_cleanup_date = datetime.fromisoformat(f.read().strip())
    
    if (datetime.now() - last_cleanup_date).days >= 30:
        for file in os.listdir(NEWS_CACHE_DIR):
            os.remove(os.path.join(NEWS_CACHE_DIR, file))
        with open(LAST_CLEANUP_FILE, "w") as f:
            f.write(datetime.now().isoformat())
        print("News cache cleaned up.")

# Get news sentiment with caching
def get_news_sentiment(symbol, days_back=3, cache_duration=timedelta(days=1)):
    company_name = symbol  # For simplicity, we're using the symbol as the company name
    cache_file = os.path.join(NEWS_CACHE_DIR, f"{symbol}_sentiment.json")
    
    # Check if cache exists and is still valid
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
            cache_timestamp = datetime.fromisoformat(cache_data["timestamp"])
            if datetime.now() - cache_timestamp < cache_duration:
                print(f"Using cached sentiment for {symbol}")
                return cache_data["sentiment"]

    # Fetch new sentiment data if cache is expired or does not exist
    from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')

    try:
        news = newsapi.get_everything(
            q=company_name,
            from_param=from_date,
            to=to_date,
            language='en',
            sort_by='relevancy',
            page=1
        )

        sia = SentimentIntensityAnalyzer()
        sentiments = []

        for article in news['articles']:
            text = article['title'] + ' ' + (article['description'] or '')
            sentiment = sia.polarity_scores(text)['compound']
            sentiments.append(sentiment)

        # Calculate average sentiment
        if sentiments:
            average_sentiment = sum(sentiments) / len(sentiments)
        else:
            print(f"No news found for {symbol}")
            average_sentiment = 0  # Neutral sentiment if no news is found

        # Cache the result
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "sentiment": average_sentiment
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        return average_sentiment

    except Exception as e:
        print(f"Error fetching news sentiment for {symbol}: {str(e)}")
        return 0  # Return neutral sentiment in case of an error

# Fetch top gainers
def fetch_top_gainers(num_of_gainers):
    url = 'https://stockanalysis.com/markets/gainers/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    gainers = []
    for row in soup.select('table tr')[1:num_of_gainers]:
        cells = row.find_all('td')
        if len(cells) > 1:
            symbol = cells[1].text.strip()
            gainers.append(symbol)
    return gainers

# Calculate positions based on available buying power and data
def calculate_positions(buying_power, data):
    positions = {}
    total_score = sum(df.iloc[-1]['combined_score'] for df in data.values() if df is not None and not df.empty)
    for symbol, df in data.items():
        if df is not None and not df.empty:
            last_row = df.iloc[-1]
            if last_row['in_uptrend'] or last_row['ml_prediction'] >= 0.6:
                score = last_row['combined_score']
                allocation = (score / total_score) * buying_power if total_score > 0 else 0
                max_position_size = min(buying_power * 0.2, 50000)  # Increased to 20% of buying power
                positions[symbol] = max(min(allocation, max_position_size), 100)  # Minimum position size of $100
    return positions

# Check if we can perform a day trade
def can_day_trade(date):
    global day_trade_count, day_trade_dates
    # Remove dates older than 5 business days
    day_trade_dates = [d for d in day_trade_dates if (date - d).days <= 5]
    day_trade_count = len(day_trade_dates)
    return day_trade_count < 3

# Update day trade count
def update_day_trade_count(date):
    global day_trade_dates
    day_trade_dates.append(date)

# Execute buy orders
def execute_buy_orders(api, positions, buying_power):
    global settled_cash, unsettled_cash
    for symbol, amount in positions.items():
        try:
            asset = api.get_asset(symbol)
            if not asset.tradable:
                print(f"Asset {symbol} is not tradable.")
                continue

            current_price = fetch_latest_price(api, symbol)  # Corrected call to fetch_latest_price
            
            # Calculate quantity based on available buying power and position size
            max_quantity = min(amount, buying_power) / current_price
            
            # Scale in approach: Start with 50% of intended position size
            initial_quantity = max_quantity * 0.5
            if not asset.fractionable:
                initial_quantity = int(initial_quantity)

            if initial_quantity > 0:
                order = api.submit_order(
                    symbol=symbol,
                    qty=initial_quantity,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                
                print(f"Buy {initial_quantity} shares of {symbol} at market price ${current_price:.2f}.")
                print(f"Money used: ${current_price * initial_quantity:.2f}")

                filled_order = wait_for_order_fill(api, order.id)
                if filled_order:
                    fill_price = float(filled_order.filled_avg_price)
                    log_transaction('buy', symbol, initial_quantity, fill_price, datetime.now())
                    unsettled_cash += fill_price * initial_quantity
                    settled_cash -= fill_price * initial_quantity
                else:
                    print(f"Order for {symbol} was not filled within the timeout period.")

        except Exception as e:
            print(f"Error executing buy order for {symbol}: {str(e)}")


# Execute sell orders
def execute_sell_orders(api, data):
    global settled_cash, unsettled_cash
    positions = api.list_positions()
    for position in positions:
        symbol = position.symbol
        quantity = float(position.qty)
        position_details = api.get_position(symbol)
        entry_date = datetime.now(pytz.UTC)
        days_held = (datetime.now(pytz.UTC) - entry_date).days
        df = data.get(symbol)
        if df is None or df.empty:
            continue
        
        last_row = df.iloc[-1]
        current_price = fetch_latest_price(api, symbol)
        average_entry = float(position.avg_entry_price)
        unrealized_plpc = (current_price - average_entry) / average_entry

        # Implement trailing stop loss
        trailing_stop_loss = max(average_entry * 0.90, current_price * 0.95)

        # Determine whether to sell based on multiple factors
        should_sell = (
            not last_row['in_uptrend'] or 
            last_row['ml_prediction'] == 0 or 
            current_price <= trailing_stop_loss or
            unrealized_plpc >= 0.20
        )

        if should_sell:
            try:
                asset = api.get_asset(symbol)
                if not asset.tradable:
                    print(f"Asset {symbol} is not tradable.")
                    continue

                if not asset.fractionable:
                    quantity = int(quantity)
                
                if quantity > 0:
                    sell_quantity = min(quantity, quantity * 0.5)
                    limit_price = round(current_price * 0.99, 2)

                    order = api.submit_order(
                        symbol=symbol,
                        qty=sell_quantity,
                        side='sell',
                        type='limit',
                        time_in_force='day',
                        limit_price=limit_price
                    )
                    print(f"Placed limit order to sell {sell_quantity} shares of {symbol} at ${limit_price:.2f}")
                    print(f"Reason: {'Trend reversal' if not last_row['in_uptrend'] else 'ML prediction' if last_row['ml_prediction'] == 0 else 'Stop loss hit' if current_price <= trailing_stop_loss else 'Time-based exit' if days_held >= 3 else 'Take profit hit'}")

                    filled_order = wait_for_order_fill(api, order.id)
                    if filled_order:
                        fill_price = float(filled_order.filled_avg_price)
                        log_transaction('sell', symbol, sell_quantity, fill_price, datetime.now())
                        unsettled_cash += fill_price * sell_quantity
                        print(f"Sold at ${fill_price:.2f}. P/L: {(fill_price - average_entry) * sell_quantity:.2f}")
                    else:
                        print(f"Sell order for {symbol} was not filled within the timeout period.")

            except Exception as e:
                print(f"Error executing sell order for {symbol}: {str(e)}")



def calculate_combined_score(df):
    last_row = df.iloc[-1]
    technical_score = (last_row['in_uptrend'] * 0.3 + 
                       (last_row['Close'] > last_row['ma']) * 0.2 + 
                       (last_row['rsi'] > 50) * 0.1 + 
                       (last_row['macd'] > 0) * 0.1)
    ml_score = last_row['ml_prediction'] * 0.2
    sentiment_score = (last_row['sentiment'] + 1) / 2 * 0.1  # Normalize sentiment to [0, 1]
    return technical_score + ml_score + sentiment_score

def run_strategy():
    global day_trade_count, peak_cash
    today = datetime.now().date()
    
    top_gainers = fetch_top_gainers(5)
    symbols = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'NVDA'] + top_gainers
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=4)).strftime('%Y-%m-%d')
    
    data = {}
    for symbol in symbols:
        df = fetch_data(symbol, start_date, end_date)
        if df is not None:
            df = supertrend(df, 10, 3.0)
            df = moving_average(df, 20, 'sma')
            df = add_indicators(df)
            model = train_ml_model(df)
            df['ml_prediction'] = model.predict_proba(df[['ma', 'rsi', 'macd', 'bollinger_hband', 'bollinger_lband', 'atr']])[:, 1]
            df['sentiment'] = get_news_sentiment(symbol)
            df['combined_score'] = calculate_combined_score(df)
            data[symbol] = df
    
    account = api.get_account()
    buying_power = float(account.buying_power)
    
    positions = calculate_positions(buying_power, data)
    execute_buy_orders(api, positions, buying_power)
    execute_sell_orders(api, data)
    
    if not can_day_trade(today):
        print("Day trading limit reached. Skipping day trades for today.")
    
    total_cash = settled_cash + unsettled_cash
    peak_cash = max(peak_cash, total_cash)
    if total_cash < peak_cash * 0.9:
        print("Portfolio value dropped more than 10% from peak. Initiating panic sale.")
        initiate_panic_sale(data)
    
    print(f"Completed strategy run at {datetime.now()}.")

def initiate_panic_sale(data):
    positions = api.list_positions()
    for position in positions:
        symbol = position.symbol
        quantity = float(position.qty)
        current_price = fetch_latest_price(symbol)
        if quantity > 0:
            try:
                order = api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                print(f"Placed market order to sell all shares of {symbol} at ${current_price:.2f}")
                filled_order = wait_for_order_fill(api, order.id)
                if filled_order:
                    fill_price = float(filled_order.filled_avg_price)
                    log_transaction('sell', symbol, quantity, fill_price, datetime.now())
                    print(f"Sold at ${fill_price:.2f}.")
                else:
                    print(f"Sell order for {symbol} was not filled within the timeout period.")
            except Exception as e:
                print(f"Error executing sell order for {symbol}: {str(e)}")

def normalize_score(outcome, min_outcome, max_outcome):
    range_outcome = max_outcome - min_outcome
    normalized = (outcome - min_outcome) / range_outcome
    score = normalized * 0.5 + 0.5
    return score

# Analyze transactions
def analyze_transactions(log_file='transactions_log.csv'):
    transactions = pd.read_csv(log_file, names=['timestamp', 'type', 'symbol', 'quantity', 'price', 'outcome'])
    
    buy_transactions = transactions[transactions['type'] == 'buy']
    sell_transactions = transactions[transactions['type'] == 'sell']
    
    total_buys = len(buy_transactions)
    total_sells = len(sell_transactions)
    avg_profit_loss = sell_transactions['outcome'].mean()
    
    min_outcome = transactions['outcome'].min()
    max_outcome = transactions['outcome'].max()
    
    transactions['score'] = transactions['outcome'].apply(normalize_score, args=(min_outcome, max_outcome))
    total_score = transactions['score'].sum()
    
    print(f"Total Buy Transactions: {total_buys}")
    print(f"Total Sell Transactions: {total_sells}")
    print(f"Average Profit/Loss per Trade: {avg_profit_loss:.2%}")
    print(f"Total Score of Transactions: {total_score:.2f}")

# Settle trades (simulate T+2 settlement)
def settle_trades():
    global settled_cash, unsettled_cash
    today = datetime.now().date()
    settlement_date = today - timedelta(days=2)
    transactions = pd.read_csv('transactions_log.csv', names=['timestamp', 'type', 'symbol', 'quantity', 'price'])
    for index, row in transactions.iterrows():
        transaction_date = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S').date()
        if transaction_date == settlement_date:
            if row['type'] == 'buy':
                unsettled_cash -= float(row['quantity']) * float(row['price'])
                settled_cash += float(row['quantity']) * float(row['price'])
            elif row['type'] == 'sell':
                unsettled_cash -= float(row['quantity']) * float(row['price'])
                settled_cash += float(row['quantity']) * float(row['price'])



monitoring_thread = start_monitoring(api)

# In your main loop
while True:
    try:
        monitoring_thread = ensure_monitoring_thread(monitoring_thread, api)
        cleanup_news_cache()  # Clean up the news cache if needed
        if is_trading_hours(api):
            settle_trades()
            run_strategy()
        else:
            time_until_target = get_time_until_ny_930()
            hours, remainder = divmod(time_until_target.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"Outside trading hours opens in: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

            # Sleep until the target time
            time.sleep(time_until_target.total_seconds())
   
        analyze_transactions()
    except Exception as e:
        print(f"An error occurred in the main loop: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        traceback.print_exc()
    time.sleep(300)
