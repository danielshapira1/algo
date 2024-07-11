import alpaca_trade_api as tradeapi
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

# Alpaca API credentials
API_KEY = 'PK4ADVFBU94YSR4DD2ME'
SECRET_KEY = '60rwNrizyPMY58EqfOSx6rgxluQ91iFVMeimzIt7'
BASE_URL = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# Supertrend Calculation
def supertrend(df, atr_period, multiplier):
    hl2 = (df['High'] + df['Low']) / 2
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=atr_period)
    df['upperband'] = hl2 + (multiplier * df['atr'])
    df['lowerband'] = hl2 - (multiplier * df['atr'])
    df['in_uptrend'] = True

    for current in range(1, len(df.index)):
        previous = current - 1

        if df['Close'].iloc[current] > df['upperband'].iloc[previous]:
            df.loc[df.index[current], 'in_uptrend'] = True
        elif df['Close'].iloc[current] < df['lowerband'].iloc[previous]:
            df.loc[df.index[current], 'in_uptrend'] = False
        else:
            df.loc[df.index[current], 'in_uptrend'] = df['in_uptrend'].iloc[previous]
            if df['in_uptrend'].iloc[current] and df['lowerband'].iloc[current] < df['lowerband'].iloc[previous]:
                df.loc[df.index[current], 'lowerband'] = df['lowerband'].iloc[previous]
            if not df['in_uptrend'].iloc[current] and df['upperband'].iloc[current] > df['upperband'].iloc[previous]:
                df.loc[df.index[current], 'upperband'] = df['upperband'].iloc[previous]
                
    return df

# Moving Average Calculation
def moving_average(df, length, ma_type):
    if ma_type == 'sma':
        df['ma'] = df['Close'].rolling(window=length).mean()
    elif ma_type == 'ema':
        df['ma'] = ta.trend.ema_indicator(df['Close'], window=length)
    elif ma_type == 'wma':
        df['ma'] = ta.trend.wma_indicator(df['Close'], window=length)
    elif ma_type == 'hullma':
        df['ma'] = ta.trend.wma_indicator(2 * ta.trend.wma_indicator(df['Close'], window=int(length/2)) - ta.trend.wma_indicator(df['Close'], window=length), window=int(np.sqrt(length)))
    elif ma_type == 'tema':
        ema1 = ta.trend.ema_indicator(df['Close'], window=length)
        ema2 = ta.trend.ema_indicator(ema1, window=length)
        ema3 = ta.trend.ema_indicator(ema2, window=length)
        df['ma'] = 3 * (ema1 - ema2) + ema3
    return df

# Fetch historical data
def fetch_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, interval='1m')
        if df is None or df.empty:
            print(f"No data returned by yfinance for {symbol}.")
            return None
        print(f"Fetched {len(df)} rows for symbol {symbol}.")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Log transactions
def log_transaction(transaction_type, symbol, quantity, price):
    with open('transactions_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), transaction_type, symbol, quantity, price])

# Fetch top gainers, num_of_gainers can be only up to 50
def fetch_top_gainers(num_of_gainers):
    url = 'https://stockanalysis.com/markets/gainers/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    gainers = []
    for row in soup.select('table tr')[1:num_of_gainers]:
        cells = row.find_all('td')
        if len(cells) > 1:
            symbol = cells[1].text.strip()
            print(symbol)
            gainers.append(symbol)
    return gainers

# Fetch latest stock price from Alpaca
def fetch_latest_price(symbol):
    quote = api.get_latest_trade(symbol)
    return quote.price

# Calculate positions based on available buying power and data
def calculate_positions(buying_power, data):
    positions = {}
    for symbol, df in data.items():
        if df is not None and not df.empty:
            last_row = df.iloc[-1]
            if last_row['in_uptrend']:
                positions[symbol] = buying_power * 0.9 / len(data)
    return positions

# Execute buy orders
def execute_buy_orders(api, positions, buying_power):
    for symbol, amount in positions.items():
        try:
            asset = api.get_asset(symbol)
            if not asset.tradable:
                print(f"Asset {symbol} is not tradable.")
                continue

            price = fetch_latest_price(symbol)
            if asset.fractionable:
                quantity = amount / price
            else:
                quantity = int(amount / price)
                # Ensure there is enough buying power for non-fractionable asset
                if price * quantity > buying_power:
                    print(f"Not enough buying power to buy {quantity} shares of non-fractionable asset {symbol}.")
                    continue

            # Ensure the order value is at least $1
            if price * quantity < 1:
                print(f"The order value for {symbol} is less than $1. Skipping the order.")
                continue

            if quantity > 0:
                order = api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                print(f"Buy {quantity} shares of {symbol} at market price ${price:.2f}.")
                print(f"Money used: ${amount:.2f}")

                while True:
                    order_status = api.get_order(order.id)
                    if order_status.status == 'filled':
                        break
                    time.sleep(1)

                log_transaction('buy', symbol, quantity, price)
                buying_power -= price * quantity  # Update buying power after the purchase
        except Exception as e:
            print(f"Error executing trade for {symbol}: {str(e)}")

# Execute sell orders
def execute_sell_orders(api, data):
    positions = api.list_positions()
    for position in positions:
        symbol = position.symbol
        quantity = float(position.qty)  # Ensure quantity is a float for compatibility
        df = data.get(symbol)
        if df is None or df.empty:
            continue
        last_row = df.iloc[-1]
        if last_row['trend'] == -1:
            try:
                asset = api.get_asset(symbol)
                if not asset.tradable:
                    print(f"Asset {symbol} is not tradable.")
                    continue

                if not asset.fractionable:
                    quantity = int(quantity)  # Convert to integer for non-fractionable assets
                
                if quantity > 0:
                    order = api.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    print(f"Sell {quantity} shares of {symbol} at market price.")

                    while True:
                        order_status = api.get_order(order.id)
                        if order_status.status == 'filled':
                            break
                        time.sleep(1)

                    price = fetch_latest_price(symbol)
                    log_transaction('sell', symbol, quantity, price)
            except Exception as e:
                print(f"Error executing sell order for {symbol}: {str(e)}")

# Define the main function to run the strategy
def run_strategy():
    top_gainers = fetch_top_gainers(100)
    symbols = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'NVDA'] + top_gainers
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    
    data = {}
    for symbol in symbols:
        df = fetch_data(symbol, start_date, end_date)
        if df is not None:
            df = supertrend(df, 10, 3.0)
            df = moving_average(df, 20, 'sma')
            df['trend'] = np.where((df['Close'] < df['lowerband']) & (df['Close'].shift(1) >= df['lowerband']), -1, 
                                   np.where((df['Close'] > df['upperband']) & (df['Close'].shift(1) <= df['upperband']), 1, 0))
            data[symbol] = df
    
    account = api.get_account()
    buying_power = float(account.buying_power)
    
    positions = calculate_positions(buying_power, data)
    execute_buy_orders(api, positions, buying_power)
    execute_sell_orders(api, data)  # Execute sell orders after buy orders
    
    print(f"Completed strategy run at {datetime.now()}.")

# Analyze transactions
def normalize_score(outcome, min_outcome, max_outcome):
    # Normalize outcome to range [0, 1] where 0.5 is no profit/loss
    range_outcome = max_outcome - min_outcome
    normalized = (outcome - min_outcome) / range_outcome
    score = normalized * 0.5 + 0.5  # Adjust to make 0.5 the neutral point
    return score

def analyze_transactions(log_file='transactions_log.csv'):
    # Read the transaction log file
    transactions = pd.read_csv(log_file, names=['timestamp', 'type', 'symbol', 'quantity', 'price', 'outcome'])
    
    # Filter buy and sell transactions
    buy_transactions = transactions[transactions['type'] == 'buy']
    sell_transactions = transactions[transactions['type'] == 'sell']
    
    # Calculate statistics
    total_buys = len(buy_transactions)
    total_sells = len(sell_transactions)
    avg_profit_loss = sell_transactions['outcome'].mean()
    
    # Determine min and max outcomes for normalization
    min_outcome = transactions['outcome'].min()
    max_outcome = transactions['outcome'].max()
    
    # Calculate score for each transaction and total score
    transactions['score'] = transactions['outcome'].apply(normalize_score, args=(min_outcome, max_outcome))
    total_score = transactions['score'].sum()
    
    # Display results
    print(f"Total Buy Transactions: {total_buys}")
    print(f"Total Sell Transactions: {total_sells}")
    print(f"Average Profit/Loss per Trade: {avg_profit_loss:.2%}")
    print(f"Total Score of Transactions: {total_score:.2f}")

# Check if current time is within trading hours
def is_trading_hours():
    eastern = pytz.timezone('US/Eastern')
    current_time = datetime.now(eastern)
    market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= current_time <= market_close

# Run the strategy in a loop every minute during trading hours
while True:
    if is_trading_hours():
        run_strategy()
    else:
        print("Outside trading hours. Waiting for the market to open.")
    time.sleep(60)  # Wait for 60 seconds before checking again
    analyze_transactions()  # Analyze transactions after each run or at desired intervals