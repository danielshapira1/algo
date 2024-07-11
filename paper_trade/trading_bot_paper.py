import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
from datetime import datetime, timedelta
import csv
import requests
from bs4 import BeautifulSoup

# Alpaca API credentials
API_KEY = 'PKQZBT1CMG0S1IYQOV0W'
SECRET_KEY = 'q3hv2ir8uDttW2flo42gyGFAyeFeUwEQi7sQpj0l'
BASE_URL = 'https://paper-api.alpaca.markets'  # Use the paper trading endpoint

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# Define stop-loss and take-profit thresholds
STOP_LOSS_PERCENT = 0.05  # 5% loss
TAKE_PROFIT_PERCENT = 0.10  # 10% gain

def fetch_top_gainers():
    url = 'https://stockanalysis.com/markets/gainers/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    gainers = []
    for row in soup.select('table tr')[1:6]:
        cells = row.find_all('td')
        if len(cells) > 1:
            symbol = cells[1].text.strip()
            print(symbol)
            gainers.append(symbol)
    return gainers

# Fetch historical data using yfinance
def fetch_historical_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    data.reset_index(inplace=True)
    data.columns = ['time', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    data = data[['time', 'open', 'high', 'low', 'close', 'volume']]
    return data

# Function to fetch recent data
def fetch_recent_data():
    top_gainers = fetch_top_gainers()
    symbols = ['AAPL', 'GOOG', 'MSFT', 'AMZN'] + top_gainers
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=180)).strftime('%Y-%m-%d')  # 6 months of data
    data = {symbol: fetch_historical_data(symbol, start_date, end_date) for symbol in symbols}
    return data

# Function to log transactions
def log_transaction(transaction_type, symbol, quantity, price, outcome=None):
    with open('transactions_log_paper.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), transaction_type, symbol, quantity, price, outcome])

# Function to analyze transaction logs
def normalize_score(outcome, min_outcome, max_outcome):
    # Normalize outcome to range [0, 1] where 0.5 is no profit/loss
    range_outcome = max_outcome - min_outcome
    normalized = (outcome - min_outcome) / range_outcome
    score = normalized * 0.5 + 0.5  # Adjust to make 0.5 the neutral point
    return score

def analyze_transactions(log_file='transactions_log_paper.csv'):
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

# Candlestick pattern functions
def is_hammer(df):
    return (df['low'] < df['open']) & (df['close'] > df['open'])

def is_engulfing(df):
    return (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))

def is_shooting_star(df):
    return (df['high'] > df['open']) & (df['close'] < df['open'])

def is_doji(df):
    return (abs(df['open'] - df['close']) < 0.1 * (df['high'] - df['low']))

# Create features including SMA20 and candlestick patterns
def create_features(df):
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(window=10).std()
    df['momentum'] = df['close'].pct_change(periods=10)
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['hammer'] = is_hammer(df)
    df['engulfing'] = is_engulfing(df)
    df['shooting_star'] = is_shooting_star(df)
    df['doji'] = is_doji(df)
    df = df.dropna()
    return df

# Prepare data for training
def prepare_training_data(data):
    X = []
    y = []
    for df in data.values():
        X.append(df[['return', 'volatility', 'momentum', 'SMA20', 'hammer', 'engulfing', 'shooting_star', 'doji']])
        y.append((df['return'].shift(-1) > 0).astype(int))
    X = pd.concat(X)
    y = pd.concat(y).dropna()
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Grid Search for Hyperparameter Tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best parameters found: ", grid_search.best_params_)
    return best_model

# Function to fetch the latest stock price from Alpaca
def fetch_latest_price(symbol):
    quote = api.get_latest_trade(symbol)
    return quote.price

# Function to calculate positions based on model predictions and current data
def calculate_positions(buying_power, model, data):
    positions = {}
    for symbol, df in data.items():
        if df is not None and not df.empty:
            last_day_features = df[['return', 'volatility', 'momentum', 'SMA20', 'hammer', 'engulfing', 'shooting_star', 'doji']].tail(1)
            print(f"Features for {symbol}: {last_day_features}")  # Debugging
            prediction = model.predict(last_day_features)[0]
            print(f"Prediction for {symbol}: {prediction}")  # Debugging
            if prediction == 1:
                positions[symbol] = buying_power * 0.9 / len(data)
    return positions

# Function to execute buy orders
def execute_buy_orders(api, positions):
    for symbol, amount in positions.items():
        try:
            price = fetch_latest_price(symbol)
            quantity = amount / price
            if quantity > 0:
                order = api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                print(f"Buy {quantity:.4f} shares of {symbol} at market price ${price:.2f}.")
                print(f"Money used: ${amount:.2f}")

                while True:
                    order_status = api.get_order(order.id)
                    if order_status.status == 'filled':
                        break
                    time.sleep(1)

                # Log the transaction
                log_transaction('buy', symbol, quantity, price)

                account = api.get_account()
                balance = float(account.cash)
                buying_power = float(account.buying_power)

                current_time = datetime.now().strftime('%m/%d/%H/%M/%S')
                print(f"Remaining balance: ${balance:.2f} as of {current_time}")
                print(f"Remaining buying power: ${buying_power:.2f} as of {current_time}")
            else:
                print(f"Not enough balance to buy {symbol}.")
        except Exception as e:
            print(f"Error executing trade for {symbol}: {str(e)}")

# Function to execute sell orders
def execute_sell_orders(api, holdings, model, data):
    for symbol, details in holdings.items():
        try:
            price = fetch_latest_price(symbol)
            quantity = details['quantity']
            cost_basis = details['cost_basis']
            current_value = quantity * price
            profit_loss = (current_value - cost_basis) / cost_basis

            df = data[symbol]
            last_day_features = df[['return', 'volatility', 'momentum', 'SMA20', 'hammer', 'engulfing', 'shooting_star', 'doji']].tail(1)
            sell_prediction = model.predict(last_day_features)[0]

            if profit_loss <= -STOP_LOSS_PERCENT or profit_loss >= TAKE_PROFIT_PERCENT or sell_prediction == 0:
                order = api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                print(f"Sell {quantity:.4f} shares of {symbol} at market price ${price:.2f}.")
                print(f"Profit/Loss: {profit_loss:.2%}")

                while True:
                    order_status = api.get_order(order.id)
                    if order_status.status == 'filled':
                        break
                    time.sleep(1)

                # Log the transaction
                log_transaction('sell', symbol, quantity, price, profit_loss)

                del holdings[symbol]

                account = api.get_account()
                balance = float(account.cash)
                buying_power = float(account.buying_power)

                current_time = datetime.now().strftime('%m/%d/%H/%M/%S')
                print(f"Remaining balance: ${balance:.2f} as of {current_time}")
                print(f"Remaining buying power: ${buying_power:.2f} as of {current_time}")
        except Exception as e:
            print(f"Error executing trade for {symbol}: {str(e)}")

# Main loop for paper trading
def main():
    data = fetch_recent_data()
    print("Fetched recent historical data.")
    for symbol, df in data.items():
        print(f"Data for {symbol}: {df.tail()}")

    data = {symbol: create_features(df) for symbol, df in data.items()}
    for symbol, df in data.items():
        print(f"Features for {symbol}: {df.tail()}")

    X_train, X_test, y_train, y_test = prepare_training_data(data)
    print("Prepared training data.")

    model = train_model(X_train, y_train)
    print("Trained the model.")

    predictions = model.predict(X_test)
    print(f"Model accuracy: {accuracy_score(y_test, predictions):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    holdings = {}

    while True:
        account = api.get_account()
        buying_power = float(account.buying_power)
        balance = float(account.cash)
        print(f"Current balance: ${balance:.2f}")
        print(f"Current buying power: ${buying_power:.2f}")

        real_time_data = fetch_recent_data()
        print("Fetched real-time data.")
        for symbol, df in real_time_data.items():
            print(f"Fetched real-time data for {symbol}")

        for symbol, df in real_time_data.items():
            if symbol in data:
                data[symbol] = pd.concat([data[symbol], df]).drop_duplicates().reset_index(drop=True)
            else:
                data[symbol] = df

        data = {symbol: create_features(df) for symbol, df in data.items()}
        positions = calculate_positions(buying_power, model, data)
        execute_buy_orders(api, positions)

        positions = api.list_positions()
        for position in positions:
            holdings[position.symbol] = {
                'quantity': float(position.qty),
                'cost_basis': float(position.cost_basis)
            }

        execute_sell_orders(api, holdings, model, data)
        
        # Analyze transactions after every iteration or at desired intervals
        analyze_transactions()

        time.sleep(10 * 60)

if __name__ == "__main__":
    main()