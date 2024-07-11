import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import ta
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import pytz
from dotenv import load_dotenv


load_dotenv()

# Alpaca API credentials
PAPER_API_KEY_ID = os.getenv('APCA_API_KEY_ID_PAPER')
PAPER_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY_PAPER')
PAPER_BASE_URL = os.getenv('APCA_API_BASE_URL_PAPER')

# if KEY or ID not match delete venv and recreate it
print(f"PAPER_API_KEY_ID: {PAPER_API_KEY_ID}")
print(f"PAPER_SECRET_KEY: {PAPER_SECRET_KEY}")
print(f"PAPER_BASE_URL: {PAPER_BASE_URL}")


api = tradeapi.REST(PAPER_API_KEY_ID, PAPER_SECRET_KEY, PAPER_BASE_URL, api_version='v2')

def fetch_latest_price(api, symbol):
    try:
        quote = api.get_latest_trade(symbol)
        return quote.price
    except Exception as e:
        print(f"Error fetching latest price for {symbol}: {str(e)}")
        return None


def wait_for_order_fill(api, order_id, timeout=60):
    import time
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            order = api.get_order(order_id)
            if order.status == 'filled':
                return order
            elif order.status in ['canceled', 'expired', 'rejected']:
                print(f"Order {order_id} {order.status}.")
                return None
            time.sleep(1)
        except Exception as e:
            print(f"Error checking order status: {str(e)}")
            return None
    print(f"Order {order_id} not filled within {timeout} seconds.")
    return None

def log_transaction(transaction_type, symbol, quantity, price, date):
    with open('transactions_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([date.strftime('%Y-%m-%d %H:%M:%S'), transaction_type, symbol, quantity, price])

def is_trading_hours(api):
    clock = api.get_clock()
    return clock.is_open

# Fetch historical data
def fetch_data(symbol, start, end, interval='1m'):
    try:
        df = yf.download(symbol, start=start, end=end, interval=interval)
        if df is None or df.empty:
            print(f"No data returned by yfinance for {symbol}.")
            return None
        print(f"Fetched {len(df)} rows for symbol {symbol}.")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    
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

def add_indicators(df):
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    df['macd'] = ta.trend.macd_diff(df['Close'])
    df['bollinger_hband'] = ta.volatility.bollinger_hband(df['Close'])
    df['bollinger_lband'] = ta.volatility.bollinger_lband(df['Close'])
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    return df

def train_ml_model(df):
    features = ['ma', 'rsi', 'macd', 'bollinger_hband', 'bollinger_lband', 'atr']
    X = df[features]
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

ny_tz = pytz.timezone('America/New_York')
jerusalem_tz = pytz.timezone('Asia/Jerusalem')

def get_time_until_ny_930():
    # Get current time in Jerusalem
    now_jerusalem = datetime.now(jerusalem_tz)

    # Get current time in New York
    now_ny = datetime.now(ny_tz)

    # Define the next 9:30 AM in New York
    ny_target_time = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)

    # If the target time has already passed today, set it to the next day
    if now_ny >= ny_target_time:
        ny_target_time += timedelta(days=1)

    # Convert the target time to Jerusalem time
    jerusalem_target_time = ny_target_time.astimezone(jerusalem_tz)

    # Calculate the time difference
    time_until_target = jerusalem_target_time - now_jerusalem

    return time_until_target