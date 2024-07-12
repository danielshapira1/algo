import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
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

api = tradeapi.REST(PAPER_API_KEY_ID, PAPER_SECRET_KEY, PAPER_BASE_URL, api_version='v2')

def fetch_latest_price(api, symbol):
    try:
        quote = api.get_latest_trade(symbol)
        return quote.price
    except Exception as e:
        print(f"Error fetching latest price for {symbol}: {str(e)}")
        return None

def wait_for_order_fill(api, order_id, timeout=60):
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

def fetch_data(symbol, start, end, interval='1m'):
    try:
        df = yf.download(symbol, start=start, end=end, interval=interval)
        if df is None or df.empty:
            print(f"No data returned by yfinance for {symbol}.")
            return None
        print(f"Fetched {len(df)} rows for symbol {symbol}.")
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def supertrend(df, atr_period, multiplier):
    if 'High' not in df.columns or 'Low' not in df.columns or 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns.")

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

def moving_average(df, length, ma_type='sma'):
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
    try:
        df = supertrend(df, 10, 3.0)
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        df['macd'] = ta.trend.macd_diff(df['Close'])
        df['bollinger_hband'] = ta.volatility.bollinger_hband(df['Close'])
        df['bollinger_lband'] = ta.volatility.bollinger_lband(df['Close'])
        df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        df['roc'] = ta.momentum.roc(df['Close'], window=10)
        df['ppo'] = ta.momentum.ppo(df['Close'])
        df['cci'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
        df['dmi_plus'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'], window=14)
        df['dmi_minus'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'], window=14)
        df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        
        df = moving_average(df, 20, 'sma')
        
    except Exception as e:
        print(f"Error adding indicators: {str(e)}")

    return df

def normalize_indicator(series, min_val, max_val):
    return (series - min_val) / (max_val - min_val)

def calculate_momentum_score(df):
    roc_score = normalize_indicator(df['roc'], -10, 10)
    rsi_score = normalize_indicator(df['rsi'], 0, 100)
    return (roc_score + rsi_score) / 2

def calculate_mean_reversion_score(df):
    ppo_score = 1 - abs(normalize_indicator(df['ppo'], -1, 1))
    cci_score = 1 - abs(normalize_indicator(df['cci'], -100, 100))
    return (ppo_score + cci_score) / 2

def calculate_trend_score(df):
    supertrend_score = df['in_uptrend'].astype(int)
    adx_score = normalize_indicator(df['adx'], 0, 100)
    dmi_score = normalize_indicator(df['dmi_plus'] - df['dmi_minus'], -100, 100)
    return (supertrend_score + adx_score + dmi_score) / 3

def calculate_performance_metrics(returns):
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized Sharpe Ratio
    sortino_ratio = returns.mean() / returns[returns < 0].std() * np.sqrt(252)  # Annualized Sortino Ratio
    max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown
    }

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
    now_jerusalem = datetime.now(jerusalem_tz)
    now_ny = datetime.now(ny_tz)
    ny_target_time = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)

    if now_ny >= ny_target_time:
        ny_target_time += timedelta(days=1)

    jerusalem_target_time = ny_target_time.astimezone(jerusalem_tz)
    time_until_target = jerusalem_target_time - now_jerusalem

    return time_until_target

def process_symbol_data(symbol, start_date, end_date):
    df = fetch_data(symbol, start_date, end_date)
    if df is not None and not df.empty:
        df = add_indicators(df)
        if 'in_uptrend' not in df.columns:
            print(f"Warning: 'in_uptrend' not found in DataFrame for {symbol}")
        model = train_ml_model(df)
        try:
            features = ['ma', 'rsi', 'macd', 'bollinger_hband', 'bollinger_lband', 'atr']
            for feature in features:
                if feature not in df.columns:
                    print(f"Missing feature '{feature}' in DataFrame for {symbol}")
                    return None
            print(f"Features for {symbol} before prediction: {df[features].tail()}")
            df['ml_prediction'] = model.predict_proba(df[features])[:, 1]
            print(f"ML predictions for {symbol}: {df['ml_prediction'].tail()}")
        except Exception as e:
            print(f"Error predicting ML values for {symbol}: {str(e)}")
            return None
        return df
    return None
