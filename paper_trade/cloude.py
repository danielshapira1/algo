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
from utils import api, fetch_latest_price, wait_for_order_fill, get_news_sentiment, log_transaction, is_trading_hours, fetch_data, supertrend, moving_average, add_indicators, train_ml_model, get_time_until_ny_930, calculate_momentum_score, calculate_mean_reversion_score, calculate_trend_score, calculate_performance_metrics, process_symbol_data, cleanup_news_cache
from p_m import start_monitoring, ensure_monitoring_thread
import traceback
import json
import nltk

nltk.download('vader_lexicon', quiet=True)

load_dotenv()


settled_cash = 100000  # Initialize with your starting settled cash amount
unsettled_cash = 0
peak_cash = settled_cash
day_trade_count = 0
day_trade_dates = []


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

def calculate_positions(buying_power, data):
    positions = {}
    total_score = sum(df.iloc[-1]['combined_score'] for df in data.values() if df is not None and not df.empty)
    for symbol, df in data.items():
        if df is not None and not df.empty:
            last_row = df.iloc[-1]
            if 'in_uptrend' not in last_row:
                print(f"'in_uptrend' not found for {symbol}")
                continue
            if last_row['in_uptrend'] or last_row['ml_prediction'] >= 0.6:
                score = last_row['combined_score']
                allocation = (score / total_score) * buying_power if total_score > 0 else 0
                max_position_size = min(buying_power * 0.2, 50000)
                positions[symbol] = max(min(allocation, max_position_size), 100)
    return positions

def can_day_trade(date):
    global day_trade_count, day_trade_dates
    day_trade_dates = [d for d in day_trade_dates if (date - d).days <= 5]
    day_trade_count = len(day_trade_dates)
    return day_trade_count < 3

def update_day_trade_count(date):
    global day_trade_dates
    day_trade_dates.append(date)

def execute_buy_orders(api, positions, buying_power):
    global settled_cash, unsettled_cash
    for symbol, amount in positions.items():
        try:
            asset = api.get_asset(symbol)
            if not asset.tradable:
                print(f"Asset {symbol} is not tradable.")
                continue

            current_price = fetch_latest_price(api, symbol)
            
            # Use process_symbol_data to get the processed DataFrame
            df = process_symbol_data(symbol, (datetime.now() - timedelta(days=6)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
            if df is None or df.empty:
                print(f"Invalid or incomplete data for {symbol}. Skipping buy order.")
                continue
            
            required_columns = ['in_uptrend', 'ma', 'rsi', 'macd', 'bollinger_hband', 'bollinger_lband', 'atr', 'ml_prediction']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Missing columns for {symbol}: {missing_columns}. Skipping buy order.")
                continue

            atr = df['atr'].iloc[-1]
            combined_score = calculate_combined_score(df.iloc[-1])
            risk_per_trade = buying_power * 0.01
            position_size = risk_per_trade / (atr * 2)
            position_size *= combined_score

            max_quantity = min(position_size, buying_power) / current_price
            
            initial_quantity = max_quantity * 0.5
            if not asset.fractionable:
                initial_quantity = int(initial_quantity)

            min_order_value = 1
            if initial_quantity * current_price < min_order_value:
                print(f"Skipping {symbol} as the order value (${initial_quantity * current_price:.2f}) is below ${min_order_value}.")
                continue

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
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {traceback.format_exc()}")

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

        trailing_stop_loss = max(average_entry * 0.90, current_price * 0.95)

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

def calculate_combined_score(row):
    technical_score = (row['in_uptrend'] * 0.3 + 
                       (row['Close'] > row['ma']) * 0.2 + 
                       (row['rsi'] > 50) * 0.1 + 
                       (row['macd'] > 0) * 0.1)
    ml_score = row['ml_prediction'] * 0.2
    sentiment_score = (row['sentiment'] + 1) / 2 * 0.1
    return technical_score + ml_score + sentiment_score

def run_baseline_strategy(data):
    for symbol, df in data.items():
        df['sma_short'] = df['Close'].rolling(window=10).mean()
        df['sma_long'] = df['Close'].rolling(window=50).mean()
        df['position'] = np.where(df['sma_short'] > df['sma_long'], 1, 0)
        df['returns'] = df['Close'].pct_change()
        df['strategy_returns'] = df['position'].shift(1) * df['returns']
    
    return data

def calculate_strategy_returns(data):
    all_returns = []
    for symbol, df in data.items():
        if 'strategy_returns' in df.columns:
            all_returns.append(df['strategy_returns'])
    if not all_returns:
        print("No strategy returns available. Skipping performance calculation.")
        return pd.Series()
    return pd.concat(all_returns, axis=1).mean(axis=1)

def run_strategy():
    global day_trade_count, peak_cash
    today = datetime.now().date()
    
    top_gainers = fetch_top_gainers(5)
    symbols = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'NVDA'] + top_gainers
    end_date = datetime.now()
    start_date = end_date - timedelta(days=6)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    data = {}
    for symbol in symbols:
        df = process_symbol_data(symbol, start_date_str, end_date_str)
        if df is not None:
            df['sentiment'] = get_news_sentiment(symbol)
            df['combined_score'] = df.apply(calculate_combined_score, axis=1)
            data[symbol] = df
        else:
            print(f"Data not available for {symbol}")
    
    account = api.get_account()
    buying_power = float(account.buying_power)
    
    positions = calculate_positions(buying_power, data)
    execute_buy_orders(api, positions, buying_power)
    execute_sell_orders(api, data)

    strategy_returns = calculate_strategy_returns(data)
    strategy_metrics = calculate_performance_metrics(strategy_returns)

    baseline_data = run_baseline_strategy({symbol: df.copy() for symbol, df in data.items()})
    baseline_returns = calculate_strategy_returns(baseline_data)
    baseline_metrics = calculate_performance_metrics(baseline_returns)

    print("Enhanced Strategy Metrics:")
    print(strategy_metrics)
    print("Baseline Strategy Metrics:")
    print(baseline_metrics)
    
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

while True:
    try:
        monitoring_thread = ensure_monitoring_thread(monitoring_thread, api)
        cleanup_news_cache()
        if is_trading_hours(api):
            settle_trades()
            run_strategy()
        else:
            time_until_target = get_time_until_ny_930()
            hours, remainder = divmod(time_until_target.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"Outside trading hours opens in: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
            time.sleep(time_until_target.total_seconds())
   
        analyze_transactions()
    except Exception as e:
        print(f"An error occurred in the main loop: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        traceback.print_exc()
    time.sleep(60)
