import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import your trading functions
from utils import supertrend, moving_average, add_indicators, train_ml_model

def fetch_historical_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date, interval="1d")
    return df

def backtest(symbol, start_date, end_date, initial_capital=100):
    # Fetch historical data
    df = fetch_historical_data(symbol, start_date, end_date)
    
    # Apply indicators
    df = supertrend(df, 10, 3.0)
    df = moving_average(df, 20, 'sma')
    df = add_indicators(df)
    
    # Add volatility indicator
    df['volatility'] = df['Close'].pct_change().rolling(window=20).std()
    
    # Train ML model
    model = train_ml_model(df)
    df['ml_prediction'] = model.predict(df[['ma', 'rsi', 'macd', 'bollinger_hband', 'bollinger_lband', 'atr']])
    
    # Initialize variables
    capital = initial_capital
    shares = 0
    trades = []
    entry_price = 0
    stop_loss = 0
    
    # Simulate trading
    for i in range(1, len(df)):
        yesterday = df.iloc[i-1]
        today = df.iloc[i]
        print(f"Date: {today.name}, Close: {today['Close']}, in_uptrend: {yesterday['in_uptrend']}, ml_prediction: {yesterday['ml_prediction']}, volatility: {yesterday['volatility']}, mean_volatility: {yesterday['volatility'].mean()}")
    
    print(f"Date: {today.name}, Close: {today['Close']}, in_uptrend: {yesterday['in_uptrend']}, ml_prediction: {yesterday['ml_prediction']}, volatility: {yesterday['volatility']}")
    
    if yesterday['in_uptrend'] and yesterday['ml_prediction'] == 1 and shares == 0 and yesterday['volatility'] > yesterday['volatility'].mean():
        print("Buy signal generated!")
        # Update trailing stop loss if in a position
        if shares > 0:
            stop_loss = max(stop_loss, today['Close'] * 0.95)  # 5% trailing stop loss
        
        # Buy signal
        if yesterday['in_uptrend'] and yesterday['ml_prediction'] == 1 and shares == 0 and yesterday['volatility'] > yesterday['volatility'].mean():
            shares = capital // today['Open']
            cost = shares * today['Open']
            capital -= cost
            entry_price = today['Open']
            stop_loss = entry_price * 0.95  # Initial stop loss at 5% below entry
            trades.append({
                'date': today.name,
                'action': 'BUY',
                'price': today['Open'],
                'shares': shares,
                'cost': cost,
                'capital': capital
            })
        
        # Sell signal
        elif ((not today['in_uptrend'] or today['ml_prediction'] == 0 or today['Close'] <= stop_loss) and shares > 0):
            sell_price = today['Open']
            revenue = shares * sell_price
            capital += revenue
            profit = revenue - (shares * entry_price)
            trades.append({
                'date': today.name,
                'action': 'SELL',
                'price': sell_price,
                'shares': shares,
                'revenue': revenue,
                'profit': profit,
                'capital': capital
            })
            shares = 0
            stop_loss = 0
        
        # Risk management: Cut losses if drawdown exceeds 20%
        elif shares > 0 and (today['Close'] / entry_price - 1) <= -0.2:
            sell_price = today['Open']
            revenue = shares * sell_price
            capital += revenue
            profit = revenue - (shares * entry_price)
            trades.append({
                'date': today.name,
                'action': 'SELL (Stop Loss)',
                'price': sell_price,
                'shares': shares,
                'revenue': revenue,
                'profit': profit,
                'capital': capital
            })
            shares = 0
            stop_loss = 0
    
    # Close any open position at the end
    if shares > 0:
        final_price = df.iloc[-1]['Close']
        revenue = shares * final_price
        capital += revenue
        profit = revenue - (shares * entry_price)
        trades.append({
            'date': df.index[-1],
            'action': 'SELL (Final)',
            'price': final_price,
            'shares': shares,
            'revenue': revenue,
            'profit': profit,
            'capital': capital
        })
    
    return trades, capital

def print_results(trades, final_capital, initial_capital):
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Capital: ${final_capital:.2f}")
    print(f"Total Return: {((final_capital - initial_capital) / initial_capital) * 100:.2f}%")
    print(f"Number of Trades: {len(trades)}")
    
    if not trades:
        print("\nNo trades were executed.")
        return

    print("\nTrade History:")
    total_profit = 0
    winning_trades = 0
    losing_trades = 0
    for trade in trades:
        if trade['action'] == 'BUY':
            print(f"{trade['date']}: BUY {trade['shares']} shares at ${trade['price']:.2f}, Cost: ${trade['cost']:.2f}")
        else:
            print(f"{trade['date']}: {trade['action']} {trade['shares']} shares at ${trade['price']:.2f}, Revenue: ${trade['revenue']:.2f}, Profit: ${trade['profit']:.2f}")
            total_profit += trade['profit']
            if trade['profit'] > 0:
                winning_trades += 1
            elif trade['profit'] < 0:
                losing_trades += 1

    print(f"\nTotal Profit: ${total_profit:.2f}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Break-Even Trades: {len(trades) - winning_trades - losing_trades}")
    
    if winning_trades + losing_trades > 0:
        win_rate = winning_trades / (winning_trades + losing_trades)
        print(f"Win Rate: {win_rate:.2%}")
    
    if trades:
        avg_profit = total_profit / len(trades)
        print(f"Average Profit per Trade: ${avg_profit:.2f}")

    profits = [trade['profit'] for trade in trades if 'SELL' in trade['action']]
    if profits:
        max_drawdown = min(profits)
        max_profit = max(profits)
        print(f"Max Drawdown: ${max_drawdown:.2f}")
        print(f"Max Profit: ${max_profit:.2f}")

    # Calculate Sharpe Ratio
    if trades:
        returns = [trade['profit'] / trade['cost'] for trade in trades if 'SELL' in trade['action']]
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Assuming 252 trading days in a year
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # Calculate Maximum Drawdown
    cumulative_returns = np.cumsum([trade['profit'] for trade in trades if 'SELL' in trade['action']])
    max_drawdown = 0
    peak = cumulative_returns[0]
    for return_ in cumulative_returns:
        if return_ > peak:
            peak = return_
        drawdown = (peak - return_) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

if __name__ == "__main__":
    symbol = "GME"  # Example stock, you can change this
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years ago
    initial_capital = 300

    trades, final_capital = backtest(symbol, start_date, end_date, initial_capital)
    print_results(trades, final_capital, initial_capital)