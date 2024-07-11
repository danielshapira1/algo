import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import your trading functions
from paper_trade.utils import supertrend, moving_average, add_indicators, train_ml_model

def fetch_historical_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date, interval="1d")
    return df

def backtest(symbol, start_date, end_date, initial_capital=100):
    # Fetch historical data
    df = fetch_historical_data(symbol, start_date, end_date)

    min_price = df['Open'].min()
    if initial_capital < min_price:
        print(f"Initial capital (${initial_capital:.2f}) is less than the minimum share price (${min_price:.2f}). Increase initial capital.")
        return [], initial_capital
    
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
        
        print(f"Date: {today.name}, Close: {today['Close']:.2f}, in_uptrend: {yesterday['in_uptrend']}, "
              f"ml_prediction: {yesterday['ml_prediction']}, volatility: {yesterday['volatility']:.4f}, "
              f"mean_volatility: {df['volatility'].mean():.4f}, shares: {shares}, capital: {capital:.2f}")
        
        # Update trailing stop loss if in a position
        if shares > 0:
            stop_loss = max(stop_loss, today['Close'] * 0.95)  # 5% trailing stop loss
        
        # Buy signal
        if yesterday['in_uptrend'] and yesterday['ml_prediction'] == 1 and shares == 0:
            print("Buy signal generated!")
            shares_to_buy = capital / today['Open']  # This allows fractional shares
            if shares_to_buy > 0:
                cost = shares_to_buy * today['Open']
                capital -= cost
                shares = shares_to_buy
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
                print(f"Bought {shares:.2f} shares at ${today['Open']:.2f}")
            else:
                print("Not enough capital to buy shares")
            
        # Sell signal
        elif (not today['in_uptrend'] or today['ml_prediction'] == 0 or today['Close'] <= stop_loss) and shares > 0:
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
            print(f"Sold {shares:.2f} shares at ${sell_price:.2f}, Profit: ${profit:.2f}")
            shares = 0
            stop_loss = 0
        
        # Risk management: Cut losses if drawdown exceeds 15%
        elif shares > 0 and (today['Close'] / entry_price - 1) <= -0.15:
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
            print(f"{trade['date']}: BUY {trade['shares']:.2f} shares at ${trade['price']:.2f}, Cost: ${trade['cost']:.2f}")
        else:
            print(f"{trade['date']}: {trade['action']} {trade['shares']:.2f} shares at ${trade['price']:.2f}, Revenue: ${trade['revenue']:.2f}, Profit: ${trade['profit']:.2f}")
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
        buy_trades = [trade for trade in trades if trade['action'] == 'BUY']
        sell_trades = [trade for trade in trades if 'SELL' in trade['action']]
        if len(buy_trades) == len(sell_trades):
            returns = [sell_trade['profit'] / buy_trade['cost'] for buy_trade, sell_trade in zip(buy_trades, sell_trades)]
            if len(returns) > 1:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Assuming 252 trading days in a year
                print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        # Calculate Maximum Drawdown
        cumulative_returns = np.cumsum(profits)
        max_drawdown = 0
        peak = cumulative_returns[0]
        for return_ in cumulative_returns:
            if return_ > peak:
                peak = return_
            drawdown = (peak - return_) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
    else:
        print("No sell trades executed, unable to calculate additional metrics.")

def plot_results(df, trades, initial_capital):
    plt.figure(figsize=(12, 8))
    
    # Plot stock price
    plt.plot(df.index, df['Close'], label='Stock Price', alpha=0.5)
    
    # Plot buy and sell points
    buy_dates = [trade['date'] for trade in trades if trade['action'] == 'BUY']
    buy_prices = [trade['price'] for trade in trades if trade['action'] == 'BUY']
    sell_dates = [trade['date'] for trade in trades if 'SELL' in trade['action']]
    sell_prices = [trade['price'] for trade in trades if 'SELL' in trade['action']]
    
    plt.scatter(buy_dates, buy_prices, color='green', marker='^', label='Buy')
    plt.scatter(sell_dates, sell_prices, color='red', marker='v', label='Sell')
    
    # Plot capital
    capital = [initial_capital] + [trade['capital'] for trade in trades]
    capital_dates = [df.index[0]] + [trade['date'] for trade in trades]
    plt.plot(capital_dates, capital, label='Capital', color='orange')
    
    plt.title(f'Backtesting Results for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price / Capital')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    symbol = "GOOG"  # Example stock, you can change this
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years ago
    initial_capital = 150  # Increased from 100 to 10000

    trades, final_capital = backtest(symbol, start_date, end_date, initial_capital)
    print_results(trades, final_capital, initial_capital)
    
    # Fetch historical data again for plotting
    df = fetch_historical_data(symbol, start_date, end_date)
    
    # Plot the results
    plot_results(df, trades, initial_capital)