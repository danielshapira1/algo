import pandas as pd
import yfinance as yf
from trader import Trader
from position_manager import PositionManager
from performance_tracker import PerformanceTracker
from datetime import datetime, timedelta
import pytz

class MockAPI:
    def __init__(self, data):
        if data.empty:
            raise ValueError("No data available for backtesting")
        self.data = data
        self.current_date = data.index[0]

    def get_barset(self, symbol, timeframe, limit):
        if timeframe != 'minute':
            raise ValueError("Only minute data is supported in this mock")
        return {symbol: [type('Bar', (), {'c': self.data.loc[self.current_date, (symbol, 'Close')]}),]}

    def get_account(self):
        return type('Account', (), {'portfolio_value': 10000})()

    def submit_order(self, symbol, qty, side, type, time_in_force, limit_price=None):
        price = self.data.loc[self.current_date, (symbol, 'Close')]
        return type('Order', (), {'id': 'mock_order', 'status': 'filled', 'filled_avg_price': price, 'filled_qty': qty})()

    def get_order(self, order_id):
        return type('Order', (), {'status': 'filled'})()

class BacktestTrader(Trader):
    def __init__(self, symbols, start_date, end_date, initial_balance=10000):
        self.symbols = symbols
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        self.account_balance = initial_balance
        self.performance_tracker = PerformanceTracker()
        self.data = self._load_data()
        if self.data.empty:
            raise ValueError("No data available for the specified date range and symbols")
        self.api = MockAPI(self.data)
        self.position_manager = PositionManager(self)
        self.current_date = self.data.index[0]
        self.ny_tz = pytz.timezone('America/New_York')
        self.jerusalem_tz = pytz.timezone('Asia/Jerusalem')
        self.update_interval = 60
        self.risk_per_trade = 0.01
        self.last_stock_update = None

    def _load_data(self):
        data = {}
        end_date = min(self.end_date, datetime.now().date())
        start_date = max(self.start_date, end_date - timedelta(days=365))
        
        for symbol in self.symbols:
            df = yf.download(symbol, start=start_date, end=end_date, interval="1d")
            if not df.empty:
                data[symbol] = df
            else:
                print(f"No data available for {symbol}")
        
        if not data:
            return pd.DataFrame()  # Return empty DataFrame if no data is available
        
        combined_data = pd.concat(data, axis=1, keys=self.symbols)
        
        # Remove the last row if it contains NaN values
        if combined_data.iloc[-1].isnull().any():
            combined_data = combined_data.iloc[:-1]
        
        return combined_data

    def run_backtest(self):
        for date in self.data.index:
            self.current_date = date
            self.api.current_date = date
            self.trade()
            current_value = self.account_balance + sum(self.get_current_price(symbol) * position['volume'] 
                                                       for symbol, position in self.position_manager.positions.items())
            self.performance_tracker.update(current_value)
        
        return self.performance_tracker.get_performance_summary()

    def get_current_price(self, symbol):
        try:
            return self.data.loc[self.current_date, (symbol, 'Close')]
        except KeyError:
            return None

    def calculate_position_size(self, stock, price, volatility):
        risk_amount = self.account_balance * self.risk_per_trade
        position_size = risk_amount / (price * volatility)
        shares = int(position_size / price)
        return max(1, shares)

    def get_time_until_ny_930(self):
        # This method is not needed for backtesting, return a dummy value
        return timedelta(seconds=0)

    def detect_market_regime(self):
        # Simplified market regime detection for backtesting
        return "Trending Up (Low Volatility)"

    def buy(self, symbol, qty):
        try:
            current_price = self.get_current_price(symbol)
            if current_price is None:
                return None

            limit_price = round(current_price * 1.01, 2)
            order = type('Order', (), {
                'id': 'mock_order',
                'status': 'filled',
                'filled_avg_price': limit_price,
                'filled_qty': qty
            })
            print(f"Buy order placed for {qty} shares of {symbol} at ${limit_price}")
            return order
        except Exception as e:
            print(f"Error placing buy order for {symbol}: {e}")
            return None

    def sell(self, symbol, qty):
        try:
            current_price = self.get_current_price(symbol)
            if current_price is None:
                return None

            limit_price = round(current_price * 0.99, 2)
            order = type('Order', (), {
                'id': 'mock_order',
                'status': 'filled',
                'filled_avg_price': limit_price,
                'filled_qty': qty
            })
            print(f"Sell order placed for {qty} shares of {symbol} at ${limit_price}")
            return order
        except Exception as e:
            print(f"Error placing sell order for {symbol}: {e}")
            return None

class Backtester:
    def __init__(self, symbols, start_date, end_date, initial_balance=10000):
        try:
            self.trader = BacktestTrader(symbols, start_date, end_date, initial_balance)
        except ValueError as e:
            print(f"Error initializing backtester: {e}")
            self.trader = None

    def run(self):
        if self.trader is None:
            print("Cannot run backtest due to initialization error")
            return

        results = self.trader.run_backtest()
        print("Backtest Results:")
        for key, value in results.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']  # Add your symbols here
    start_date = '2022-06-01'  # Adjust this date as needed
    end_date = '2023-06-01'    # Adjust this date as needed
    backtester = Backtester(symbols, start_date, end_date)
    backtester.run()