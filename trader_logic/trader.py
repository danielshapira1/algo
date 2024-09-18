from datetime import datetime, timedelta
from datetime import time as datetime_time
import time as time_module
import pytz
from collections import defaultdict
import yfinance as yf
import numpy as np
import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from .position_manager import PositionManager
from .stock_finder import get_stocks_to_trade
from .trader_utils import get_project_root
import pandas as pd
from .performance_tracker import PerformanceTracker
import logging
import threading
import signal

load_dotenv()

class Trader:
    def __init__(self, update_interval=60, risk_per_trade=0.01, min_trade_value=5, environment='paper'):

        #get environment from .env
        self.environment = environment
        self.api = None
        self.setup_api()
       
        # Get initial account balance
        account = self.api.get_account()
        self.account_balance = float(account.portfolio_value)
        
        self.risk_per_trade = risk_per_trade
        self.min_trade_value = min_trade_value
        max_positions = max(5, 5+int(self.account_balance / 5000))
        self.position_manager = PositionManager(self, max_positions=max_positions)
        self.performance_tracker = PerformanceTracker()
        self.update_interval = update_interval
        self.local_tz = pytz.timezone('Asia/Jerusalem')
        self.ny_tz = pytz.timezone('America/New_York')
        self.market_open_time = datetime_time(9, 30)
        self.market_close_time = datetime_time(16, 0)
        self.running = False
        self.last_stock_update = None
        self.account_balance  # Initialize with actual balance
        self.update_thread = None
        self.error_log_times = defaultdict(float)
        self.error_log_interval = 60
        self.logger = logging.getLogger(__name__)

        self.trading_lock = threading.Lock()
        self.is_stopping = False
        self.positions_file = os.path.join(get_project_root(), 'data', 'csv', 'open_positions.csv')
        self.stocks_to_trade_file = os.path.join(get_project_root(), 'data', 'csv', 'stocks_to_trade.csv')

    def setup_api(self):
        if self.environment == 'paper':
            api_key = os.getenv('APCA_API_KEY_ID_PAPER')
            api_secret = os.getenv('APCA_API_SECRET_KEY_PAPER')
            base_url = os.getenv('APCA_API_BASE_URL_PAPER')
        else:
            api_key = os.getenv('APCA_API_KEY_ID_LIVE')
            api_secret = os.getenv('APCA_API_SECRET_KEY_LIVE')
            base_url = os.getenv('APCA_API_BASE_URL_LIVE')
        
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

    def switch_environment(self, new_environment):
        if new_environment != self.environment:
            self.environment = new_environment
            self.setup_api()
            # Reset any necessary state here
            self.account_balance = float(self.api.get_account().portfolio_value)

    def start(self):
        self.running = True
        self.is_stopping = False
        self.position_manager.start()
        self.logger.info("Trader started. Entering main loop.")
        
        # Start the account balance update thread
        self.account_update_thread = threading.Thread(target=self._update_account_balance_loop, daemon=True)
        self.account_update_thread.start()
        
        # Start the main trading loop in a separate thread
        self.update_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.update_thread.start()

    def _main_loop(self):
        last_market_status = None
        while self.running:
            try:
                if self.is_stopping:
                    break
                
                current_market_status = self.is_market_open()
                if current_market_status != last_market_status:
                    self._log_market_status()
                    last_market_status = current_market_status
                
                if current_market_status:
                    self.logger.info("Market is open. Starting trade cycle.")
                    with self.trading_lock:
                        self.trade()
                else:
                    wait_time = self.time_until_market_open()
                    self.logger.info(f"Market is closed. Waiting {wait_time/3600:.2f} hours until next market open.")
                    time_module.sleep(min(wait_time, 3600))  # Sleep for the wait time or 1 hour, whichever is shorter
                
                time_module.sleep(self.update_interval)
            except Exception as e:
                error_message = f"Unexpected error in main loop: {str(e)}"
                self.log_error_with_rate_limit(error_message)
                if not self.is_stopping:
                    self.logger.info("Continuing to run despite error.")
                else:
                    break
        
        self.logger.info("Trader main loop exited.")

    def stop(self):
        self.logger.info("Initiating trader shutdown...")
        self.is_stopping = True
        with self.trading_lock:
            self.running = False
        
        self.position_manager.stop()
        self.position_manager.join(timeout=10)  # Wait up to 10 seconds for the thread to finish
        if self.position_manager.is_alive():
            self.logger.warning("PositionManager thread did not exit cleanly.")
        
        if self.update_thread:
            self.update_thread.join(timeout=10)
            if self.update_thread.is_alive():
                self.logger.warning("Main trading thread did not exit cleanly.")
        
        if self.account_update_thread:
            self.account_update_thread.join(timeout=10)
            if self.account_update_thread.is_alive():
                self.logger.warning("Account update thread did not exit cleanly.")
        
        self.logger.info("Waiting for ongoing operations to complete...")
        time_module.sleep(5)  # Use time_module.sleep instead of time.sleep
        self.logger.info("Trader shutdown complete.")

    def _update_account_balance_loop(self):
        while self.running:
            try:
                if not self.is_market_open():
                    self._update_account_balance()
                    # Sleep for 1 hour (3600 seconds)
                    time_module.sleep(3600)
                else:
                    # When market is open, check more frequently (e.g., every minute)
                    time_module.sleep(60)
            except Exception as e:
                self.logger.error(f"Error updating account balance: {e}", exc_info=True)
                if self.is_stopping:
                    break
                time_module.sleep(60)  

    def _update_account_balance(self):
        try:
            account = self.api.get_account()
            new_balance = float(account.portfolio_value)
            if new_balance != self.account_balance:
                self.account_balance = new_balance
                if not self.is_market_open():
                    self.logger.info(f"Updated account balance (market closed): ${self.account_balance:.2f}")
                else:
                    self.logger.debug(f"Updated account balance: ${self.account_balance:.2f}")
        except Exception as e:
            self.logger.error(f"Error fetching account balance: {e}")

    def trade(self):
        if self.is_stopping:   
            self.logger.info("Trader is stopping. Skipping trade cycle.")
            return
        if not self.is_market_open():
            self.logger.warning("Trade method called when market is closed. Skipping trade cycle.")
            return   
        if self.account_balance < 100:  # Set a minimum account balance threshold
            self.logger.warning(f"Account balance (${self.account_balance:.2f}) is too low. Skipping trade cycle.")
            return

        try:
            self.position_manager.update_positions()
            now = datetime.now(self.ny_tz)
            current_date = now.date()

            if self.last_stock_update is None or current_date > self.last_stock_update:
                self.logger.info("Updating stocks to trade...")
                get_stocks_to_trade()
                self.last_stock_update = current_date
                
                
            market_regime = self.detect_market_regime()

            if "Bullish" in market_regime:
                self.risk_per_trade = 0.02
            elif "Bearish" in market_regime:
                self.risk_per_trade = 0.005
            else:
                self.risk_per_trade = 0.01

            self.check_day_trade_status()

            try:
                stocks_to_trade = pd.read_csv(self.stocks_to_trade_file)
            except FileNotFoundError:
                self.logger.error("stocks_to_trade.csv not found. Skipping trade.")
                return
            except pd.errors.EmptyDataError:
                self.logger.error("stocks_to_trade.csv is empty. Skipping trade.")
                return
            except Exception as e:
                self.logger.error(f"Error reading stocks_to_trade.csv: {e}. Skipping trade.")
                return

            for _, stock in stocks_to_trade.iterrows():
                if self.is_stopping:
                    self.logger.info("Trader is stopping. Interrupting trade cycle.")
                    return
                
                symbol = stock['Symbol']
                price = stock['Price']
                score = stock['Score']
                volatility = stock['Volatility']
                
                self.logger.info(f"Evaluating trade for {symbol} (Price: ${price:.2f}, Score: {score}, Volatility: {volatility:.4f})")
                
                if symbol not in self.position_manager.positions:
                    volume = self.calculate_position_size(symbol, price, volatility)
                    total_cost = price * volume
                    if volume > 0 and total_cost <= self.account_balance * 0.95:  # Use at most 95% of available balance
                        try:
                            success, message = self.position_manager.open_position(symbol, price, volume, "Auto", datetime.now().date(), volatility)
                            if not success:
                                self.logger.warning(f"Failed to open position for {symbol}: {message}")
                            else:
                                self.account_balance -= total_cost
                                self.logger.info(f"Opened position for {symbol}: {volume} shares at ${price:.2f}. New balance: ${self.account_balance:.2f}")
                        except Exception as e:
                            self.logger.error(f"Error opening position for {symbol}: {e}")
                    else:
                        self.logger.info(f"Skipping {symbol} due to insufficient funds or invalid volume. Required: ${total_cost:.2f}, Available: ${self.account_balance:.2f}")
                else:
                    self.logger.info(f"Position already exists for {symbol}")
                    try:
                        current_value = self.account_balance + sum(self.get_current_price(symbol) * position['volume'] 
                                                                for symbol, position in list(self.position_manager.positions.items()))
                        self.performance_tracker.update(current_value)
                    except Exception as e:
                        self.log_error_once(f"Error updating performance tracker: {str(e)}")

            # Perform rebalancing check
            try:
                self.position_manager.check_and_rebalance()
            except Exception as e:
                self.logger.error(f"Error during portfolio rebalancing: {e}")

            # Update performance tracker
            try:
                current_value = self.account_balance + sum(self.get_current_price(symbol) * position['volume'] 
                                                        for symbol, position in list(self.position_manager.positions.items()))
                self.performance_tracker.update(current_value)
            except Exception as e:
                self.log_error_once(f"Error updating performance tracker: {str(e)}")

        except Exception as e:
            self.logger.error(f"Unexpected error in trade method: {e}", exc_info=True)
    
    def end_trading_day(self):
        self.position_manager.close_all_positions("End of trading day")
        self.position_manager.reset_day_trades()
        account = self.api.get_account()
        self.account_balance = float(account.portfolio_value)
        self.performance_tracker.log_performance()
        self.logger.info(f"Updated account balance: ${self.account_balance:.2f}")

    def wait_for_next_trading_day(self):
        now = datetime.now(self.ny_tz)
        next_day = now + datetime.timedelta(days=1)
        next_day = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
        wait_time = (next_day - now).total_seconds()
        self.logger.info(f"Waiting {wait_time:.2f} seconds until next trading day.")
        time_module.sleep(wait_time)

    def get_current_price(self, symbol):
        try:
            # Get the latest trade
            trades = self.api.get_latest_trade(symbol)
            return trades.price
        except Exception as e:
            self.logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    def calculate_position_size(self, symbol, price, volatility):
        risk_amount = min(self.account_balance * self.risk_per_trade, self.account_balance * 0.01)  # Even more conservative
        position_size = risk_amount / (price * max(volatility, 0.01))
        shares = max(1, int(position_size))
        total_cost = shares * price
        if total_cost > self.account_balance * 0.80:  # More conservative
            shares = max(1, int((self.account_balance * 0.80) / price))
            total_cost = shares * price
        
        if total_cost > self.account_balance * 0.80 or total_cost < 10:  # Minimum trade value of $10
            self.logger.info(f"Skipping {symbol} due to insufficient funds or too small position size. Required: ${total_cost:.2f}, Available: ${self.account_balance:.2f}")
            return 0
        
        return shares
        
    def update_account_balance(self):
        try:
            account = self.api.get_account()
            self.account_balance = float(account.portfolio_value)
            self.logger.info(f"Updated account balance: ${self.account_balance:.2f}")
        except Exception as e:
            self.logger.error(f"Error updating account balance: {e}")

    def buy(self, symbol, qty):
        if self.is_stopping or not self.is_market_open() or qty <= 0:
            self.logger.info(f"Skipping buy order for {symbol}. Stopping: {self.is_stopping}, Market open: {self.is_market_open()}, Quantity: {qty}")
            return None
        
        try:
            current_price = self.get_current_price(symbol)
            if current_price is None:
                return None

            total_cost = current_price * qty
            if total_cost > self.account_balance * 0.95:  # Use at most 95% of available balance
                self.logger.warning(f"Insufficient funds to buy {qty} shares of {symbol}. Required: ${total_cost:.2f}, Available: ${self.account_balance:.2f}")
                return None

            limit_price = round(current_price * 1.01, 2)
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='limit',
                time_in_force='day',
                limit_price=limit_price
            )
            self.logger.info(f"Buy limit order placed for {qty} shares of {symbol} at ${limit_price:.2f}")
            return order
        except Exception as e:
            if "insufficient buying power" in str(e).lower():
                self.log_error_once(f"Insufficient buying power for {symbol}")
            else:
                self.log_error_once(f"Error placing buy order for {symbol}: {str(e)}")
            return None

    def sell(self, symbol, qty):
        try:
            position = self.position_manager.positions.get(symbol)
            if not position:
                self.logger.warning(f"No position found for {symbol}. Cannot sell.")
                return None
            
            available_qty = position['volume']
            if qty > available_qty:
                self.logger.warning(f"Requested sell quantity ({qty}) exceeds available quantity ({available_qty}) for {symbol}. Adjusting to available quantity.")
                qty = available_qty

            current_price = self.get_current_price(symbol)
            if current_price is None:
                raise Exception(f"Unable to get current price for {symbol}")

            limit_price = round(current_price * 0.99, 2)
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='limit',
                time_in_force='day',
                limit_price=limit_price
            )
            self.logger.info(f"Sell limit order placed for {qty} shares of {symbol} at ${limit_price:.2f}")
            return order
        except Exception as e:
            self.logger.error(f"Error placing sell order for {symbol}: {e}")
            return None
        
    def detect_market_regime(self):
        try:
            # Get S&P 500 data
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="6mo")
            
            # Get data for currently held stocks
            stock_data = {}
            for symbol in self.position_manager.positions.keys():
                stock = yf.Ticker(symbol)
                stock_data[symbol] = stock.history(period="6mo")
            
            # Calculate metrics for S&P 500
            spy_returns = spy_hist['Close'].pct_change()
            spy_volatility = spy_returns.std() * np.sqrt(252)
            spy_ma50 = spy_hist['Close'].rolling(window=50).mean()
            spy_ma200 = spy_hist['Close'].rolling(window=200).mean()
            
            # Determine S&P 500 trend
            if spy_hist['Close'].iloc[-1] > spy_ma50.iloc[-1] > spy_ma200.iloc[-1]:
                spy_trend = "Uptrend"
            elif spy_hist['Close'].iloc[-1] < spy_ma50.iloc[-1] < spy_ma200.iloc[-1]:
                spy_trend = "Downtrend"
            else:
                spy_trend = "Sideways"
            
            # Calculate average metrics for held stocks
            stock_volatilities = []
            stock_trends = []
            for symbol, data in stock_data.items():
                returns = data['Close'].pct_change()
                volatility = returns.std() * np.sqrt(252)
                ma50 = data['Close'].rolling(window=50).mean()
                ma200 = data['Close'].rolling(window=200).mean()
                
                stock_volatilities.append(volatility)
                if data['Close'].iloc[-1] > ma50.iloc[-1] > ma200.iloc[-1]:
                    stock_trends.append("Uptrend")
                elif data['Close'].iloc[-1] < ma50.iloc[-1] < ma200.iloc[-1]:
                    stock_trends.append("Downtrend")
                else:
                    stock_trends.append("Sideways")
            
            avg_stock_volatility = np.mean(stock_volatilities) if stock_volatilities else spy_volatility
            
            # Determine overall market regime
            if spy_trend == "Uptrend" and "Downtrend" not in stock_trends:
                regime = "Bullish"
            elif spy_trend == "Downtrend" and "Uptrend" not in stock_trends:
                regime = "Bearish"
            else:
                regime = "Mixed"
            
            if spy_volatility > 0.2 or avg_stock_volatility > 0.2:
                regime += " (High Volatility)"
            else:
                regime += " (Low Volatility)"
            
            self.logger.info(f"Current market regime: {regime}")
            return regime
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return "Unknown"

    def _log_market_status(self):
        if self.is_market_open():
            self.logger.info("Market is now open.")
        else:
            self.logger.info("Market is now closed.")

    def check_day_trade_status(self):
        remaining_day_trades = self.position_manager.get_remaining_day_trades()
        self.logger.info(f"Remaining day trades: {remaining_day_trades}")
        if remaining_day_trades != "Unlimited" and remaining_day_trades <= 1:
            self.logger.warning("CAUTION: You are close to reaching the day trade limit. "
                                "Further day trades may result in Pattern Day Trader restrictions.")

    def is_market_open(self):
        now = datetime.now(self.ny_tz)
        
        if now.weekday() > 4:  # Saturday or Sunday
            return False
        
        current_time = now.time()
        if self.market_open_time <= current_time < self.market_close_time:
            return True
        else:
            return False
    
    def time_until_market_open(self):
        now = datetime.now(self.ny_tz)
        if now.weekday() > 4:  # It's the weekend
            days_until_monday = (7 - now.weekday()) % 7
            next_market_open = now + timedelta(days=days_until_monday)
            next_market_open = next_market_open.replace(hour=self.market_open_time.hour, minute=self.market_open_time.minute, second=0, microsecond=0)
        elif now.time() < self.market_open_time:
            next_market_open = now.replace(hour=self.market_open_time.hour, minute=self.market_open_time.minute, second=0, microsecond=0)
        else:
            next_market_open = now + timedelta(days=1)
            next_market_open = next_market_open.replace(hour=self.market_open_time.hour, minute=self.market_open_time.minute, second=0, microsecond=0)
        
        wait_time = (next_market_open - now).total_seconds()
        return wait_time
    
    # def setup_logging(self):
    #     log_dir = os.path.join(get_project_root(), 'data', 'logs')
    #     os.makedirs(log_dir, exist_ok=True)

    #     self.logger = logging.getLogger(__name__)
    #     self.logger.setLevel(logging.DEBUG)

    #     # Info handler
    #     info_handler = logging.FileHandler(os.path.join(log_dir, 'trader.log'))
    #     info_handler.setLevel(logging.INFO)
    #     info_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    #     info_handler.setFormatter(info_formatter)

    #     # Debug handler
    #     debug_handler = logging.FileHandler(os.path.join(log_dir, 'trader_debug.log'))
    #     debug_handler.setLevel(logging.DEBUG)
    #     debug_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    #     debug_handler.setFormatter(debug_formatter)

    #     self.logger.addHandler(info_handler)
    #     self.logger.addHandler(debug_handler)
    
    def log_error_once(self, error_message):
        if error_message not in self.logged_errors:
            self.logger.error(error_message)
            self.logged_errors.add(error_message)
    
    def log_error_with_rate_limit(self, error_message):
        current_time = time_module.time()
        if current_time - self.error_log_times[error_message] > self.error_log_interval:
            self.logger.error(error_message)
            self.error_log_times[error_message] = current_time

def signal_handler(signum, frame):
    global trader
    trader.logger.info("Received interrupt signal. Stopping the Trader...")
    trader.stop()

if __name__ == "__main__":
    # Set up logging
    log_dir = os.path.join(get_project_root(), 'data', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(log_dir, "trader.log")),
                            logging.StreamHandler()
                        ])
    
    # Create a separate handler for debug logs
    debug_handler = logging.FileHandler(os.path.join(log_dir, "trader_debug.log"))
    debug_handler.setLevel(logging.DEBUG)
    debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    debug_handler.setFormatter(debug_formatter)
    
    # Get the root logger and add the debug handler
    root_logger = logging.getLogger()
    root_logger.addHandler(debug_handler)
    
    trader = Trader(update_interval=60)
    print(trader.account_balance)
    trader.logger.info(f"Initial account balance: ${trader.account_balance:.2f}")
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        trader.start()
    except KeyboardInterrupt:
        trader.logger.info("stop received. Stopping the Trader...")
    finally:
        trader.stop()
        trader.logger.info(f"Final account balance: ${trader.account_balance:.2f}")
        trader.logger.info(f"Positions: {trader.position_manager.positions}")
        trader.logger.info(f"Day trades made: {len(trader.position_manager.day_trades)}")
        trader.logger.info(f"Is pattern day trader: {trader.position_manager.is_pattern_day_trader}")