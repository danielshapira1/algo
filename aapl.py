import yfinance as yf
import time

def fetch_aapl_stock_price():
    ticker = yf.Ticker('AAPL')
    while True:
        data = ticker.history(period='1d')  # Fetches data for the last trading day
        if not data.empty:
            latest_price = data['Close'].iloc[-1]  # Get the latest closing price
            print(f"Apple Stock Price (AAPL): ${latest_price:.2f}")
        else:
            print("No data available for Apple stock.")
        time.sleep(5)  # Wait for 5 seconds

# def trade(self):
#         self.logger.debug("Entering trade method")
#         if not self.is_market_open():
#             self.logger.warning("Trade method called when market is closed. Skipping trade cycle.")
#             return
# 
#         try:
#             self.position_manager.update_positions()
#             now = datetime.now(self.ny_tz)
#             current_date = now.date()
#             market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
#             market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
# 
#             if now < market_open:
#                 time_to_open = (market_open - now).total_seconds()
#                 self.logger.info(f"Market not open yet. Waiting {time_to_open:.2f} seconds until market opens.")
#                 return
# 
#             if now > market_close:
#                 self.logger.info("Market is closed. Waiting for next trading day.")
#                 self.last_stock_update = None
#                 self.wait_for_next_trading_day()
#                 return
# 
#             if self.last_stock_update is None or current_date > self.last_stock_update:
#                 self.logger.info("Updating stocks to trade...")
#                 get_stocks_to_trade()
#                 self.last_stock_update = current_date
#                 market_regime = self.detect_market_regime()
#                 self.logger.info(f"Current market regime: {market_regime}")
# 
#             if "Trending Up" in market_regime:
#                 self.risk_per_trade = 0.02
#             elif "Trending Down" in market_regime:
#                 self.risk_per_trade = 0.005
#             else:
#                 self.risk_per_trade = 0.01
# 
#             self.check_day_trade_status()
# 
#             try:
#                 stocks_to_trade = pd.read_csv('stocks_to_trade.csv')
#             except FileNotFoundError:
#                 self.logger.error("stocks_to_trade.csv not found. Skipping trade.")
#                 return
#             except pd.errors.EmptyDataError:
#                 self.logger.error("stocks_to_trade.csv is empty. Skipping trade.")
#                 return
#             except Exception as e:
#                 self.logger.error(f"Error reading stocks_to_trade.csv: {e}. Skipping trade.")
#                 return
# 
#             for _, stock in stocks_to_trade.iterrows():
#                 if self.is_stopping:
#                     self.logger.info("Trader is stopping. Interrupting trade cycle.")
#                     return
#                 
#                 symbol = stock['Symbol']
#                 price = stock['Price']
#                 score = stock['Score']
#                 volatility = stock['Volatility']
#                 
#                 self.logger.info(f"Evaluating trade for {symbol} (Price: ${price:.2f}, Score: {score}, Volatility: {volatility:.4f})")
#                 
#                 if symbol not in self.position_manager.positions:
#                     volume = self.calculate_position_size(symbol, price, volatility)
#                     total_cost = price * volume
#                     if volume > 0 and total_cost <= self.account_balance * 0.95:  # Use at most 95% of available balance
#                         try:
#                             success, message = self.position_manager.open_position(symbol, price, volume, "Auto", datetime.now().date(), volatility)
#                             if not success:
#                                 self.logger.warning(f"Failed to open position for {symbol}: {message}")
#                             else:
#                                 self.account_balance -= total_cost
#                                 self.logger.info(f"Opened position for {symbol}: {volume} shares at ${price:.2f}. New balance: ${self.account_balance:.2f}")
#                         except Exception as e:
#                             self.logger.error(f"Error opening position for {symbol}: {e}")
#                     else:
#                         self.logger.info(f"Skipping {symbol} due to insufficient funds or invalid volume. Required: ${total_cost:.2f}, Available: ${self.account_balance:.2f}")
#                 else:
#                     self.logger.info(f"Position already exists for {symbol}")
#                     try:
#                         current_price = self.get_current_price(symbol)
#                         if current_price is not None:
#                             success, message = self.position_manager.update_position(symbol, current_price, datetime.now().date())
#                             if not success:
#                                 self.logger.warning(f"Failed to update position for {symbol}: {message}")
#                     except Exception as e:
#                         self.logger.error(f"Error updating position for {symbol}: {e}")
# 
#             # Perform rebalancing check
#             try:
#                 self.position_manager.check_and_rebalance()
#             except Exception as e:
#                 self.logger.error(f"Error during portfolio rebalancing: {e}")
# 
#             # Update performance tracker
#             try:
#                 current_value = self.account_balance + sum(self.get_current_price(symbol) * position['volume'] 
#                                                            for symbol, position in self.position_manager.positions.items())
#                 self.performance_tracker.update(current_value)
#             except Exception as e:
#                 self.logger.error(f"Error updating performance tracker: {e}")
# 
#         except Exception as e:
#             self.logger.error(f"Unexpected error in trade method: {e}")
if __name__ == "__main__":
    fetch_aapl_stock_price()
