import threading
from datetime import datetime, date, timedelta
from datetime import time as datetime_time
import time as time_module
import yfinance as yf
import os
from trader_logic.trader_utils import get_project_root
import numpy as np
import csv
import logging

class PositionManager(threading.Thread):
    def __init__(self, trader, max_positions=5, max_sector_exposure=0.30):
        threading.Thread.__init__(self)
        self.trader = trader
        self.positions = {}
        self.sector_allocation = {}
        self.day_trades = []
        self.last_sell_time = {}
        self.is_pattern_day_trader = False
        self.max_positions = max_positions
        self.max_sector_exposure = max_sector_exposure
        self.logger = logging.getLogger(__name__)
        self.positions_file = os.path.join(get_project_root(), 'data', 'csv', 'open_positions.csv')
        self.day_trade_warning_threshold = 3
        self.last_rebalance_time = datetime.now()
        self.market_open_time = datetime_time(9, 30)
        self.market_close_time = datetime_time(16, 0)
        self.load_positions()
        self.running = True
        self.shutdown_event = threading.Event()


    def run(self):
        while self.running:
            self.update_positions()
            self.check_portfolio_health()
            
            current_time = datetime.now()
            if self.is_trading_hours():
                if (current_time - self.last_rebalance_time).total_seconds() >= 300:  # 5 minutes
                    self.check_and_rebalance()
                    self.last_rebalance_time = current_time
            else:
                if (current_time - self.last_rebalance_time).total_seconds() >= 3600:  # 1 hour
                    self.check_and_rebalance()
                    self.last_rebalance_time = current_time
            
            self.check_pattern_day_trader_status()
            
            # Wait for the update interval or until the shutdown event is set
            self.shutdown_event.wait(timeout=self.trader.update_interval)
            if self.shutdown_event.is_set():
                break

        self.logger.info("PositionManager thread exited.")

    def stop(self):
        self.logger.info("PositionManager stopping...")
        self.running = False
        self.shutdown_event.set()

    def save_positions(self):
        try:
            with open(self.positions_file, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['symbol', 'entry_price', 'current_price', 'volume', 'sector', 'entry_date', 'market_value', 'unrealized_pl', 'unrealized_plpc', 'highest_price', 'trailing_stop_loss', 'take_profit', 'volatility'])
                writer.writeheader()
                for symbol, position in self.positions.items():
                    position_data = position.copy()
                    position_data['symbol'] = symbol
                    position_data['entry_date'] = position_data['entry_date'].isoformat()  # Convert date to string
                    writer.writerow(position_data)
            self.logger.info(f"Positions saved to {self.positions_file}")
        except Exception as e:
            self.logger.error(f"Error saving positions: {e}")

    def load_positions(self):
        try:
            with open(self.positions_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    symbol = row.pop('symbol')
                    row['entry_date'] = date.fromisoformat(row['entry_date'])
                    row['entry_price'] = float(row['entry_price'])
                    row['volume'] = int(row['volume'])
                    row['highest_price'] = float(row['highest_price'])
                    row['trailing_stop_loss'] = float(row['trailing_stop_loss'])
                    row['take_profit'] = float(row['take_profit'])
                    row['volatility'] = float(row['volatility'])
                    self.positions[symbol] = row
            self.logger.info(f"Positions loaded from {self.positions_file}")
        except FileNotFoundError:
            self.logger.info(f"No existing positions file found at {self.positions_file}. Starting with an empty portfolio.")
        except Exception as e:
            self.logger.error(f"Error loading positions: {e}")

    def can_open_position(self, stock, price, volume, sector, volatility, market_cap, avg_daily_volume):
        if len(self.positions) >= self.max_positions:
            return False, "Max number of positions reached"
        
        position_value = price * volume
        total_portfolio_value = self.trader.account_balance + sum(pos['entry_price'] * pos['volume'] for pos in self.positions.values())
        
        if self.trader.account_balance * 0.1 < price * volume:
            return False, "Position would exceed 10% of portfolio"
        
        sector_exposure = (self.sector_allocation.get(sector, 0) + position_value) / total_portfolio_value
        if sector_exposure > self.max_sector_exposure:
            return False, f"Sector exposure would exceed {self.max_sector_exposure*100}%"
        
        if volatility > 0.4:
            return False, "Stock volatility too high"
        
        if avg_daily_volume < 500000:
            return False, "Insufficient liquidity"
        
        if price < 10 or price > 1000:
            return False, "Stock price outside allowed range"
        
        if market_cap < 1e9 or market_cap > 5e11:
            return False, "Market cap outside allowed range"
        
        if self.trader.account_balance < 25000 and len(self.day_trades) >= 3:
            return False, "Day trade limit reached for accounts under $25,000"
        
        return True, "Position can be opened"

    def open_position(self, stock, price, volume, sector, entry_date, volatility):
        if stock in self.last_sell_time:
            time_since_sell = datetime.now() - self.last_sell_time[stock]
            if time_since_sell.total_seconds() < 86400:  # 24 hours in seconds
                self.trader.logger.warning(f"Cannot buy {stock} due to recent sale (wash sale prevention)")
                return False, "Recent sale prevents buying (wash sale prevention)"
        try:
            can_open, reason = self.can_open_position(stock, price, volume, sector, volatility, 5e9, 1e6)
            if not can_open:
                self.logger.warning(f"Cannot open position for {stock}: {reason}")
                return False, reason
            
            stop_loss_pct = max(0.05, min(0.15, volatility))
            take_profit_pct = max(0.1, min(0.3, volatility * 2))
            
            adjusted_volume = int(volume * (0.2 / volatility))
            
            order = self.trader.buy(stock, adjusted_volume)
            if order is None:
                raise Exception("Failed to place buy order")
            
            entry_date = datetime.now().date()
            self.positions[stock] = {
                'entry_price': price,
                'volume': adjusted_volume,
                'sector': sector,
                'entry_date': entry_date,
                'highest_price': price,
                'trailing_stop_loss': price * (1 - stop_loss_pct),
                'take_profit': price * (1 + take_profit_pct),
                'volatility': volatility
            }
            
            self.trader.account_balance -= price * adjusted_volume
            self.sector_allocation[sector] = self.sector_allocation.get(sector, 0) + price * adjusted_volume
            
            self.save_positions()
            self.logger.info(f"Position opened for {stock}: {adjusted_volume} shares at ${price}")
            return True, "Position opened successfully"
        except Exception as e:
            self.logger.error(f"Error opening position for {stock}: {e}")
            return False, str(e)

    def update_position(self, stock, current_price, current_date):
        if stock not in self.positions:
            return False, "Position does not exist"
        
        position = self.positions[stock]
        
        if current_price <= position['trailing_stop_loss']:
            return self.close_position(stock, current_price, "Trailing stop loss triggered")
        
        if current_price >= position['take_profit']:
            return self.close_position(stock, current_price, "Take profit triggered")
        
        if (current_date - position['entry_date']).days >= 1:
            return self.close_position(stock, current_price, "Max time in position reached")
        
        if current_price > position['highest_price']:
            position['highest_price'] = current_price
            stop_loss_pct = max(0.05, min(0.15, position['volatility']))
            position['trailing_stop_loss'] = current_price * (1 - stop_loss_pct)
            self.save_positions()
        
        return True, "Position updated"
    
    def update_positions(self):
        try:
            alpaca_positions = self.trader.api.list_positions()
            self.positions = {}
            for position in alpaca_positions:
                current_price = float(position.current_price)
                entry_price = float(position.avg_entry_price)
                unrealized_pl = float(position.unrealized_pl)
                unrealized_plpc = float(position.unrealized_plpc)
                
                self.positions[position.symbol] = {
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'volume': int(position.qty),
                    'sector': self.get_sector(position.symbol),
                    'entry_date': self.get_entry_date(position.symbol),
                    'market_value': float(position.market_value),
                    'unrealized_pl': unrealized_pl,
                    'unrealized_plpc': unrealized_plpc,
                    'highest_price': max(current_price, self.positions.get(position.symbol, {}).get('highest_price', current_price)),
                    'trailing_stop_loss': self.calculate_stop_loss(position),
                    'take_profit': self.calculate_take_profit(position),
                    'volatility': self.calculate_volatility(position.symbol)
                }
            self.save_positions()
            self.logger.info(f"Updated positions: {self.positions}")
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")

    def add_day_trade(self, stock):
        current_date = datetime.now(self.trader.ny_tz).date()
        self.day_trades.append((stock, current_date))
        
        five_days_ago = current_date - datetime.timedelta(days=5)
        self.day_trades = [(s, d) for s, d in self.day_trades if d > five_days_ago]
        
        day_trade_count = len(self.day_trades)
        
        if self.trader.account_balance < 25000:
            if day_trade_count == self.day_trade_warning_threshold:
                self.logger.warning(f"WARNING: You have made {day_trade_count} day trades. "
                                    f"Making one more day trade will mark you as a Pattern Day Trader.")
            elif day_trade_count >= 4:
                self.logger.warning("ALERT: You have been marked as a Pattern Day Trader. "
                                    "Further day trading may be restricted.")
                self.is_pattern_day_trader = True
        
        self.logger.info(f"Day trade added. Total day trades in last 5 trading days: {day_trade_count}")

    def close_position(self, stock, exit_price, reason):
        try:
            if stock not in self.positions:
                return False, "Position does not exist"
        
            position = self.positions[stock]
            order = self.trader.sell(stock, position['volume'])
            if order is None:
                return False, "Failed to place sell order"
            
            profit_loss = (exit_price - position['entry_price']) * position['volume']
            self.trader.account_balance += exit_price * position['volume']
            self.sector_allocation[position['sector']] -= position['entry_price'] * position['volume']
            
            if (datetime.date.today() - position['entry_date']).days == 0:
                self.add_day_trade(stock)
            
            del self.positions[stock]
            self.last_sell_time[stock] = datetime.now()  # Record the sell time
            self.save_positions()
            
            self.logger.info(f"Position closed: {stock}. Reason: {reason}. Profit/Loss: {profit_loss}")
            return True, f"Position closed: {reason}. Profit/Loss: {profit_loss}"
        except Exception as e:
            self.logger.error(f"Error closing position for {stock}: {e}")
            return False, str(e)

    def close_all_positions(self, reason):
        for stock in list(self.positions.keys()):
            try:
                current_price = self.trader.get_current_price(stock)
                if current_price is not None:
                    self.close_position(stock, current_price, reason)
            except Exception as e:
                self.logger.error(f"Error closing position for {stock}: {e}")

    def get_sector(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            sector = stock.info.get('sector', 'Unknown')
            return sector
        except Exception as e:
            self.logger.error(f"Error fetching sector for {symbol}: {e}")
            return "Unknown"

    def get_entry_date(self, symbol):
        return self.positions.get(symbol, {}).get('entry_date', datetime.now().date())

    def calculate_stop_loss(self, position):
        volatility = self.calculate_volatility(position.symbol)
        stop_loss_pct = max(0.05, min(0.15, volatility))
        return float(position.avg_entry_price) * (1 - stop_loss_pct)

    def calculate_take_profit(self, position):
        volatility = self.calculate_volatility(position.symbol)
        take_profit_pct = max(0.1, min(0.3, volatility * 2))
        return float(position.avg_entry_price) * (1 + take_profit_pct)
    
    def calculate_volatility(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")
            if hist.empty:
                return 0.2  # Default value if no data
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            return volatility
        except Exception as e:
            self.logger.error(f"Error calculating volatility for {symbol}: {e}")
            return 0.2  # Default value in case of error
        
    def get_historical_data(self, symbol, period='1y'):
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            stock = yf.Ticker(symbol)
            hist = stock.history(start=start_date, end=end_date)
            return hist
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None

    def calculate_returns(self, historical_data):
        return historical_data['Close'].pct_change().dropna()

    def calculate_win_rate(self, returns):
        positive_returns = returns[returns > 0]
        return len(positive_returns) / len(returns)

    def calculate_win_loss_ratio(self, returns):
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 1
        return avg_win / avg_loss

    def kelly_criterion(self, win_rate, win_loss_ratio):
        return win_rate - ((1 - win_rate) / win_loss_ratio)

    def check_portfolio_health(self):
        total_value = self.trader.account_balance + sum(pos['entry_price'] * pos['volume'] for pos in self.positions.values())
        if total_value < self.trader.account_balance * 0.85:
            self.close_all_positions("Maximum drawdown reached")
            self.logger.warning("Trading halted due to maximum drawdown")
            return False, "Trading halted due to maximum drawdown"
        return True, "Portfolio health check passed"

    def check_and_rebalance(self):
        self.logger.info("Portfolio rebalancing check initiated")
        try:
            total_value = sum(position['market_value'] for position in self.positions.values())
            target_value_per_position = total_value / len(self.positions)
            
            for symbol, position in list(self.positions.items()):
                current_value = position['market_value']
                if abs(current_value - target_value_per_position) / target_value_per_position > 0.1:
                    if current_value > target_value_per_position:
                        # Sell
                        shares_to_sell = int((current_value - target_value_per_position) / position['current_price'])
                        if shares_to_sell > 0:
                            try:
                                self.trader.sell(symbol, shares_to_sell)
                                self.logger.info(f"Rebalancing: Sold {shares_to_sell} shares of {symbol}")
                            except Exception as e:
                                self.trader.log_error_once(f"Error selling {symbol} for rebalancing: {e}")
                    else:
                        # Buy
                        shares_to_buy = int((target_value_per_position - current_value) / position['current_price'])
                        if shares_to_buy > 0:
                            try:
                                self.trader.buy(symbol, shares_to_buy)
                                self.logger.info(f"Rebalancing: Bought {shares_to_buy} shares of {symbol}")
                            except Exception as e:
                                self.trader.log_error_once(f"Error buying {symbol} for rebalancing: {e}")
            
            self.save_positions()
            self.logger.info("Portfolio rebalancing completed")
        except Exception as e:
            self.trader.log_error_once(f"Error during portfolio rebalancing: {e}")

    def is_trading_hours(self):
        now = datetime.now(self.trader.ny_tz)
        return (now.weekday() < 5 and  # Monday to Friday
                self.market_open_time <= now.time() < self.market_close_time)

    def check_pattern_day_trader_status(self):
        if len(self.day_trades) >= 4 and self.trader.account_balance >= 25000:
            if not self.is_pattern_day_trader:
                self.logger.info("Account now qualifies as a Pattern Day Trader.")
            self.is_pattern_day_trader = True
        elif self.trader.account_balance < 25000:
            if self.is_pattern_day_trader:
                self.logger.warning("Account balance has fallen below $25,000. "
                                    "Pattern Day Trading restrictions may apply.")
            self.is_pattern_day_trader = False

    def get_remaining_day_trades(self):
        if self.trader.account_balance >= 25000:
            return "Unlimited"
        else:
            return max(0, 3 - len(self.day_trades))

    def reset_day_trades(self):
        self.day_trades = []