import numpy as np
from datetime import datetime
import logging

class PerformanceTracker:
    def __init__(self):
        self.daily_returns = []
        self.trades = []
        self.peak_value = 0
        self.start_value = 0
        self.start_date = datetime.now()
        self.logger = logging.getLogger(__name__)

    def update(self, current_value):
        if self.start_value == 0:
            self.start_value = current_value
        if self.peak_value == 0:
            self.peak_value = current_value
        
        if len(self.daily_returns) == 0:
            daily_return = 0
        else:
            daily_return = (current_value - self.peak_value) / self.peak_value
        
        self.daily_returns.append(daily_return)
        
        if current_value > self.peak_value:
            self.peak_value = current_value

    def add_trade(self, entry_price, exit_price, entry_date, exit_date):
        trade_return = (exit_price - entry_price) / entry_price
        self.trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'return': trade_return
        })
        self.logger.info(f"Trade added: Entry: {entry_date}, Exit: {exit_date}, Return: {trade_return:.2%}")

    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        if len(self.daily_returns) < 2:
            return 0
        
        returns = np.array(self.daily_returns)
        excess_returns = returns - (risk_free_rate / 252)  # Assuming 252 trading days in a year
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe_ratio

    def calculate_max_drawdown(self):
        peak = self.start_value
        max_dd = 0
        
        for value in self.daily_returns:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd

    def calculate_win_rate(self):
        if not self.trades:
            return 0
        
        winning_trades = sum(1 for trade in self.trades if trade['return'] > 0)
        return winning_trades / len(self.trades)

    def get_performance_summary(self):
        if not self.daily_returns or self.start_value == 0:
            return {
                'Total Return': "0.00%",
                'Sharpe Ratio': "0.00",
                'Max Drawdown': "0.00%",
                'Win Rate': "0.00%",
                'Number of Trades': 0
            }

        total_return = (self.peak_value / self.start_value) - 1
        sharpe_ratio = self.calculate_sharpe_ratio()
        max_drawdown = self.calculate_max_drawdown()
        win_rate = self.calculate_win_rate()
        
        return {
            'Total Return': f"{total_return:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Win Rate': f"{win_rate:.2%}",
            'Number of Trades': len(self.trades)
        }

    def log_performance(self):
        summary = self.get_performance_summary()
        self.logger.info("Performance Summary:")
        for key, value in summary.items():
            self.logger.info(f"{key}: {value}")