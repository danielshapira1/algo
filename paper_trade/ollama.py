import pandas as pd
import numpy as np

from utils import api

class MeanReversionStrategy:
    def __init__(self):
        self.short_ma = 50
        self.long_ma = 200
        self.rsi_period = 14
        self.bollinger_window = 20
        self.upper_band = 2
        self.lower_band = -2
        self.max_position_size = 0.1

    def calculate_cloud(self, df):
        cloud = (df['short_ma'] - df['long_ma']) / df['close']
        return cloud

    def calculate_rsi(self, df):
        delta = df['close'].diff(1)
        up = delta.copy()
        down = delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        rsi = (up.mean() / abs(down).mean()) * 100
        return rsi

    def calculate_bollinger_band(self, df):
        bb_upper = df['close'] + (self.upper_band * df['std_dev'])
        bb_lower = df['close'] - (self.lower_band * df['std_dev'])
        return bb_upper, bb_lower

    def execute_trade(self, df, cloud, rsi, bb_upper, bb_lower):
        if cloud > 0 and rsi < 30:
            return 'long'
        elif cloud < 0 and rsi > 70:
            return 'short'
        elif cloud > self.upper_band and df['close'] > bb_upper:
            return 'long'
        elif cloud < self.lower_band and df['close'] < bb_lower:
            return 'short'

    def backtest(self):
        data = pd.read_csv('stock_data.csv')

        cloud = self.calculate_cloud(data)
        rsi = self.calculate_rsi(data)
        bb_upper, bb_lower = self.calculate_bollinger_band(data)

        trade_signal = self.execute_trade(data, cloud, rsi, bb_upper, bb_lower)

        # Backtest the strategy using the Alpaca API
        api.place_order('AAPL', 100, 'long')

if __name__ == "__main__":
    strategy = MeanReversionStrategy()
    strategy.backtest()