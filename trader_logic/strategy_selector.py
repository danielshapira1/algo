class StrategySelector:
    def __init__(self, mean_reversion, momentum, sentiment):
        self.mean_reversion = mean_reversion
        self.momentum = momentum
        self.sentiment = sentiment

    def select_strategy(self, market_data):
        volatility = market_data['Close'].pct_change().std()
        trend = (market_data['Close'].iloc[-1] - market_data['Close'].iloc[0]) / market_data['Close'].iloc[0]
        volume = market_data['Volume'].mean()

        if volatility > 0.02 and abs(trend) < 0.05:
            return self.mean_reversion
        elif abs(trend) > 0.1 and volume > 1000000:
            return self.momentum
        else:
            return self.sentiment