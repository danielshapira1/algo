from sklearn.linear_model import LogisticRegression
import pandas as pd

class MLPredictor:
    def __init__(self):
        self.model = LogisticRegression()

    def prepare_features(self, data):
        data['Returns'] = data['Close'].pct_change()
        data['SMA'] = data['Close'].rolling(window=20).mean()
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        
        features = ['Returns', 'SMA', 'Volatility']
        X = data[features].dropna()
        y = (data['Close'].shift(-1) > data['Close']).dropna().astype(int)
        
        return X, y

    def train(self, data):
        X, y = self.prepare_features(data)
        self.model.fit(X, y)

    def predict(self, data):
        X, _ = self.prepare_features(data)
        return self.model.predict(X.iloc[-1].to_frame().T)[0]