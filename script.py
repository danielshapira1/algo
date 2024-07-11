import alpaca_trade_api as tradeapi

# Alpaca API credentials
API_KEY = 'AK10E2CGO4W38YHIXY3R'
SECRET_KEY = 'HyrHuaUI7YeMZ83FnAFhldXm1KL9AaVx9q2ZvcUw'
BASE_URL = 'https://api.alpaca.markets'

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# Check account details to verify API credentials
try:
    account = api.get_account()
    print(f"Account status: {account.status}")
except Exception as e:
    print(f"Failed to connect to Alpaca: {e}")

# Fetch recent bar data to verify data access
try:
    bars = api.get_bars('AAPL', tradeapi.TimeFrame.Day, limit=5).df
    print(bars)
except Exception as e:
    print(f"Failed to fetch bar data: {e}")
