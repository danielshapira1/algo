import alpaca_trade_api as tradeapi

# Alpaca API credentials
API_KEY = 'PK7FWRNN8VUZF61OD401'
SECRET_KEY = 'bZLOjm8hn2htnZK4c5F34qsqAsLwZDPPlfLSfTqD'
BASE_URL = 'https://paper-api.alpaca.markets' 

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
