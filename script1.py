import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

# Alpaca API credentials
API_KEY = 'PK7FWRNN8VUZF61OD401'
SECRET_KEY = 'bZLOjm8hn2htnZK4c5F34qsqAsLwZDPPlfLSfTqD'
BASE_URL = 'https://paper-api.alpaca.markets' 

api_key = os.getenv('APCA_API_KEY_ID_PAPER')
api_secret = os.getenv('APCA_API_SECRET_KEY_PAPER')
base_url = os.getenv('APCA_API_BASE_URL_PAPER')
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

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
