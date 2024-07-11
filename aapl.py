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

if __name__ == "__main__":
    fetch_aapl_stock_price()
