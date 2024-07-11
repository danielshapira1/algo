import yfinance as yf
import time

def fetch_google_stock_price():
    ticker = yf.Ticker('GOOG')
    while True:
        try:
            # Fetch the current stock price
            current_price = ticker.fast_info['last_price']
            print(f"Google Stock Price (GOOG): ${current_price:.2f}")
        except KeyError:
            print("Could not fetch the current price.")
        except Exception as e:
            print(f"An error occurred: {e}")
        time.sleep(5)  # Wait for 5 seconds

if __name__ == "__main__":
    fetch_google_stock_price()
