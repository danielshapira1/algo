import threading
import time
from datetime import datetime, timedelta
from utils import api, fetch_latest_price, wait_for_order_fill, log_transaction, is_trading_hours, fetch_data, supertrend, moving_average, add_indicators, train_ml_model, get_time_until_ny_930

def manage_open_positions(api):
    positions = api.list_positions()
    for position in positions:
        symbol = position.symbol
        quantity = float(position.qty)
        current_price = fetch_latest_price(api, symbol)
        if current_price is None:
            print(f"Skipping {symbol} due to price fetch error")
            continue
        average_entry = float(position.avg_entry_price)
        unrealized_plpc = (current_price - average_entry) / average_entry

        # Fetch recent data for analysis
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        df = fetch_data(symbol, start_date, end_date)
        
        if df is not None and not df.empty:
            df = supertrend(df, 10, 3.0)
            df = moving_average(df, 20, 'sma')
            df = add_indicators(df)
            model = train_ml_model(df)
            df['ml_prediction'] = model.predict(df[['ma', 'rsi', 'macd', 'bollinger_hband', 'bollinger_lband', 'atr']])
            
            last_row = df.iloc[-1]

            # Dynamic trailing stop loss
            atr = last_row['atr']
            trailing_stop_loss = max(average_entry * 0.95, current_price - 2 * atr)

            # Take profit levels
            take_profit_1 = average_entry * 1.05  # 5% profit
            take_profit_2 = average_entry * 1.10  # 10% profit
            take_profit_3 = average_entry * 1.20  # 20% profit

            # Check for sell signals
            should_sell = (
                current_price <= trailing_stop_loss or
                not last_row['in_uptrend'] or
                last_row['ml_prediction'] == 0 or
                unrealized_plpc <= -0.05  # 5% stop loss
            )

            # Implement partial profit-taking strategy
            if current_price >= take_profit_3:
                sell_quantity = quantity
            elif current_price >= take_profit_2:
                sell_quantity = quantity * 0.5
            elif current_price >= take_profit_1:
                sell_quantity = quantity * 0.25
            elif should_sell:
                sell_quantity = quantity
            else:
                sell_quantity = 0

            if sell_quantity > 0:
                try:
                    order = api.submit_order(
                        symbol=symbol,
                        qty=sell_quantity,
                        side='sell',
                        type='limit',
                        time_in_force='day',
                        limit_price=current_price * 0.99  # Set limit price 1% below current price
                    )
                    print(f"Placing order to sell {sell_quantity} shares of {symbol} at ${current_price * 0.99:.2f}")

                    filled_order = wait_for_order_fill(order.id)
                    if filled_order:
                        fill_price = float(filled_order.filled_avg_price)
                        log_transaction('sell', symbol, sell_quantity, fill_price, datetime.now())
                        print(f"Sold at ${fill_price:.2f}. P/L: {(fill_price - average_entry) * sell_quantity:.2f}")
                    else:
                        print(f"Sell order for {symbol} was not filled within the timeout period.")
                except Exception as e:
                    print(f"Error executing sell order for {symbol}: {str(e)}")

            # Implement position sizing strategy
            account = api.get_account()
            buying_power = float(account.buying_power)
            if last_row['in_uptrend'] and last_row['ml_prediction'] == 1 and unrealized_plpc > 0:
                try:
                    additional_quantity = min(buying_power / current_price, quantity * 0.5)
                    if additional_quantity > 0:
                        order = api.submit_order(
                            symbol=symbol,
                            qty=additional_quantity,
                            side='buy',
                            type='limit',
                            time_in_force='day',
                            limit_price=current_price * 1.01  # Set limit price 1% above current price
                        )
                        print(f"Placing order to buy additional {additional_quantity} shares of {symbol} at ${current_price * 1.01:.2f}")
                except Exception as e:
                    print(f"Error executing buy order for {symbol}: {str(e)}")

def monitor_positions(api):
    while True:
        try:
            time_until_target = get_time_until_ny_930()
            hours, remainder = divmod(time_until_target.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            if is_trading_hours(api):
                manage_open_positions(api)
            else:
                print(f"Outside trading hours opens in: {int(hours):02}:{int(minutes):02}:{int(seconds):02}. Skipping position management.")
            
            # Sleep for 5 minutes or until next trading session, whichever is shorter
            sleep_time = min(time_until_target.total_seconds(), 300)
            time.sleep(sleep_time)
        except Exception as e:
            print(f"Error in monitor_positions: {str(e)}")
            time.sleep(60)  # Wait for 1 minute before retrying if there's an error

def start_monitoring(api):
    monitoring_thread = threading.Thread(target=monitor_positions, args=(api,), daemon=True)
    monitoring_thread.start()
    return monitoring_thread

def ensure_monitoring_thread(monitoring_thread, api):
    if not monitoring_thread.is_alive():
        print("Monitoring thread died. Restarting...")
        monitoring_thread = start_monitoring(api)
    return monitoring_thread