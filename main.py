import streamlit as st
import os
import sys
import time
from threading import Thread

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from trader_logic.trader import Trader
from trader_logic.logging_config import setup_logging
from gui.dashboard import create_dashboard

setup_logging()

@st.cache_resource
def get_trader():
    return Trader(update_interval=60)

def run_trader(trader, stop):
    try:
        trader.start()
        while not stop():
            time.sleep(1)
    finally:
        trader.stop()
        st.session_state.trading = False

def main():
    st.set_page_config(page_title="AlgoTrader Dashboard", layout="wide")
    
    trader = get_trader()

    if 'trading' not in st.session_state:
        st.session_state.trading = False

    create_dashboard(trader)

    # Start/Stop trading logic
    if st.session_state.trading:
        if st.sidebar.button("Stop Trading", key="stop_trading_main"):
            st.session_state.trading = False
            st.experimental_rerun()
    else:
        if st.sidebar.button("Start Trading", key="start_trading_main"):
            st.session_state.trading = True
            stop_flag = False
            thread = Thread(target=run_trader, args=(trader, lambda: stop_flag))
            thread.start()
            st.experimental_rerun()

if __name__ == "__main__":
    main()