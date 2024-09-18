import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_dashboard(trader):
    # Custom CSS for a modern look
    st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        color: #FFFFFF;
        background-color: #4CAF50;
        border-radius: 5px;
    }
    .stMetric {
        background-color: #2D2D2D;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("AlgoTrader Dashboard")

    # Sidebar
    st.sidebar.header("Controls")
    if st.sidebar.button("Start Trading", key="start_trading_dashboard"):
        if not trader.running:
            trader.start()
            st.sidebar.success("Trading started!")
        else:
            st.sidebar.warning("Trading is already running!")
    if st.sidebar.button("Stop Trading", key="stop_trading_dashboard"):
        if trader.running:
            trader.stop()
            st.sidebar.error("Trading stopped!")
        else:
            st.sidebar.warning("Trading is not running!")

    # Add a trading status indicator
    trading_status = "Running" if trader.running else "Stopped"
    st.sidebar.metric("Trading Status", trading_status)

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Account Balance", f"${trader.account_balance:.2f}")
    with col2:
        st.metric("Number of Positions", len(trader.position_manager.positions))

    # Current Positions
    st.subheader("Current Positions")
    positions = trader.position_manager.positions
    if positions:
        positions_df = pd.DataFrame({
            'Stock': positions.keys(),
            'Size': [pos['volume'] for pos in positions.values()],
            'Worth': [float(pos['current_price']) * float(pos['volume']) for pos in positions.values()]
        })
        positions_df['Worth'] = positions_df['Worth'].apply(lambda x: f"${x:,.2f}")
        positions_df['Total'] = positions_df['Worth']
        st.table(positions_df)
        total_worth = sum(float(pos['current_price']) * float(pos['volume']) for pos in positions.values())
        st.metric("Total Portfolio Worth", f"${total_worth:.2f}")
    else:
        st.info("No open positions")

    # Performance Metrics
    st.subheader("Performance Metrics")
    performance = trader.performance_tracker.get_performance_summary()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return", performance['Total Return'])
    col2.metric("Sharpe Ratio", performance['Sharpe Ratio'])
    col3.metric("Max Drawdown", performance['Max Drawdown'])

if __name__ == "__main__":
    st.error("This file should not be run directly. Please run main.py instead.")