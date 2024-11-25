import streamlit as st
import pandas as pd
from Utils.arcticdb_reader import ArcticReader
from Utils.chart_generator import ChartGenerator
from datetime import datetime, timedelta
import os
from pathlib import Path

# Initialize
@st.cache_resource
def init_arctic():
    current_dir = Path(os.getcwd())
    arctic_dir = current_dir.parent.parent / 'arcticdb'
    return ArcticReader(arctic_dir)


def main():
    st.title("Market Data Visualization")

    # Initialize ArcticReader
    arctic = init_arctic()

    # Sidebar controls
    st.sidebar.header("Controls")

    # Library selection
    libraries = arctic.get_libraries()
    selected_lib = st.sidebar.selectbox("Select Library", libraries)

    # Symbol selection
    symbols = arctic.get_symbols(selected_lib)
    selected_symbol = st.sidebar.selectbox("Select Symbol", symbols)

    # Date range selection
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(start_date, end_date),
        max_value=end_date
    )

    if len(date_range) == 2:
        start_date, end_date = date_range

        # Get data
        df = arctic.get_data(selected_lib, selected_symbol,
                             start_date=start_date,
                             end_date=end_date)

        # Display basic info
        st.subheader("Data Summary")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Days", len(df))
        with col2:
            st.metric("Price Change",
                      f"{((df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100):.2f}%")
        with col3:
            st.metric("Avg Volume",
                      f"{df['volume'].mean():,.0f}")

        # Candlestick chart
        st.subheader("Price Chart")
        fig = ChartGenerator.create_candlestick(
            df, f"{selected_symbol} Price Chart"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Returns analysis
        st.subheader("Returns Analysis")
        returns_charts = ChartGenerator.create_returns_analysis(df)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(returns_charts['distribution'],
                            use_container_width=True)
        with col2:
            st.plotly_chart(returns_charts['qq_plot'],
                            use_container_width=True)

        # Raw data
        st.subheader("Raw Data")
        st.dataframe(df)


if __name__ == "__main__":
    main()