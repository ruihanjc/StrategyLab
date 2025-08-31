import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import arcticdb as adb
import os


# Initialize ArcticDB connection
def get_arctic_connection():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    arctic_path = os.path.join(project_dir, 'arcticdb')
    return adb.Arctic(f"lmdb://{arctic_path}")


def main():
    st.title("📈 Market Data Viewer")
    st.sidebar.header("Data Selection")

    # Get ArcticDB connection
    arctic = get_arctic_connection()

    try:
        # Get available libraries
        libraries = arctic.list_libraries()
        if not libraries:
            st.error("No libraries found in ArcticDB")
            return

        # Library selection
        default_index = libraries.index('equity') if 'equity' in libraries else 0
        selected_lib = st.sidebar.selectbox("Select Class", libraries, index=default_index)

        # Get symbols from selected library
        lib = arctic.get_library(selected_lib)

        if lib.name != "metadata":
            symbols = lib.list_symbols()

            if not symbols:
                st.error(f"No symbols found in {selected_lib} library")
                return

            # Symbol selection
            selected_symbol = st.sidebar.selectbox("Select Symbol", symbols)

            # Date range selection
            end_date = datetime.now().date()
            start_date = datetime(2020, 1, 1, 12, 0, 0).date()

            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(start_date, end_date),
                max_value=end_date
            )

            if len(date_range) == 2:
                start_date, end_date = date_range

                # Load data
                with st.spinner('Loading data...'):
                    try:
                        data = lib.read(selected_symbol).data

                        # Convert date column if exists
                        if 'date' in data.columns:
                            data['date'] = pd.to_datetime(data['date'])

                            if data['date'][0] > pd.Timestamp(start_date):
                                start_date = data['date'][1]

                            data = data[(data['date'] >= pd.Timestamp(start_date)) &
                                        (data['date'] <= pd.Timestamp(end_date))]

                            data.set_index('date', inplace=True)

                        if data.empty:
                            st.warning("No data found for selected date range")
                            return

                        # Display basic info
                        st.subheader("Data Summary")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Total Records", len(data))
                        with col2:
                            if 'close' in data.columns:
                                price_change = ((data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100)
                                st.metric("Price Change", f"{price_change:.2f}%")
                        with col3:
                            if 'volume' in data.columns:
                                st.metric("Avg Volume", f"{data['volume'].mean():,.0f}")

                        # Create price chart
                        st.subheader(f"{selected_symbol} Price Chart")

                        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                            # Candlestick chart
                            fig = go.Figure(data=go.Candlestick(
                                x=data.index,
                                open=data['open'],
                                high=data['high'],
                                low=data['low'],
                                close=data['close'],
                                name=selected_symbol
                            ))
                            fig.update_layout(
                                title=f"{selected_symbol} Candlestick Chart",
                                yaxis_title="Price",
                                xaxis_title="Date",
                                template="plotly_white"
                            )
                        elif 'close' in data.columns:
                            # Line chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data['close'],
                                mode='lines',
                                name='Close Price'
                            ))
                            fig.update_layout(
                                title=f"{selected_symbol} Price Chart",
                                yaxis_title="Price",
                                xaxis_title="Date",
                                template="plotly_white"
                            )
                        else:
                            st.error("No price columns found in data")
                            return

                        st.plotly_chart(fig, use_container_width=True)

                        # Volume chart if available
                        if 'volume' in data.columns:
                            st.subheader("Volume")
                            vol_fig = go.Figure()
                            vol_fig.add_trace(go.Bar(
                                x=data.index,
                                y=data['volume'],
                                name='Volume'
                            ))
                            vol_fig.update_layout(
                                title=f"{selected_symbol} Volume",
                                yaxis_title="Volume",
                                xaxis_title="Date",
                                template="plotly_white"
                            )
                            st.plotly_chart(vol_fig, use_container_width=True)

                        # Returns analysis if price data available
                        if 'close' in data.columns:
                            st.subheader("Returns Analysis")
                            returns = data['close'].pct_change().dropna()

                            col1, col2 = st.columns(2)

                            with col1:
                                # Returns distribution
                                hist_fig = px.histogram(
                                    x=returns,
                                    nbins=50,
                                    title="Daily Returns Distribution"
                                )
                                hist_fig.update_layout(template="plotly_white")
                                st.plotly_chart(hist_fig, use_container_width=True)

                            with col2:
                                # Cumulative returns
                                cum_returns = (1 + returns).cumprod()
                                cum_fig = go.Figure()
                                cum_fig.add_trace(go.Scatter(
                                    x=cum_returns.index,
                                    y=cum_returns,
                                    mode='lines',
                                    name='Cumulative Returns'
                                ))
                                cum_fig.update_layout(
                                    title="Cumulative Returns",
                                    yaxis_title="Cumulative Return",
                                    xaxis_title="Date",
                                    template="plotly_white"
                                )
                                st.plotly_chart(cum_fig, use_container_width=True)

                        # Raw data
                        with st.expander("Raw Data"):
                            st.dataframe(data)

                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
        else:
            st.subheader("Raw Data")
            st.dataframe(lib.read("instrument_contracts").data)

    except Exception as e:
        st.error(f"Error connecting to ArcticDB: {str(e)}")


if __name__ == "__main__":
    main()
