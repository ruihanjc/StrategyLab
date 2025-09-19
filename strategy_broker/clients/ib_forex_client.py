from strategy_broker.clients.ib_client import IBClient
from ib_insync import Forex, BarData
import pandas as pd
import datetime
from typing import List

# Define a constant for the start date filter
FILTER_START_DATE = datetime.date(2020, 1, 1)


class IBForexClient(IBClient):
    """
    A client for fetching Forex data from Interactive Brokers.
    """

    def __init__(self, bar_size_setting: str = "1 day", what_to_show: str = "MIDPOINT", timeout: int = 20,
                 format_date: int = 2):
        """
        Initializes the IBForexClient.

        Args:
            bar_size_setting: The size of the bars to request (e.g., "1 day", "1 hour").
            what_to_show: The type of data to request (e.g., "TRADES", "MIDPOINT").
            timeout: The timeout for the request in seconds.
            format_date: The format for the date in the returned data.
                         1 for yyyyMMdd hh:mm:ss, 2 for system time zone format.
        """
        super().__init__()
        self.barSizeSetting = bar_size_setting
        self.whatToShow = what_to_show
        self.timeout = timeout
        self.formatDate = format_date

    def get_ib_data(self, ticker: str, duration_str: str, filter_by_date: bool = False) -> pd.DataFrame:
        """
        Gets historical data for a given Forex ticker.

        Args:
            ticker: The Forex ticker symbol (e.g., "EURUSD").
            duration_str: The duration of the data to fetch (e.g., "1 Y", "10 D").
            filter_by_date: If True, filters the data to start from FILTER_START_DATE.

        Returns:
            A pandas DataFrame with the historical data, or an empty DataFrame on error.
        """
        try:
            contract = Forex(ticker)

            bars: List[BarData] = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration_str,
                barSizeSetting=self.barSizeSetting,
                whatToShow=self.whatToShow,
                useRTH=True,
                formatDate=self.formatDate,
                timeout=self.timeout
            )

            if not bars:
                print(f"No historical data returned for {ticker}")
                return pd.DataFrame()

            if filter_by_date:
                bars = [bar for bar in bars if bar.date >= FILTER_START_DATE]

            return self._process_bars_data(bars, ticker)

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    def _process_bars_data(self, bars: List[BarData], ticker: str) -> pd.DataFrame:
        """
        Processes the raw bar data from IB into a pandas DataFrame.

        Args:
            bars: A list of BarData objects from ib_insync.
            ticker: The ticker symbol for the data.

        Returns:
            A pandas DataFrame with the processed data.
        """
        data = [{
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
            'ticker': ticker,
            'timestamp': datetime.datetime.now(),
            'service': 'Forex',
            'source': 'IBKR'
        } for bar in bars]

        df = pd.DataFrame(data)
        return df
