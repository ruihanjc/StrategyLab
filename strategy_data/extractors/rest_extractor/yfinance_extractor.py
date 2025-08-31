from abc import ABC
from strategy_data.extractors.base_extractor import BaseRestExtractor

import yfinance as yf
import datetime


class YahooFinanceExtractor(BaseRestExtractor, ABC):
    def __init__(self, instrument, api_config=None) -> None:
        super().__init__(instrument, api_config or {})

    def run(self):
        try:
            return self.process_data()
        except Exception as e:
            raise

    def process_data(self):
        start_date, end_date, has_history = self.get_history()

        # Filter to 2020+ if no historical data
        if not has_history:
            start_date = max(start_date, datetime.date(2020, 1, 1))

        ticker_obj = yf.Ticker(f"{self.ticker}=X")
        data = ticker_obj.history(start=start_date, end=end_date)

        datapoints = []
        for date, row in data.iterrows():
            datapoints.append({
                'ticker': self.ticker,
                'service': 'forex',
                'source': 'YahooFinance',
                'date': date,
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'timestamp': datetime.date.today()
            })

        return self.create_dataframe(datapoints)

    def get_eod_data(self, ticker, start, end):
        # Required by abstract method but not used in yfinance implementation
        pass
