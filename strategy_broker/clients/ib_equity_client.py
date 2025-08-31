from strategy_broker.clients.ib_client import IBClient
from ib_insync import Stock

import pandas as pd
import datetime


class IBEquityClient(IBClient):

    def __init__(self, account_type: str):
        super().__init__(account_type)
        self.barSizeSetting = "1 day"
        self.whatToShow = "TRADES"
        self.timeout = 20
        self.formatDate = 2

    def get_ib_data(self, ticker: str, durationStr: (str, bool)):
        try:
            stock = Stock(ticker, exchange="SMART", currency="USD")

            bars = self.ib.reqHistoricalData(
                stock,
                endDateTime='',
                durationStr=durationStr[0],
                barSizeSetting=self.barSizeSetting,
                whatToShow=self.whatToShow,
                useRTH=True,
                formatDate=self.formatDate,
                timeout=self.timeout
            )

            if not durationStr[1]:
                bars = [bar for bar in bars if bar.date >= datetime.date(2020, 1, 1)]

            return self.process_bars_data(bars, ticker)

        except Exception as e:
            print(f"This is an error: {e}")

    def process_bars_data(self, bars, ticker):
        data = []

        for bar in bars:
            data.append({
                'date': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'ticker': ticker,
                'timestamp': datetime.date.today(),
                'service': 'equity'
            })

        df = pd.DataFrame(data)
        return df
