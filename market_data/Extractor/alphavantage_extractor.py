from .base_extractor import BaseExtractor
import datetime


class AlphaVantageExtractor(BaseExtractor):
    def __init__(self, config, api_config) -> None:
        super().__init__(config, api_config)
        self.alphavantage_base_url = api_config["alphavantage_api"]
        self.file_format = "&datatype=csv"
        self.api_key = api_config["alphavantage_api_key"]

    def run(self):
        try:
            match self.service:
                case "Equity":
                    return self.process_data()
                case _:
                    raise ValueError(f"Unsupported service: {self.service}")
        except Exception as e:
            raise

    def process_data(self):
        response = self.get_eod_data(self.ticker)
        datapoints = response['Time Series (Daily)']

        records = [
            {
                'ticker': self.ticker,
                'date': date,
                'service': 'StockEquity',
                'source': 'AlphaVantage',
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': int(values['5. volume']),
                'timestamp': datetime.date.today()
            }
            for date, values in datapoints.items()
        ]

        return self.create_dataframe(records)

    def get_eod_data(self, ticker):
        url = f"{self.alphavantage_base_url}function=TIME_SERIES_DAILY&symbol={ticker}&apikey={self.api_key}"
        return self.make_request(url)