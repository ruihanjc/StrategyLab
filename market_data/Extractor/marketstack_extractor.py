from .base_extractor import BaseExtractor
import datetime


class MarketStackExtractor(BaseExtractor):
    def __init__(self, config, api_config) -> None:
        super().__init__(config, api_config)
        self.mktstack_eod_base_url = api_config["mktstack_api"]
        self.api_key = api_config["mktstack_api_key"]

    def run(self):
        try:
            match self.service:
                case "MarketIndex":
                    return self.get_eod_data(self.ticker)
                case "Equity":
                    return self.process_data()
                case _:
                    raise ValueError(f"Unsupported service: {self.service}")
        except Exception as e:
            raise

    def process_data(self):
        response = self.get_eod_data(self.ticker)
        datapoints = response['data']

        records = [
            {
                'ticker': self.ticker,
                'date': values['date'],
                'service': 'Equity',
                'source': 'MarketStack',
                'open': float(values['open']),
                'high': float(values['high']),
                'low': float(values['low']),
                'close': float(values['close']),
                'volume': int(values['volume']),
                'timestamp': datetime.date.today()
            }
            for values in datapoints
        ]

        return self.create_dataframe(records)

    def get_eod_data(self, ticker):
        url = f"{self.mktstack_eod_base_url}symbols={ticker}&access_key={self.api_key}"
        return self.make_request(url)