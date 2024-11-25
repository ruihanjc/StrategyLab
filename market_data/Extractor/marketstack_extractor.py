from .base_extractor import BaseExtractor
from market_data.Database.arcticdb_reader import ArcticReader
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

        check_historical = ArcticReader().get_historical_range(self.service.lower(), self.ticker)

        if check_historical[0].date() is not datetime.datetime.strptime("2010-01-01","Y%-m%-d%").date():
            check_historical[0] = datetime.datetime.strptime("2010-01-01","Y%-m%-d%").date()

        if check_historical[1].date() is not datetime.datetime.today():
            check_historical[1] = datetime.datetime.today()

        datapoints = []

        while check_historical[0] < check_historical[1]:
            response = self.get_eod_data(self.ticker, check_historical[0], check_historical[1])

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

    def get_eod_data(self, ticker, start, end):
        url = f"{self.mktstack_eod_base_url}symbols={ticker}&access_key={self.api_key}&date_from={start}&date_to={end}"
        return self.make_request(url)