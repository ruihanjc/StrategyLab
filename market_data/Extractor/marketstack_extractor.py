from calendar import month
from turtledemo.penrose import start

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

        start_date = end_date = None

        if check_historical[0].date() is not datetime.datetime(2010, 1, 1, 12, 0, 0):
            start_date = datetime.datetime(2020, 1, 1, 12, 0, 0)
        if check_historical[1].date() is not datetime.datetime.today():
            end_date = datetime.datetime.today()

        datapoints = []
        while start_date < end_date:
            temp_end = start_date+datetime.timedelta(days=90)
            response = self.get_eod_data(self.ticker, start_date.date(), temp_end.date())
            range_points = response['data']
            for values in range_points:
                datapoints.append(
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
                )
                
            start_date = temp_end + datetime.timedelta(days=1)


        return self.create_dataframe(datapoints)

    def get_eod_data(self, ticker, start, end):
        url = f"{self.mktstack_eod_base_url}symbols={ticker}&access_key={self.api_key}&date_from={start}&date_to={end}"
        return self.make_request(url)
