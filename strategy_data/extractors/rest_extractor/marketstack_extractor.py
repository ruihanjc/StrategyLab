from abc import ABC
from strategy_data.extractors.base_extractor import BaseRestExtractor

import datetime


class MarketStackExtractor(BaseRestExtractor, ABC):
    def __init__(self, config, api_config) -> None:
        super().__init__(config, api_config)
        self.mktstack_eod_base_url = api_config["mktstack_api"]
        self.api_key = api_config["mktstack_api_key"]

    def run(self):
        try:
            return self.process_data()
        except Exception as e:
            raise

    def process_data(self):
        start_date, end_date, has_history = self.get_history()

        datapoints = []
        while start_date < end_date:
            temp_end = start_date + datetime.timedelta(days=90)
            response = self.get_eod_data(self.ticker, start_date, temp_end)
            range_points = response['data']
            for values in range_points:
                datapoints.append(
                    {
                        'ticker': self.ticker,
                        'date': values['date'],
                        'service': 'Equity',
                        'source': 'MarketStack',
                        'open': float(values['open']) if values['open'] else 0,
                        'high': float(values['high']) if values['high'] else 0,
                        'low': float(values['low']) if values['low'] else 0,
                        'close': float(values['close']) if values['close'] else 0,
                        'volume': int(values['volume']) if values['volume'] else 0,
                        'timestamp': datetime.date.today()
                    }
                )

            start_date = temp_end + datetime.timedelta(days=1)

        if not datapoints:
            return

        return self.create_dataframe(datapoints)

    def get_eod_data(self, ticker, start, end):
        url = f"{self.mktstack_eod_base_url}?symbols={ticker}&access_key={self.api_key}&date_from={start}&date_to={end}"
        return self.make_request(url)
