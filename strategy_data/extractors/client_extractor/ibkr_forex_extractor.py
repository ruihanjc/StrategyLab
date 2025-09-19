from abc import ABC
from strategy_broker.clients import ib_forex_client
from strategy_data.extractors.base_extractor import BaseClientExtractor

import datetime


class IBKRForexExtractor(BaseClientExtractor, ABC):
    def __init__(self, config) -> None:
        super().__init__(config)

    def run(self):
        try:
            return self.process_data()
        except Exception as e:
            raise

    def get_client(self):
        return ib_forex_client.IBForexClient()

    def process_data(self):
        client = self.get_client()
        try:
            start_date, end_date, has_history = self.get_history()
            duration_str, filter_by_date = self.get_duration(start_date, end_date, has_history)
            return client.get_ib_data(self.ticker, duration_str, filter_by_date)
        finally:
            if hasattr(client, 'ib_connection'):
                client.ib_connection.disconnect()
