from abc import ABC
from strategy_broker.clients import ib_equity_client
from strategy_data.extractors.base_extractor import BaseClientExtractor

import datetime


class IBKREquityExtractor(BaseClientExtractor, ABC):
    def __init__(self, config) -> None:
        super().__init__(config)

    def run(self):
        try:
            return self.process_data()
        except Exception as e:
            raise

    def get_client(self):
        return ib_equity_client.IBEquityClient("IBKR_DUMMY_ACCOUNT")

    def process_data(self):
        client = self.get_client()

        start_date, end_date, has_history = self.get_history()

        return client.get_ib_data(self.ticker, self.get_duration(start_date, end_date, has_history))
