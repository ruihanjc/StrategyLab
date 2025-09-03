from abc import ABC
from strategy_broker.clients import ib_equity_client
from strategy_data.extractors.base_extractor import BaseClientExtractor


class IBKREquityExtractor(BaseClientExtractor, ABC):
    def __init__(self, config) -> None:
        super().__init__(config)

    def run(self):
        try:
            return self.process_data()
        except Exception as e:
            raise

    def get_client(self):
        return ib_equity_client.IBEquityClient()

    def process_data(self):
        client = self.get_client()

        # get_history returns: start_date, end_date, has_history
        _, _, has_history = self.get_history()

        # If there is history, we are NOT backfilling.
        # If there is NO history, we ARE backfilling.
        should_backfill = not has_history

        return client.get_ib_data(self.ticker, self.service, backfill=should_backfill)
