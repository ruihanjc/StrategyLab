from abc import abstractmethod
from strategy_broker.ib_connection import IBConnection
from ib_insync import Contract

import logging


def setup_logging():
    """Initialize logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


class IBClient(object):

    def __init__(self):
        setup_logging()
        self.ib_conn = IBConnection()
        self.ib_conn.connect()
        self.logger = logging.getLogger(__name__)

    @property
    def ib_connection(self) -> IBConnection:
        return self.ib_conn

    @abstractmethod
    def get_ib_data(self, asset_class, duration):
        pass

    @abstractmethod
    def process_bars_data(self, bars, ticker, service):
        pass
