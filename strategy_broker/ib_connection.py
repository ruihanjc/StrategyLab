from ib_insync import IB, Stock
from dotenv import load_dotenv
import yaml
import os
import re
import time

from ibapi.wrapper import *


class IBConnection():
    def __init__(self):
        load_dotenv()
        path = os.path.abspath(__file__ + '/../config/private_config.yaml')
        with open(path, 'r') as file:
            content = file.read()
            # Substitute environment variables ${VAR_NAME} using same method as ConfigManager
            content = re.sub(r'\$\{([^}]+)}', lambda m: os.getenv(m.group(1), ''), content)
            self.config = yaml.safe_load(content)
        self.account_number = None
        self.ib = IB()
        self._client_id = None

    def set_client_id(self, client_id):
        self._client_id = client_id

    def _generate_client_id(self):
        if self._client_id is None:
            self._client_id = int((time.time() * 1000) % 10000) + os.getpid() % 1000
        return self._client_id

    def connect(self):
        if not self.ib.isConnected():
            account_config = self.config["IBKR_ACCOUNT"]
            client_id = self._generate_client_id()
            self.ib.connect(
                account_config['ib_ipaddress'],
                account_config['ib_port'],
                clientId=client_id
            )
            self.account_number = account_config["broker_account"]
        return self.ib

    def disconnect(self):
        self.ib.disconnect()

    def get_account_summary(self):
        account_summary = self.ib.accountSummary(self.account_number)
        return {
            'account_summary': account_summary
        }

    def get_current_price(self, symbol):
        """Get current market price for a symbol."""
        try:
            if not self.ib.isConnected():
                self.connect()

            # Create a stock contract
            contract = Stock(symbol, 'SMART', 'USD')

            # Qualify the contract
            self.ib.qualifyContracts(contract)

            # Request market data
            ticker = self.ib.reqMktData(contract, '', False, False)

            # Wait for price data
            self.ib.sleep(2)  # Give it time to get data

            # Get the price (bid/ask midpoint or last price)
            if ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
                price = (ticker.bid + ticker.ask) / 2
            elif ticker.last and ticker.last > 0:
                price = ticker.last
            elif ticker.close and ticker.close > 0:
                price = ticker.close
            else:
                print(f"No valid price data available for {symbol}")
                return None

            # Cancel the market data subscription
            self.ib.cancelMktData(contract)

            return price

        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return None
