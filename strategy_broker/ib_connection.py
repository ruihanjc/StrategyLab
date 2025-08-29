from ib_insync import IB
import yaml
import os

from ibapi.wrapper import *


class IBConnection():
    def __init__(self, config_path='private_config.yaml'):
        path = os.path.abspath(__file__ + '/../private_config.yaml')
        with open(path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.ib = IB()

    def connect(self):
        if not self.ib.isConnected():
            self.ib.connect(
                self.config['ib_ipaddress'],
                self.config['ib_port'],
                clientId=1  # This can be dynamic if needed
            )
        return self.ib

    def disconnect(self):
        self.ib.disconnect()

    def get_account_summary(self, account_number):
        return self.ib.accountSummary(account_number)



if __name__ == "__main__":

    app = IBConnection()
    ib = app.connect()

    mycontract = Contract()
    mycontract.symbol = "AAPL"
    mycontract.secType = "STK"
    mycontract.exchange = "SMART"
    mycontract.currency = "USD"
    app.reqHistoricalData(app.nextId(), mycontract, "20240523 16:00:00 US/Eastern", "1 D", "1 hour", "TRADES", 1, 1, False, [])