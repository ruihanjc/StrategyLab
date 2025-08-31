from ib_insync import IB
import yaml
import os

from ibapi.wrapper import *


class IBConnection():
    def __init__(self):
        path = os.path.abspath(__file__ + '/../config/private_config.yaml')
        with open(path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.ib = IB()

    def connect(self, account_type : str):
        account_config = self.config[account_type]
        if not self.ib.isConnected():
            self.ib.connect(
                account_config['ib_ipaddress'],
                account_config['ib_port'],
                clientId=1  # This can be dynamic if needed
            )
        return self.ib

    def disconnect(self):
        self.ib.disconnect()

    def get_account_summary(self, account_number):
        return self.ib.accountSummary(account_number)