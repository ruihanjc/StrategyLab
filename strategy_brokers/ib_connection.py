from ib_insync import IB
import yaml
import os


class IBConnection:
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
