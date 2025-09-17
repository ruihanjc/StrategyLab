from ib_insync import IB
import yaml
import os
import re
import time

from ibapi.wrapper import *


class IBConnection():
    def __init__(self):
        path = os.path.abspath(__file__ + '/../config/private_config.yaml')
        with open(path, 'r') as file:
            content = file.read()
            # Substitute environment variables ${VAR_NAME} using same method as ConfigManager
            content = re.sub(r'\$\{([^}]+)}', lambda m: os.getenv(m.group(1), ''), content)
            self.config = yaml.safe_load(content)

        self.ib = IB()
        self._client_id = None

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
        return self.ib

    def disconnect(self):
        self.ib.disconnect()

    def get_account_summary(self, account_number):
        return self.ib.accountSummary(account_number)