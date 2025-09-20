from ib_insync import IB
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
