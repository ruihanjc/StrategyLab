import requests
import urllib3
import yaml
import os
import re

class IBConnection:
    def __init__(self):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config', 'private_config.yaml'))
        with open(path, 'r') as file:
            content = file.read()
            content = re.sub(r'\$\{([^}]+)\}', lambda m: os.getenv(m.group(1), ''), content)
            self.config = yaml.safe_load(content)

        account_config = self.config.get("IBKR_ACCOUNT", {})
        self.base_url = account_config.get('client_portal_url', 'https://localhost:5000/v1/api/')
        
        # Disable SSL warnings for the self-signed certificate used by the gateway
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        self.session = requests.Session()
        self.session.verify = False

    def connect(self):
        """
        Checks authentication status against the Client Portal API.
        Returns True if authenticated, False otherwise.
        """
        endpoint = "iserver/auth/status"
        try:
            response = self.session.get(f"{self.base_url}{endpoint}")
            if response.status_code == 200:
                auth_status = response.json()
                if auth_status.get("loggedIn"):
                    print("Successfully connected and authenticated with IBKR Client Portal.")
                    return True
                else:
                    print("Connected to IBKR Client Portal, but not logged in.")
                    return False
            else:
                print(f"Failed to connect. Status code: {response.status_code}")
                print("Response:", response.text)
                return False
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            print("Please ensure the IBKR Client Portal Gateway is running.")
            return False

    def disconnect(self):
        """
        Logs out from the Client Portal API session.
        """
        endpoint = "logout"
        try:
            response = self.session.post(f"{self.base_url}{endpoint}", json={})
            if response.status_code == 200:
                print("Successfully logged out from IBKR Client Portal.")
            else:
                print(f"Failed to logout. Status code: {response.status_code}")
                print("Response:", response.text)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during logout: {e}")

    def get_account_summary(self):
        """
        Retrieves account information.
        """
        endpoint = "iserver/accounts"
        try:
            response = self.session.get(f"{self.base_url}{endpoint}")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get account summary. Status code: {response.status_code}")
                print("Response:", response.text)
                return None
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching account summary: {e}")
            return None

    def get_conid(self, ticker):
        """
        Gets the contract ID for a given ticker symbol.
        """
        endpoint = "iserver/secdef/search"
        params = {"symbol": ticker, "secType": "STK"}
        try:
            response = self.session.post(f"{self.base_url}{endpoint}", json=params)
            if response.status_code == 200:
                results = response.json()
                if results:
                    return results[0]['conid']
                else:
                    print(f"No conid found for ticker: {ticker}")
                    return None
            else:
                print(f"Failed to get conid. Status code: {response.status_code}")
                print("Response:", response.text)
                return None
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching conid: {e}")
            return None

    def get_historical_data(self, conid, period, bar):
        """
        Fetches historical market data.
        """
        endpoint = "iserver/marketdata/history"
        params = {"conid": conid, "period": period, "bar": bar}
        try:
            response = self.session.get(f"{self.base_url}{endpoint}", params=params)
            if response.status_code == 200:
                return response.json().get('data', [])
            else:
                print(f"Failed to get historical data. Status code: {response.status_code}")
                print("Response:", response.text)
                return []
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching historical data: {e}")
            return []
