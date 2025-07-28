from abc import ABC, abstractmethod
import requests
import pandas as pd
from typing import Dict, Any
import datetime



class BaseExtractor(ABC):
    def __init__(self, config: Any):
        self.service = config[0] #Service
        self.ticker = config[2] #Ticker

    @abstractmethod
    def run(self):
        """Main execution method"""
        pass

    @abstractmethod
    def process_data(self):
        """Process the raw data into DataFrame"""
        pass

    @abstractmethod
    def get_eod_data(self, ticker: str, start: datetime, to: datetime):
        """Get end of day data for a ticker"""
        pass



class BaseRestExtractor(BaseExtractor, ABC):
    """Base class for all REST extractors"""

    def __init__(self, config: Any, api_config: dict[str, str]) -> None:
        super().__init__(config)
        self.api_config = api_config

    @staticmethod
    def make_request(url: str, headers: Dict = None, payload: str = "") -> Dict:
        """Make HTTP request with error handling"""
        try:
            headers = headers or {}
            response = requests.request("GET", url, headers=headers, data=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

    @staticmethod
    def create_dataframe(records: list) -> pd.DataFrame:
        """Create and format DataFrame from records"""
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')