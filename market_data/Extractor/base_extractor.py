from abc import ABC, abstractmethod
import requests
import pandas as pd
from typing import Dict, Any
import datetime


class BaseExtractor(ABC):
    """Base class for all data extractors"""

    def __init__(self, config: Any, api_config: Dict[str, str]) -> None:
        self.service = config.service
        self.ticker = config.ticker
        self.api_config = api_config

    @abstractmethod
    def run(self):
        """Main execution method"""
        pass

    @abstractmethod
    def process_data(self):
        """Process the raw data into DataFrame"""
        pass

    @abstractmethod
    def get_eod_data(self, ticker: str, start: datetime, to:datetime):
        """Get end of day data for a ticker"""
        pass

    def make_request(self, url: str, headers: Dict = None, payload: str = "") -> Dict:
        """Make HTTP request with error handling"""
        try:
            headers = headers or {}
            response = requests.request("GET", url, headers=headers, data=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

    def create_dataframe(self, records: list) -> pd.DataFrame:
        """Create and format DataFrame from records"""
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')