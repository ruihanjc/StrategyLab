from abc import ABC, abstractmethod
from typing import Dict, List
from strategy_core.sysobjects import Instrument
from strategy_data.database import ArcticReader

import requests
import pandas as pd
import datetime


class BaseExtractor(ABC):
    def __init__(self, instrument: Instrument):
        self.service = instrument.asset_class  # Service
        self.ticker = instrument.ticker  # Ticker

    @abstractmethod
    def run(self):
        """Main execution method"""
        pass

    @abstractmethod
    def process_data(self):
        """Process the raw data into DataFrame"""
        pass

    def get_history(self) -> (datetime, datetime, str, bool):
        check_end, has_history = ArcticReader().has_historical_range(self.service.lower(), self.ticker)

        if has_history:
            start_date = check_end.date()
            end_date = datetime.date.today()
        else:
            start_date = datetime.date(2020, 1, 1)
            end_date = datetime.date.today()

        return start_date, end_date, has_history


class BaseClientExtractor(BaseExtractor, ABC):
    def __init__(self, instrument: Instrument):
        super().__init__(instrument)

    @abstractmethod
    def get_client(self):
        pass

    @staticmethod
    def get_duration(start_date: datetime.date, end_date: datetime.date, has_history : bool) -> (str, bool):
        if has_history:
            days = (end_date - start_date).days.__str__()
            return f"{days} D", True

        years = 0
        while end_date > start_date:
            years += 1
            end_date = datetime.date(end_date.year - 1, end_date.month, end_date.day)

        return f"{years} Y", False


class BaseRestExtractor(BaseExtractor, ABC):
    """Base class for all REST extractors"""

    def __init__(self, instrument: Instrument, api_config: dict[str, str]) -> None:
        super().__init__(instrument)
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

    @abstractmethod
    def get_eod_data(self, ticker: str, start: datetime, to: datetime):
        """Get end of day data for a ticker"""
        pass
