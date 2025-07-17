"""
Instrument data structures for backtesting
Based on pysystemtrade instrument handling
"""

import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Instrument:
    """
    Represents a trading instrument with metadata
    """
    name: str
    currency: str = "USD"
    asset_class: str = "equity"
    point_size: float = 1.0
    description: str = ""
    meta_data: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.description == "":
            self.description = f"{self.asset_class.title()} instrument: {self.name}"
    
    def __str__(self):
        return f"Instrument({self.name}, {self.currency}, {self.asset_class})"
    
    def __repr__(self):
        return self.__str__()


class InstrumentList:
    """
    Container for multiple instruments with validation and filtering
    """
    
    def __init__(self, instruments: Union[List[Instrument], Dict[str, Instrument]] = None):
        self._instruments = {}
        
        if instruments is not None:
            if isinstance(instruments, list):
                for instrument in instruments:
                    self.add_instrument(instrument)
            elif isinstance(instruments, dict):
                for name, instrument in instruments.items():
                    self.add_instrument(instrument)
    
    def add_instrument(self, instrument: Instrument):
        """Add an instrument to the list"""
        if not isinstance(instrument, Instrument):
            raise ValueError("Must be an Instrument object")
        self._instruments[instrument.name] = instrument
    
    def get_instrument(self, name: str) -> Optional[Instrument]:
        """Get instrument by name"""
        return self._instruments.get(name)
    
    def remove_instrument(self, name: str):
        """Remove instrument by name"""
        if name in self._instruments:
            del self._instruments[name]
    
    def get_instrument_list(self) -> List[str]:
        """Get list of instrument names"""
        return list(self._instruments.keys())
    
    def get_instruments_by_asset_class(self, asset_class: str) -> List[Instrument]:
        """Filter instruments by asset class"""
        return [
            instrument for instrument in self._instruments.values()
            if instrument.asset_class == asset_class
        ]
    
    def get_instruments_by_currency(self, currency: str) -> List[Instrument]:
        """Filter instruments by currency"""
        return [
            instrument for instrument in self._instruments.values()
            if instrument.currency == currency
        ]
    
    def __len__(self):
        return len(self._instruments)
    
    def __iter__(self):
        return iter(self._instruments.values())
    
    def __getitem__(self, name: str):
        return self._instruments[name]
    
    def __contains__(self, name: str):
        return name in self._instruments
    
    def __str__(self):
        return f"InstrumentList({len(self._instruments)} instruments)"
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        data = []
        for instrument in self._instruments.values():
            data.append({
                'name': instrument.name,
                'currency': instrument.currency,
                'asset_class': instrument.asset_class,
                'point_size': instrument.point_size,
                'description': instrument.description
            })
        return pd.DataFrame(data)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'InstrumentList':
        """Create InstrumentList from DataFrame"""
        instruments = []
        for _, row in df.iterrows():
            instrument = Instrument(
                name=row['name'],
                currency=row.get('currency', 'USD'),
                asset_class=row.get('asset_class', 'equity'),
                point_size=row.get('point_size', 1.0),
                description=row.get('description', '')
            )
            instruments.append(instrument)
        return cls(instruments)


def create_sample_instruments() -> InstrumentList:
    """Create sample instruments for testing"""
    instruments = [
        Instrument("AAPL", "USD", "equity", 1.0, "Apple Inc."),
        Instrument("GOOGL", "USD", "equity", 1.0, "Alphabet Inc."),
        Instrument("MSFT", "USD", "equity", 1.0, "Microsoft Corp."),
        Instrument("TSLA", "USD", "equity", 1.0, "Tesla Inc."),
        Instrument("SPY", "USD", "etf", 1.0, "SPDR S&P 500 ETF"),
        Instrument("QQQ", "USD", "etf", 1.0, "Invesco QQQ Trust"),
        Instrument("EURUSD", "USD", "forex", 100000.0, "EUR/USD Currency Pair"),
        Instrument("GBPUSD", "USD", "forex", 100000.0, "GBP/USD Currency Pair"),
    ]
    return InstrumentList(instruments)