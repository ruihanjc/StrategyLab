"""
Instrument data structures for backtesting
"""

import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class Instrument:
    source: str
    ticker: str
    asset_class: str


class InstrumentList:

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
        self._instruments[instrument.ticker] = instrument

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
        data = []
        for instrument in self._instruments.values():
            data.append({
                'ticker': instrument.ticker,
                'asset_class': instrument.asset_class,
                'source': instrument.source
            })
        return pd.DataFrame(data)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'InstrumentList':
        instruments = []
        for _, row in df.iterrows():
            instrument = Instrument(
                ticker=row['ticker'],
                asset_class=row['asset_class'],
                source=row["source"]
            )
            instruments.append(instrument)
        return cls(instruments)
