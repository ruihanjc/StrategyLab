import arcticdb as adb
import pandas as pd
from typing import List, Optional


class ArcticReader:
    def __init__(self, arctic_path: str):
        self.arctic_uri = f"lmdb://{arctic_path}"
        self.arctic = adb.Arctic(self.arctic_uri)

    def get_libraries(self) -> List[str]:
        """Get all available libraries"""
        return self.arctic.list_libraries()

    def get_symbols(self, library: str) -> List[str]:
        """Get all symbols in a library"""
        lib = self.arctic.get_library(library)
        return lib.list_symbols()

    def get_data(self, library: str, symbol: str,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None) -> pd.DataFrame:
        """Get data for a symbol with optional date range"""
        lib = self.arctic.get_library(library)
        data = lib.read(symbol).data

        if start_date:
            data = data[data['date'] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data['date'] <= pd.to_datetime(end_date)]

        return data
