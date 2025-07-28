import datetime
import arcticdb as adb
import pandas as pd
import argparse
from pathlib import Path
import os
from datetime import date
import sys

from wheel.macosx_libfile import read_data

from market_data.database.arctic_connection import get_arcticdb_connection


from dateutil.utils import today



class ArcticReader:
    def __init__(self):
        current_dir = Path(os.getcwd())
        arctic_dir = current_dir / 'arcticdb'
        self.arctic = get_arcticdb_connection(arctic_dir)

    def list_libraries(self):
        """List all libraries"""
        return self.arctic.list_libraries()

    def list_symbols(self, library):
        """List all symbols in a library"""
        lib = self.arctic.get_library(library)
        return lib.list_symbols()

    def get_historical_range(self, library, symbol):
        """Read data from a symbol"""
        lib = self.arctic.get_library(library)
        if (lib.has_symbol(symbol)):
            data = lib.read(symbol).data
            if 'date' in data.columns:
                latest_date = data['date'].max()
                return latest_date, True
        else:
            return datetime.date.today(), False
        return None


    def read_data(self, library, symbol, head):
        """Read data from a symbol"""
        lib = self.arctic.get_library(library)
        data = lib.read(symbol).data
        return data.head(head) if head else data


    def get_info(self, library, symbol):
        """Get information about a symbol"""
        lib = self.arctic.get_library(library)
        data = lib.read(symbol).data
        return {
            'rows': len(data),
            'columns': list(data.columns),
            'date_range': f"{data['date'].min()} to {data['date'].max()}" if 'date' in data.columns else 'N/A'
        }



if __name__ == '__main__':
    x = ArcticReader()
    print(x.read_data('equity', 'TSLA', ''))
