import arcticdb as adb
import pandas as pd
import argparse
from pathlib import Path
import os
import sys


class ArcticReader:
    def __init__(self):
        current_dir = Path(os.getcwd())
        self.arctic_dir = current_dir.parent.parent / 'arcticdb'
        self.arctic_uri = f"lmdb://{self.arctic_dir}"
        self.arctic = adb.Arctic(self.arctic_uri)

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
        data = lib.read(symbol).data
        if 'date' in data.columns:
            earliest_date = data['date'].min()
            latest_date = data['date'].max()
            return earliest_date, latest_date
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