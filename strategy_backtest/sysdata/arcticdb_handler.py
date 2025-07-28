import pandas as pd
from datetime import datetime
import arcticdb as adb
import argparse
from pathlib import Path
import os
import sys
from market_data.database.arctic_connection import get_arcticdb_connection


class ArcticdDBHandler:
    def __init__(self, type, dir):
        self.arctic = get_arcticdb_connection(dir)
        print(self.arctic.list_libraries())
        self.library = self.arctic.get_library(type)

    def load_from_arcticdb(self, symbols, start_date, end_date):
        # Connect to local ArcticDB

        data_dict = {}
        for symbol in symbols:
            try:
                # Read versioned item from ArcticDB
                item = self.library.read(symbol)
                df = item.data

                # Filter by date
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                else:
                    # Convert string date column to datetime if needed
                    date_col = next((col for col in df.columns if 'date' in col))
                    if date_col:
                        df[date_col] = pd.to_datetime(df[date_col])
                        df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
                        df.set_index(date_col, inplace=True)

                data_dict[symbol] = df
                print(f"Loaded {symbol} data: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {symbol}: {str(e)}")

        return data_dict

if __name__ == "__main__":
    symbols = ['PLTR']
    start_date = '2020-01-01'
    end_date = '2024-12-31'

    arcticlibrary = ArcticdDBHandler('equity', f"{Path(os.getcwd()).parent.parent}/arcticdb")

    print(arcticlibrary.arctic.get_library('equity').read('TSLA'))

    data = arcticlibrary.load_from_arcticdb(symbols, start_date, end_date)
