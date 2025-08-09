import datetime
import pandas as pd
import os

from strategy_data.database.arctic_connection import get_arcticdb_connection


class ArcticReader:
    def __init__(self):
        project_dir = os.path.abspath(__file__ + "/../../../")
        self.arctic_path = os.path.join(project_dir, 'arcticdb')
        self.arctic = get_arcticdb_connection(self.arctic_path)

    def list_libraries(self):
        """List all libraries"""
        return self.arctic.list_libraries()

    def list_symbols(self, library):
        """List all symbols in a library"""
        lib = self.arctic.get_library(library)
        return lib.list_symbols()

    def has_historical_range(self, library, symbol):
        """Read data from a symbol"""
        lib = self.arctic.get_library(library)
        if lib.has_symbol(symbol):
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

    def load_from_arcticdb(self, service, source_tickers, start_date, end_date):
        # Connect to local ArcticDB

        data_dict = {}
        for source_ticker in source_tickers:
            try:
                # Read versioned item from ArcticDB
                lib = self.arctic.get_library(service)
                item = lib.read(source_ticker["ticker"])
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

                data_dict[source_ticker["ticker"]] = df
                print(f"Loaded {source_ticker["ticker"]} data: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {source_ticker["ticker"]}: {str(e)}")

        return data_dict
