import datetime
import pandas as pd
import os

from collections import defaultdict
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

    def has_historical_range(self, service, symbol):
        """Read data from a symbol"""
        lib = self.arctic.get_library(service)
        if lib.has_symbol(symbol):
            data = lib.read(symbol).data
            if 'date' in data.columns:
                latest_date = data['date'].max()
                return latest_date, True
        else:
            return datetime.date.today(), False
        return None

    def get_latest_date_for_ticker(self, service, symbol):
        pass

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

    def get_metadata_info(self, symbol):
        """Get information about a symbol"""
        lib = self.arctic.get_library("metadata")
        data = lib.read("instrument_contracts").data
        df = pd.DataFrame(data)
        return df.loc[symbol]

    def load_from_arcticdb(self, service, ticker, start_date, end_date):
        try:
            # Read versioned item from ArcticDB
            lib = self.arctic.get_library(service)
            item = lib.read(ticker)
            df = item.data

            # Filter by date
            if isinstance(df.index, pd.DatetimeIndex):
                df = df[(df.index >= start_date) & (df.index <= end_date)]
            else:
                # Convert string date column to datetime if needed
                date_col = next((col for col in df.columns if 'date' in col.lower()), None)
                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
                    df.set_index(date_col, inplace=True)

            return df
        except Exception as e:
            print(f"Error loading {ticker}: {str(e)}")

    def load_multiple_from_arcticdb(self, instruments, start_date, end_date):
        # Group instruments by asset class/service
        grouped_instruments = defaultdict(list)

        for instrument in instruments:
            asset_class = instrument.asset_class
            grouped_instruments[asset_class].append(instrument)

        data_dict = {}

        # Process each asset class group
        for asset_class, instrument_list in grouped_instruments.items():
            try:
                lib = self.arctic.get_library(asset_class)
                for instrument in instrument_list:
                    try:
                        # Read versioned item from ArcticDB
                        item = lib.read(instrument.ticker)
                        df = item.data

                        # Filter by date
                        if isinstance(df.index, pd.DatetimeIndex):
                            df = df[(df.index >= start_date) & (df.index <= end_date)]
                        else:
                            # Convert string date column to datetime if needed
                            date_col = next((col for col in df.columns if 'date' in col.lower()), None)
                            if date_col:
                                df[date_col] = pd.to_datetime(df[date_col])
                                df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
                                df.set_index(date_col, inplace=True)

                        data_dict[instrument.ticker] = df
                    except Exception as e:
                        print(f"Error loading {instrument.ticker} from {asset_class}: {str(e)}")

            except Exception as e:
                print(f"Error accessing {asset_class} library: {str(e)}")

        return data_dict
