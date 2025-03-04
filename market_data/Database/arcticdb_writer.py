import arcticdb as adb
import pandas as pd
from datetime import datetime
from arcticdb import Arctic

from market_data.Database.arctic_connection import get_arcticdb_connection


class MarketDataStore:
    def __init__(self, arctic_path):
        self.arctic = get_arcticdb_connection(arctic_path['local_storage'])
        self._initialize_libraries()

    def _initialize_libraries(self):
        """Initialize libraries by service type"""
        # Service-based organization
        self.services = {
            'equity': 'Stock market equities',
            'futures': 'Futures contracts',
            'commodity': 'Commodity prices',
            'forex': 'Foreign exchange rates',
            'crypto': 'Cryptocurrency pairs'
        }

        for service, description in self.services.items():
            if not self.arctic.has_library(service):
                self.arctic.create_library(service)

    def store_market_data(self, df):
        """Store market data in appropriate service library"""
        if df is not None:
            ticker = df['ticker'][0]
            service = df['service'][0].lower()


        if service not in self.services:
            raise ValueError(f"Unknown service type: {service}")

        try:
            # Convert to DataFrame if needed
            if isinstance(df, dict):
                df = pd.DataFrame([df])
            elif isinstance(df, list):
                df = pd.DataFrame(df)
            else:
                df = df.copy()

            # Get the appropriate library
            lib = self.arctic.get_library(service)

            # Store data
            symbol = f"{ticker}"
            lib.write(symbol, self.normalize_dataframe(df))
            return True

        except Exception as e:
            print(f"Error storing data: {str(e)}")
            return False


    def get_service_stats(self):
        """Get statistics about each service library"""
        stats = {}
        for service in self.services:
            lib = self.arctic.get_library(service)
            symbols = lib.list_symbols()

            # Calculate total size and number of records
            total_records = 0
            symbols_info = []

            for symbol in symbols:
                data = lib.read(symbol).data
                total_records += len(data)
                symbols_info.append({
                    'symbol': symbol,
                    'records': len(data),
                    'start_date': data['date'].min() if 'date' in data.columns else None,
                    'end_date': data['date'].max() if 'date' in data.columns else None
                })

            stats[service] = {
                'total_symbols': len(symbols),
                'total_records': total_records,
                'symbols_info': symbols_info
            }

        return stats

    def normalize_dataframe(self, df):
        """Normalize DataFrame to ensure proper data types"""
        # Convert timestamp to pandas Timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Convert date to pandas Timestamp
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # Ensure numeric columns are float
        numeric_columns = ['open', 'high', 'low', 'close']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Ensure volume is integer
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('int64')

        # Ensure categorical columns are string
        categorical_columns = ['ticker', 'service']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)

        return df

