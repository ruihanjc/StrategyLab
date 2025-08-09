import os
import pandas as pd

from strategy_data.database.arctic_connection import get_arcticdb_connection


class ArcticWriter:
    def __init__(self):
        project_dir = os.path.abspath(__file__ + "/../../../")
        self.arctic_path = os.path.join(project_dir, 'arcticdb')
        self.arctic = get_arcticdb_connection(self.arctic_path)

    def _initialize_libraries(self):
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

    def store_market_data(self, fetched_data):
        service, ticker = '', ''
        if fetched_data is not None:
            ticker = fetched_data['ticker'][0]
            service = fetched_data['service'][0].lower()

        if not self.arctic.has_library(service):
            raise ValueError(f"Unknown service type: {service}")

        try:
            if isinstance(fetched_data, dict):
                fetched_data = pd.DataFrame([fetched_data])
            elif isinstance(fetched_data, list):
                fetched_data = pd.DataFrame(fetched_data)

            lib = self.arctic.get_library(service)

            symbol = f"{ticker}"
            lib.write(symbol, self.normalize_dataframe(fetched_data))
            return True

        except Exception as e:
            print(f"Error storing data: {str(e)}")
            return False

    def get_service_stats(self):
        stats = {}
        for service in self.services:
            lib = self.arctic.get_library(service)
            symbols = lib.list_symbols()

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

    @staticmethod
    def normalize_dataframe(df):
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
