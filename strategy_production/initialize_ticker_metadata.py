import pandas as pd
import os
from strategy_data.database.arctic_connection import get_arcticdb_connection


def initialize_instrument_metadata():
    """
    Initializes and stores instrument metadata in ArcticDB.
    """

    # Define instrument metadata
    metadata = [
        {'symbol': 'SPY', 'asset_type': 'STK', 'exchange': 'SMART', 'currency': 'USD', 'primary_exchange': 'ARCA'},
        {'symbol': 'NVDA', 'asset_type': 'STK', 'exchange': 'SMART', 'currency': 'USD', 'primary_exchange': 'NASDAQ'},
        {'symbol': 'TSLA', 'asset_type': 'STK', 'exchange': 'SMART', 'currency': 'USD', 'primary_exchange': 'NASDAQ'},
        {'symbol': 'PLTR', 'asset_type': 'STK', 'exchange': 'SMART', 'currency': 'USD', 'primary_exchange': 'NASDAQ'},
        {'symbol': 'AUDUSD', 'asset_type': 'CASH', 'exchange': 'SMART', 'currency': 'USD', 'primary_exchange': 'IDEALPRO'}
    ]

    df = pd.DataFrame(metadata)
    df.set_index('symbol', inplace=True)

    # Connect to ArcticDB
    project_dir = os.path.abspath(__file__ + "/../../")
    arctic_path = os.path.join(project_dir, 'arcticdb')
    conn = get_arcticdb_connection(arctic_path)
    metadata_lib = conn.get_library('metadata')

    # Write to the library
    metadata_lib.append('instrument_contracts', df)

    print(
        "Instrument metadata initialized and stored in ArcticDB library 'metadata' with symbol 'instrument_contracts'.")


if __name__ == "__main__":
    initialize_instrument_metadata()
