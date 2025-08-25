import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy_data.database.arctic_connection import ArcticDBConnection

def initialize_instrument_metadata():
    """
    Initializes and stores instrument metadata in ArcticDB.
    """
    
    # Define instrument metadata
    metadata = [
        {'symbol': 'SPY', 'asset_type': 'STK', 'exchange': 'SMART', 'currency': 'USD', 'primary_exchange': 'ARCA'},
        {'symbol': 'ES', 'asset_type': 'FUT', 'exchange': 'CME', 'currency': 'USD', 'multiplier': '50'},
        {'symbol': 'EURUSD', 'asset_type': 'CASH', 'exchange': 'IDEALPRO', 'currency': 'USD'}
    ]
    
    df = pd.DataFrame(metadata)
    df.set_index('symbol', inplace=True)
    
    # Connect to ArcticDB
    conn = ArcticDBConnection()
    metadata_lib = conn.get_library('metadata')
    
    # Write to the library
    metadata_lib.write('instrument_contracts', df)
    
    print("Instrument metadata initialized and stored in ArcticDB library 'metadata' with symbol 'instrument_contracts'.")

if __name__ == "__main__":
    initialize_instrument_metadata()
