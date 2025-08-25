from ib_insync import Contract, Stock, Future, Forex
from strategy_data.database.arctic_connection import ArcticConnection


def create_contract(symbol: str) -> Contract:
    """
    Creates an ib_insync Contract object for a given symbol by fetching its metadata from ArcticDB.
    """

    # Connect to ArcticDB and get metadata
    conn = ArcticConnection()
    metadata_lib = conn.get_library('metadata')
    df = metadata_lib.read('instrument_contracts').data

    # Find the instrument
    instrument_data = df.loc[symbol]

    asset_type = instrument_data.get('asset_type')

    if asset_type == 'STK':
        return Stock(
            symbol=instrument_data.name,
            exchange=instrument_data.get('exchange'),
            currency=instrument_data.get('currency'),
            primaryExchange=instrument_data.get('primary_exchange')
        )
    elif asset_type == 'FUT':
        # Note: This requires a more sophisticated way to determine the last trading date
        # or contract month for futures. This is a simplified example.
        return Future(
            symbol=instrument_data.name,
            exchange=instrument_data.get('exchange'),
            currency=instrument_data.get('currency'),
            multiplier=instrument_data.get('multiplier'),
            lastTradeDateOrContractMonth="202412"  # This needs to be dynamic
        )
    elif asset_type == 'CASH':
        # For Forex, the symbol is the currency pair, e.g., EURUSD
        # and the local symbol is EUR.USD
        return Forex(
            pair=instrument_data.name,
            exchange=instrument_data.get('exchange')
        )
    else:
        raise ValueError(f"Unsupported asset type: {asset_type}")
