import os

from strategy_core.sysdata.arcticdb_handler import ArcticdDBHandler
from strategy_core.sysobjects import InstrumentList, Instrument, MultiplePrices


def create_instruments_from_config(instrument):
    """Create instruments from configuration"""
    instruments = []

    # Validate that symbols is a list of dictionaries
    if not isinstance(source_tickers, list):
        raise ValueError(f"Expected list of symbols for service '{service}', got {type(source_tickers)}")

    for source_ticker in source_tickers:
        # Validate that source_symbol is a dictionary with 'ticker' key
        if not isinstance(source_ticker, dict) or 'ticker' not in source_ticker:
            raise ValueError(f"Expected dict with 'ticker' key, got {source_ticker}")
        instrument = Instrument(
            name=source_ticker["ticker"],
            asset_class=service,
            point_size=1.0,
            description=f"{service.title()} instrument: {source_ticker['ticker']}"
        )
        instruments.append(instrument)

    return InstrumentList(instruments)


def load_price_data(backtest_settings, service, source_tickers):
    """Load price data from ArcticDB or create sample data"""
    price_data = {}

    # Initialize ArcticDB handler - use fixed path relative to this script
    current_dir = os.path.abspath(__file__ + "/../../")
    arcticdb_path = current_dir + "/arcticdb"
    arcticdb = ArcticdDBHandler(service, str(arcticdb_path))

    raw_data = arcticdb.load_from_arcticdb(
        source_tickers=source_tickers,
        start_date=backtest_settings.get('start_date'),
        end_date=backtest_settings.get('end_date')
    )

    for source_ticker in source_tickers:
        try:
            if raw_data is not None and source_ticker['ticker'] in raw_data:
                # Convert to MultiplePrices format
                df = raw_data[source_ticker['ticker']]
                if df is not None and not df.empty:
                    # Ensure we have OHLCV columns
                    required_cols = ['open', 'high', 'low', 'close']
                    if all(col in df.columns for col in required_cols):
                        price_data[source_ticker['ticker']] = MultiplePrices(df)
                    else:
                        # Use close price only
                        if 'close' in df.columns:
                            price_data[source_ticker['ticker']] = df['close']
                        else:
                            print(f"Warning: No usable price data for {source_ticker['ticker']}")

        except Exception as e:
            print(f"Error loading data for {source_ticker['ticker']}: {e}")
    return price_data
