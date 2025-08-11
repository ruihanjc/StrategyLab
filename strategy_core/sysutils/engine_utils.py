from strategy_core.sysobjects import InstrumentList, Instrument, MultiplePrices
from strategy_core.sysobjects.strategy import Strategy
from strategy_core.sysrules import ewmac
from strategy_core.sysrules.trading_rule import TradingRule
from strategy_data.database import ArcticReader


def create_instruments_from_config(instrument_settings):
    """Create instruments from configuration"""
    created_instruments = []

    # Validate that symbols is a list of dictionaries
    if not isinstance(instrument_settings, dict):
        raise ValueError(f"Expected a dictionary of service with instruments.")

    for service in instrument_settings.keys():
        for source_ticker in instrument_settings.get(service):
            # Validate that source_symbol is a dictionary with 'ticker' key
            if not isinstance(source_ticker, dict) or 'ticker' not in source_ticker:
                raise ValueError(f"Expected dict with 'ticker' key, got {source_ticker}")
            instrument = Instrument(
                ticker=source_ticker["ticker"],
                asset_class=service,
                source=source_ticker["source"]
            )
            created_instruments.append(instrument)

    return InstrumentList(created_instruments)


def create_strategy_from_config(instruments, config):
    rules = []

    price_datas = load_price_data(instruments, config)
    for rule in config.get("strategy").items():
        for ticker in price_datas:
            # Pass both price data and ticker information
            rules.append(parse_rule(rule, price_datas[ticker], ticker))

    return Strategy(rules)


def load_price_data(instruments: InstrumentList, config):
    price_data = {}

    arcticdb = ArcticReader()

    raw_data = arcticdb.load_multiple_from_arcticdb(
        instruments=instruments,
        start_date=config.get('start_date'),
        end_date=config.get('end_date')
    )

    for ticker, data in raw_data.items():
        try:
            df = data
            if df is not None and not df.empty:
                required_cols = ['open', 'high', 'low', 'close']
                if all(col in df.columns for col in required_cols):
                    price_data[ticker] = MultiplePrices(df)
                else:
                    if 'close' in df.columns:
                        price_data[ticker] = df['close']
                    else:
                        print(f"Warning: No usable price data for {ticker}")

        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")
    return price_data


def parse_rule(rule, data, ticker=None):
    match rule[0]:
        case "ewmac":
            # Import the actual function, not the module
            from strategy_core.sysrules.ewmac import ewmac
            # Create a wrapper data object that includes ticker info
            rule_data = {'price_data': data, 'ticker': ticker}
            return TradingRule(ewmac, rule[1], rule_data)
        case _:
            raise Exception("No such rule in current project")
