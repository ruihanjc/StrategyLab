#!/usr/bin/env python
"""
Main script for running backtests using enhanced pysystemtrade-style system
Demonstrates complete workflow with integrated components
"""

# Standard library imports
import logging
import os
import sys
from pathlib import Path

from strategy_core.sysdata.arcticdb_handler import ArcticdDBHandler
from strategy_core.sysobjects.instruments import Instrument, InstrumentList
from strategy_core.sysobjects.prices import MultiplePrices
from strategy_backtest.enhanced_backtest_system import EnhancedBacktestSystem
from strategy_backtest.simple_backtest_system import SimpleBacktestSystem
from strategy_core.sysutils.config_manager import ConfigManager


def setup_logging():
    """Initialize logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main(config_arguments=None):
    # Setup logging first
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize configuration
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))

        logger.info("Initializing configuration...")
        config = ConfigManager()

        # Run enhanced backtest
        run_enhanced_backtest(config)

    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        raise


def run_enhanced_backtest(config_manager):
    """Run enhanced backtest using integrated pysystemtrade components"""

    try:
        # Load configurations
        backtest_settings = config_manager.get_backtest_settings()
        services = backtest_settings.get("backtest_services")



        # Create tickers from config
        if not isinstance(services, dict):
            raise ValueError("There are no services set for the backtest.")

        for service, source_tickers in services.items():
            # Skip non-asset class configurations
            if service in ['preprocessing', 'metadata']:
                continue

            tickers = create_instruments_from_config(service, source_tickers)
            # Load price data
            price_data = load_price_data(backtest_settings, service, source_tickers)

            # Choose backtest system based on config
            backtest_type = backtest_settings.get('backtest_type', 'enhanced_backtest_system')
            
            if backtest_type == 'simple_backtest_system':
                system = SimpleBacktestSystem(
                    instruments=tickers,
                    initial_capital=backtest_settings.get('initial_capital', 100000),
                    commission=backtest_settings.get('commission', 0.001),
                    volatility_target=backtest_settings.get('volatility_target', 0.15)
                )
                
                # Run simple backtest
                print(f"\nRunning simple backtest for {service}...")
                simple_config = backtest_settings.get('simple_backtest', {})
                results = system.run_simple_backtest(
                    price_data=price_data,
                    strategy_name=simple_config.get('strategy_name', 'simple_momentum'),
                    lookback_days=simple_config.get('lookback_days', 20),
                    start_date=backtest_settings.get('start_date'),
                    end_date=backtest_settings.get('end_date')
                )
                
                # Display simple results
                print("\n=== SIMPLE BACKTEST RESULTS ===")
                print(system.get_simple_summary())
                
            else:
                system = EnhancedBacktestSystem(
                    instruments=tickers,
                    initial_capital=backtest_settings.get('initial_capital', 1000000),
                    volatility_target=backtest_settings.get('volatility_target', 0.25),
                    max_leverage=backtest_settings.get('max_leverage', 1.0),
                    risk_free_rate=backtest_settings.get('risk_free_rate', 0.02)
                )
                
                # Run enhanced backtest
                print(f"\nRunning enhanced backtest for {service}...")
                results = None
                
                # Display enhanced results
                print("\n=== ENHANCED BACKTEST RESULTS ===")
                summary = system.get_performance_summary()
                print(summary)


            # Plot results based on system type
            try:
                if backtest_type == 'simple_backtest_system':
                    system.plot_simple_results()
                else:
                    if 'reports' in system.results:
                        # Strategy performance report (PRIMARY PURPOSE)
                        if 'rule_performance' in system.results['reports']:
                            print("\n" + "=" * 60)
                            print("RULE PERFORMANCE ANALYSIS:")
                            print(system.results['reports']['rule_performance'])

                        print("\n" + "=" * 60)
                        print("PORTFOLIO PERFORMANCE REPORT:")
                        print(system.results['reports']['performance'])

                        print("\n" + "=" * 60)
                        print("RISK REPORT:")
                        print(system.results['reports']['risk'])
                    
                    system.plot_results()
            except Exception as e:
                print(f"Could not plot results: {e}")

    except Exception as e:
        print(f"Error in run_enhanced_backtest: {e}")
        import traceback
        traceback.print_exc()


def create_instruments_from_config(service, source_tickers):
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


if __name__ == '__main__':
    try:
        success = main(sys.argv)
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Application failed: {str(e)}", exc_info=True)
        sys.exit(1)
