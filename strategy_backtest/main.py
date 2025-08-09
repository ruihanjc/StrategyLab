#!/usr/bin/env python
"""
Main script for running backtests using enhanced pysystemtrade-style system
Demonstrates complete workflow with integrated components
"""

# Standard library imports
import logging
import sys
from pathlib import Path

from strategy_core.sysobjects import Portfolio
from strategy_core.sysobjects.engine import TradingEngine
from strategy_core.sysobjects.rules.strategy import Strategy
from strategy_core.sysutils.config_manager import ConfigManager
from strategy_core.sysutils.engine_utils import *


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
        logger.info("Initializing configuration...")
        config = ConfigManager("strategy_backtest", "backtest")

        backtest_config = config.get_settings()

        create_instruments_from_config(backtest_config.get("instruments"))

        portfolio = Portfolio()

        strategy = Strategy()

        engine = TradingEngine()


    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        raise


def run_enhanced_backtest(config_manager):
    """Run enhanced backtest using integrated pysystemtrade components"""
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


if __name__ == '__main__':
    try:
        success = main(sys.argv)
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Application failed: {str(e)}", exc_info=True)
        sys.exit(1)
