#!/usr/bin/env python
"""
Main script for running backtests using enhanced pysystemtrade-style system
Demonstrates complete workflow with integrated components
"""

# Standard library imports
import logging
import os
import sys

from strategy_core.sysobjects import Portfolio
from strategy_core.sysobjects.engine import BacktestEngine
from strategy_core.sysutils.config_manager import ConfigManager
from strategy_core.sysutils.engine_utils import *

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


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

        # Create instruments from configuration
        logger.info("Creating instruments...")
        instruments = create_instruments_from_config(backtest_config.get("instruments"))
        logger.info(f"Created {len(instruments)} instruments")

        # Load price data for instruments
        logger.info("Loading price data...")
        price_data = load_price_data(instruments, backtest_config)
        logger.info(f"Loaded price data for {len(price_data)} instruments")

        # Create strategy
        logger.info("Creating strategy...")
        strategy = create_strategy_from_config(instruments, backtest_config)

        # Validate strategy
        if not strategy.validate_strategy(price_data):
            logger.error("Strategy validation failed")
            return False

        # Create portfolio
        logger.info("Creating portfolio...")
        initial_capital = backtest_config.get("initial_capital")
        position_sizing_config = backtest_config.get("position_sizing", {})
        portfolio = Portfolio(instruments, 
                            initial_capital=initial_capital,
                            position_sizing_config=position_sizing_config)

        # Get date range from config
        start_date = backtest_config.get("start_date")
        end_date = backtest_config.get("end_date")

        logger.info(f"Backtest period: {start_date} to {end_date}")

        # Create and run backtest engine
        logger.info("Creating backtest engine...")
        engine = BacktestEngine(
            portfolio=portfolio,
            strategy=strategy,
            data_handler=price_data,
            start_date=start_date,
            end_date=end_date
        )

        # Run the backtest
        logger.info("Starting backtest execution...")
        results = engine.run()

        # Display results
        logger.info("Backtest completed successfully!")

        # Print performance summary
        performance = results.get_performance_summary()
        logger.info("=== BACKTEST RESULTS ===")
        logger.info(f"Total Return: {performance.get('total_return', 0):.2%}")
        logger.info(f"Annualized Return: {performance.get('annualized_return', 0):.2%}")
        logger.info(f"Annualized Volatility: {performance.get('annualized_volatility', 0):.2%}")
        logger.info(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Maximum Drawdown: {performance.get('max_drawdown', 0):.2%}")
        logger.info(f"Win Rate: {performance.get('win_rate', 0):.2%}")
        logger.info(f"Final Capital: ${performance.get('final_capital', 0):,.2f}")
        
        # Print yearly performance
        yearly_performance = performance.get('yearly_performance', {})
        if yearly_performance:
            logger.info("\n=== YEAR-BY-YEAR PERFORMANCE ===")
            for year, year_metrics in yearly_performance.items():
                logger.info(f"{year}: Return: {year_metrics.get('total_return', 0):.2%}, "
                           f"Sharpe: {year_metrics.get('sharpe_ratio', 0):.2f}, "
                           f"Volatility: {year_metrics.get('volatility', 0):.2%}")
        
        logger.info("========================")
        return True

    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    try:
        success = main(sys.argv)
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Application failed: {str(e)}", exc_info=True)
        sys.exit(1)
