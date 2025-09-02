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
from strategy_core.sysobjects.engine import ProductionEngine
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


def generate_daily_signals(end_date=None, lookback_days=252):
    """
    Generate daily trading signals using the production engine
    
    Parameters:
    -----------
    end_date: str or None
        End date for signal generation (defaults to today)
    lookback_days: int
        Number of days to look back for data (default: 252 trading days)
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Set dates
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Calculate start date based on lookback
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=lookback_days * 1.5)  # Extra buffer for weekends/holidays
        start_date = start_dt.strftime('%Y-%m-%d')

        logger.info(f"Generating signals for period: {start_date} to {end_date}")

        # Initialize configuration
        logger.info("Initializing configuration...")
        config = ConfigManager("strategy_backtest", "backtest")
        backtest_config = config.get_settings()

        # Override dates in config
        backtest_config['start_date'] = start_date
        backtest_config['end_date'] = end_date

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

        logger.info(f"Signal generation period: {start_date} to {end_date}")

        # Create and run production engine
        logger.info("Creating production engine for signal generation...")
        engine = ProductionEngine(
            portfolio=portfolio,
            strategy=strategy,
            data_handler=price_data,
            start_date=start_date,
            end_date=end_date
        )

        # Run signal generation
        logger.info("Generating trading signals...")
        results = engine.run()

        # Log completion
        logger.info("Daily signal generation completed successfully!")
        
        # Log key metrics for verification
        performance = results.get_performance_summary()
        logger.info(f"Signal generation complete - Final positions generated for {len(instruments)} instruments")

        # optimize_ewmac(engine)
        #
        # # Plot results in CTA style
        # if backtest_config.get("plot_results", True):
        #     logger.info("Generating plots...")
        #     try:
        #         # Use the CTA-style plotting with the price data from the engine
        #         plot_path = "strategy_backtest/results/backtest_results.png"
        #         # Show the plot interactively AND save it
        #         results.plot_cta_style(results.price_data, save_path=plot_path, show_plot=True)
        #         logger.info("Interactive plot displayed and saved to backtest_results.png")
        #     except Exception as e:
        #         logger.warning(f"Error generating plots: {str(e)}")
        #
        # # Save results if requested
        # if backtest_config.get("save_results", False):
        #     results_file = backtest_config.get("results_file", "backtest_results.pkl")
        #     try:
        #         import pickle
        #         with open(results_file, 'wb') as f:
        #             pickle.dump(results, f)
        #         logger.info(f"Results saved to {results_file}")
        #     except Exception as e:
        #         logger.error(f"Error saving results: {str(e)}")

        return True

    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    # Parse command line arguments
    end_date = None
    lookback_days = 252
    
    if len(sys.argv) > 1:
        end_date = sys.argv[1]
    if len(sys.argv) > 2:
        lookback_days = int(sys.argv[2])
    
    try:
        success = generate_daily_signals(end_date=end_date, lookback_days=lookback_days)
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Daily signal generation failed: {str(e)}", exc_info=True)
        sys.exit(1)
