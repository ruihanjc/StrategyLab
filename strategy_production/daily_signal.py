#!/usr/bin/env python
"""
Main script for running backtests using enhanced pysystemtrade-style system
Demonstrates complete workflow with integrated components
"""
from datetime import datetime
# Standard library imports
import logging
import os
import sys
import yaml
import json

from strategy_core.sysobjects import Portfolio
from strategy_core.sysobjects.engine import ProductionEngine
from strategy_core.sysutils.engine_utils import *
from strategy_production.calculate_positions import calculate_positions

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def setup_logging():
    """Initialize logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def generate_daily_signals():
    """
    Generate daily trading signals using the production engine
    Uses default configuration path and today's date
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Use default configuration
        project_dir = os.path.abspath(__file__ + "/../../")
        config_path = os.path.join(project_dir, "strategy_backtest/config/backtest_config.yaml")


        # Initialize configuration
        logger.info(f"Initializing configuration from {config_path}...")
        with open(config_path, 'r') as file:
            backtest_config = yaml.safe_load(file)

        # Use config start_date and override end_date
        start_date = backtest_config.get('start_date')
        end_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"Generating signals for period: {start_date} to {end_date}")

        # Create instruments from configuration
        logger.info("Creating instruments...")
        instruments = create_instruments_from_config(backtest_config.get("instruments"))
        logger.info(f"Created {len(instruments)} instruments")

        # Load price data for instruments
        logger.info("Loading price data...")
        price_data = load_price_data(instruments, start_date, end_date)
        logger.info(f"Loaded price data for {len(price_data)} instruments")

        # Create strategy
        logger.info("Creating strategy...")
        strategy = create_strategy_from_config(instruments, backtest_config, price_data)

        # Validate strategy
        if not strategy.validate_strategy(price_data):
            logger.error("Strategy validation failed")
            return False

        # Create portfolio
        logger.info("Creating portfolio...")
        position_sizing_config = backtest_config.get("position_sizing", {})
        portfolio = Portfolio(instruments,
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

        # --- Save target positions to file for execution ---
        logger.info("Saving target positions for execution...")
        target_positions = results.daily_positions.iloc[-1].to_dict()

        calculate_positions(target_positions)

        # Define the output path
        output_dir = os.path.join(project_dir, "strategy_production/order_signal")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "target_positions.json")
        
        with open(output_path, 'w') as f:
            json.dump(target_positions, f, indent=4)
            
        logger.info(f"Target positions saved to {output_path}")
        # --- End of save ---

        return True

    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    try:
        success = generate_daily_signals()
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Daily signal generation failed: {str(e)}", exc_info=True)
        sys.exit(1)
