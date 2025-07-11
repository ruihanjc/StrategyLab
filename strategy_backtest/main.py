#!/usr/bin/env python
"""
Main script for running backtests using ArcticDB data
Demonstrates complete workflow with YAML configuration
"""
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from pathlib import Path
import logging
from strategy_backtest.sysrules.ewma import EqualWeightMovingAverage
# Import components from our framework
from sysutils.config import ConfigManager
from sysdata.arcticdb_handler import ArcticdDBHandler
from systems.base_engine import BacktestEngine


def setup_logging():
    """Initialize logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main(config_arguments):
    # Setup logging first
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize configuration
        logger.info("Initializing configuration...")
        config = ConfigManager()

        run_single_backtest(config)


    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        raise


def run_single_backtest(config_manager):
    """Run a single-asset backtest using configuration"""

    # Load configurations
    backtest_settings = config_manager.get_backtest_settings()
    data_settings = config_manager.get_data_settings()

    arcticdb = ArcticdDBHandler('equity', f"{Path(os.getcwd())}/arcticdb")


    # Load from ArcticDB
    arcticdb_settings = data_settings.get('arcticdb', {})
    symbols = arcticdb_settings.get('symbols', [])

    for ticker in symbols:
        raw_data_dict = arcticdb.load_from_arcticdb(
            symbols=[ticker],
            start_date=backtest_settings.get('start_date'),
            end_date=backtest_settings.get('end_date')
        )
        ticker_data = raw_data_dict

        # Preprocess data


        # Create strategy from config
        strategy = EqualWeightMovingAverage(
            ma_periods = [16,32],
            threshold=0.6  # 60% of MAs must agree for a signal
        )

        # Initialize backtesting engine
        engine = BacktestEngine(
            initial_capital=backtest_settings.get('initial_capital', 100000),
            commission=backtest_settings.get('commission', 0.001),
            slippage=backtest_settings.get('slippage', 0.0005)
        )

        # Run backtest
        results, metrics = engine.run(ticker_data, strategy, price_column='close')

        # Display results
        print("\n=== Backtest Results ===")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Number of Trades: {metrics['num_trades']}")

        # Plot results
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(ticker_data.index, ticker_data['close'])
        plt.title(f"{ticker} Close Price")
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(results.index, results['equity'])
        plt.title('Equity Curve')
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(results.index, results['equity']/results['equity'].cummax() - 1)
        plt.title('Drawdown')
        plt.grid(True)

        plt.tight_layout()
        plt.show()



def run_multi_strategy_backtest(config_manager=None, use_sample_data=True):
    """Run a backtest with a multi-factor strategy"""
    if config_manager is None:
        config_manager = ConfigManager()

    # Load configurations
    backtest_settings = config_manager.get



if __name__ == '__main__':
    try:
        success = main(sys.argv)
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Application failed: {str(e)}", exc_info=True)
        sys.exit(1)