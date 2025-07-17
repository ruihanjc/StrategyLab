#!/usr/bin/env python
"""
Main script for running backtests using enhanced pysystemtrade-style system
Demonstrates complete workflow with integrated components
"""
import os
import sys

import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Add parent directory to path for market_data imports
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

# Import enhanced components
try:
    from sysutils.config import ConfigManager
    from systems.enhanced_backtest_system import EnhancedBacktestSystem
    from sysobjects.instruments import create_sample_instruments, Instrument, InstrumentList
    from sysobjects.prices import MultiplePrices, create_sample_price_data
    
    # Try to import ArcticDB handler, but don't fail if it's not available
    try:
        from sysdata.arcticdb_handler import ArcticdDBHandler
        ARCTICDB_AVAILABLE = True
    except ImportError:
        ARCTICDB_AVAILABLE = False
        print("ArcticDB not available, will use sample data")
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run from the strategy_backtest directory")
    sys.exit(1)


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
        data_settings = config_manager.get_data_settings()

        # Create instruments from config
        instruments = create_instruments_from_config(data_settings)
        
        # Load price data
        price_data = load_price_data(data_settings, backtest_settings)
        
        # Create trading rules configuration
        trading_rules = create_trading_rules_config()
        
        # Create forecast weights
        forecast_weights = create_forecast_weights()
        
        print("=== ENHANCED BACKTEST SYSTEM ===")
        print(f"Instruments: {instruments.get_instrument_list()}")
        print(f"Trading Rules: {list(trading_rules.keys())}")
        print(f"Forecast Weights: {forecast_weights}")
        
        # Create enhanced backtest system
        system = EnhancedBacktestSystem(
            instruments=instruments,
            initial_capital=backtest_settings.get('initial_capital', 1000000),
            volatility_target=backtest_settings.get('volatility_target', 0.25),
            max_leverage=backtest_settings.get('max_leverage', 1.0),
            risk_free_rate=backtest_settings.get('risk_free_rate', 0.02)
        )
        
        # Run backtest
        print("\nRunning enhanced backtest...")
        results = system.run_backtest(
            price_data=price_data,
            trading_rules=trading_rules,
            forecast_weights=forecast_weights,
            start_date=backtest_settings.get('start_date'),
            end_date=backtest_settings.get('end_date')
        )
        
        # Display results
        print("\n=== ENHANCED BACKTEST RESULTS ===")
        summary = system.get_performance_summary()
        print(summary)
        
        if 'reports' in system.results:
            print("\n" + "="*60)
            print("PERFORMANCE REPORT:")
            print(system.results['reports']['performance'])
            
            print("\n" + "="*60)
            print("RISK REPORT:")
            print(system.results['reports']['risk'])
        
        # Plot results
        try:
            system.plot_results()
        except Exception as e:
            print(f"Could not plot results: {e}")
            
    except Exception as e:
        print(f"Enhanced backtest failed: {e}")
        print("Falling back to sample data test...")
        
        # Create simple test with sample data
        from sysobjects.instruments import create_sample_instruments
        from sysobjects.prices import create_sample_price_data
        
        instruments = create_sample_instruments()
        price_data = {}
        
        for instrument_name in instruments.get_instrument_list()[:2]:  # Test with 2 instruments
            price_data[instrument_name] = create_sample_price_data(instrument_name)
        
        trading_rules = create_trading_rules_config()
        forecast_weights = create_forecast_weights()
        
        system = EnhancedBacktestSystem(
            instruments=instruments,
            initial_capital=1000000,
            volatility_target=0.25,
            max_leverage=1.0
        )
        
        print("Running with sample data...")
        results = system.run_backtest(
            price_data=price_data,
            trading_rules=trading_rules,
            forecast_weights=forecast_weights
        )
        
        print("\n=== SAMPLE DATA BACKTEST RESULTS ===")
        print(system.get_performance_summary())


def create_instruments_from_config(data_settings):
    """Create instruments from configuration"""
    instruments = []
    
    # Get symbols from config
    arcticdb_settings = data_settings.get('arcticdb', {})
    symbols = arcticdb_settings.get('symbols', ['AAPL'])
    
    for symbol in symbols:
        instrument = Instrument(
            name=symbol,
            currency="USD",
            asset_class="equity",
            point_size=1.0,
            description=f"Equity instrument: {symbol}"
        )
        instruments.append(instrument)
    
    return InstrumentList(instruments)


def run_single_backtest(config_manager):
    """Legacy single-asset backtest function with enhanced trading rules"""
    from sysrules.trading_rules import EWMACRule, BreakoutRule, MomentumRule
    from systems.backtest_engine import BacktestEngine
    
    # Load configurations
    backtest_settings = config_manager.get_backtest_settings()
    data_settings = config_manager.get_data_settings()

    # Initialize ArcticDB handler - use fixed path relative to this script
    script_dir = Path(__file__).parent
    arcticdb_path = script_dir.parent / "arcticdb"
    
    try:
        arcticdb = ArcticdDBHandler('equity', str(arcticdb_path))
        use_arcticdb = True
    except Exception as e:
        print(f"Could not initialize ArcticDB: {e}")
        print("Using sample data instead...")
        use_arcticdb = False

    # Get symbols from config
    arcticdb_settings = data_settings.get('arcticdb', {})
    symbols = arcticdb_settings.get('symbols', ['AAPL', 'GOOGL', 'MSFT'])

    for ticker in symbols:
        print(f"\n=== Running backtest for {ticker} ===")
        
        if use_arcticdb:
            try:
                raw_data_dict = arcticdb.load_from_arcticdb(
                    symbols=[ticker],
                    start_date=backtest_settings.get('start_date'),
                    end_date=backtest_settings.get('end_date')
                )
                ticker_data = raw_data_dict.get(ticker)
                
                if ticker_data is None or (hasattr(ticker_data, 'empty') and ticker_data.empty):
                    print(f"No data found for {ticker}, using sample data")
                    from sysobjects.prices import create_sample_price_data
                    price_obj = create_sample_price_data(ticker)
                    ticker_data = price_obj.get_price_data()
                    
            except Exception as e:
                print(f"Error loading {ticker}: {e}, using sample data")
                from sysobjects.prices import create_sample_price_data
                price_obj = create_sample_price_data(ticker)
                ticker_data = price_obj.get_price_data()
        else:
            # Use sample data
            from sysobjects.prices import create_sample_price_data
            price_obj = create_sample_price_data(ticker)
            ticker_data = price_obj.get_price_data()

        # Create enhanced trading rules
        ewmac_rule = EWMACRule(16, 64)
        breakout_rule = BreakoutRule(20)
        momentum_rule = MomentumRule(20)
        
        # Use close price for signals
        close_prices = ticker_data['close'] if 'close' in ticker_data.columns else ticker_data.iloc[:, 0]
        
        # Generate signals from multiple rules
        ewmac_signal = ewmac_rule(close_prices)
        breakout_signal = breakout_rule(close_prices)
        momentum_signal = momentum_rule(close_prices)
        
        # Combine signals (simple average)
        combined_signal = (ewmac_signal + breakout_signal + momentum_signal) / 3
        
        # Create simple strategy wrapper
        class CombinedStrategy:
            def __init__(self, signal):
                self.signal = signal
                
            def generate_signals(self, data):
                # Convert forecast to position signal
                positions = self.signal / 20.0  # Scale down from forecast range
                positions = positions.clip(-1, 1)  # Limit to -1, 1
                return positions
        
        strategy = CombinedStrategy(combined_signal)

        # Initialize backtesting engine
        engine = BacktestEngine(
            initial_capital=backtest_settings.get('initial_capital', 100000),
            commission=backtest_settings.get('commission', 0.001),
            slippage=backtest_settings.get('slippage', 0.0005)
        )

        # Run backtest
        results, metrics = engine.run(ticker_data, strategy, price_column='close')

        # Display results
        print(f"\n=== Backtest Results for {ticker} ===")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Number of Trades: {metrics['num_trades']}")

        # Plot results
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(ticker_data.index, close_prices)
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


def load_price_data(data_settings, backtest_settings):
    """Load price data from ArcticDB or create sample data"""
    price_data = {}
    
    # Get symbols from config
    arcticdb_settings = data_settings.get('arcticdb', {})
    symbols = arcticdb_settings.get('symbols', ['PLTR'])
    
    if ARCTICDB_AVAILABLE:
        try:
            # Initialize ArcticDB handler - use fixed path relative to this script
            script_dir = Path(__file__).parent
            arcticdb_path = script_dir.parent / "arcticdb"
            arcticdb = ArcticdDBHandler('equity', str(arcticdb_path))

            raw_data = arcticdb.load_from_arcticdb(
                symbols=symbols,
                start_date=backtest_settings.get('start_date'),
                end_date=backtest_settings.get('end_date')
            )

            for symbol in symbols:
                try:
                    if raw_data is not None and symbol in raw_data:
                        # Convert to MultiplePrices format
                        df = raw_data[symbol]
                        if df is not None and not df.empty:
                            # Ensure we have OHLCV columns
                            required_cols = ['open', 'high', 'low', 'close']
                            if all(col in df.columns for col in required_cols):
                                price_data[symbol] = MultiplePrices(df)
                            else:
                                # Use close price only
                                if 'close' in df.columns:
                                    price_data[symbol] = df['close']
                                else:
                                    print(f"Warning: No usable price data for {symbol}")
                                    
                except Exception as e:
                    print(f"Error loading data for {symbol}: {e}")
                    # Create sample data as fallback
                    price_data[symbol] = create_sample_price_data(symbol)
                    
        except Exception as e:
            print(f"ArcticDB initialization failed: {e}")
            print("Using sample data instead...")
            # Use sample data instead
            pass
    
    # If ArcticDB not available or no data loaded, create sample data
    if not ARCTICDB_AVAILABLE or not price_data:
        print("Creating sample data...")
        for symbol in symbols:
            price_data[symbol] = create_sample_price_data(symbol)
    
    return price_data


def create_trading_rules_config():
    """Create trading rules configuration"""
    return {
        'ewmac_16_64': {
            'rule_class': 'EWMAC',
            'parameters': {
                'Lfast': 16,
                'Lslow': 64,
                'vol_days': 35
            },
            'forecast_scalar': 7.5
        }
    }


def create_forecast_weights():
    """Create forecast weights configuration"""
    return {
        'ewmac_16_64': 1.0,
    }


if __name__ == '__main__':
    try:
        success = main(sys.argv)
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Application failed: {str(e)}", exc_info=True)
        sys.exit(1)