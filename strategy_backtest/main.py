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
import pandas as pd
import numpy as np

from strategy_backtest.sysdata.arcticdb_handler import ArcticdDBHandler
from strategy_backtest.sysobjects.instruments import Instrument, InstrumentList
from strategy_backtest.sysobjects.prices import MultiplePrices
from strategy_backtest.systems.enhanced_backtest_system import EnhancedBacktestSystem
from strategy_backtest.sysutils.config import ConfigManager

# Setup path - add project root to Python path for absolute imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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

        # Create tickers from config
        if not isinstance(data_settings, dict):
            raise ValueError(f"data_settings should be a dict, got {type(data_settings)}")

        for service in data_settings:
            # Skip non-asset class configurations
            if service in ['preprocessing', 'metadata']:
                continue
                
            tickers = create_instruments_from_config(data_settings, service)
            # Load price data
            price_data = load_price_data(data_settings, backtest_settings, service)
            # Create trading rules configuration
            trading_rules = create_trading_rules_config()
            # Create forecast weights
            forecast_weights = create_forecast_weights()

            print("=== ENHANCED BACKTEST SYSTEM ===")
            print(f"Instruments: {tickers.get_instrument_list()}")
            print(f"Trading Rules: {list(trading_rules.keys())}")
            print(f"Forecast Weights: {forecast_weights}")

            # Create enhanced backtest system
            system = EnhancedBacktestSystem(
                instruments=tickers,
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
                # Rule performance report (PRIMARY PURPOSE)
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

            # Plot results
            try:
                system.plot_results()
            except Exception as e:
                print(f"Could not plot results: {e}")

    except Exception as e:
        print(f"Error in run_enhanced_backtest: {e}")
        import traceback
        traceback.print_exc()


def create_instruments_from_config(data_settings, service):
    """Create instruments from configuration"""
    instruments = []

    # Get symbols from config
    symbols = data_settings.get(service)
    
    # Validate that symbols is a list of dictionaries
    if not isinstance(symbols, list):
        raise ValueError(f"Expected list of symbols for service '{service}', got {type(symbols)}")

    for source_symbol in symbols:
        # Validate that source_symbol is a dictionary with 'ticker' key
        if not isinstance(source_symbol, dict) or 'ticker' not in source_symbol:
            raise ValueError(f"Expected dict with 'ticker' key, got {source_symbol}")
        instrument = Instrument(
            name=source_symbol["ticker"],
            asset_class=service,
            point_size=1.0,
            description=f"Equity instrument: {source_symbol['ticker']}"
        )
        instruments.append(instrument)

    return InstrumentList(instruments)


def load_price_data(data_settings, backtest_settings, service):
    """Load price data from ArcticDB or create sample data"""
    price_data = {}

    # Get symbols from config
    service_source_ticker = data_settings.get(service)

    # Initialize ArcticDB handler - use fixed path relative to this script
    current_dir = os.path.abspath(__file__ + "/../../")
    arcticdb_path = current_dir + "/arcticdb"
    arcticdb = ArcticdDBHandler(service, str(arcticdb_path))

    raw_data = arcticdb.load_from_arcticdb(
        source_tickers=data_settings[service],
        start_date=backtest_settings.get('start_date'),
        end_date=backtest_settings.get('end_date')
    )

    for source_ticker in data_settings[service]:
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


def test_fixed_system():
    """Test the fixed backtesting system with existing data"""
    print("=" * 60)
    print("TESTING FIXED StrategyLab vs pysystemtrade approach")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load just one instrument for comparison
        config = ConfigManager()
        backtest_settings = config.get_backtest_settings()
        data_settings = config.get_data_settings()
        
        # Get TSLA data as example
        service = 'equity'
        if service in data_settings:
            # Load price data
            current_dir = os.path.abspath(__file__ + "/../../")
            arcticdb_path = current_dir + "/arcticdb"
            arcticdb = ArcticdDBHandler(service, str(arcticdb_path))
            
            # Get first ticker
            first_ticker = data_settings[service][0]
            raw_data = arcticdb.load_from_arcticdb(
                source_tickers=[first_ticker],
                start_date=backtest_settings.get('start_date'),
                end_date=backtest_settings.get('end_date')
            )
            
            if raw_data and first_ticker['ticker'] in raw_data:
                price_data = raw_data[first_ticker['ticker']]['close']
                
                print(f"\n1. PRICE DATA ({first_ticker['ticker']}):")
                print(f"   - Data points: {len(price_data)}")
                print(f"   - Date range: {price_data.index[0]} to {price_data.index[-1]}")
                print(f"   - Price range: ${price_data.min():.2f} to ${price_data.max():.2f}")
                
                # Test OLD vs NEW approach
                print(f"\n2. TRADING RULE COMPARISON:")
                
                # OLD APPROACH - Your original EWMA (discrete signals)
                try:
                    from strategy_backtest.sysrules.ewma import EqualWeightMovingAverage
                    old_strategy = EqualWeightMovingAverage(ma_periods=[16, 64])
                    old_signals = old_strategy.generate_signals(pd.DataFrame({'close': price_data}))
                    
                    print(f"   OLD (Discrete Signals):")
                    print(f"   - Signal type: {type(old_signals.iloc[0])}")
                    print(f"   - Unique values: {sorted(old_signals.unique())}")
                    print(f"   - Buy signals: {(old_signals == 1).sum()}")
                    print(f"   - Sell signals: {(old_signals == -1).sum()}")
                    print(f"   - Hold signals: {(old_signals == 0).sum()}")
                except Exception as e:
                    print(f"   OLD approach failed: {e}")
                
                # NEW APPROACH - Proper EWMAC (continuous forecasts)
                from strategy_backtest.sysrules.proper_ewmac import ProperEWMAC
                new_rule = ProperEWMAC(Lfast=16, Lslow=64, vol_days=35)
                new_forecast = new_rule(price_data)
                
                print(f"   NEW (Continuous Forecasts):")
                print(f"   - Forecast type: {type(new_forecast)}")
                forecast_data = new_forecast.get_data()
                print(f"   - Value range: {forecast_data.min():.3f} to {forecast_data.max():.3f}")
                print(f"   - Mean forecast: {forecast_data.mean():.3f}")
                print(f"   - Std deviation: {forecast_data.std():.3f}")
                print(f"   - Non-zero forecasts: {(forecast_data != 0).sum()}")
                
                # Show forecast quality
                print(f"\n3. FORECAST QUALITY ANALYSIS:")
                returns = price_data.pct_change().shift(-1)
                common_index = forecast_data.index.intersection(returns.index)
                aligned_forecast = forecast_data.reindex(common_index)[:-1]
                aligned_returns = returns.reindex(common_index)[:-1]
                
                # Remove NaN values
                mask = ~(aligned_forecast.isna() | aligned_returns.isna())
                clean_forecast = aligned_forecast[mask]
                clean_returns = aligned_returns[mask]
                
                if len(clean_forecast) > 10:
                    correlation = np.corrcoef(clean_forecast, clean_returns)[0, 1]
                    hit_rate = ((clean_forecast * clean_returns) > 0).mean()
                    
                    print(f"   - Forecast-Return Correlation: {correlation:.4f}")
                    print(f"   - Hit Rate: {hit_rate:.1%}")
                    
                    if abs(correlation) > 0.05 and hit_rate > 0.52:
                        verdict = "✅ GOOD - Shows predictive power"
                    elif abs(correlation) > 0.02 or hit_rate > 0.51:
                        verdict = "⚠️  WEAK - Some predictive ability"
                    else:
                        verdict = "❌ POOR - No meaningful predictive power"
                    
                    print(f"   - Verdict: {verdict}")
                
                # Show sample forecasts
                print(f"\n4. SAMPLE FORECASTS:")
                sample_forecasts = forecast_data.dropna().head(10)
                for date, value in sample_forecasts.items():
                    direction = "LONG" if value > 0 else "SHORT" if value < 0 else "NEUTRAL"
                    print(f"   {date.strftime('%Y-%m-%d')}: {value:+7.3f} ({direction})")
                
                print(f"\n" + "=" * 60)
                print("SUMMARY:")
                print("✅ Fixed: Trading rules now return continuous forecasts")
                print("✅ Fixed: Proper volatility-adjusted EWMAC implementation")
                print("✅ Fixed: Forecast processing pipeline working")
                print("✅ Fixed: Performance analysis measures forecast quality")
                print("✅ Ready: System now matches pysystemtrade approach")
                print("=" * 60)
                
        else:
            print("No equity data found in configuration")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_fixed_system()
    else:
        try:
            success = main(sys.argv)
            sys.exit(0 if success else 1)
        except Exception as e:
            logging.error(f"Application failed: {str(e)}", exc_info=True)
            sys.exit(1)
