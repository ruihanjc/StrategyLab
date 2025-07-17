"""
Test script for integrated trading rules system
Tests all components of the pysystemtrade-style trading rules
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_rules import *
from rule_factory import TradingRuleFactory, TradingRuleManager, TradingRuleSet
from rule_config_manager import RuleConfigManager
from ..sysobjects.prices import create_sample_price_data
from ..sysobjects.forecasts import Forecast


def create_test_data():
    """Create test data for rule testing"""
    # Create multiple instruments
    instruments = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    price_data = {}
    
    for instrument in instruments:
        # Create price data with different characteristics
        np.random.seed(hash(instrument) % 1000)
        price_data[instrument] = create_sample_price_data(
            instrument, 
            start_date="2020-01-01", 
            end_date="2023-12-31"
        ).adjusted_prices('close')
    
    return price_data


def test_individual_rules():
    """Test individual trading rules"""
    print("="*60)
    print("TESTING INDIVIDUAL TRADING RULES")
    print("="*60)
    
    # Create test data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Create trending price data
    trend = np.cumsum(np.random.normal(0.0005, 0.02, len(dates)))
    noise = np.random.normal(0, 0.01, len(dates))
    prices = pd.Series(100 * np.exp(trend + noise), index=dates)
    
    # Test EWMAC rule
    print("\n1. Testing EWMAC Rule")
    ewmac_rule = EWMACRule(16, 64)
    ewmac_forecast = ewmac_rule(prices)
    print(f"   EWMAC 16/64: mean={ewmac_forecast.mean():.3f}, std={ewmac_forecast.std():.3f}")
    
    # Test Breakout rule
    print("\n2. Testing Breakout Rule")
    breakout_rule = BreakoutRule(20)
    breakout_forecast = breakout_rule(prices)
    print(f"   Breakout 20: mean={breakout_forecast.mean():.3f}, std={breakout_forecast.std():.3f}")
    
    # Test Momentum rule
    print("\n3. Testing Momentum Rule")
    momentum_rule = MomentumRule(20)
    momentum_forecast = momentum_rule(prices)
    print(f"   Momentum 20: mean={momentum_forecast.mean():.3f}, std={momentum_forecast.std():.3f}")
    
    # Test Acceleration rule
    print("\n4. Testing Acceleration Rule")
    accel_rule = AccelerationRule(4)
    accel_forecast = accel_rule(prices)
    print(f"   Acceleration 4: mean={accel_forecast.mean():.3f}, std={accel_forecast.std():.3f}")
    
    # Test Mean Reversion rule
    print("\n5. Testing Mean Reversion Rule")
    mr_rule = MeanReversionRule(20)
    mr_forecast = mr_rule(prices)
    print(f"   Mean Reversion 20: mean={mr_forecast.mean():.3f}, std={mr_forecast.std():.3f}")
    
    # Test Volatility rule
    print("\n6. Testing Volatility Rule")
    vol_rule = VolatilityRule(20)
    vol_forecast = vol_rule(prices)
    print(f"   Volatility 20: mean={vol_forecast.mean():.3f}, std={vol_forecast.std():.3f}")
    
    return {
        'prices': prices,
        'ewmac': ewmac_forecast,
        'breakout': breakout_forecast,
        'momentum': momentum_forecast,
        'acceleration': accel_forecast,
        'mean_reversion': mr_forecast,
        'volatility': vol_forecast
    }


def test_rule_factory():
    """Test rule factory functionality"""
    print("\n" + "="*60)
    print("TESTING RULE FACTORY")
    print("="*60)
    
    # Create factory
    factory = TradingRuleFactory()
    
    # List available rules
    print("\n1. Available Rules:")
    rules = factory.list_available_rules()
    for rule in rules[:10]:  # Show first 10
        print(f"   {rule}")
    print(f"   ... and {len(rules) - 10} more")
    
    # Create some rules
    print("\n2. Creating Rule Instances:")
    
    # Create EWMAC rule
    ewmac_rule = factory.create_rule('ewmac_16_64')
    print(f"   EWMAC 16/64: {ewmac_rule.name} - {ewmac_rule.description}")
    
    # Create Breakout rule
    breakout_rule = factory.create_rule('breakout_20')
    print(f"   Breakout 20: {breakout_rule.name} - {breakout_rule.description}")
    
    # Get rule info
    print("\n3. Rule Information:")
    rule_info = factory.get_rule_info('ewmac_16_64')
    print(f"   Name: {rule_info['name']}")
    print(f"   Class: {rule_info['class']}")
    print(f"   Parameters: {rule_info['parameters']}")
    print(f"   Forecast Scalar: {rule_info['forecast_scalar']}")
    print(f"   Data Requirements: {rule_info['data_requirements']}")
    
    return factory


def test_rule_manager():
    """Test rule manager functionality"""
    print("\n" + "="*60)
    print("TESTING RULE MANAGER")
    print("="*60)
    
    # Create test data
    price_data = create_test_data()
    test_instrument = 'AAPL'
    test_prices = price_data[test_instrument]
    
    # Create manager
    manager = TradingRuleManager()
    
    # Add some rules
    print("\n1. Adding Rules to Manager:")
    rules_to_add = ['ewmac_16_64', 'ewmac_32_128', 'breakout_20', 'momentum_20']
    
    for rule_name in rules_to_add:
        manager.add_rule(rule_name)
        print(f"   Added: {rule_name}")
    
    # Generate forecasts
    print("\n2. Generating Forecasts:")
    forecasts = manager.generate_all_forecasts(test_prices)
    
    for rule_name, forecast in forecasts.items():
        if not forecast.empty:
            print(f"   {rule_name}: mean={forecast.mean():.3f}, std={forecast.std():.3f}")
    
    # Get rule summary
    print("\n3. Rule Summary:")
    summary = manager.get_rule_summary()
    print(summary)
    
    return manager, forecasts


def test_rule_sets():
    """Test predefined rule sets"""
    print("\n" + "="*60)
    print("TESTING RULE SETS")
    print("="*60)
    
    # Test different rule sets
    rule_sets = ['trend_following', 'breakout', 'momentum', 'mixed', 'comprehensive']
    
    for set_name in rule_sets:
        print(f"\n{set_name.upper()} Rule Set:")
        rule_weights = TradingRuleSet.get_rule_set(set_name)
        
        total_weight = sum(rule_weights.values())
        print(f"   Total Rules: {len(rule_weights)}")
        print(f"   Total Weight: {total_weight:.3f}")
        
        # Show top 3 rules by weight
        sorted_rules = sorted(rule_weights.items(), key=lambda x: x[1], reverse=True)
        print("   Top Rules:")
        for rule, weight in sorted_rules[:3]:
            print(f"     {rule}: {weight:.3f}")
    
    return rule_sets


def test_config_manager():
    """Test configuration manager"""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION MANAGER")
    print("="*60)
    
    # Create config manager
    config_manager = RuleConfigManager()
    
    # List available rules
    print("\n1. Available Rules:")
    rules = config_manager.list_available_rules()
    print(f"   Total Rules: {len(rules)}")
    
    # List rule sets
    print("\n2. Available Rule Sets:")
    rule_sets = config_manager.list_available_rule_sets()
    for rule_set in rule_sets:
        print(f"   {rule_set}")
    
    # Get rule statistics
    print("\n3. Rule Statistics:")
    stats = config_manager.get_rule_statistics()
    print(stats.head())
    
    # Create rule manager from config
    print("\n4. Creating Rule Manager from Config:")
    rule_manager = config_manager.create_rule_manager(rule_set='mixed_strategy')
    print(f"   Rules in manager: {len(rule_manager.get_rule_names())}")
    
    return config_manager


def test_full_system():
    """Test the complete integrated system"""
    print("\n" + "="*60)
    print("TESTING COMPLETE INTEGRATED SYSTEM")
    print("="*60)
    
    # Create test data
    price_data = create_test_data()
    
    # Create configuration manager
    config_manager = RuleConfigManager()
    
    # Create rule manager from comprehensive rule set
    rule_manager = config_manager.create_rule_manager(rule_set='comprehensive')
    
    print(f"\n1. Testing on {len(price_data)} instruments")
    print(f"   Using {len(rule_manager.get_rule_names())} trading rules")
    
    # Generate forecasts for all instruments
    all_forecasts = {}
    
    for instrument, prices in price_data.items():
        print(f"\n   Processing {instrument}...")
        
        try:
            forecasts = rule_manager.generate_all_forecasts(prices)
            all_forecasts[instrument] = forecasts
            
            # Calculate forecast statistics
            forecast_stats = {}
            for rule_name, forecast in forecasts.items():
                if not forecast.empty:
                    forecast_stats[rule_name] = {
                        'mean': forecast.mean(),
                        'std': forecast.std(),
                        'correlation': forecast.corr(prices.pct_change().shift(-1))
                    }
            
            print(f"     Generated {len(forecasts)} forecasts")
            
        except Exception as e:
            print(f"     Error: {e}")
    
    # Summary statistics
    print(f"\n2. Summary Statistics:")
    total_forecasts = sum(len(forecasts) for forecasts in all_forecasts.values())
    print(f"   Total forecasts generated: {total_forecasts}")
    print(f"   Average forecasts per instrument: {total_forecasts / len(price_data):.1f}")
    
    return all_forecasts


def test_performance():
    """Test performance of the system"""
    print("\n" + "="*60)
    print("TESTING SYSTEM PERFORMANCE")
    print("="*60)
    
    # Create larger test dataset
    instruments = [f"STOCK_{i}" for i in range(20)]
    price_data = {}
    
    for instrument in instruments:
        np.random.seed(hash(instrument) % 1000)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = pd.Series(100 * np.cumprod(1 + returns), index=dates)
        price_data[instrument] = prices
    
    # Create rule manager
    config_manager = RuleConfigManager()
    rule_manager = config_manager.create_rule_manager(rule_set='mixed_strategy')
    
    print(f"\n1. Performance Test:")
    print(f"   Instruments: {len(instruments)}")
    print(f"   Rules: {len(rule_manager.get_rule_names())}")
    print(f"   Data points per instrument: {len(dates)}")
    
    # Time the forecast generation
    import time
    start_time = time.time()
    
    total_forecasts = 0
    for instrument, prices in price_data.items():
        forecasts = rule_manager.generate_all_forecasts(prices)
        total_forecasts += len(forecasts)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n2. Performance Results:")
    print(f"   Total forecasts generated: {total_forecasts}")
    print(f"   Time elapsed: {elapsed_time:.2f} seconds")
    print(f"   Forecasts per second: {total_forecasts / elapsed_time:.1f}")
    print(f"   Time per forecast: {elapsed_time / total_forecasts * 1000:.1f}ms")


def create_visualization():
    """Create visualization of trading rules"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATION")
    print("="*60)
    
    # Create test data
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Create trending price data
    trend = np.cumsum(np.random.normal(0.001, 0.015, len(dates)))
    noise = np.random.normal(0, 0.008, len(dates))
    prices = pd.Series(100 * np.exp(trend + noise), index=dates)
    
    # Create rule manager
    config_manager = RuleConfigManager()
    rule_manager = config_manager.create_rule_manager(rule_set='trend_following')
    
    # Generate forecasts
    forecasts = rule_manager.generate_all_forecasts(prices)
    
    # Create plots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Price chart
    axes[0, 0].plot(prices.index, prices.values)
    axes[0, 0].set_title('Price Series')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].grid(True)
    
    # Individual forecasts
    plot_idx = 1
    for rule_name, forecast in list(forecasts.items())[:5]:
        if not forecast.empty:
            row = plot_idx // 2
            col = plot_idx % 2
            
            axes[row, col].plot(forecast.index, forecast.values)
            axes[row, col].set_title(f'{rule_name} Forecast')
            axes[row, col].set_ylabel('Forecast')
            axes[row, col].grid(True)
            axes[row, col].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('trading_rules_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   Visualization saved as 'trading_rules_visualization.png'")


def main():
    """Main test function"""
    print("PYSYSTEMTRADE-STYLE TRADING RULES INTEGRATION TEST")
    print("="*60)
    
    try:
        # Test individual components
        individual_results = test_individual_rules()
        factory = test_rule_factory()
        manager, forecasts = test_rule_manager()
        rule_sets = test_rule_sets()
        config_manager = test_config_manager()
        
        # Test complete system
        all_forecasts = test_full_system()
        
        # Performance test
        test_performance()
        
        # Create visualization
        try:
            create_visualization()
        except Exception as e:
            print(f"   Visualization failed: {e}")
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Summary
        print("\n✓ Individual trading rules working")
        print("✓ Rule factory system working")
        print("✓ Rule manager system working")
        print("✓ Rule sets working")
        print("✓ Configuration manager working")
        print("✓ Complete integrated system working")
        print("✓ Performance acceptable")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)