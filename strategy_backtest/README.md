# Enhanced Strategy Backtest System

A comprehensive backtesting system that integrates the most important components from pysystemtrade to create a robust, production-ready backtesting framework.

## Features

### Core Components Integrated from pysystemtrade

1. **Enhanced Data Structures**
   - `Instrument` and `InstrumentList` for instrument metadata
   - `AdjustedPrices` and `MultiplePrices` for price data handling
   - `Forecast` and `ForecastCombination` for signal processing
   - `Position` and `PositionSeries` for position management
   - `TradingCosts` and `CostCalculator` for cost modeling

2. **Professional Trading Rules System**
   - **15+ Trading Rules**: EWMAC, Breakout, Momentum, Acceleration, Mean Reversion, Carry, Volatility
   - **Rule Factory**: Dynamic rule creation and management
   - **Rule Manager**: Forecast generation and combination
   - **Configuration System**: YAML-based rule configuration
   - **Predefined Rule Sets**: Trend following, breakout, momentum, mixed, comprehensive
   - **Parameter Optimization**: Automatic forecast scalar optimization

3. **Portfolio Management**
   - `PortfolioOptimizer` for correlation-based optimization
   - `PositionSizer` for volatility-based position sizing
   - `VolatilityEstimator` for dynamic volatility calculation
   - `RiskBudgeter` for risk allocation constraints

4. **Risk Management**
   - `RiskManager` for comprehensive risk overlays
   - `VolatilityTargeting` for dynamic leverage adjustment
   - `CorrelationMonitor` for correlation analysis
   - `RiskReporter` for risk monitoring and reporting

5. **Forecast Processing**
   - `ForecastScaler` for forecast normalization
   - `ForecastCombiner` for multi-signal combination
   - `ForecastMapper` for forecast-to-position mapping
   - `ForecastProcessor` for complete forecast pipeline

6. **Performance Analytics**
   - `PerformanceAnalyzer` for comprehensive metrics
   - `PerformanceReporter` for detailed reporting
   - Advanced statistics and risk-adjusted returns

7. **Enhanced Backtest System**
   - `EnhancedBacktestSystem` - complete integrated system
   - Modular architecture for easy customization
   - Comprehensive logging and error handling

## Quick Start

```python
from systems.enhanced_backtest_system import EnhancedBacktestSystem
from sysobjects.instruments import create_sample_instruments
from sysobjects.prices import create_sample_price_data
from sysrules.rule_config_manager import RuleConfigManager

# Create instruments
instruments = create_sample_instruments()

# Create price data
price_data = {}
for instrument_name in instruments.get_instrument_list()[:3]:
    price_data[instrument_name] = create_sample_price_data(instrument_name)

# Define trading rules using new system
trading_rules = {
    'ewmac_16_64': {
        'rule_class': 'EWMAC',
        'parameters': {'Lfast': 16, 'Lslow': 64, 'vol_days': 35},
        'forecast_scalar': 7.5
    },
    'ewmac_32_128': {
        'rule_class': 'EWMAC',
        'parameters': {'Lfast': 32, 'Lslow': 128, 'vol_days': 35},
        'forecast_scalar': 7.5
    },
    'breakout_20': {
        'rule_class': 'Breakout',
        'parameters': {'lookback': 20, 'smooth': 5},
        'forecast_scalar': 1.0
    },
    'momentum_20': {
        'rule_class': 'Momentum',
        'parameters': {'lookback': 20, 'vol_days': 35},
        'forecast_scalar': 2.0
    }
}

# Create system
system = EnhancedBacktestSystem(
    instruments=instruments,
    initial_capital=1000000,
    volatility_target=0.25,
    max_leverage=1.0
)

# Run backtest
results = system.run_backtest(
    price_data=price_data,
    trading_rules=trading_rules
)

# View results
print(system.get_performance_summary())
system.plot_results()
```

### Alternative: Using Rule Configuration Manager

```python
from sysrules.rule_config_manager import RuleConfigManager

# Create rule configuration manager
config_manager = RuleConfigManager()

# Create rule manager with predefined rule set
rule_manager = config_manager.create_rule_manager(rule_set='mixed_strategy')

# Generate forecasts for multiple instruments
for instrument_name, price_data in price_data.items():
    forecasts = rule_manager.generate_all_forecasts(price_data)
    print(f"{instrument_name}: {len(forecasts)} forecasts generated")
```

## System Architecture

### Data Flow

1. **Data Preparation**: Load and align price data
2. **Forecast Generation**: Generate trading signals from rules
3. **Forecast Processing**: Scale and combine forecasts
4. **Position Calculation**: Convert forecasts to positions
5. **Risk Management**: Apply risk overlays and constraints
6. **Cost Calculation**: Calculate transaction costs
7. **Performance Analysis**: Analyze results and generate reports

### Key Components

#### sysobjects/
- **instruments.py**: Instrument metadata and management
- **prices.py**: Price data structures with validation
- **forecasts.py**: Forecast handling and combination
- **positions.py**: Position management and analysis
- **costs.py**: Cost modeling and calculation

#### systems/
- **portfolio.py**: Portfolio optimization and position sizing
- **risk_management.py**: Risk overlays and monitoring
- **forecast_processing.py**: Forecast scaling and combination
- **performance_analytics.py**: Performance analysis and reporting
- **enhanced_backtest_system.py**: Main integrated system

#### sysrules/
- **trading_rules.py**: Core trading rule implementations
- **rule_factory.py**: Rule factory and management system
- **rule_config_manager.py**: Configuration management
- **test_trading_rules.py**: Comprehensive testing suite

## Configuration

The system uses YAML configuration files:

### backtest_config.yaml
```yaml
initial_capital: 1000000
volatility_target: 0.25
max_leverage: 1.0
risk_free_rate: 0.02
start_date: '2020-01-01'
end_date: '2023-12-31'
```

### data_config.yaml
```yaml
arcticdb:
  symbols:
    - AAPL
    - GOOGL
    - MSFT
    - TSLA
```

## Advanced Features

### Professional Trading Rules System

#### Core Trading Rules (15+ implemented)

1. **EWMAC (Exponential Weighted Moving Average Crossover)**
   - Multiple speed combinations (2/8, 4/16, 8/32, 16/64, 32/128, 64/256)
   - Volatility-normalized signals
   - Trend following methodology from pysystemtrade

2. **Breakout Rules**
   - Multiple lookback periods (10, 20, 40, 80, 160, 320 days)
   - Smoothed signals to reduce noise
   - Position relative to recent high/low range

3. **Momentum Rules**
   - Various horizons (10, 20, 40, 80, 250 days)
   - Volatility-normalized returns
   - Time series momentum implementation

4. **Acceleration Rules**
   - Rate of change of EWMAC signals
   - Multiple speeds (4, 8, 16 day base periods)
   - Momentum acceleration detection

5. **Mean Reversion Rules**
   - Threshold-based activation
   - Multiple lookback periods
   - Fade extreme movements

6. **Carry Rules**
   - Yield-based trading for appropriate instruments
   - Multiple smoothing periods
   - Carry data integration

7. **Relative Momentum Rules**
   - Outperformance vs benchmark/asset class
   - Cross-sectional momentum
   - Multiple horizons

8. **Volatility Rules**
   - Volatility regime detection
   - Fade volatility spikes
   - Multiple measurement periods

#### Rule Management System

- **Rule Factory**: Dynamic rule creation and parameterization
- **Rule Manager**: Forecast generation and combination
- **Configuration System**: YAML-based rule configuration
- **Predefined Rule Sets**: 
  - Trend Following (EWMAC + Breakout focused)
  - Momentum (Momentum + Acceleration focused)
  - Mixed Strategy (Balanced approach)
  - Comprehensive (All rule types)
  - Conservative (Longer-term signals)
  - Aggressive (Shorter-term signals)

#### Advanced Features

- **Automatic Forecast Scaling**: Dynamic optimization of forecast scalars
- **Rule Validation**: Parameter validation and data requirements checking
- **Performance Testing**: Comprehensive test suite for all rules
- **Configuration Management**: Save/load rule configurations
- **Rule Statistics**: Detailed analytics on rule performance

### Multi-Asset Portfolio Management
- Correlation-based optimization
- Dynamic instrument weights
- Risk parity allocation
- Leverage constraints

### Sophisticated Risk Management
- Portfolio-level risk overlays
- Volatility targeting
- Drawdown control
- Correlation monitoring

### Professional Cost Modeling
- Bid-ask spreads
- Market impact
- Commission structures
- Slippage estimation

### Comprehensive Analytics
- Risk-adjusted returns
- Drawdown analysis
- Trade-level statistics
- Attribution analysis

## Performance Metrics

The system calculates comprehensive performance metrics:

### Return Metrics
- Total return
- Annualized return
- Volatility
- Sharpe ratio
- Sortino ratio
- Calmar ratio

### Risk Metrics
- Maximum drawdown
- Value at Risk (VaR)
- Expected shortfall
- Beta analysis

### Trade Metrics
- Win rate
- Profit factor
- Average trade size
- Trade frequency

### Portfolio Metrics
- Leverage analysis
- Turnover metrics
- Concentration measures
- Risk attribution

## Running the System

### From Command Line
```bash
cd StrategyLab/strategy_backtest
python standard_update_data.py
```

### From Python Script
```python
from main import run_enhanced_backtest
from sysutils.config import ConfigManager

config = ConfigManager()
run_enhanced_backtest(config)
```

## Integration with Existing Data

The system integrates with your existing ArcticDB data:

1. **Automatic Data Loading**: Reads from your ArcticDB setup
2. **Fallback to Sample Data**: Uses sample data if ArcticDB unavailable
3. **Flexible Price Formats**: Handles OHLCV or close-only data
4. **Date Range Support**: Configurable start/end dates

## Extensibility

The modular design allows easy extension:

1. **Custom Trading Rules**: Add new forecast generation rules
2. **Risk Overlays**: Implement custom risk management logic
3. **Cost Models**: Create asset-specific cost models
4. **Performance Metrics**: Add custom performance calculations

## Error Handling

- Comprehensive logging throughout the system
- Graceful handling of missing data
- Validation of inputs and configurations
- Detailed error reporting

## Dependencies

- pandas
- numpy
- matplotlib
- pyyaml
- arcticdb (for data storage)

## Future Enhancements

1. **Machine Learning Integration**: ML-based forecast generation
2. **Real-time Execution**: Live trading capabilities
3. **Alternative Data**: Integration with alternative data sources
4. **Advanced Optimization**: More sophisticated portfolio optimization
5. **Regime Detection**: Adaptive strategies based on market regimes

## Support

For questions or issues:
1. Check the comprehensive logging output
2. Review the sample implementations
3. Examine the test functions in each module
4. Refer to the pysystemtrade documentation for theoretical background

This system provides a professional-grade backtesting framework that combines the best practices from pysystemtrade with modern Python development patterns.