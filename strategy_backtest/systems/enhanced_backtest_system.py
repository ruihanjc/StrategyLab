"""
Enhanced backtesting system integrating pysystemtrade components
Complete system that combines all major pysystemtrade functionality
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, Any
from datetime import datetime

from ..sysobjects.instruments import Instrument, InstrumentList
from ..sysobjects.prices import AdjustedPrices, MultiplePrices
from ..sysobjects.forecasts import Forecast, ForecastCombination
from ..sysobjects.positions import Position, PositionSeries
from ..sysobjects.costs import TradingCosts, CostCalculator

from .portfolio import PortfolioOptimizer, PositionSizer, VolatilityEstimator, RiskBudgeter
from .risk_management import RiskManager, VolatilityTargeting, CorrelationMonitor, RiskReporter
from .forecast_processing import ForecastScaler, ForecastCombiner, ForecastMapper, ForecastProcessor
from .performance_analytics import PerformanceAnalyzer, PerformanceReporter



class EnhancedBacktestSystem:
    """
    Complete backtesting system with pysystemtrade-style functionality
    """
    
    def __init__(self, 
                 instruments: InstrumentList,
                 initial_capital: float = 1000000,
                 volatility_target: float = 0.25,
                 forecast_cap: float = 20.0,
                 risk_free_rate: float = 0.02,
                 max_leverage: float = 1.0,
                 trading_costs: Union[TradingCosts, Dict[str, TradingCosts]] = None):
        """
        Initialize enhanced backtest system
        
        Parameters:
        -----------
        instruments: InstrumentList
            List of instruments to trade
        initial_capital: float
            Initial capital
        volatility_target: float
            Target portfolio volatility
        forecast_cap: float
            Forecast cap for scaling
        risk_free_rate: float
            Risk-free rate for Sharpe calculation
        max_leverage: float
            Maximum portfolio leverage
        trading_costs: TradingCosts or Dict
            Trading cost configuration
        """
        self.instruments = instruments
        self.initial_capital = initial_capital
        self.volatility_target = volatility_target
        self.forecast_cap = forecast_cap
        self.risk_free_rate = risk_free_rate
        self.max_leverage = max_leverage
        
        # Initialize components
        self._initialize_components(trading_costs)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Storage for backtest results
        self.results = {}
    
    def _initialize_components(self, trading_costs):
        """Initialize all system components"""
        # Cost calculator
        if trading_costs is None:
            trading_costs = TradingCosts()
        self.cost_calculator = CostCalculator(trading_costs)
        
        # Portfolio components
        self.portfolio_optimizer = PortfolioOptimizer(max_portfolio_leverage=self.max_leverage)
        self.position_sizer = PositionSizer(volatility_target=self.volatility_target)
        self.volatility_estimator = VolatilityEstimator()
        self.risk_budgeter = RiskBudgeter(max_leverage=self.max_leverage)
        
        # Risk management
        self.risk_manager = RiskManager(
            max_portfolio_leverage=self.max_leverage,
            volatility_target=self.volatility_target
        )
        self.volatility_targeting = VolatilityTargeting(target_volatility=self.volatility_target)
        self.correlation_monitor = CorrelationMonitor()
        self.risk_reporter = RiskReporter()
        
        # Forecast processing
        self.forecast_scaler = ForecastScaler(forecast_cap=self.forecast_cap)
        self.forecast_combiner = ForecastCombiner(forecast_cap=self.forecast_cap)
        self.forecast_mapper = ForecastMapper(forecast_cap=self.forecast_cap)
        self.forecast_processor = ForecastProcessor(
            self.forecast_scaler, self.forecast_combiner, self.forecast_mapper
        )
        
        # Performance analytics
        self.performance_analyzer = PerformanceAnalyzer(risk_free_rate=self.risk_free_rate)
        self.performance_reporter = PerformanceReporter(self.performance_analyzer)
    
    def run_backtest(self, 
                    price_data: Dict[str, Union[pd.Series, MultiplePrices]],
                    trading_rules: Dict[str, Any],
                    forecast_weights: Dict[str, float] = None,
                    start_date: datetime = None,
                    end_date: datetime = None) -> Dict:
        """
        Run complete backtest
        
        Parameters:
        -----------
        price_data: Dict[str, Union[pd.Series, MultiplePrices]]
            Price data for each instrument
        trading_rules: Dict[str, Any]
            Trading rules configuration
        forecast_weights: Dict[str, float]
            Weights for combining forecasts
        start_date: datetime
            Start date for backtest
        end_date: datetime
            End date for backtest
            
        Returns:
        --------
        Dict
            Complete backtest results
        """
        self.logger.info("Starting enhanced backtest...")
        
        # Step 1: Prepare data
        self.logger.info("Preparing data...")
        prepared_data = self._prepare_data(price_data, start_date, end_date)
        
        # Step 2: Generate forecasts
        self.logger.info("Generating forecasts...")
        raw_forecasts = self._generate_forecasts(prepared_data, trading_rules)
        
        # Step 3: Process forecasts
        self.logger.info("Processing forecasts...")
        processed_forecasts = self._process_forecasts(raw_forecasts, prepared_data, forecast_weights)
        
        # Step 4: Calculate positions
        self.logger.info("Calculating positions...")
        raw_positions = self._calculate_positions(processed_forecasts, prepared_data)
        
        # Step 5: Apply risk management
        self.logger.info("Applying risk management...")
        risk_adjusted_positions = self._apply_risk_management(raw_positions, prepared_data)
        
        # Step 6: Calculate costs
        self.logger.info("Calculating costs...")
        costs = self._calculate_costs(risk_adjusted_positions, prepared_data)
        
        # Step 7: Analyze performance
        self.logger.info("Analyzing performance...")
        performance = self._analyze_performance(risk_adjusted_positions, prepared_data, costs)
        
        # Step 8: Generate reports
        self.logger.info("Generating reports...")
        reports = self._generate_reports(risk_adjusted_positions, prepared_data, costs)
        
        # Compile results
        self.results = {
            'prepared_data': prepared_data,
            'raw_forecasts': raw_forecasts,
            'processed_forecasts': processed_forecasts,
            'raw_positions': raw_positions,
            'final_positions': risk_adjusted_positions,
            'costs': costs,
            'performance': performance,
            'reports': reports,
            'configuration': self._get_configuration()
        }
        
        self.logger.info("Backtest completed successfully!")
        
        return self.results
    
    def _prepare_data(self, 
                     price_data: Dict[str, Union[pd.Series, MultiplePrices]],
                     start_date: datetime = None,
                     end_date: datetime = None) -> Dict:
        """Prepare and align data"""
        prepared = {}
        
        # Convert all price data to AdjustedPrices
        for instrument_name, data in price_data.items():
            if instrument_name in self.instruments:
                if isinstance(data, MultiplePrices):
                    prepared[instrument_name] = data.adjusted_prices('close')
                elif isinstance(data, pd.Series):
                    prepared[instrument_name] = AdjustedPrices(data)
                else:
                    prepared[instrument_name] = AdjustedPrices(data)
                
                # Apply date filter if specified
                if start_date or end_date:
                    prepared[instrument_name] = prepared[instrument_name].get_data_for_period(
                        start_date or prepared[instrument_name].index[0],
                        end_date or prepared[instrument_name].index[-1]
                    )
        
        # Calculate volatilities
        volatilities = self.volatility_estimator.estimate_portfolio_volatilities(prepared)
        
        return {
            'prices': prepared,
            'volatilities': volatilities
        }
    
    def _generate_forecasts(self, 
                          prepared_data: Dict,
                          trading_rules: Dict[str, Any]) -> Dict[str, Dict[str, Forecast]]:
        """Generate raw forecasts from trading rules"""
        try:
            from ..sysrules.rule_factory import TradingRuleManager, TradingRuleSet
        except ImportError:
            from sysrules.rule_factory import TradingRuleManager, TradingRuleSet
        
        forecasts = {}
        
        for instrument_name in prepared_data['prices'].keys():
            if instrument_name in self.instruments:
                instrument_forecasts = {}
                
                # Create rule manager for this instrument
                rule_manager = TradingRuleManager()
                
                # Add rules from trading_rules config
                for rule_name, rule_config in trading_rules.items():
                    try:
                        # Add rule to manager
                        if isinstance(rule_config, dict):
                            rule_manager.add_rule(rule_name, rule_config)
                        else:
                            # Use default config if just rule name provided
                            rule_manager.add_rule(rule_name)
                    except Exception as e:
                        self.logger.warning(f"Failed to add rule {rule_name}: {e}")
                
                # Generate forecasts for all rules
                try:
                    price_data = prepared_data['prices'][instrument_name]
                    additional_data = prepared_data.get('additional_data', {}).get(instrument_name, {})
                    
                    instrument_forecasts = rule_manager.generate_all_forecasts(
                        price_data, additional_data
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Failed to generate forecasts for {instrument_name}: {e}")
                
                if instrument_forecasts:
                    forecasts[instrument_name] = instrument_forecasts
        
        return forecasts
    
    def _generate_single_forecast(self, 
                                price_data: AdjustedPrices,
                                rule_name: str,
                                rule_config: Dict) -> Forecast:
        """Generate a single forecast (simplified example)"""
        # This is a simplified example - you would implement actual trading rules here
        if rule_name == 'ewmac':
            # Simple EWMAC implementation
            fast_span = rule_config.get('fast_span', 16)
            slow_span = rule_config.get('slow_span', 64)
            
            fast_ma = price_data.ewm(span=fast_span).mean()
            slow_ma = price_data.ewm(span=slow_span).mean()
            
            # Calculate forecast
            forecast_values = (fast_ma - slow_ma) / slow_ma * 100
            
            return Forecast(forecast_values, forecast_cap=self.forecast_cap)
        
        elif rule_name == 'momentum':
            # Simple momentum
            lookback = rule_config.get('lookback', 20)
            forecast_values = price_data.pct_change(lookback) * 100
            
            return Forecast(forecast_values, forecast_cap=self.forecast_cap)
        
        else:
            # Default: random forecast for demonstration
            np.random.seed(42)
            forecast_values = pd.Series(
                np.random.normal(0, 5, len(price_data)),
                index=price_data.index
            )
            
            return Forecast(forecast_values, forecast_cap=self.forecast_cap)
    
    def _process_forecasts(self, 
                          raw_forecasts: Dict,
                          prepared_data: Dict,
                          forecast_weights: Dict[str, float] = None) -> Dict[str, Forecast]:
        """Process raw forecasts through scaling and combination"""
        processed_forecasts = {}
        
        for instrument_name, instrument_forecasts in raw_forecasts.items():
            if len(instrument_forecasts) == 1:
                # Single forecast - just scale it
                rule_name, forecast = list(instrument_forecasts.items())[0]
                processed_forecasts[instrument_name] = self.forecast_scaler.scale_forecast(
                    forecast, prepared_data['prices'][instrument_name]
                )
            else:
                # Multiple forecasts - combine them
                combined_forecast = self.forecast_combiner.combine_forecasts(
                    instrument_forecasts, forecast_weights
                )
                processed_forecasts[instrument_name] = combined_forecast
        
        return processed_forecasts
    
    def _calculate_positions(self, 
                           processed_forecasts: Dict[str, Forecast],
                           prepared_data: Dict) -> PositionSeries:
        """Calculate raw positions from processed forecasts"""
        positions = {}
        
        for instrument_name, forecast in processed_forecasts.items():
            if instrument_name in self.instruments:
                instrument = self.instruments[instrument_name]
                price_data = prepared_data['prices'][instrument_name]
                volatility = prepared_data['volatilities'].get(instrument_name)
                
                # Calculate position
                position = self.position_sizer.calculate_position_size(
                    forecast, price_data, instrument, volatility=volatility
                )
                
                positions[instrument_name] = position
        
        return PositionSeries(positions)
    
    def _apply_risk_management(self, 
                             raw_positions: PositionSeries,
                             prepared_data: Dict) -> PositionSeries:
        """Apply risk management overlay"""
        # Apply portfolio risk management
        risk_adjusted = self.risk_manager.apply_risk_overlay(
            raw_positions, prepared_data['prices'], self.instruments
        )
        
        # Apply risk budgeting
        budgeted = self.risk_budgeter.apply_risk_budgets(
            risk_adjusted, self.instruments
        )
        
        return budgeted
    
    def _calculate_costs(self, 
                        positions: PositionSeries,
                        prepared_data: Dict) -> Dict[str, pd.Series]:
        """Calculate transaction costs"""
        costs = {}
        
        for instrument_name in positions.get_instruments():
            position = positions.get_position(instrument_name)
            price_data = prepared_data['prices'][instrument_name]
            
            if position is not None and not price_data.empty:
                instrument_costs = self.cost_calculator.calculate_position_costs(
                    position, price_data, instrument_name
                )
                costs[instrument_name] = instrument_costs
        
        return costs
    
    def _analyze_performance(self, 
                           positions: PositionSeries,
                           prepared_data: Dict,
                           costs: Dict[str, pd.Series]) -> Dict:
        """Analyze performance"""
        return self.performance_analyzer.analyze_performance(
            positions, prepared_data['prices'], costs, self.initial_capital
        )
    
    def _generate_reports(self, 
                         positions: PositionSeries,
                         prepared_data: Dict,
                         costs: Dict[str, pd.Series]) -> Dict[str, str]:
        """Generate comprehensive reports"""
        reports = {}
        
        # Performance report
        reports['performance'] = self.performance_reporter.generate_performance_report(
            positions, prepared_data['prices'], costs, self.initial_capital
        )
        
        # Risk report
        reports['risk'] = self.risk_reporter.generate_risk_report(
            positions, prepared_data['prices'], self.instruments
        )
        
        return reports
    
    def _get_configuration(self) -> Dict:
        """Get system configuration"""
        return {
            'initial_capital': self.initial_capital,
            'volatility_target': self.volatility_target,
            'forecast_cap': self.forecast_cap,
            'risk_free_rate': self.risk_free_rate,
            'max_leverage': self.max_leverage,
            'num_instruments': len(self.instruments),
            'instruments': self.instruments.get_instrument_list()
        }
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get performance summary table"""
        if not self.results:
            raise ValueError("No backtest results available. Run backtest first.")
        
        return self.performance_reporter.generate_summary_table(
            self.results['final_positions'],
            self.results['prepared_data']['prices'],
            self.results['costs']
        )
    
    def plot_results(self):
        """Plot backtest results"""
        if not self.results:
            raise ValueError("No backtest results available. Run backtest first.")
        
        self.performance_reporter.plot_performance(
            self.results['final_positions'],
            self.results['prepared_data']['prices'],
            self.results['costs']
        )
    
    def save_results(self, filepath: str):
        """Save results to file"""
        if not self.results:
            raise ValueError("No backtest results available. Run backtest first.")
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
    
    def load_results(self, filepath: str):
        """Load results from file"""
        import pickle
        with open(filepath, 'rb') as f:
            self.results = pickle.load(f)


def create_sample_enhanced_backtest():
    """Create a sample enhanced backtest for demonstration"""
    from ..sysobjects.instruments import create_sample_instruments
    from ..sysobjects.prices import create_sample_price_data
    
    # Create sample instruments
    instruments = create_sample_instruments()
    
    # Create sample price data
    price_data = {}
    for instrument_name in instruments.get_instrument_list()[:4]:  # First 4 instruments
        price_data[instrument_name] = create_sample_price_data(instrument_name)
    
    # Create trading rules
    trading_rules = {
        'ewmac_fast': {'fast_span': 16, 'slow_span': 64},
        'ewmac_slow': {'fast_span': 32, 'slow_span': 128},
        'momentum': {'lookback': 20}
    }
    
    # Create forecast weights
    forecast_weights = {
        'ewmac_fast': 0.4,
        'ewmac_slow': 0.4,
        'momentum': 0.2
    }
    
    # Create enhanced backtest system
    system = EnhancedBacktestSystem(
        instruments=instruments,
        initial_capital=1000000,
        volatility_target=0.25,
        max_leverage=1.0
    )
    
    # Run backtest
    results = system.run_backtest(
        price_data=price_data,
        trading_rules=trading_rules,
        forecast_weights=forecast_weights
    )
    
    return {
        'system': system,
        'results': results,
        'instruments': instruments,
        'price_data': price_data,
        'trading_rules': trading_rules,
        'forecast_weights': forecast_weights
    }


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run sample backtest
    sample = create_sample_enhanced_backtest()
    system = sample['system']
    
    # Print performance summary
    print("Performance Summary:")
    print(system.get_performance_summary())
    
    # Print reports
    print("\n" + "="*50)
    print(system.results['reports']['performance'])
    
    print("\n" + "="*50)
    print(system.results['reports']['risk'])
    
    # Plot results (if matplotlib is available)
    try:
        system.plot_results()
    except ImportError:
        print("Matplotlib not available for plotting")