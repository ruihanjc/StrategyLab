"""
Enhanced backtesting system integrating pysystemtrade components
Complete system that combines all major pysystemtrade functionality
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Union, Any
from datetime import datetime

from strategy_core.sysobjects.instruments import InstrumentList
from strategy_core.sysobjects.prices import AdjustedPrices, MultiplePrices
from strategy_core.sysobjects.forecasts import Forecast
from strategy_core.sysobjects.positions import PositionSeries
from strategy_core.sysobjects.costs import TradingCosts, CostCalculator

from strategy_core.sysobjects.portfolio import Portfolio
from strategy_core.sysobjects.position_sizer import PositionSizer
from strategy_core.sysriskutils.portfolio_optimizer import PortfolioOptimizer
from strategy_core.sysriskutils.volatility_estimator import VolatilityEstimator
from strategy_core.sysriskutils.risk_budgeter import RiskBudgeter
from strategy_core.sysriskutils.risk_management import RiskManager, VolatilityTargeting, CorrelationMonitor, RiskReporter
from strategy_core.sysriskutils.forecast_processing import ForecastScaler, ForecastCombiner, ForecastMapper, ForecastProcessor
from strategy_core.sysriskutils.performance_analytics import PerformanceAnalyzer, PerformanceReporter


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
        self.logger.info("Preparing price data...")
        prepared_data = self._prepare_data(price_data, start_date, end_date)

        # Step 2: Test trading rules (CORE PURPOSE)
        self.logger.info("Testing trading rules...")
        rule_signals = self._test_trading_rules(prepared_data, trading_rules)

        # Step 3: Evaluate rule performance (CORE PURPOSE) 
        self.logger.info("Evaluating rule signal quality...")
        rule_performance = self._evaluate_rule_signals(rule_signals, prepared_data)

        # Step 4: Combine rule signals (if multiple rules)
        self.logger.info("Combining rule signals...")
        combined_signals = self._combine_rule_signals(rule_signals, prepared_data, forecast_weights)

        # Step 5: Apply position sizing (SECONDARY: How much to bet)
        self.logger.info("Calculating position sizes...")
        raw_positions = self._calculate_position_sizes(combined_signals, prepared_data)

        # Step 6: Apply risk management (SECONDARY: Risk controls)
        self.logger.info("Applying risk controls...")
        final_positions = self._apply_risk_controls(raw_positions, prepared_data)

        # Step 7: Calculate trading costs
        self.logger.info("Calculating trading costs...")
        costs = self._calculate_costs(final_positions, prepared_data)

        # Step 8: Analyze backtest results
        self.logger.info("Analyzing backtest results...")
        performance = self._analyze_performance(final_positions, prepared_data, costs)

        # Step 9: Generate reports
        self.logger.info("Generating reports...")
        reports = self._generate_reports(final_positions, prepared_data, costs, rule_performance, rule_signals)

        # Compile results
        self.results = {
            'prepared_data': prepared_data,
            'rule_signals': rule_signals,
            'rule_performance': rule_performance,  # NEW: Core rule testing results
            'combined_signals': combined_signals,
            'raw_positions': raw_positions,
            'final_positions': final_positions,
            'costs': costs,
            'portfolio_performance': performance,  # Portfolio performance (secondary)
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

    def _test_trading_rules(self,
                            prepared_data: Dict,
                            trading_rules: Dict[str, Any]) -> Dict[str, Dict[str, Forecast]]:
        """Test trading rules and generate forecasts for each instrument"""

        forecasts = {}

        for instrument_name in prepared_data['prices'].keys():
            if instrument_name in self.instruments:
                instrument_forecasts = {}

                # Create rule manager for this instrument
                rule_manager = None

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
                    
                    # Convert AdjustedPrices to pandas Series if needed
                    if hasattr(price_data, 'get_data'):
                        price_series = price_data.get_data()
                    elif hasattr(price_data, 'data'):
                        price_series = price_data.data
                    else:
                        price_series = price_data
                    
                    additional_data = prepared_data.get('additional_data', {}).get(instrument_name, {})

                    instrument_forecasts = rule_manager.generate_all_forecasts(
                        price_series, additional_data
                    )

                except Exception as e:
                    self.logger.warning(f"Failed to generate forecasts for {instrument_name}: {e}")
                    import traceback
                    traceback.print_exc()

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
    
    def _evaluate_rule_signals(self,
                               rule_forecasts: Dict,
                               prepared_data: Dict) -> Dict:
        """Evaluate the quality of trading rule forecasts (CORE PURPOSE)"""
        rule_performance = {}
        
        for instrument_name, instrument_forecasts in rule_forecasts.items():
            if instrument_name not in prepared_data['prices']:
                continue
                
            price_data = prepared_data['prices'][instrument_name]
            
            # Convert AdjustedPrices to pandas Series if needed
            if hasattr(price_data, 'get_data'):
                price_series = price_data.get_data()
            elif hasattr(price_data, 'data'):
                price_series = price_data.data
            else:
                price_series = price_data
                
            instrument_rule_performance = {}
            
            for rule_name, forecast in instrument_forecasts.items():
                # Test forecast quality
                forecast_performance = self._test_forecast_quality(forecast, price_series, rule_name)
                instrument_rule_performance[rule_name] = forecast_performance
            
            rule_performance[instrument_name] = instrument_rule_performance
        
        return rule_performance
    
    def _test_forecast_quality(self, forecast: Forecast, price_data: pd.Series, rule_name: str) -> Dict:
        """Test the quality of individual trading rule forecasts"""
        if forecast.empty or price_data.empty:
            return {'valid': False, 'reason': 'Empty forecast or price data'}
        
        # Get forecast data
        if hasattr(forecast, 'get_data'):
            forecast_values = forecast.get_data()
        elif hasattr(forecast, 'data'):
            forecast_values = forecast.data
        else:
            forecast_values = forecast
        
        # Align forecast and price data
        common_index = forecast_values.index.intersection(price_data.index)
        if len(common_index) < 10:
            return {'valid': False, 'reason': 'Insufficient overlapping data'}
            
        aligned_forecast = forecast_values.reindex(common_index).fillna(0)
        aligned_price = price_data.reindex(common_index).ffill()
        
        # Calculate future returns for forecast evaluation
        price_returns = aligned_price.pct_change().shift(-1)  # Next period return
        
        # Test forecast predictive power
        forecast_strength = aligned_forecast.abs()
        forecast_direction = np.sign(aligned_forecast)
        
        # Calculate forecast-return correlation (key test)
        try:
            # Remove last observation (no future return available)
            test_forecast = aligned_forecast[:-1]
            test_returns = price_returns[:-1]
            
            # Remove NaN values
            valid_mask = ~(test_forecast.isna() | test_returns.isna())
            test_forecast = test_forecast[valid_mask]
            test_returns = test_returns[valid_mask]
            
            if len(test_forecast) > 10:
                forecast_return_corr = np.corrcoef(test_forecast, test_returns)[0, 1]
                if np.isnan(forecast_return_corr):
                    forecast_return_corr = 0.0
            else:
                forecast_return_corr = 0.0
        except (ValueError, IndexError, TypeError):
            forecast_return_corr = 0.0
        
        # Calculate hit rate (% of correct directional calls)
        try:
            direction_forecast = forecast_direction[:-1]
            direction_returns = price_returns[:-1]
            
            valid_mask = ~(direction_forecast.isna() | direction_returns.isna())
            direction_forecast = direction_forecast[valid_mask]
            direction_returns = direction_returns[valid_mask]
            
            if len(direction_forecast) > 0:
                correct_direction = (direction_forecast * direction_returns) > 0
                hit_rate = correct_direction.sum() / len(correct_direction)
            else:
                hit_rate = 0.0
        except (ValueError, IndexError, TypeError):
            hit_rate = 0.0
        
        # Calculate average return when forecast is strong
        try:
            strong_forecasts = forecast_strength > forecast_strength.quantile(0.75)
            strong_forecast_returns = price_returns[:-1][strong_forecasts[:-1]]
            avg_strong_forecast_return = strong_forecast_returns.mean() if len(strong_forecast_returns) > 0 else 0
        except (ValueError, IndexError, TypeError):
            avg_strong_forecast_return = 0.0
        
        # Forecast volatility (measure of activity)
        forecast_volatility = aligned_forecast.std()
        
        return {
            'valid': True,
            'rule_name': rule_name,
            'forecast_return_correlation': forecast_return_corr,
            'hit_rate': hit_rate,
            'avg_strong_forecast_return': avg_strong_forecast_return,
            'forecast_volatility': forecast_volatility,
            'total_observations': len(aligned_forecast),
            'forecast_mean': aligned_forecast.mean(),
            'forecast_std': aligned_forecast.std()
        }
    
    def _combine_rule_signals(self,
                             rule_signals: Dict,
                             prepared_data: Dict,
                             signal_weights: Dict[str, float] = None) -> Dict[str, Forecast]:
        """Combine multiple rule signals (renamed from _process_forecasts)"""
        return self._process_forecasts(rule_signals, prepared_data, signal_weights)

    def _calculate_position_sizes(self,
                             processed_signals: Dict[str, Forecast],
                             prepared_data: Dict) -> PositionSeries:
        """Calculate position sizes based on trading signals (SECONDARY PURPOSE)"""
        positions = {}

        for instrument_name, signal in processed_signals.items():
            if instrument_name in self.instruments:
                instrument = self.instruments[instrument_name]
                price_data = prepared_data['prices'][instrument_name]
                volatility = prepared_data['volatilities'].get(instrument_name)

                # Calculate position size based on signal
                position = self.position_sizer.calculate_position_size(
                    signal, price_data, instrument, volatility=volatility
                )

                positions[instrument_name] = position

        return PositionSeries(positions)

    def _apply_risk_controls(self,
                               raw_positions: PositionSeries,
                               prepared_data: Dict) -> PositionSeries:
        """Apply risk controls to position sizes (SECONDARY PURPOSE)"""
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
