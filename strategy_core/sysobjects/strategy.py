import pandas as pd
import logging
from typing import Dict, List
from datetime import datetime

from .forecasts import Forecast, ForecastCombination


class Strategy:
    """
    Trading strategy that combines multiple trading rules
    Generates signals and forecasts for portfolio
    """

    def __init__(self, trading_rules: list):
        """
        Initialize strategy with trading rules

        Parameters:
        -----------
        trading_rules: list
            List of TradingRule objects
        """
        self.parameters = {}
        self.trading_rules = trading_rules
        self.logger = logging.getLogger(f"{__name__}.Strategy")
        
    def generate_signals(self, current_date: datetime, price_data: Dict) -> Dict[str, Forecast]:
        """
        Generate trading signals/forecasts for all instruments
        
        Parameters:
        -----------
        current_date: datetime
            Current date for signal generation
        price_data: Dict
            Current price data for all instruments
            
        Returns:
        --------
        Dict[str, Forecast]
            Forecasts for each instrument
        """
        raw_forecasts = {}
        
        self.logger.debug(f"Generating signals for {len(self.trading_rules)} rules")
        
        # First, generate individual rule forecasts
        for rule in self.trading_rules:
            try:
                # Get the rule's data and parameters
                rule_data = rule.get_data()
                rule_function = rule.get_rule()
                rule_params = rule.params
                
                # Slice the price data to current point in time for point-in-time calculation
                sliced_rule_data = self._slice_data_to_current_date(rule_data, current_date)
                
                # Extract instrument name from rule data
                if isinstance(rule_data, dict) and 'ticker' in rule_data:
                    instrument_name = rule_data['ticker']
                elif hasattr(rule_data, 'ticker'):
                    instrument_name = rule_data.ticker
                elif hasattr(rule_data, 'name'):
                    instrument_name = rule_data.name
                else:
                    # Try to get from price data keys - rule_data might be the price data itself
                    if len(price_data) == 1:
                        instrument_name = list(price_data.keys())[0]
                    else:
                        # Last resort: try to match by checking what rule_data contains
                        self.logger.debug(f"Rule data type: {type(rule_data)}")
                        # Skip if can't determine instrument
                        self.logger.warning(f"Could not determine instrument for rule {rule_function.__name__}")
                        continue
                
                # Ensure instrument_name is a string, not a Series
                if isinstance(instrument_name, pd.Series):
                    self.logger.error(f"instrument_name is a Series: {instrument_name}")
                    continue
                elif not isinstance(instrument_name, str):
                    instrument_name = str(instrument_name)
                
                # Get price series for this instrument
                if instrument_name in price_data:
                    # Execute the trading rule to get forecast
                    try:
                        forecast_value = rule_function(sliced_rule_data, rule_params)
                        
                        if forecast_value is not None:
                            # Create forecast series with single value
                            if isinstance(forecast_value, (int, float)):
                                forecast_series = pd.Series([forecast_value], index=[current_date])
                            elif isinstance(forecast_value, pd.Series):
                                # Take the latest value if series provided
                                latest_value = forecast_value.iloc[-1] if not forecast_value.empty else 0
                                forecast_series = pd.Series([latest_value], index=[current_date])
                            else:
                                forecast_series = pd.Series([0], index=[current_date])
                            
                            # Create Forecast object
                            forecast = Forecast(forecast_series)
                            
                            # Store raw forecasts by rule_instrument key
                            rule_key = f"{rule_function.__name__}_{instrument_name}"
                            raw_forecasts[rule_key] = forecast
                                
                            self.logger.debug(f"Generated forecast for {rule_key}: {forecast.iloc[-1]:.2f}")
                        
                    except Exception as e:
                        self.logger.error(f"Error in rule execution for {instrument_name}: {str(e)}")
                        continue
            except Exception as e:
                self.logger.error(f"Error executing trading rule {rule_function.__name__}: {str(e)}")
                continue
        
        # Now combine forecasts using weights
        return self._combine_forecasts_with_weights(raw_forecasts)
    
    def _combine_forecasts_with_weights(self, raw_forecasts: Dict[str, Forecast]) -> Dict[str, Forecast]:
        """Combine raw forecasts using configured weights"""
        if not hasattr(self, 'rule_weights'):
            # No weights configured, use simple average
            return self._simple_average_combination(raw_forecasts)
        
        # Group forecasts by instrument
        instrument_forecasts = {}
        for rule_key, forecast in raw_forecasts.items():
            instrument = rule_key.split('_', 1)[1]  # Extract instrument from rule_key
            if instrument not in instrument_forecasts:
                instrument_forecasts[instrument] = {}
            instrument_forecasts[instrument][rule_key] = forecast
        
        # Combine forecasts for each instrument using weights
        final_forecasts = {}
        for instrument, forecasts in instrument_forecasts.items():
            if len(forecasts) == 1:
                # Single forecast, no combination needed
                final_forecasts[instrument] = list(forecasts.values())[0]
            else:
                # Multiple forecasts, combine with weights
                from ..sysriskutils.forecast_processing import ForecastCombiner
                combiner = ForecastCombiner()
                
                # Extract weights for this instrument's rules
                weights = {}
                for rule_key in forecasts.keys():
                    if rule_key in self.rule_weights:
                        weights[rule_key] = self.rule_weights[rule_key]
                    else:
                        weights[rule_key] = 1.0
                
                # Combine forecasts
                combined_forecast = combiner.combine_forecasts(forecasts, weights)
                final_forecasts[instrument] = combined_forecast
        
        return final_forecasts
    
    def _simple_average_combination(self, raw_forecasts: Dict[str, Forecast]) -> Dict[str, Forecast]:
        """Simple average combination when no weights are available"""
        instrument_forecasts = {}
        for rule_key, forecast in raw_forecasts.items():
            instrument = rule_key.split('_', 1)[1]
            if instrument not in instrument_forecasts:
                instrument_forecasts[instrument] = []
            instrument_forecasts[instrument].append(forecast)
        
        final_forecasts = {}
        for instrument, forecasts in instrument_forecasts.items():
            if len(forecasts) == 1:
                final_forecasts[instrument] = forecasts[0]
            else:
                # Simple average
                combined_value = sum(f.iloc[-1] for f in forecasts) / len(forecasts)
                combined_series = pd.Series([combined_value], index=[forecasts[0].index[-1]])
                final_forecasts[instrument] = Forecast(combined_series)
        
        return final_forecasts
    
    def _slice_data_to_current_date(self, rule_data, current_date):
        """
        Slice price data to only include data up to current_date to avoid look-ahead bias
        
        Parameters:
        -----------
        rule_data: dict or other
            Rule data containing price information
        current_date: datetime
            Current date for point-in-time slicing
            
        Returns:
        --------
        dict or other
            Sliced rule data
        """
        if not isinstance(rule_data, dict) or 'price_data' not in rule_data:
            return rule_data
            
        try:
            price_data = rule_data['price_data']
            
            # Handle MultiplePrices object
            if hasattr(price_data, 'data') and hasattr(price_data.data, 'index'):
                # Check if current_date is within the data range
                if current_date < price_data.data.index[0]:
                    # Current date is before this instrument's data starts
                    # Return empty/minimal data to avoid processing
                    empty_df = pd.DataFrame(index=[current_date], columns=price_data.data.columns)
                    empty_df = empty_df.fillna(0)  # Only in this case we use 0 to signal no data
                    sliced_rule_data = rule_data.copy()
                    sliced_price_obj = type(price_data)(empty_df)
                    sliced_rule_data['price_data'] = sliced_price_obj
                    return sliced_rule_data
                
                # Slice the underlying DataFrame
                sliced_df = price_data.data.loc[:current_date].copy()
                
                # Handle NaN values properly - forward fill, don't use 0
                for col in sliced_df.columns:
                    if sliced_df[col].dtype in ['float64', 'int64']:
                        sliced_df[col] = sliced_df[col].ffill()
                
                # Create new price data object with sliced data
                sliced_rule_data = rule_data.copy()
                
                # Create new MultiplePrices-like object with sliced data
                sliced_price_obj = type(price_data)(sliced_df)
                sliced_rule_data['price_data'] = sliced_price_obj
                
                return sliced_rule_data
                
            # Handle Series data
            elif hasattr(price_data, 'index'):
                # Check if current_date is within the data range
                if len(price_data.index) > 0 and current_date < price_data.index[0]:
                    # Current date is before this instrument's data starts
                    empty_series = pd.Series([0], index=[current_date])  # Signal no data
                    sliced_rule_data = rule_data.copy()
                    sliced_rule_data['price_data'] = empty_series
                    return sliced_rule_data
                    
                sliced_data = price_data.loc[:current_date].copy()
                
                # Forward fill NaN values for price data (handle both Series and DataFrame)
                if isinstance(sliced_data, pd.DataFrame):
                    # Handle DataFrame - check each column
                    for col in sliced_data.columns:
                        if sliced_data[col].dtype in ['float64', 'int64', 'float32', 'int32'] or pd.api.types.is_numeric_dtype(sliced_data[col]):
                            sliced_data[col] = sliced_data[col].astype('float64').ffill()
                        else:
                            sliced_data[col] = sliced_data[col].infer_objects(copy=False).ffill()
                else:
                    # Handle Series
                    if sliced_data.dtype in ['float64', 'int64', 'float32', 'int32'] or pd.api.types.is_numeric_dtype(sliced_data):
                        sliced_data = sliced_data.astype('float64').ffill()
                    else:
                        sliced_data = sliced_data.infer_objects(copy=False).ffill()
                
                sliced_rule_data = rule_data.copy()
                sliced_rule_data['price_data'] = sliced_data
                return sliced_rule_data
                
            else:
                return rule_data
                
        except Exception as e:
            self.logger.warning(f"Could not slice data to current date: {str(e)}")
            return rule_data
    
    def add_trading_rule(self, trading_rule):
        """
        Add a new trading rule to the strategy
        
        Parameters:
        -----------
        trading_rule: TradingRule
            Trading rule to add
        """
        self.trading_rules.append(trading_rule)
        self.logger.info(f"Added trading rule: {trading_rule.get_rule().__name__}")
    
    def remove_trading_rule(self, rule_index: int):
        """
        Remove a trading rule by index
        
        Parameters:
        -----------
        rule_index: int
            Index of rule to remove
        """
        if 0 <= rule_index < len(self.trading_rules):
            removed_rule = self.trading_rules.pop(rule_index)
            self.logger.info(f"Removed trading rule: {removed_rule.get_rule().__name__}")
        else:
            raise IndexError(f"Rule index {rule_index} out of range")
    
    def get_strategy_summary(self) -> Dict:
        """
        Get summary of strategy configuration
        
        Returns:
        --------
        Dict
            Strategy summary
        """
        rule_info = []
        for rule in self.trading_rules:
            rule_info.append({
                'function': rule.get_rule().__name__,
                'params': rule.params,
                'has_data': rule.get_data() is not None
            })
        
        return {
            'total_rules': len(self.trading_rules),
            'parameters': self.parameters,
            'rules': rule_info
        }
    
    def validate_strategy(self, price_data: Dict) -> bool:
        """
        Validate that strategy can execute with given price data
        
        Parameters:
        -----------
        price_data: Dict
            Price data to validate against
            
        Returns:
        --------
        bool
            True if strategy is valid
        """
        if not self.trading_rules:
            self.logger.error("No trading rules defined")
            return False
        
        valid_rules = 0
        for rule in self.trading_rules:
            try:
                rule_data = rule.get_data()
                if rule_data is not None:
                    valid_rules += 1
                else:
                    self.logger.warning(f"Rule {rule.get_rule().__name__} has no data")
            except Exception as e:
                self.logger.error(f"Error validating rule: {str(e)}")
        
        if valid_rules == 0:
            self.logger.error("No valid trading rules found")
            return False
        
        self.logger.info(f"Strategy validation passed: {valid_rules}/{len(self.trading_rules)} rules valid")
        return True
