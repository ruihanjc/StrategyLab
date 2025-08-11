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
        forecasts = {}
        
        self.logger.debug(f"Generating signals for {len(self.trading_rules)} rules")
        
        for rule in self.trading_rules:
            try:
                # Get the rule's data and parameters
                rule_data = rule.get_data()
                rule_function = rule.get_rule()
                rule_params = rule.params
                
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
                        forecast_value = rule_function(rule_data, rule_params)
                        
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
                            
                            # Combine forecasts for the same instrument if multiple rules exist
                            if instrument_name in forecasts:
                                # Simple average combination for now
                                existing_forecast = forecasts[instrument_name]
                                combined_value = (existing_forecast.iloc[-1] + forecast.iloc[-1]) / 2
                                combined_series = pd.Series([combined_value], index=[current_date])
                                forecasts[instrument_name] = Forecast(combined_series)
                            else:
                                forecasts[instrument_name] = forecast
                                
                            self.logger.debug(f"Generated forecast for {instrument_name}: {forecast.iloc[-1]:.2f}")
                        
                    except Exception as e:
                        self.logger.error(f"Error in rule execution for {instrument_name}: {str(e)}")
                        continue
            except Exception as e:
                self.logger.error(f"Error executing trading rule {rule_function.__name__}: {str(e)}")
                continue
        
        return forecasts
    
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
