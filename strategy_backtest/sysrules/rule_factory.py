"""
Trading rule factory and management system
Based on pysystemtrade trading rules architecture
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any, Type
from datetime import datetime
import warnings
import yaml
import json

try:
    from .trading_rules import (
        TradingRuleBase, EWMACRule, BreakoutRule, MomentumRule, 
        AccelerationRule, MeanReversionRule, CarryRule, 
        RelativeMomentumRule, VolatilityRule, create_standard_trading_rules
    )
    from .proper_ewmac import ProperEWMAC
    from ..sysobjects.forecasts import Forecast
    from ..sysobjects.prices import AdjustedPrices
    from ..sysutils.math_algorithms import calculate_forecast_scalar
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    
    from sysrules.trading_rules import (
        TradingRuleBase, EWMACRule, BreakoutRule, MomentumRule, 
        AccelerationRule, MeanReversionRule, CarryRule, 
        RelativeMomentumRule, VolatilityRule, create_standard_trading_rules
    )
    from sysrules.proper_ewmac import ProperEWMAC
    from sysobjects.forecasts import Forecast
    from sysobjects.prices import AdjustedPrices
    from sysutils.math_algorithms import calculate_forecast_scalar


class TradingRuleFactory:
    """
    Factory for creating and managing trading rules
    Similar to pysystemtrade rule management
    """
    
    def __init__(self):
        self.rule_registry = {}
        self.config_templates = {}
        self._register_standard_rules()
    
    def _register_standard_rules(self):
        """Register standard trading rules"""
        # Register rule classes
        self.rule_registry.update({
            'EWMAC': ProperEWMAC,  # Use proper EWMAC implementation
            'Breakout': BreakoutRule,
            'Momentum': MomentumRule,
            'Acceleration': AccelerationRule,
            'MeanReversion': MeanReversionRule,
            'Carry': CarryRule,
            'RelativeMomentum': RelativeMomentumRule,
            'Volatility': VolatilityRule
        })
        
        # Register config templates
        self.config_templates.update({
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
            'ewmac_8_32': {
                'rule_class': 'EWMAC',
                'parameters': {'Lfast': 8, 'Lslow': 32, 'vol_days': 35},
                'forecast_scalar': 7.5
            },
            'ewmac_64_256': {
                'rule_class': 'EWMAC',
                'parameters': {'Lfast': 64, 'Lslow': 256, 'vol_days': 35},
                'forecast_scalar': 7.5
            },
            'breakout_20': {
                'rule_class': 'Breakout',
                'parameters': {'lookback': 20, 'smooth': 5},
                'forecast_scalar': 1.0
            },
            'breakout_40': {
                'rule_class': 'Breakout',
                'parameters': {'lookback': 40, 'smooth': 10},
                'forecast_scalar': 1.0
            },
            'breakout_80': {
                'rule_class': 'Breakout',
                'parameters': {'lookback': 80, 'smooth': 20},
                'forecast_scalar': 1.0
            },
            'breakout_160': {
                'rule_class': 'Breakout',
                'parameters': {'lookback': 160, 'smooth': 40},
                'forecast_scalar': 1.0
            },
            'momentum_20': {
                'rule_class': 'Momentum',
                'parameters': {'lookback': 20, 'vol_days': 35},
                'forecast_scalar': 2.0
            },
            'momentum_40': {
                'rule_class': 'Momentum',
                'parameters': {'lookback': 40, 'vol_days': 35},
                'forecast_scalar': 2.0
            },
            'momentum_80': {
                'rule_class': 'Momentum',
                'parameters': {'lookback': 80, 'vol_days': 35},
                'forecast_scalar': 2.0
            },
            'acceleration_4': {
                'rule_class': 'Acceleration',
                'parameters': {'Lfast': 4, 'vol_days': 35},
                'forecast_scalar': 2.0
            },
            'mean_reversion_20': {
                'rule_class': 'MeanReversion',
                'parameters': {'lookback': 20, 'threshold': 2.0, 'vol_days': 35},
                'forecast_scalar': 1.0
            },
            'carry_90': {
                'rule_class': 'Carry',
                'parameters': {'smooth_days': 90},
                'forecast_scalar': 1.0
            },
            'relative_momentum_40': {
                'rule_class': 'RelativeMomentum',
                'parameters': {'horizon': 40, 'ewma_span': 10},
                'forecast_scalar': 1.0
            },
            'volatility_20': {
                'rule_class': 'Volatility',
                'parameters': {'vol_lookback': 20, 'signal_lookback': 5},
                'forecast_scalar': 1.0
            }
        })
    
    def register_rule(self, name: str, rule_class: Type[TradingRuleBase], config: Dict = None):
        """Register a new trading rule"""
        self.rule_registry[name] = rule_class
        if config:
            self.config_templates[name] = config
    
    def create_rule(self, rule_name: str, **kwargs) -> TradingRuleBase:
        """Create a trading rule instance"""
        if rule_name in self.config_templates:
            config = self.config_templates[rule_name]
            rule_class_name = config['rule_class']
            parameters = config.get('parameters', {})
            
            # Override with provided kwargs
            parameters.update(kwargs)
            
            if rule_class_name in self.rule_registry:
                rule_class = self.rule_registry[rule_class_name]
                return rule_class(**parameters)
            else:
                raise ValueError(f"Unknown rule class: {rule_class_name}")
        else:
            raise ValueError(f"Unknown rule name: {rule_name}")
    
    def get_rule_config(self, rule_name: str) -> Dict:
        """Get configuration for a rule"""
        if rule_name in self.config_templates:
            return self.config_templates[rule_name].copy()
        else:
            raise ValueError(f"Unknown rule name: {rule_name}")
    
    def list_available_rules(self) -> List[str]:
        """List all available rules"""
        return list(self.config_templates.keys())
    
    def get_rule_info(self, rule_name: str) -> Dict:
        """Get detailed information about a rule"""
        if rule_name not in self.config_templates:
            raise ValueError(f"Unknown rule name: {rule_name}")
        
        config = self.config_templates[rule_name]
        rule_class_name = config['rule_class']
        rule_class = self.rule_registry[rule_class_name]
        
        # Create temporary instance to get info
        parameters = config.get('parameters', {})
        rule_instance = rule_class(**parameters)
        
        return {
            'name': rule_name,
            'class': rule_class_name,
            'description': rule_instance.description,
            'parameters': parameters,
            'forecast_scalar': config.get('forecast_scalar', 1.0),
            'data_requirements': rule_instance.get_data_requirements()
        }


class TradingRuleManager:
    """
    Manages trading rules execution and forecasting
    Similar to pysystemtrade rule execution
    """
    
    def __init__(self, 
                 rule_factory: TradingRuleFactory = None,
                 auto_calculate_scalars: bool = True):
        self.rule_factory = rule_factory or TradingRuleFactory()
        self.auto_calculate_scalars = auto_calculate_scalars
        self.rule_instances = {}
        self.rule_configs = {}
        self.forecast_scalars = {}
    
    def add_rule(self, rule_name: str, rule_config: Dict = None, **kwargs):
        """Add a trading rule to the manager"""
        if rule_config is None:
            # Use factory default config
            rule_config = self.rule_factory.get_rule_config(rule_name)
        
        # Create rule instance
        rule_instance = self.rule_factory.create_rule(rule_name, **kwargs)
        
        # Store rule and config
        self.rule_instances[rule_name] = rule_instance
        self.rule_configs[rule_name] = rule_config
        
        # Store forecast scalar
        self.forecast_scalars[rule_name] = rule_config.get('forecast_scalar', 1.0)
    
    def remove_rule(self, rule_name: str):
        """Remove a trading rule"""
        if rule_name in self.rule_instances:
            del self.rule_instances[rule_name]
        if rule_name in self.rule_configs:
            del self.rule_configs[rule_name]
        if rule_name in self.forecast_scalars:
            del self.forecast_scalars[rule_name]
    
    def get_rule_names(self) -> List[str]:
        """Get list of configured rule names"""
        return list(self.rule_instances.keys())
    
    def generate_forecast(self, 
                         rule_name: str, 
                         price_data: pd.Series,
                         additional_data: Dict[str, pd.Series] = None,
                         **kwargs) -> Forecast:
        """
        Generate forecast for a specific rule
        
        Parameters:
        -----------
        rule_name: str
            Name of the trading rule
        price_data: pd.Series
            Price data
        additional_data: Dict[str, pd.Series]
            Additional data required by rule (e.g., carry data)
        **kwargs: dict
            Additional parameters for the rule
            
        Returns:
        --------
        Forecast
            Generated forecast
        """
        if rule_name not in self.rule_instances:
            raise ValueError(f"Rule {rule_name} not configured")
        
        rule_instance = self.rule_instances[rule_name]
        
        try:
            # Get data requirements
            data_requirements = rule_instance.get_data_requirements()
            
            # Prepare data arguments
            if 'carry' in data_requirements:
                if additional_data is None or 'carry' not in additional_data:
                    raise ValueError(f"Rule {rule_name} requires carry data")
                raw_forecast = rule_instance(additional_data['carry'], **kwargs)
            elif 'benchmark' in data_requirements:
                benchmark_data = additional_data.get('benchmark') if additional_data else None
                raw_forecast = rule_instance(price_data, benchmark_data, **kwargs)
            else:
                # Standard price-based rule
                raw_forecast = rule_instance(price_data, **kwargs)
            
            # Apply forecast scalar
            forecast_scalar = self.forecast_scalars[rule_name]
            if self.auto_calculate_scalars:
                # Calculate dynamic scalar
                dynamic_scalar = calculate_forecast_scalar(raw_forecast)
                scaled_forecast = raw_forecast * dynamic_scalar
            else:
                # Use fixed scalar
                scaled_forecast = raw_forecast * forecast_scalar
            
            # Create forecast object
            forecast = Forecast(scaled_forecast, forecast_cap=20.0)
            
            return forecast
            
        except Exception as e:
            warnings.warn(f"Error generating forecast for rule {rule_name}: {str(e)}")
            # Return empty forecast
            return Forecast(pd.Series(dtype=float, index=price_data.index))
    
    def generate_all_forecasts(self, 
                             price_data: pd.Series,
                             additional_data: Dict[str, pd.Series] = None,
                             rule_subset: List[str] = None) -> Dict[str, Forecast]:
        """
        Generate forecasts for all configured rules
        
        Parameters:
        -----------
        price_data: pd.Series
            Price data
        additional_data: Dict[str, pd.Series]
            Additional data for rules
        rule_subset: List[str]
            Subset of rules to run (optional)
            
        Returns:
        --------
        Dict[str, Forecast]
            Dictionary of forecasts by rule name
        """
        forecasts = {}
        
        # Determine which rules to run
        if rule_subset is None:
            rule_names = self.get_rule_names()
        else:
            rule_names = [name for name in rule_subset if name in self.rule_instances]
        
        # Generate forecasts
        for rule_name in rule_names:
            forecast = self.generate_forecast(
                rule_name, price_data, additional_data
            )
            forecasts[rule_name] = forecast
        
        return forecasts
    
    def get_rule_summary(self) -> pd.DataFrame:
        """Get summary of all configured rules"""
        summary_data = []
        
        for rule_name in self.rule_instances:
            rule_info = self.rule_factory.get_rule_info(rule_name)
            summary_data.append({
                'Rule Name': rule_name,
                'Class': rule_info['class'],
                'Description': rule_info['description'],
                'Parameters': str(rule_info['parameters']),
                'Forecast Scalar': self.forecast_scalars[rule_name],
                'Data Requirements': ', '.join(rule_info['data_requirements'])
            })
        
        return pd.DataFrame(summary_data)
    
    def update_forecast_scalar(self, rule_name: str, new_scalar: float):
        """Update forecast scalar for a rule"""
        if rule_name in self.forecast_scalars:
            self.forecast_scalars[rule_name] = new_scalar
        else:
            raise ValueError(f"Rule {rule_name} not configured")
    
    def save_config(self, filepath: str):
        """Save rule configuration to file"""
        config_data = {
            'rules': {},
            'auto_calculate_scalars': self.auto_calculate_scalars
        }
        
        for rule_name in self.rule_instances:
            config_data['rules'][rule_name] = {
                'config': self.rule_configs[rule_name],
                'forecast_scalar': self.forecast_scalars[rule_name]
            }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    
    def load_config(self, filepath: str):
        """Load rule configuration from file"""
        with open(filepath, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Clear existing rules
        self.rule_instances.clear()
        self.rule_configs.clear()
        self.forecast_scalars.clear()
        
        # Load settings
        self.auto_calculate_scalars = config_data.get('auto_calculate_scalars', True)
        
        # Load rules
        for rule_name, rule_data in config_data.get('rules', {}).items():
            rule_config = rule_data['config']
            forecast_scalar = rule_data.get('forecast_scalar', 1.0)
            
            # Add rule
            self.add_rule(rule_name, rule_config)
            self.forecast_scalars[rule_name] = forecast_scalar


class TradingRuleSet:
    """
    Predefined sets of trading rules
    Similar to pysystemtrade rule configurations
    """
    
    @staticmethod
    def get_trend_following_rules() -> Dict[str, float]:
        """Get trend following rule set with weights"""
        return {
            'ewmac_16_64': 0.25,
            'ewmac_32_128': 0.25,
            'ewmac_8_32': 0.25,
            'ewmac_64_256': 0.25
        }
    
    @staticmethod
    def get_breakout_rules() -> Dict[str, float]:
        """Get breakout rule set with weights"""
        return {
            'breakout_20': 0.25,
            'breakout_40': 0.25,
            'breakout_80': 0.25,
            'breakout_160': 0.25
        }
    
    @staticmethod
    def get_momentum_rules() -> Dict[str, float]:
        """Get momentum rule set with weights"""
        return {
            'momentum_20': 0.4,
            'momentum_40': 0.4,
            'acceleration_4': 0.2
        }
    
    @staticmethod
    def get_mixed_rules() -> Dict[str, float]:
        """Get mixed rule set with weights"""
        return {
            'ewmac_16_64': 0.2,
            'ewmac_32_128': 0.2,
            'breakout_20': 0.15,
            'breakout_80': 0.15,
            'momentum_20': 0.15,
            'momentum_40': 0.15
        }
    
    @staticmethod
    def get_comprehensive_rules() -> Dict[str, float]:
        """Get comprehensive rule set with weights"""
        return {
            'ewmac_16_64': 0.15,
            'ewmac_32_128': 0.15,
            'ewmac_8_32': 0.1,
            'breakout_20': 0.1,
            'breakout_40': 0.1,
            'breakout_80': 0.1,
            'momentum_20': 0.1,
            'momentum_40': 0.1,
            'acceleration_4': 0.05,
            'mean_reversion_20': 0.05
        }
    
    @staticmethod
    def get_rule_set(set_name: str) -> Dict[str, float]:
        """Get a specific rule set"""
        rule_sets = {
            'trend_following': TradingRuleSet.get_trend_following_rules(),
            'breakout': TradingRuleSet.get_breakout_rules(),
            'momentum': TradingRuleSet.get_momentum_rules(),
            'mixed': TradingRuleSet.get_mixed_rules(),
            'comprehensive': TradingRuleSet.get_comprehensive_rules()
        }
        
        if set_name in rule_sets:
            return rule_sets[set_name]
        else:
            raise ValueError(f"Unknown rule set: {set_name}")


def create_default_rule_manager(rule_set: str = 'mixed') -> TradingRuleManager:
    """Create a default rule manager with predefined rule set"""
    manager = TradingRuleManager()
    
    # Get rule set
    rules_and_weights = TradingRuleSet.get_rule_set(rule_set)
    
    # Add rules to manager
    for rule_name in rules_and_weights.keys():
        manager.add_rule(rule_name)
    
    return manager


def create_sample_trading_system():
    """Create sample trading system for testing"""
    # Create rule manager
    manager = create_default_rule_manager('mixed')
    
    # Create sample price data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = pd.Series(100 * np.cumprod(1 + returns), index=dates)
    
    # Generate forecasts
    forecasts = manager.generate_all_forecasts(prices)
    
    # Print summary
    print("Rule Summary:")
    print(manager.get_rule_summary())
    
    print("\nForecast Statistics:")
    for rule_name, forecast in forecasts.items():
        if not forecast.empty:
            print(f"{rule_name}: mean={forecast.mean():.3f}, std={forecast.std():.3f}")
    
    return {
        'manager': manager,
        'prices': prices,
        'forecasts': forecasts
    }


if __name__ == "__main__":
    # Test the trading rule system
    sample_system = create_sample_trading_system()
    print("Trading rule system created successfully!")