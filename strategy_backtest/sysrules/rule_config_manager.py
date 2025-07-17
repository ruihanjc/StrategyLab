"""
Trading rule configuration management system
Based on pysystemtrade configuration patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import yaml
import json
from pathlib import Path
import warnings

from .rule_factory import TradingRuleFactory, TradingRuleManager, TradingRuleSet


class RuleConfigManager:
    """
    Manages trading rule configurations
    Similar to pysystemtrade configuration management
    """
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config_data = {}
        self.rule_factory = TradingRuleFactory()
        
        # Load default config if path provided
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
        else:
            self._load_default_config()
    
    def _load_default_config(self):
        """Load default configuration"""
        # Get default config path
        default_config_path = Path(__file__).parent.parent / 'sysconfigs' / 'trading_rules_config.yaml'
        
        if default_config_path.exists():
            self.load_config(str(default_config_path))
        else:
            # Create minimal default config
            self.config_data = {
                'auto_calculate_scalars': True,
                'default_forecast_cap': 20.0,
                'default_vol_days': 35,
                'rules': {},
                'rule_sets': {},
                'validation': {
                    'max_forecast_scalar': 20.0,
                    'min_forecast_scalar': 0.1,
                    'max_lookback_periods': 500,
                    'min_lookback_periods': 2,
                    'required_data_points': 100
                }
            }
    
    def load_config(self, config_path: str):
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            self.config_data = yaml.safe_load(f)
        
        self.config_path = config_path
        
        # Validate configuration
        self._validate_config()
    
    def save_config(self, config_path: str = None):
        """Save configuration to file"""
        if config_path is None:
            config_path = self.config_path
        
        if config_path is None:
            raise ValueError("No config path specified")
        
        with open(config_path, 'w') as f:
            yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
    
    def _validate_config(self):
        """Validate configuration structure"""
        required_sections = ['rules', 'rule_sets', 'validation']
        
        for section in required_sections:
            if section not in self.config_data:
                warnings.warn(f"Missing configuration section: {section}")
                self.config_data[section] = {}
        
        # Validate individual rules
        for rule_name, rule_config in self.config_data.get('rules', {}).items():
            self._validate_rule_config(rule_name, rule_config)
        
        # Validate rule sets
        for set_name, rule_weights in self.config_data.get('rule_sets', {}).items():
            self._validate_rule_set(set_name, rule_weights)
    
    def _validate_rule_config(self, rule_name: str, rule_config: Dict):
        """Validate individual rule configuration"""
        required_fields = ['rule_class', 'parameters']
        
        for field in required_fields:
            if field not in rule_config:
                raise ValueError(f"Rule {rule_name} missing required field: {field}")
        
        # Validate forecast scalar
        forecast_scalar = rule_config.get('forecast_scalar', 1.0)
        validation = self.config_data.get('validation', {})
        
        min_scalar = validation.get('min_forecast_scalar', 0.1)
        max_scalar = validation.get('max_forecast_scalar', 20.0)
        
        if not (min_scalar <= forecast_scalar <= max_scalar):
            warnings.warn(f"Rule {rule_name} forecast scalar {forecast_scalar} outside valid range [{min_scalar}, {max_scalar}]")
    
    def _validate_rule_set(self, set_name: str, rule_weights: Dict):
        """Validate rule set configuration"""
        # Check that all rules in set exist
        available_rules = set(self.config_data.get('rules', {}).keys())
        
        for rule_name in rule_weights.keys():
            if rule_name not in available_rules:
                warnings.warn(f"Rule set {set_name} references unknown rule: {rule_name}")
        
        # Check weights sum to approximately 1.0
        total_weight = sum(rule_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            warnings.warn(f"Rule set {set_name} weights sum to {total_weight}, not 1.0")
    
    def get_rule_config(self, rule_name: str) -> Dict:
        """Get configuration for a specific rule"""
        if rule_name not in self.config_data.get('rules', {}):
            raise ValueError(f"Unknown rule: {rule_name}")
        
        return self.config_data['rules'][rule_name].copy()
    
    def get_rule_set(self, set_name: str) -> Dict[str, float]:
        """Get rule set configuration"""
        if set_name not in self.config_data.get('rule_sets', {}):
            raise ValueError(f"Unknown rule set: {set_name}")
        
        return self.config_data['rule_sets'][set_name].copy()
    
    def add_rule(self, rule_name: str, rule_config: Dict):
        """Add a new rule configuration"""
        # Validate the rule config
        self._validate_rule_config(rule_name, rule_config)
        
        # Add to configuration
        if 'rules' not in self.config_data:
            self.config_data['rules'] = {}
        
        self.config_data['rules'][rule_name] = rule_config
    
    def remove_rule(self, rule_name: str):
        """Remove a rule configuration"""
        if rule_name in self.config_data.get('rules', {}):
            del self.config_data['rules'][rule_name]
        
        # Remove from any rule sets
        for set_name, rule_weights in self.config_data.get('rule_sets', {}).items():
            if rule_name in rule_weights:
                del rule_weights[rule_name]
                # Renormalize weights
                total_weight = sum(rule_weights.values())
                if total_weight > 0:
                    for rule in rule_weights:
                        rule_weights[rule] /= total_weight
    
    def add_rule_set(self, set_name: str, rule_weights: Dict[str, float]):
        """Add a new rule set"""
        # Validate the rule set
        self._validate_rule_set(set_name, rule_weights)
        
        # Add to configuration
        if 'rule_sets' not in self.config_data:
            self.config_data['rule_sets'] = {}
        
        self.config_data['rule_sets'][set_name] = rule_weights
    
    def remove_rule_set(self, set_name: str):
        """Remove a rule set"""
        if set_name in self.config_data.get('rule_sets', {}):
            del self.config_data['rule_sets'][set_name]
    
    def list_available_rules(self) -> List[str]:
        """List all available rules"""
        return list(self.config_data.get('rules', {}).keys())
    
    def list_available_rule_sets(self) -> List[str]:
        """List all available rule sets"""
        return list(self.config_data.get('rule_sets', {}).keys())
    
    def get_rule_info(self, rule_name: str) -> Dict:
        """Get detailed information about a rule"""
        if rule_name not in self.config_data.get('rules', {}):
            raise ValueError(f"Unknown rule: {rule_name}")
        
        rule_config = self.config_data['rules'][rule_name]
        
        return {
            'name': rule_name,
            'class': rule_config['rule_class'],
            'parameters': rule_config['parameters'],
            'forecast_scalar': rule_config.get('forecast_scalar', 1.0),
            'description': rule_config.get('description', ''),
            'data_requirements': self._get_rule_data_requirements(rule_config['rule_class'])
        }
    
    def _get_rule_data_requirements(self, rule_class: str) -> List[str]:
        """Get data requirements for a rule class"""
        data_requirements = {
            'EWMAC': ['price'],
            'Breakout': ['price'],
            'Momentum': ['price'],
            'Acceleration': ['price'],
            'MeanReversion': ['price'],
            'Carry': ['carry'],
            'RelativeMomentum': ['price', 'benchmark'],
            'Volatility': ['price']
        }
        
        return data_requirements.get(rule_class, ['price'])
    
    def create_rule_manager(self, rule_set: str = None, rules: List[str] = None) -> TradingRuleManager:
        """Create a rule manager from configuration"""
        manager = TradingRuleManager(
            auto_calculate_scalars=self.config_data.get('auto_calculate_scalars', True)
        )
        
        if rule_set is not None:
            # Use predefined rule set
            if rule_set not in self.config_data.get('rule_sets', {}):
                raise ValueError(f"Unknown rule set: {rule_set}")
            
            rule_weights = self.config_data['rule_sets'][rule_set]
            rules_to_add = list(rule_weights.keys())
        elif rules is not None:
            # Use specified rules
            rules_to_add = rules
        else:
            # Use all available rules
            rules_to_add = self.list_available_rules()
        
        # Add rules to manager
        for rule_name in rules_to_add:
            if rule_name in self.config_data.get('rules', {}):
                rule_config = self.config_data['rules'][rule_name]
                manager.add_rule(rule_name, rule_config)
        
        return manager
    
    def get_rule_statistics(self) -> pd.DataFrame:
        """Get statistics about all rules"""
        stats_data = []
        
        for rule_name, rule_config in self.config_data.get('rules', {}).items():
            rule_class = rule_config['rule_class']
            parameters = rule_config['parameters']
            
            # Extract key parameters
            lookback = None
            if 'lookback' in parameters:
                lookback = parameters['lookback']
            elif 'Lslow' in parameters:
                lookback = parameters['Lslow']
            
            vol_days = parameters.get('vol_days', self.config_data.get('default_vol_days', 35))
            
            stats_data.append({
                'Rule Name': rule_name,
                'Rule Class': rule_class,
                'Lookback': lookback,
                'Vol Days': vol_days,
                'Forecast Scalar': rule_config.get('forecast_scalar', 1.0),
                'Description': rule_config.get('description', '')
            })
        
        return pd.DataFrame(stats_data)
    
    def get_rule_set_statistics(self) -> pd.DataFrame:
        """Get statistics about rule sets"""
        stats_data = []
        
        for set_name, rule_weights in self.config_data.get('rule_sets', {}).items():
            num_rules = len(rule_weights)
            max_weight = max(rule_weights.values()) if rule_weights else 0
            min_weight = min(rule_weights.values()) if rule_weights else 0
            weight_concentration = max_weight / (1.0 / num_rules) if num_rules > 0 else 0
            
            stats_data.append({
                'Rule Set': set_name,
                'Number of Rules': num_rules,
                'Max Weight': max_weight,
                'Min Weight': min_weight,
                'Weight Concentration': weight_concentration
            })
        
        return pd.DataFrame(stats_data)
    
    def optimize_forecast_scalars(self, 
                                price_data: Dict[str, pd.Series],
                                rule_subset: List[str] = None,
                                target_abs_forecast: float = 10.0) -> Dict[str, float]:
        """
        Optimize forecast scalars based on historical data
        
        Parameters:
        -----------
        price_data: Dict[str, pd.Series]
            Historical price data by instrument
        rule_subset: List[str]
            Subset of rules to optimize (optional)
        target_abs_forecast: float
            Target absolute forecast level
            
        Returns:
        --------
        Dict[str, float]
            Optimized forecast scalars by rule
        """
        from ..sysutils.math_algorithms import calculate_forecast_scalar
        
        optimized_scalars = {}
        
        # Get rules to optimize
        if rule_subset is None:
            rules_to_optimize = self.list_available_rules()
        else:
            rules_to_optimize = rule_subset
        
        # Create temporary manager
        temp_manager = self.create_rule_manager(rules=rules_to_optimize)
        
        # Calculate optimal scalars for each rule
        for rule_name in rules_to_optimize:
            rule_forecasts = []
            
            # Generate forecasts for each instrument
            for instrument, prices in price_data.items():
                try:
                    forecast = temp_manager.generate_forecast(rule_name, prices)
                    if not forecast.empty:
                        rule_forecasts.append(forecast)
                except Exception as e:
                    warnings.warn(f"Failed to generate forecast for {rule_name} on {instrument}: {e}")
            
            # Combine forecasts and calculate scalar
            if rule_forecasts:
                combined_forecast = pd.concat(rule_forecasts)
                optimal_scalar = calculate_forecast_scalar(
                    combined_forecast, target_abs_forecast
                )
                optimized_scalars[rule_name] = optimal_scalar
        
        return optimized_scalars
    
    def update_forecast_scalars(self, new_scalars: Dict[str, float]):
        """Update forecast scalars in configuration"""
        for rule_name, scalar in new_scalars.items():
            if rule_name in self.config_data.get('rules', {}):
                self.config_data['rules'][rule_name]['forecast_scalar'] = scalar
    
    def export_config_summary(self, filepath: str = None) -> str:
        """Export configuration summary"""
        summary = []
        
        summary.append("=== TRADING RULES CONFIGURATION SUMMARY ===\n")
        
        # Global settings
        summary.append("Global Settings:")
        summary.append(f"  Auto Calculate Scalars: {self.config_data.get('auto_calculate_scalars', True)}")
        summary.append(f"  Default Forecast Cap: {self.config_data.get('default_forecast_cap', 20.0)}")
        summary.append(f"  Default Vol Days: {self.config_data.get('default_vol_days', 35)}")
        
        # Rule statistics
        summary.append(f"\nRules: {len(self.config_data.get('rules', {}))}")
        
        # Rule sets
        summary.append(f"\nRule Sets: {len(self.config_data.get('rule_sets', {}))}")
        for set_name, rule_weights in self.config_data.get('rule_sets', {}).items():
            summary.append(f"  {set_name}: {len(rule_weights)} rules")
        
        # Rule class breakdown
        rule_classes = {}
        for rule_config in self.config_data.get('rules', {}).values():
            rule_class = rule_config['rule_class']
            rule_classes[rule_class] = rule_classes.get(rule_class, 0) + 1
        
        summary.append("\nRule Class Breakdown:")
        for rule_class, count in sorted(rule_classes.items()):
            summary.append(f"  {rule_class}: {count} rules")
        
        summary_text = "\n".join(summary)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(summary_text)
        
        return summary_text


def create_sample_config_manager():
    """Create sample configuration manager for testing"""
    manager = RuleConfigManager()
    
    # Print configuration summary
    print(manager.export_config_summary())
    
    # Create rule manager
    rule_manager = manager.create_rule_manager(rule_set='mixed_strategy')
    
    # Print rule summary
    print("\nRule Manager Summary:")
    print(rule_manager.get_rule_summary())
    
    return {
        'config_manager': manager,
        'rule_manager': rule_manager
    }


if __name__ == "__main__":
    # Test the configuration manager
    sample = create_sample_config_manager()
    print("Configuration manager created successfully!")