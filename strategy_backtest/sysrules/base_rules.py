from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import yaml

# Import config manager
from strategy_backtest.sysutils.config import ConfigManager


class Strategy(ABC):
    """
    Abstract base class for all trading strategies
    Now with YAML configuration support
    """

    def __init__(self, config=None):
        """
        Initialize strategy with optional config

        Parameters:
        -----------
        config: dict or str
            Either a config dict or a string with strategy name to load from YAML
        """
        self.parameters = {}

        if config is not None:
            if isinstance(config, dict):
                # Use provided config dictionary
                self.parameters = config.get('parameters', {})
                self.name = config.get('name', self.__class__.__name__)
                self.description = config.get('description', '')
                self.signal_rules = config.get('signal_rules', {})
            elif isinstance(config, str):
                # Load from YAML using strategy name
                self._load_config_from_yaml(config)
            else:
                raise ValueError("Config must be a dictionary or string (strategy name)")
        else:
            # Default parameters from child class
            self.name = self.__class__.__name__
            self.description = ''
            self.signal_rules = {}

    def _load_config_from_yaml(self, strategy_name):
        """Load configuration from YAML file"""
        config_manager = ConfigManager()
        config = config_manager.get_strategy_config(strategy_name)

        self.parameters = config.get('parameters', {})
        self.name = config.get('name', self.__class__.__name__)
        self.description = config.get('description', '')
        self.signal_rules = config.get('signal_rules', {})

    @abstractmethod
    def generate_signals(self, data, price_column='close'):
        """
        Generate trading signals based on market data
        Must be implemented by child classes
        """
        pass

    def set_parameters(self, parameters):
        """Update strategy parameters"""
        self.parameters.update(parameters)

    def get_parameters(self):
        """Get current strategy parameters"""
        return self.parameters

    def save_config(self, file_name=None):
        """
        Save current configuration to YAML file

        Parameters:
        -----------
        file_name: str
            File name (without extension) to save configuration
            If None, use strategy name
        """
        if file_name is None:
            file_name = f"../sysutils/{self.name.lower().replace(' ', '_')}"

        config = {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
            'signal_rules': self.signal_rules
        }

        config_manager = ConfigManager()
        config_manager.create_config(file_name, config)

        return file_name