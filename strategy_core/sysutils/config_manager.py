import yaml
import os
from typing import Dict, Any, List, Optional, Union
import configparser
from datetime import datetime, timedelta


class ConfigManager:
    """
    Configuration manager for loading and validating YAML configuration
    """

    def __init__(self, config_path='./config.ini'):
        """
        Initialize config manager

        Parameters:
        -----------
        project_dir: str
            Directory containing YAML config files
        """
        self.config = configparser.ConfigParser()
        self.project_dir = os.path.abspath(__file__ + "/../../../")  # Use current working directory
        self.configs = {}

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration file

        Parameters:
        -----------
        config_name: str
            Name of the config file (without .yaml extension)
            Can include subdirectories, e.g., 'strategies/ma_crossover'

        Returns:
        --------
        dict
            Configuration as a dictionary
        """
        # Check if already loaded
        if config_name in self.configs:
            return self.configs[config_name]

        # Construct file path
        file_path = os.path.join(self.project_dir, f"strategy_backtest/backtest_configs/{config_name}.yaml")

        # Load and parse YAML
        try:
            with open(file_path, 'r') as file:
                config = yaml.safe_load(file)
                self.configs[config_name] = config
                return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {file_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {file_path}: {str(e)}")

    def get_database_config(self):
        """Get database configuration section"""
        print(self.config)
        return self.config['DATABASE']

    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """
        Load a strategy configuration

        Parameters:
        -----------
        strategy_name: str
            Name of the strategy

        Returns:
        --------
        dict
            Strategy configuration
        """
        return self.load_config(f"strategy_config/{strategy_name}")

    def get_backtest_settings(self) -> Dict[str, Any]:
        """
        Load general backtest settings

        Returns:
        --------
        dict
            Backtest settings
        """

        backtest_config = self.load_config("backtest_config")

        if backtest_config['end_date'] == '':
            yesterday = datetime.today().date() - timedelta(days=1)
            backtest_config['end_date'] = yesterday.strftime('%Y-%m-%d')

        return backtest_config

    def get_data_settings(self) -> Dict[str, Any]:
        """
        Load data loading settings

        Returns:
        --------
        dict
            Data settings
        """
        return self.load_config("data_config")["backtest_services"]

    def create_config(self, config_name: str, config_data: Dict[str, Any]) -> None:
        """
        Create or update a configuration file

        Parameters:
        -----------
        config_name: str
            Name of the config file (without .yaml extension)
        config_data: dict
            Configuration data to save
        """
        # Construct file path
        file_path = os.path.join(self.project_dir, f"{config_name}.yaml")

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save YAML
        try:
            with open(file_path, 'w') as file:
                yaml.dump(config_data, file, default_flow_style=False, sort_keys=False)
                self.configs[config_name] = config_data
        except Exception as e:
            raise IOError(f"Error saving YAML file {file_path}: {str(e)}")
