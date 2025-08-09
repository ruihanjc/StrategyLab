import yaml
import os
from typing import Dict, Any
import configparser
from datetime import datetime, timedelta


class ConfigManager:
    def __init__(self, config_dir, mode):
        self.config = configparser.ConfigParser()
        self.project_dir = os.path.abspath(__file__ + "/../../../")
        self.mode = mode
        self.config_path = os.path.join(self.project_dir, f"{config_dir}/config/{mode}_config.yaml")
        self.configs = {}

    def load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.configs[self.mode] = config
                return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {self.config_path}: {str(e)}")

    def get_database_config(self):
        return self.config['DATABASE']

    def get_strategy_config(self) -> Dict[str, Any]:
        return self.load_config()

    def get_settings(self) -> Dict[str, Any]:
        current_config = self.load_config()

        if current_config['end_date'] == '':
            yesterday = datetime.today().date() - timedelta(days=1)
            current_config['end_date'] = yesterday.strftime('%Y-%m-%d')

        return current_config

    def create_config(self, config_name: str, config_data: Dict[str, Any]) -> None:
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
