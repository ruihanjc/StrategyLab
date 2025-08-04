import os
import yaml
import re
from dotenv import load_dotenv

class ConfigManager:
    def __init__(self, config_path='./config.yaml'):
        # Load environment variables from .env file
        load_dotenv()
        
        current_dir = os.path.dirname(os.path.dirname(__file__))
        self.config_path = os.path.join(current_dir, 'configuration', 'configs', 'config.yaml')
        self.arctic_dir = os.path.join(current_dir, '../arcticdb')
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file and substitute environment variables"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as file:
                content = file.read()
                # Substitute environment variables ${VAR_NAME}
                content = re.sub(r'\$\{([^}]+)\}', lambda m: os.getenv(m.group(1), ''), content)
                self.config = yaml.safe_load(content)
        else:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
    
    def get_database_config(self):
        """Get database configuration section"""
        db_config = self.config.get('database', {}).copy()
        db_config['local_storage'] = str(self.arctic_dir)
        return db_config

    def get_logging_config(self):
        """Get logging configuration section"""
        return self.config.get('logging', {})
    
    def get_api_config(self):
        """Get API configuration section"""
        return self.config.get('api', {})

    def get_daily_update_config(self):
        """Get update services configuration section"""
        return self.config.get('update_services', {})

    def update_config(self, section, key, value):
        """Update a specific configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        with open(self.config_path, 'w') as config_file:
            yaml.safe_dump(self.config, config_file, default_flow_style=False)

# Usage example