import configparser
import os
import json
import yaml
from pathlib import Path

class ConfigManager:
    def __init__(self, config_path='./config.ini'):
        self.config = configparser.ConfigParser()
        self.config_path = os.path.abspath(__file__) + '/../config.ini'
        current_dir = os.path.abspath(__file__) + '/../../'
        self.arctic_dir = current_dir + 'arcticdb'
        self.load_config()
    
    def load_config(self):
        """Load configuration from INI file"""
        if os.path.exists(self.config_path):
            self.config.read(self.config_path)
        else:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
    
    def get_database_config(self):
        """Get database configuration section"""
        self.config['DATABASE'] = {
            'local_storage': str(self.arctic_dir),
            'map_size': '500MB'
        }
        return self.config['DATABASE']

    def get_logging_config(self):
        """Get logging configuration section"""
        return dict(self.config['Logging'])
    
    def get_api_config(self):
        """Get API configuration section"""
        print(self.config)
        return dict(self.config['API'])
    
    def update_config(self, section, key, value):
        """Update a specific configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = str(value)
        with open(self.config_path, 'w') as config_file:
            self.config.write(config_file)

# Usage example