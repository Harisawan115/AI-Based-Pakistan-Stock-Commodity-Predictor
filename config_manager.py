"""
Configuration management utilities
"""

import configparser
import os
from pathlib import Path

class Config:
    """Configuration manager"""
    
    def __init__(self, config_file='config.ini'):
        self.config = configparser.ConfigParser()
        
        possible_paths = [
            config_file,
            os.path.join(os.path.dirname(__file__), config_file),
            os.path.join(os.path.expanduser('~'), '.psx_predictor', config_file),
            '/etc/psx_predictor/config.ini'
        ]
        
        config_found = False
        for path in possible_paths:
            if os.path.exists(path):
                self.config.read(path)
                config_found = True
                print(f"Configuration loaded from: {path}")
                break
        
        if not config_found:
            print("Warning: No configuration file found. Using defaults.")
    
    def get(self, section, key, fallback=None):
        """Get configuration value"""
        try:
            return self.config.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def getint(self, section, key, fallback=0):
        """Get integer configuration value"""
        try:
            return self.config.getint(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback
    
    def getfloat(self, section, key, fallback=0.0):
        """Get float configuration value"""
        try:
            return self.config.getfloat(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback
    
    def getboolean(self, section, key, fallback=False):
        """Get boolean configuration value"""
        try:
            return self.config.getboolean(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback
    
    def getlist(self, section, key, fallback=None):
        """Get list configuration value"""
        try:
            value = self.config.get(section, key)
            return [item.strip() for item in value.split(',')]
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback or []

config = Config()

if __name__ == "__main__":
    print("Configuration Test")
    print("="*50)
    print(f"Historical Years: {config.getint('DATA', 'historical_years', 5)}")
    print(f"Streamlit Port: {config.getint('DEPLOYMENT', 'streamlit_port', 8501)}")
    print(f"RSI Period: {config.getint('FEATURES', 'rsi_period', 14)}")