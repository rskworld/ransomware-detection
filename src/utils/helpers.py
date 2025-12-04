"""
Helper Functions Module
Utility functions for the ransomware detection system

Author: Molla Samser
Designer & Tester: Rima Khatun
Organization: RSK World
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Optional
import logging


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Path to log file (optional)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )


def create_directories(directories: list):
    """
    Create directories if they don't exist.
    
    Args:
        directories: List of directory paths
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created/verified directory: {directory}")


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent

