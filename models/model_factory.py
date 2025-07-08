"""
Model Factory for creating trading models.

This module provides a centralized factory for creating different
model architectures based on configuration parameters.
"""

import yaml
from typing import Dict
import logging

from models.base_model import BaseModel
from models.baseline.lstm_model import BaselineLSTMModel
from models.sigformer.sigformer_model import SigFormerModel
from models.ensemble.ensemble_model import EnsembleModel
from models.graph.gnn_model import VisibilityGraphGNN

logger = logging.getLogger(__name__)


def create_model(config: Dict) -> BaseModel:
    """
    Create model based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    architecture = config['model']['architecture']
    
    if architecture == 'baseline':
        return BaselineLSTMModel(config)
    elif architecture == 'sigformer':
        return SigFormerModel(config)
    elif architecture == 'ensemble':
        return EnsembleModel(config)
    elif architecture == 'gnn':
        return VisibilityGraphGNN(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def load_config(config_path: str = 'config/config.yaml') -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise


def get_available_models() -> Dict[str, str]:
    """
    Get available model architectures.
    
    Returns:
        Dictionary mapping model names to descriptions
    """
    return {
        'baseline': 'Baseline LSTM model with dense layers for delta hedging',
        'sigformer': 'SigFormer with signature transform and Transformer encoder',
        'ensemble': 'Ensemble model combining baseline, SigFormer, and GNN architectures',
        'gnn': 'Graph Neural Network model using time series visibility graphs'
    }


def validate_config(config: Dict) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_sections = ['model', 'training', 'data']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Validate model section
    if 'architecture' not in config['model']:
        raise ValueError("Missing 'architecture' in model configuration")
    
    available_models = get_available_models()
    if config['model']['architecture'] not in available_models:
        raise ValueError(f"Unknown architecture: {config['model']['architecture']}")
    
    # Validate required model parameters
    required_model_params = ['sequence_length', 'input_features', 'output_size']
    for param in required_model_params:
        if param not in config['model']:
            raise ValueError(f"Missing required model parameter: {param}")
    
    logger.info("Configuration validation passed")
    return True 