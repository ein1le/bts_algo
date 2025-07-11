#!/usr/bin/env python3
"""
Main entry point for the BTS Algorithmic Trading System.

This script demonstrates the functionality-based directory structure
and provides different execution modes for the trading system.
"""

import argparse
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import from new structure
from models.model_factory import create_model, load_config, validate_config, get_available_models
from preprocessing.data_loader import TradingDataLoader
from preprocessing.feature_engineering import FeatureEngineeringPipeline
from training.trainer import TradingModelTrainer, run_hyperparameter_optimization
from optimization.hyperopt import BayesianOptimizer
from simulation.sim_models import MonteCarloSimulator
from simulation.sim_functions import run_comprehensive_simulation_analysis
from visualization.graph_viz import TradingGraphVisualizer

import tensorflow as tf

# Enable memory growth for all GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled memory growth for GPUs")
    except RuntimeError as e:
        print(e)

# Enable mixed precision if desired
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("Enabled mixed precision")


def setup_directories():
    """Create necessary directories for the project."""
    directories = [
        'data/raw',
        'data/processed', 
        'data/synthetic',
        'experiments',
        'models/checkpoints',
        'logs',
        'reports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Project directories initialized")


def create_sample_data(n_days: int = 1000) -> pd.DataFrame:
    """
    Create sample market data for demonstration.
    
    Args:
        n_days: Number of days of data to generate
        
    Returns:
        Sample market data DataFrame
    """
    logger.info(f"Creating sample data with {n_days} days")
    
    # Generate sample dates
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Generate realistic price data using GBM
    np.random.seed(42)
    dt = 1/252  # Daily time step
    mu = 0.05   # Expected return
    sigma = 0.2 # Volatility
    
    # Generate price path
    returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    high_factor = 1 + np.abs(np.random.normal(0, 0.01, n_days))
    low_factor = 1 - np.abs(np.random.normal(0, 0.01, n_days))
    
    sample_data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.005, n_days)),
        'High': prices * high_factor,
        'Low': prices * low_factor,
        'Close': prices,
        'Volume': np.random.lognormal(15, 0.5, n_days).astype(int)
    }, index=dates)
    
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    sample_data['High'] = np.maximum(sample_data['High'], 
                                   np.maximum(sample_data['Open'], sample_data['Close']))
    sample_data['Low'] = np.minimum(sample_data['Low'],
                                  np.minimum(sample_data['Open'], sample_data['Close']))
    
    logger.info("Sample data created successfully")
    return sample_data


def mode_train(config_path: str, architecture: str = None, data_file: str = None) -> None:
    """
    Training mode - train a model with specified architecture.
    
    Args:
        config_path: Path to configuration file
        architecture: Model architecture to use
        data_file: Path to data file (optional, will use sample data if not provided)
    """
    logger.info("=== TRAINING MODE ===")
    
    # Load and validate configuration
    config = load_config(config_path)
    validate_config(config)
    
    # Override architecture if specified
    if architecture:
        if architecture not in get_available_models():
            logger.error(f"Unknown architecture: {architecture}")
            logger.info(f"Available architectures: {list(get_available_models().keys())}")
            return
        config['model']['architecture'] = architecture
    
    # Load or create data
    if data_file and Path(data_file).exists():
        logger.info(f"Loading data from {data_file}")
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    else:
        logger.info("No data file provided, creating sample data")
        data = create_sample_data()
    
    # Initialize trainer and train model
    trainer = TradingModelTrainer(config)
    results = trainer.train_model(data)
    
    # Display results
    logger.info("=== TRAINING COMPLETED ===")
    logger.info(f"Architecture: {config['model']['architecture']}")
    logger.info(f"Final Metrics:")
    for metric, value in results['final_metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info(f"Model saved to: {results['model_path']}")


def mode_optimize(config_path: str, data_file: str = None) -> None:
    """
    Hyperparameter optimization mode.
    
    Args:
        config_path: Path to configuration file
        data_file: Path to data file (optional)
    """
    logger.info("=== HYPERPARAMETER OPTIMIZATION MODE ===")
    
    # Load configuration
    config = load_config(config_path)
    validate_config(config)
    
    # Load or create data
    if data_file and Path(data_file).exists():
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    else:
        data = create_sample_data()
    
    # Run optimization
    results = run_hyperparameter_optimization(config, data)
    
    # Display results
    logger.info("=== OPTIMIZATION COMPLETED ===")
    logger.info(f"Best Sharpe Ratio: {results['best_sharpe']:.4f}")
    logger.info("Best Parameters:")
    for param, value in results['best_params'].items():
        logger.info(f"  {param}: {value}")


def mode_simulate(config_path: str, n_paths: int = 1000) -> None:
    """
    Monte Carlo simulation mode.
    
    Args:
        config_path: Path to configuration file
        n_paths: Number of simulation paths
    """
    logger.info("=== MONTE CARLO SIMULATION MODE ===")
    
    # Load configuration
    config = load_config(config_path)
    
    # Create sample historical data
    historical_data = create_sample_data(252)  # 1 year of data
    
    # Run comprehensive simulation analysis
    simulation_config = {'n_bootstrap_samples': 500}
    results = run_comprehensive_simulation_analysis(
        historical_data['Close'].values, 
        simulation_config
    )
    
    # Display results
    logger.info("=== SIMULATION COMPLETED ===")
    logger.info("Calibrated Parameters:")
    logger.info(f"  GBM mu: {results['calibrated_parameters']['gbm']['mu']:.4f}")
    logger.info(f"  GBM sigma: {results['calibrated_parameters']['gbm']['sigma']:.4f}")
    logger.info(f"Number of stress scenarios: {len(results['stress_scenarios'])}")
    logger.info(f"Number of regime scenarios: {len(results['regime_scenarios'])}")


def mode_feature_analysis(config_path: str, data_file: str = None) -> None:
    """
    Feature engineering analysis mode.
    
    Args:
        config_path: Path to configuration file
        data_file: Path to data file (optional)
    """
    logger.info("=== FEATURE ENGINEERING ANALYSIS MODE ===")
    
    # Load configuration
    config = load_config(config_path)
    
    # Load or create data
    if data_file and Path(data_file).exists():
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    else:
        data = create_sample_data()
    
    # Initialize feature engineering pipeline
    feature_pipeline = FeatureEngineeringPipeline(config)
    
    # Create features
    logger.info("Creating comprehensive feature set...")
    features = feature_pipeline.create_all_features(data)
    
    # Preprocess features
    logger.info("Preprocessing features...")
    processed_features = feature_pipeline.preprocess_features(features)
    
    # Display analysis
    logger.info("=== FEATURE ANALYSIS COMPLETED ===")
    logger.info(f"Original data shape: {data.shape}")
    logger.info(f"Feature matrix shape: {features.shape}")
    logger.info(f"Processed features shape: {processed_features.shape}")
    logger.info(f"Number of features created: {len(features.columns)}")
    logger.info("Feature categories:")
    
    # Categorize features
    feature_categories = {
        'momentum': [col for col in features.columns if 'roc_' in col or 'momentum_' in col or 'rsi_' in col],
        'trend': [col for col in features.columns if 'sma_' in col or 'ema_' in col or 'macd' in col or 'bb_' in col],
        'volatility': [col for col in features.columns if 'vol_' in col or 'atr' in col or 'gk_' in col],
        'volume': [col for col in features.columns if 'obv' in col or 'vwap' in col or 'mfi' in col],
        'microstructure': [col for col in features.columns if 'spread' in col or 'impact' in col or 'illiquidity' in col],
        'calendar': [col for col in features.columns if any(x in col for x in ['hour', 'day', 'month', 'weekend', 'market'])]
    }
    
    for category, feature_list in feature_categories.items():
        logger.info(f"  {category}: {len(feature_list)} features")


def mode_visualize(config_path: str, data_file: str = None) -> None:
    """
    Visualization mode - create trading analytics visualizations.
    
    Args:
        config_path: Path to configuration file  
        data_file: Path to data file (optional)
    """
    logger.info("=== VISUALIZATION MODE ===")
    
    # Load configuration
    config = load_config(config_path)
    
    # Load or create data
    if data_file and Path(data_file).exists():
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    else:
        data = create_sample_data()
    
    # Initialize visualizer
    visualizer = TradingGraphVisualizer(config)
    
    # Create sample predictions for demonstration
    np.random.seed(42)
    predictions = np.random.normal(0, 0.1, len(data))
    
    # Generate visualizations
    logger.info("Creating trading analytics visualizations...")
    
    try:
        # Create execution flow diagram
        flow_path = visualizer.create_execution_flow_diagram()
        logger.info(f"Execution flow diagram saved to: {flow_path}")
        
        # Create performance dashboard
        dashboard_path = visualizer.create_performance_dashboard(
            data, predictions, save_path='reports/performance_dashboard.html'
        )
        logger.info(f"Performance dashboard saved to: {dashboard_path}")
        
        logger.info("=== VISUALIZATION COMPLETED ===")
        
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        logger.info("Skipping visualization due to dependencies")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BTS Algorithmic Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --architecture baseline
  python main.py train --architecture sigformer --data data/market_data.csv
  python main.py optimize --config config/config.yaml
  python main.py simulate --n-paths 5000
  python main.py features --data data/market_data.csv
  python main.py visualize
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['train', 'optimize', 'simulate', 'features', 'visualize'],
        help='Execution mode'
    )
    
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--architecture',
        choices=list(get_available_models().keys()),
        help='Model architecture for training mode'
    )
    
    parser.add_argument(
        '--data',
        help='Path to data file (CSV format)'
    )
    
    parser.add_argument(
        '--n-paths',
        type=int,
        default=1000,
        help='Number of simulation paths (default: 1000)'
    )
    
    args = parser.parse_args()
    
    # Setup project directories
    setup_directories()
    
    # Check if config file exists
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Please ensure the configuration file exists")
        sys.exit(1)
    
    # Execute based on mode
    try:
        if args.mode == 'train':
            mode_train(args.config, args.architecture, args.data)
        elif args.mode == 'optimize':
            mode_optimize(args.config, args.data)
        elif args.mode == 'simulate':
            mode_simulate(args.config, args.n_paths)
        elif args.mode == 'features':
            mode_feature_analysis(args.config, args.data)
        elif args.mode == 'visualize':
            mode_visualize(args.config, args.data)
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    # Display project information
    print("=" * 60)
    print("BTS Algorithmic Trading Portfolio System")
    print("Functionality-Based Architecture")
    print("=" * 60)
    print()
    
    # Show available models
    models = get_available_models()
    print("Available Model Architectures:")
    for name, description in models.items():
        print(f"  {name}: {description}")
    print()
    
    # Show directory structure
    print("Project Structure:")
    structure = [
        "models/          - Model architectures (baseline, sigformer, ensemble)",
        "pipeline/        - Data processing, feature engineering, training",
        "optimization/    - Hyperparameter optimization",
        "simulation/      - Monte Carlo simulation and analysis", 
        "visualization/   - Trading analytics and graph visualization",
        "config/          - Configuration files",
        "utils/           - Utility functions",
        "data/            - Raw, processed, and synthetic data",
        "experiments/     - Training experiments and results"
    ]
    for item in structure:
        print(f"  {item}")
    print()
    
    main() 