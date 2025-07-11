"""
Training Pipeline for Deep Hedging Models.

This module implements the training pipeline with policy-gradient loss,
early stopping, and comprehensive monitoring for algorithmic trading models.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import logging
import os
from typing import Dict, Tuple, List, Optional, Callable
import pickle
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Local imports with updated paths
from models.model_factory import create_model, load_config
from pipeline.preprocessing.data_loader import TradingDataLoader
from optimization.hyperopt import BayesianOptimizer
from simulation.sim_models import MonteCarloSimulator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingModelTrainer:
    """
    Comprehensive trainer for deep hedging models.
    
    Implements advanced training strategies including:
    - Policy gradient loss with utility maximization
    - Monte Carlo data augmentation
    - Rolling Sharpe ratio early stopping
    - Model checkpointing and TensorBoard integration
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.history = None
        self.data_loader = TradingDataLoader(config)
        self.simulator = MonteCarloSimulator(config)
        
        # Setup logging and checkpoints
        self.experiment_dir = self._setup_experiment_directory()
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger.info(f"Trainer initialized. Experiment directory: {self.experiment_dir}")
    
    def _setup_experiment_directory(self) -> str:
        """Setup experiment directory with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{self.config['model']['architecture']}_{timestamp}"
        experiment_dir = os.path.join('experiments', experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training and validation data.
        
        Args:
            data: Raw market data DataFrame
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        logger.info("Preparing training data")
        
        # Load and preprocess data
        sequences, targets = self.data_loader.create_sequences(data)
        
        # Time series split (preserve temporal order)
        split_idx = int(len(sequences) * self.config['data']['train_split'])
        
        X_train = sequences[:split_idx]
        y_train = targets[:split_idx]
        X_val = sequences[split_idx:]
        y_val = targets[split_idx:]
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        
        return X_train, y_train, X_val, y_val
    
    def create_custom_loss(self) -> Callable:
        """
        Create custom deep hedging loss function.
        
        Returns:
            Custom loss function
        """
        def deep_hedging_loss(y_true, y_pred):
            """
            Policy gradient loss for deep hedging.
            
            Implements utility maximization with transaction costs
            as described in Buehler et al. (2019).
            """
            # Constrain hedge ratios to [-1, 1]
            hedge_ratios = tf.clip_by_value(y_pred, -1.0, 1.0)
            
            # Calculate portfolio P&L
            portfolio_returns = hedge_ratios * y_true
            
            # Transaction costs
            transaction_cost = self.config['training']['loss']['transaction_cost']
            position_changes = tf.abs(hedge_ratios - tf.concat([tf.zeros((tf.shape(hedge_ratios)[0], 1)), hedge_ratios[:, :-1]], axis=1))
            transaction_costs = transaction_cost * position_changes
            
            # Net P&L after costs
            net_pnl = portfolio_returns - transaction_costs
            
            # Utility function (exponential utility)
            risk_aversion = self.config['training']['loss']['risk_aversion']
            
            utility_type = self.config['training']['loss'].get('utility_type', 'exponential')
            
            if utility_type == 'exponential':
                # Exponential utility: U(x) = -exp(-γx)
                utility = -tf.exp(-risk_aversion * net_pnl)
            elif utility_type == 'power':
                # Power utility: U(x) = x^(1-γ)/(1-γ)
                if risk_aversion != 1:
                    utility = tf.pow(tf.maximum(net_pnl + 1, 1e-8), 1 - risk_aversion) / (1 - risk_aversion)
                else:
                    utility = tf.math.log(tf.maximum(net_pnl + 1, 1e-8))
            else:  # quadratic
                # Quadratic utility: U(x) = x - (γ/2)x^2
                utility = net_pnl - (risk_aversion / 2) * tf.square(net_pnl)
            
            # Expected utility (negative for minimization)
            expected_utility = tf.reduce_mean(utility)
            
            return -expected_utility
        
        return deep_hedging_loss
    
    def create_callbacks(self, X_val: np.ndarray, y_val: np.ndarray) -> List:
        """
        Create training callbacks.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Early stopping based on validation loss
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config['training']['early_stopping']['patience'],
            restore_best_weights=True,
                verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
            factor=self.config['training']['lr_schedule']['factor'],
            patience=self.config['training']['lr_schedule']['patience'],
            min_lr=self.config['training']['lr_schedule']['min_lr'],
                verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpointing
        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.h5')
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        callbacks.append(checkpoint)
        
        # Custom callback for rolling Sharpe ratio early stopping
        if self.config['training']['early_stopping'].get('use_sharpe_ratio', False):
            sharpe_callback = RollingSharpeEarlyStopping(
                X_val=X_val,
                y_val=y_val,
                patience=self.config['training']['early_stopping']['sharpe_patience'],
                min_sharpe=self.config['training']['early_stopping']['min_sharpe_ratio']
            )
            callbacks.append(sharpe_callback)
        
        # TensorBoard logging
        if self.config['training'].get('use_tensorboard', True):
            tensorboard_dir = os.path.join(self.experiment_dir, 'tensorboard')
            tensorboard = keras.callbacks.TensorBoard(
                log_dir=tensorboard_dir,
                    histogram_freq=1,
                    write_graph=True,
                write_images=True
            )
            callbacks.append(tensorboard)
        
        return callbacks
    
    def augment_data_with_monte_carlo(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment training data with Monte Carlo simulations.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Augmented training data
        """
        if not self.config['training'].get('use_monte_carlo_augmentation', False):
            return X_train, y_train
        
        logger.info("Augmenting training data with Monte Carlo simulations")
        
        # Generate synthetic paths
        n_synthetic = int(len(X_train) * self.config['training']['monte_carlo']['augmentation_ratio'])
        
        synthetic_paths = self.simulator.generate_synthetic_paths(
            n_paths=n_synthetic,
            path_length=X_train.shape[1],
            n_features=X_train.shape[2]
        )
        
        # Create sequences from synthetic paths
        synthetic_X, synthetic_y = self.data_loader.create_sequences_from_paths(synthetic_paths)
        
        # Combine with original data
        X_augmented = np.concatenate([X_train, synthetic_X], axis=0)
        y_augmented = np.concatenate([y_train, synthetic_y], axis=0)
        
        # Shuffle augmented data
        indices = np.random.permutation(len(X_augmented))
        X_augmented = X_augmented[indices]
        y_augmented = y_augmented[indices]
        
        logger.info(f"Data augmented from {len(X_train)} to {len(X_augmented)} samples")
        
        return X_augmented, y_augmented
    
    def train_model(self, data: pd.DataFrame) -> Dict:
        """
        Train the deep hedging model.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Training history and metrics
        """
        logger.info("Starting model training")
        
        # Create model
        self.model = create_model(self.config)
        
        # Prepare data
        X_train, y_train, X_val, y_val = self.prepare_data(data)
        
        # Augment data if enabled
        X_train, y_train = self.augment_data_with_monte_carlo(X_train, y_train)
        
        # Build and compile model
        model = self.model.get_model()
        
        # Create callbacks
        callbacks = self.create_callbacks(X_val, y_val)
        
        # Training parameters
        training_config = self.config['training']
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=training_config['epochs'],
            batch_size=training_config['batch_size'],
            callbacks=callbacks,
            verbose=1,
            shuffle=training_config.get('shuffle', True)
        )
        
        self.history = history
        
        # Calculate final metrics
        final_metrics = self._calculate_final_metrics(X_val, y_val)
        
        # Save model and results
        self._save_training_results(final_metrics)
        
        logger.info("Model training completed")
        
        return {
            'history': history.history,
            'final_metrics': final_metrics,
            'model_path': os.path.join(self.experiment_dir, 'final_model.h5')
        }
    
    def _calculate_final_metrics(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Calculate comprehensive final metrics."""
        
        # Get predictions
        predictions = self.model.model.predict(X_val, verbose=0)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - y_val))
        mse = np.mean((predictions - y_val) ** 2)
        rmse = np.sqrt(mse)
        
        # Financial metrics
        portfolio_returns = predictions.flatten() * y_val.flatten()
        sharpe_ratio = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8) * np.sqrt(252)
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        annual_return = np.mean(portfolio_returns) * 252
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar_ratio),
            'annual_return': float(annual_return),
            'win_rate': float(np.mean(portfolio_returns > 0))
        }
    
    def _save_training_results(self, metrics: Dict) -> None:
        """Save training results and model."""
        
        # Save final model
        model_path = os.path.join(self.experiment_dir, 'final_model.h5')
        self.model.model.save(model_path)
        
        # Save configuration
        config_path = os.path.join(self.experiment_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(self.config, f)
        
        # Save metrics
        metrics_path = os.path.join(self.experiment_dir, 'metrics.json')
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save training history
        if self.history:
            history_path = os.path.join(self.experiment_dir, 'history.pkl')
            with open(history_path, 'wb') as f:
                pickle.dump(self.history.history, f)
        
        logger.info(f"Training results saved to {self.experiment_dir}")


class RollingSharpeEarlyStopping(keras.callbacks.Callback):
    """Custom callback for early stopping based on rolling Sharpe ratio."""
    
    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, 
                 patience: int = 10, min_sharpe: float = 0.5):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.patience = patience
        self.min_sharpe = min_sharpe
        self.wait = 0
        self.best_sharpe = -np.inf
        
    def on_epoch_end(self, epoch, logs=None):
        # Get current predictions
        predictions = self.model.predict(self.X_val, verbose=0)
        
        # Calculate rolling Sharpe ratio
        portfolio_returns = predictions.flatten() * self.y_val.flatten()
        
        if len(portfolio_returns) > 30:  # Minimum window for Sharpe calculation
            recent_returns = portfolio_returns[-30:]  # Last 30 observations
            sharpe_ratio = np.mean(recent_returns) / (np.std(recent_returns) + 1e-8) * np.sqrt(252)
            
            if sharpe_ratio > self.best_sharpe:
                self.best_sharpe = sharpe_ratio
                self.wait = 0
            else:
                self.wait += 1
            
            # Log Sharpe ratio
            logs = logs or {}
            logs['val_sharpe'] = sharpe_ratio
            
            if self.wait >= self.patience and sharpe_ratio < self.min_sharpe:
                logger.info(f"Early stopping: Sharpe ratio {sharpe_ratio:.4f} below threshold {self.min_sharpe}")
                self.model.stop_training = True


def run_hyperparameter_optimization(config: Dict, data: pd.DataFrame) -> Dict:
    """
    Run hyperparameter optimization using Bayesian optimization.
    
    Args:
        config: Base configuration
        data: Training data
        
    Returns:
        Best hyperparameters and results
    """
    logger.info("Starting hyperparameter optimization")
    
    optimizer = BayesianOptimizer(config)
    
    def objective(params):
        # Update config with trial parameters
        trial_config = config.copy()
        trial_config.update(params)
        
        # Create and train model
        trainer = TradingModelTrainer(trial_config)
        results = trainer.train_model(data)
        
        # Return negative Sharpe ratio (for minimization)
        return -results['final_metrics']['sharpe_ratio']
    
    # Run optimization
    best_params, best_value = optimizer.optimize(objective)
    
    logger.info(f"Hyperparameter optimization completed. Best Sharpe ratio: {-best_value:.4f}")
    
    return {
        'best_params': best_params,
        'best_sharpe': -best_value,
        'optimization_history': optimizer.get_optimization_history()
    }


if __name__ == "__main__":
    # Example usage
    
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Create sample data (replace with real data loading)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    sample_data = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
        'High': 100 + np.cumsum(np.random.normal(0, 1, len(dates))) + np.random.uniform(0, 2, len(dates)),
        'Low': 100 + np.cumsum(np.random.normal(0, 1, len(dates))) - np.random.uniform(0, 2, len(dates)),
        'Close': 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
        'Volume': np.random.uniform(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Train model
    trainer = TradingModelTrainer(config)
    results = trainer.train_model(sample_data)
    
    print("Training completed!")
    print(f"Final metrics: {results['final_metrics']}")
    print(f"Model saved to: {results['model_path']}") 