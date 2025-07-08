"""
Hyperparameter optimization for trading models using Optuna.

This module provides:
- Bayesian optimization with Optuna
- Custom objective functions for financial metrics
- Model-specific parameter spaces
- Multi-objective optimization (return vs risk)
- Cross-validation with time series splits
"""

import optuna
from optuna.integration import TensorFlowKeras
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import logging
from typing import Dict, List, Tuple, Optional, Callable
import yaml
from sklearn.model_selection import TimeSeriesSplit
import warnings

from model import create_model, load_config
from data_loader import DataLoader

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialMetrics:
    """Financial performance metrics for optimization."""
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return -np.inf
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return -np.inf
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return np.inf if np.mean(excess_returns) > 0 else -np.inf
        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
    
    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 1.0
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return -np.min(drawdown)
    
    @staticmethod
    def calmar_ratio(returns: np.ndarray) -> float:
        """Calculate Calmar ratio."""
        if len(returns) == 0:
            return -np.inf
        annual_return = np.prod(1 + returns) ** (252 / len(returns)) - 1
        max_dd = FinancialMetrics.max_drawdown(returns)
        return annual_return / max_dd if max_dd > 0 else np.inf
    
    @staticmethod
    def information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate Information ratio."""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return -np.inf
        excess_returns = returns - benchmark_returns[:len(returns)]
        if np.std(excess_returns) == 0:
            return np.inf if np.mean(excess_returns) > 0 else -np.inf
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


class TradingObjective:
    """Custom objective function for trading model optimization."""
    
    def __init__(self, config: Dict, data_dict: Dict, architecture: str):
        """
        Initialize trading objective.
        
        Args:
            config: Configuration dictionary
            data_dict: Prepared data dictionary
            architecture: Model architecture ('baseline', 'sigformer', 'ensemble')
        """
        self.config = config
        self.data_dict = data_dict
        self.architecture = architecture
        self.hyperopt_config = config['hyperopt']
        
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value (higher is better)
        """
        try:
            # Sample hyperparameters based on architecture
            if self.architecture == 'baseline':
                suggested_config = self._suggest_baseline_params(trial)
            elif self.architecture == 'sigformer':
                suggested_config = self._suggest_sigformer_params(trial)
            else:
                raise ValueError(f"Architecture {self.architecture} not supported for optimization")
            
            # Update config with suggested parameters
            updated_config = self._update_config_with_params(suggested_config)
            
            # Create and train model
            model_instance = create_model(updated_config)
            model = model_instance.build_model()
            model_instance.compile_model()
            
            # Time series cross-validation
            objective_scores = []
            tscv = TimeSeriesSplit(n_splits=3)
            
            X_combined = np.concatenate([self.data_dict['X_train'], self.data_dict['X_val']], axis=0)
            y_combined = np.concatenate([self.data_dict['y_train'], self.data_dict['y_val']], axis=0)
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_combined)):
                logger.info(f"Training fold {fold + 1}/3")
                
                X_fold_train, X_fold_val = X_combined[train_idx], X_combined[val_idx]
                y_fold_train, y_fold_val = y_combined[train_idx], y_combined[val_idx]
                
                # Training callbacks
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=1e-6
                    )
                ]
                
                # Train model
                history = model.fit(
                    X_fold_train, y_fold_train,
                    validation_data=(X_fold_val, y_fold_val),
                    epochs=50,  # Reduced for optimization speed
                    batch_size=suggested_config.get('batch_size', 64),
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Generate predictions
                y_pred = model.predict(X_fold_val, verbose=0).flatten()
                
                # Calculate trading performance
                returns = self._calculate_trading_returns(y_fold_val, y_pred)
                
                # Calculate objective metric
                objective_score = self._calculate_objective_score(returns)
                objective_scores.append(objective_score)
                
                # Report intermediate value for pruning
                trial.report(objective_score, fold)
                
                # Prune if necessary
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Return mean objective score
            final_score = np.mean(objective_scores)
            logger.info(f"Trial completed with score: {final_score:.4f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Trial failed with error: {e}")
            return -np.inf
    
    def _suggest_baseline_params(self, trial: optuna.Trial) -> Dict:
        """Suggest hyperparameters for baseline model."""
        space = self.hyperopt_config['baseline_space']
        
        return {
            'lstm_units_1': trial.suggest_int('lstm_units_1', space['lstm_units_1'][0], space['lstm_units_1'][1]),
            'lstm_units_2': trial.suggest_int('lstm_units_2', space['lstm_units_2'][0], space['lstm_units_2'][1]),
            'dropout_rate': trial.suggest_float('dropout_rate', space['dropout_rate'][0], space['dropout_rate'][1]),
            'learning_rate': trial.suggest_float('learning_rate', space['learning_rate'][0], space['learning_rate'][1], log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'recurrent_dropout': trial.suggest_float('recurrent_dropout', 0.0, 0.3),
            'dense_units': trial.suggest_int('dense_units', 32, 128)
        }
    
    def _suggest_sigformer_params(self, trial: optuna.Trial) -> Dict:
        """Suggest hyperparameters for SigFormer model."""
        space = self.hyperopt_config['sigformer_space']
        
        return {
            'signature_depth': trial.suggest_int('signature_depth', space['signature_depth'][0], space['signature_depth'][1]),
            'd_model': trial.suggest_categorical('d_model', [128, 256, 512]),
            'n_heads': trial.suggest_categorical('n_heads', [4, 8, 16]),
            'n_layers': trial.suggest_int('n_layers', space['n_layers'][0], space['n_layers'][1]),
            'dropout': trial.suggest_float('dropout', space['dropout'][0], space['dropout'][1]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'd_ff': trial.suggest_categorical('d_ff', [512, 1024, 2048])
        }
    
    def _update_config_with_params(self, suggested_params: Dict) -> Dict:
        """Update configuration with suggested parameters."""
        updated_config = self.config.copy()
        
        if self.architecture == 'baseline':
            updated_config['model']['baseline']['lstm_units'] = [
                suggested_params['lstm_units_1'],
                suggested_params['lstm_units_2']
            ]
            updated_config['model']['baseline']['dropout_rate'] = suggested_params['dropout_rate']
            updated_config['model']['baseline']['recurrent_dropout'] = suggested_params.get('recurrent_dropout', 0.1)
            updated_config['model']['baseline']['dense_units'] = [suggested_params.get('dense_units', 64)]
            updated_config['training']['learning_rate'] = suggested_params['learning_rate']
            updated_config['training']['batch_size'] = suggested_params['batch_size']
            
        elif self.architecture == 'sigformer':
            updated_config['model']['sigformer']['signature_depth'] = suggested_params['signature_depth']
            updated_config['model']['sigformer']['d_model'] = suggested_params['d_model']
            updated_config['model']['sigformer']['n_heads'] = suggested_params['n_heads']
            updated_config['model']['sigformer']['n_layers'] = suggested_params['n_layers']
            updated_config['model']['sigformer']['dropout'] = suggested_params['dropout']
            updated_config['model']['sigformer']['d_ff'] = suggested_params['d_ff']
            updated_config['training']['learning_rate'] = suggested_params['learning_rate']
            updated_config['training']['batch_size'] = suggested_params['batch_size']
        
        return updated_config
    
    def _calculate_trading_returns(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate trading returns based on predictions.
        
        Args:
            y_true: True returns
            y_pred: Predicted hedge ratios
            
        Returns:
            Trading returns
        """
        # Simple trading strategy: position based on predicted hedge ratio
        # In practice, this would be more sophisticated
        positions = np.tanh(y_pred)  # Ensure positions are in [-1, 1]
        
        # Trading returns (simplified)
        # In reality, this would include transaction costs, slippage, etc.
        trading_returns = positions * y_true
        
        # Apply transaction costs
        transaction_cost = self.config['training']['loss']['transaction_cost']
        position_changes = np.abs(np.diff(np.concatenate([[0], positions])))
        transaction_costs = position_changes * transaction_cost
        
        net_returns = trading_returns - transaction_costs
        
        return net_returns
    
    def _calculate_objective_score(self, returns: np.ndarray) -> float:
        """
        Calculate objective score for optimization.
        
        Args:
            returns: Trading returns
            
        Returns:
            Objective score (higher is better)
        """
        if len(returns) == 0:
            return -np.inf
        
        # Multi-objective: Sharpe ratio with penalty for high drawdown
        sharpe = FinancialMetrics.sharpe_ratio(returns)
        max_dd = FinancialMetrics.max_drawdown(returns)
        
        # Objective function: Sharpe ratio - drawdown penalty
        objective = sharpe - 2.0 * max_dd  # Weight drawdown penalty
        
        return objective


class HyperparameterOptimizer:
    """Main hyperparameter optimization class."""
    
    def __init__(self, config: Dict):
        """
        Initialize optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.hyperopt_config = config['hyperopt']
        
    def optimize_model(self, data_dict: Dict, architecture: str) -> Dict:
        """
        Optimize hyperparameters for specified architecture.
        
        Args:
            data_dict: Prepared data dictionary
            architecture: Model architecture to optimize
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting hyperparameter optimization for {architecture} model")
        
        # Create study
        study_name = f"{architecture}_optimization"
        sampler = TPESampler(seed=self.config['environment']['random_seed'])
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=study_name
        )
        
        # Create objective function
        objective = TradingObjective(self.config, data_dict, architecture)
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.hyperopt_config['n_trials'],
            timeout=None,
            show_progress_bar=True
        )
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Optimization completed for {architecture}")
        logger.info(f"Best value: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Generate optimization report
        optimization_results = {
            'architecture': architecture,
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': len(study.trials),
            'study': study,
            'trials_df': study.trials_dataframe()
        }
        
        return optimization_results
    
    def optimize_all_architectures(self, data_dict: Dict) -> Dict:
        """
        Optimize hyperparameters for all architectures.
        
        Args:
            data_dict: Prepared data dictionary
            
        Returns:
            Dictionary with all optimization results
        """
        results = {}
        architectures = ['baseline', 'sigformer']
        
        for architecture in architectures:
            try:
                results[architecture] = self.optimize_model(data_dict, architecture)
            except Exception as e:
                logger.error(f"Optimization failed for {architecture}: {e}")
                results[architecture] = None
        
        return results
    
    def save_optimization_results(self, results: Dict, filepath: str) -> None:
        """
        Save optimization results to file.
        
        Args:
            results: Optimization results dictionary
            filepath: Output file path
        """
        # Save best parameters and summary
        summary = {}
        for arch, result in results.items():
            if result is not None:
                summary[arch] = {
                    'best_params': result['best_params'],
                    'best_value': result['best_value'],
                    'n_trials': result['n_trials']
                }
        
        # Save to YAML
        with open(filepath, 'w') as file:
            yaml.dump(summary, file, default_flow_style=False)
        
        logger.info(f"Optimization results saved to {filepath}")
    
    def load_best_config(self, results: Dict, architecture: str) -> Dict:
        """
        Load best configuration for specified architecture.
        
        Args:
            results: Optimization results
            architecture: Architecture name
            
        Returns:
            Updated configuration with best parameters
        """
        if architecture not in results or results[architecture] is None:
            logger.warning(f"No optimization results for {architecture}, using default config")
            return self.config
        
        best_params = results[architecture]['best_params']
        
        # Update configuration
        updated_config = self.config.copy()
        
        if architecture == 'baseline':
            if 'lstm_units_1' in best_params and 'lstm_units_2' in best_params:
                updated_config['model']['baseline']['lstm_units'] = [
                    best_params['lstm_units_1'],
                    best_params['lstm_units_2']
                ]
            if 'dropout_rate' in best_params:
                updated_config['model']['baseline']['dropout_rate'] = best_params['dropout_rate']
            if 'learning_rate' in best_params:
                updated_config['training']['learning_rate'] = best_params['learning_rate']
            if 'batch_size' in best_params:
                updated_config['training']['batch_size'] = best_params['batch_size']
                
        elif architecture == 'sigformer':
            for param_name, param_value in best_params.items():
                if param_name in ['signature_depth', 'd_model', 'n_heads', 'n_layers', 'dropout', 'd_ff']:
                    updated_config['model']['sigformer'][param_name] = param_value
                elif param_name == 'learning_rate':
                    updated_config['training']['learning_rate'] = param_value
                elif param_name == 'batch_size':
                    updated_config['training']['batch_size'] = param_value
        
        logger.info(f"Configuration updated with best parameters for {architecture}")
        
        return updated_config


def run_hyperparameter_optimization(symbol: str = 'AAPL', start_date: str = '2020-01-01', end_date: str = '2023-12-31') -> Dict:
    """
    Run complete hyperparameter optimization pipeline.
    
    Args:
        symbol: Trading symbol
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        Optimization results
    """
    # Load configuration
    config = load_config()
    
    # Load and prepare data
    data_loader = DataLoader(config)
    data_dict = data_loader.load_and_prepare_data(symbol, start_date, end_date)
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(config)
    
    # Run optimization
    results = optimizer.optimize_all_architectures(data_dict)
    
    # Save results
    optimizer.save_optimization_results(results, f'hyperopt_results_{symbol}.yaml')
    
    return results


if __name__ == "__main__":
    # Example usage
    results = run_hyperparameter_optimization('AAPL', '2020-01-01', '2023-12-31')
    
    for arch, result in results.items():
        if result is not None:
            print(f"\n{arch.upper()} Model Optimization Results:")
            print(f"Best Value: {result['best_value']:.4f}")
            print(f"Best Parameters: {result['best_params']}")
        else:
            print(f"\n{arch.upper()} Model: Optimization failed") 