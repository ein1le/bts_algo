"""
Simulation utility functions for Monte Carlo analysis.

This module provides helper functions for simulation analysis,
statistical testing, and scenario generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from scipy import stats
from scipy.optimize import minimize
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SimulationAnalyzer:
    """Analysis tools for Monte Carlo simulation results."""
    
    @staticmethod
    def calculate_path_statistics(paths: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive statistics for simulated paths.
        
        Args:
            paths: Array of simulated paths [n_paths, n_steps]
            
        Returns:
            Dictionary of path statistics
        """
        if paths.ndim == 1:
            paths = paths.reshape(1, -1)
        
        # Final values
        final_values = paths[:, -1]
        
        # Returns
        returns = np.diff(np.log(paths), axis=1)
        
        stats_dict = {
            'mean_final_value': np.mean(final_values),
            'std_final_value': np.std(final_values),
            'min_final_value': np.min(final_values),
            'max_final_value': np.max(final_values),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_95': np.percentile(final_values, 95),
            
            # Return statistics
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'skewness': stats.skew(returns.flatten()),
            'kurtosis': stats.kurtosis(returns.flatten()),
            
            # Path-specific metrics
            'avg_path_volatility': np.mean([np.std(path_returns) for path_returns in returns]),
            'max_drawdown_avg': np.mean([SimulationAnalyzer._calculate_max_drawdown(path) for path in paths])
        }
        
        return stats_dict
    
    @staticmethod
    def _calculate_max_drawdown(path: np.ndarray) -> float:
        """Calculate maximum drawdown for a single path."""
        peak = np.maximum.accumulate(path)
        drawdown = (path - peak) / peak
        return -np.min(drawdown)
    
    @staticmethod
    def compare_models(results_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Compare results from different simulation models.
        
        Args:
            results_dict: Dictionary mapping model names to simulation results
            
        Returns:
            DataFrame with comparison statistics
        """
        comparison_data = []
        
        for model_name, paths in results_dict.items():
            stats = SimulationAnalyzer.calculate_path_statistics(paths)
            stats['model'] = model_name
            comparison_data.append(stats)
        
        return pd.DataFrame(comparison_data)
    
    @staticmethod
    def perform_model_validation(simulated_returns: np.ndarray, 
                                historical_returns: np.ndarray) -> Dict[str, float]:
        """
        Validate simulation model against historical data.
        
        Args:
            simulated_returns: Returns from simulation
            historical_returns: Actual historical returns
            
        Returns:
            Dictionary of validation metrics
        """
        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(simulated_returns, historical_returns)
        
        # Anderson-Darling test (if samples are large enough)
        try:
            ad_statistic, ad_critical_values, ad_significance = stats.anderson_ksamp([
                simulated_returns, historical_returns
            ])
        except:
            ad_statistic = ad_significance = np.nan
        
        # Moment comparison
        sim_moments = [
            np.mean(simulated_returns),
            np.std(simulated_returns),
            stats.skew(simulated_returns),
            stats.kurtosis(simulated_returns)
        ]
        
        hist_moments = [
            np.mean(historical_returns),
            np.std(historical_returns),
            stats.skew(historical_returns),
            stats.kurtosis(historical_returns)
        ]
        
        moment_errors = [abs(s - h) / abs(h) if h != 0 else abs(s - h) 
                        for s, h in zip(sim_moments, hist_moments)]
        
        validation_metrics = {
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue,
            'ad_statistic': ad_statistic,
            'ad_significance': ad_significance,
            'mean_error_pct': moment_errors[0] * 100,
            'std_error_pct': moment_errors[1] * 100,
            'skew_error_pct': moment_errors[2] * 100,
            'kurtosis_error_pct': moment_errors[3] * 100,
            'overall_moment_error': np.mean(moment_errors) * 100
        }
        
        return validation_metrics


class ParameterCalibration:
    """Parameter calibration utilities for simulation models."""
    
    @staticmethod
    def calibrate_gbm_parameters(returns: np.ndarray) -> Dict[str, float]:
        """
        Calibrate GBM parameters using maximum likelihood estimation.
        
        Args:
            returns: Historical returns
            
        Returns:
            Dictionary with calibrated parameters
        """
        # Remove any NaN values
        returns = returns[~np.isnan(returns)]
        
        # MLE for GBM
        mu_est = np.mean(returns) + 0.5 * np.var(returns)  # Drift
        sigma_est = np.std(returns)  # Volatility
        
        # Annualize (assuming daily returns)
        mu_annual = mu_est * 252
        sigma_annual = sigma_est * np.sqrt(252)
        
        return {
            'mu': mu_annual,
            'sigma': sigma_annual,
            'log_likelihood': ParameterCalibration._gbm_log_likelihood(returns, mu_est, sigma_est)
        }
    
    @staticmethod
    def _gbm_log_likelihood(returns: np.ndarray, mu: float, sigma: float) -> float:
        """Calculate log-likelihood for GBM parameters."""
        n = len(returns)
        log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma**2)
        log_likelihood -= np.sum((returns - mu)**2) / (2 * sigma**2)
        return log_likelihood
    
    @staticmethod
    def calibrate_heston_parameters(returns: np.ndarray, 
                                  initial_guess: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calibrate Heston model parameters using method of moments.
        
        Args:
            returns: Historical returns
            initial_guess: Initial parameter guess
            
        Returns:
            Dictionary with calibrated parameters
        """
        if initial_guess is None:
            initial_guess = {
                'kappa': 2.0,
                'theta': 0.04,
                'sigma_v': 0.3,
                'rho': -0.7,
                'v0': 0.04
            }
        
        # Calculate empirical moments
        empirical_mean = np.mean(returns)
        empirical_var = np.var(returns)
        empirical_skew = stats.skew(returns)
        empirical_kurt = stats.kurtosis(returns)
        
        # Objective function for calibration
        def objective(params):
            kappa, theta, sigma_v, rho, v0 = params
            
            # Theoretical moments for Heston model (simplified)
            theoretical_mean = empirical_mean  # Use empirical for simplicity
            theoretical_var = v0
            theoretical_skew = 0  # Simplified
            theoretical_kurt = 3  # Simplified
            
            # Moment matching
            error = (
                (empirical_mean - theoretical_mean)**2 +
                (empirical_var - theoretical_var)**2 +
                0.1 * (empirical_skew - theoretical_skew)**2 +
                0.1 * (empirical_kurt - theoretical_kurt)**2
            )
            
            return error
        
        # Constraints
        bounds = [
            (0.1, 10.0),    # kappa
            (0.001, 1.0),   # theta
            (0.01, 2.0),    # sigma_v
            (-0.99, 0.99),  # rho
            (0.001, 1.0)    # v0
        ]
        
        # Initial values
        x0 = [initial_guess['kappa'], initial_guess['theta'], 
              initial_guess['sigma_v'], initial_guess['rho'], initial_guess['v0']]
        
        try:
            result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                kappa, theta, sigma_v, rho, v0 = result.x
                return {
                    'kappa': kappa,
                    'theta': theta,
                    'sigma_v': sigma_v,
                    'rho': rho,
                    'v0': v0,
                    'calibration_error': result.fun,
                    'success': True
                }
            else:
                logger.warning("Heston calibration failed, using initial guess")
                initial_guess['success'] = False
                return initial_guess
                
        except Exception as e:
            logger.error(f"Heston calibration error: {e}")
            initial_guess['success'] = False
            return initial_guess


class StressTestGenerator:
    """Generate stress test scenarios for risk management."""
    
    @staticmethod
    def generate_market_crash_scenarios(base_params: Dict, 
                                      crash_magnitudes: List[float] = None) -> List[Dict]:
        """
        Generate market crash scenarios.
        
        Args:
            base_params: Base model parameters
            crash_magnitudes: List of crash magnitudes (as volatility multipliers)
            
        Returns:
            List of stress scenario parameters
        """
        if crash_magnitudes is None:
            crash_magnitudes = [1.5, 2.0, 3.0, 5.0]  # Volatility multipliers
        
        scenarios = []
        
        for magnitude in crash_magnitudes:
            stress_params = base_params.copy()
            
            # Increase volatility
            if 'sigma' in stress_params:
                stress_params['sigma'] *= magnitude
            
            # Add negative drift
            if 'mu' in stress_params:
                stress_params['mu'] = min(stress_params['mu'], -0.1 * magnitude)
            
            # Add jump parameters if not present
            if 'lambda_j' not in stress_params:
                stress_params['lambda_j'] = 0.1 * magnitude
                stress_params['mu_j'] = -0.1 * magnitude
                stress_params['sigma_j'] = 0.2 * magnitude
            
            stress_params['scenario_name'] = f'crash_{magnitude}x'
            stress_params['stress_magnitude'] = magnitude
            
            scenarios.append(stress_params)
        
        return scenarios
    
    @staticmethod
    def generate_regime_change_scenarios(base_params: Dict, 
                                       n_regimes: int = 3) -> List[Dict]:
        """
        Generate regime change scenarios.
        
        Args:
            base_params: Base model parameters
            n_regimes: Number of different regimes
            
        Returns:
            List of regime scenario parameters
        """
        scenarios = []
        
        # Define regime characteristics
        regimes = [
            {'name': 'low_vol', 'vol_mult': 0.5, 'drift_adj': 0.02},
            {'name': 'normal', 'vol_mult': 1.0, 'drift_adj': 0.0},
            {'name': 'high_vol', 'vol_mult': 2.0, 'drift_adj': -0.05},
            {'name': 'crisis', 'vol_mult': 3.0, 'drift_adj': -0.15}
        ]
        
        for i, regime in enumerate(regimes[:n_regimes]):
            regime_params = base_params.copy()
            
            # Adjust parameters for regime
            if 'sigma' in regime_params:
                regime_params['sigma'] *= regime['vol_mult']
            
            if 'mu' in regime_params:
                regime_params['mu'] += regime['drift_adj']
            
            regime_params['scenario_name'] = f"regime_{regime['name']}"
            regime_params['regime_type'] = regime['name']
            
            scenarios.append(regime_params)
        
        return scenarios


class BootstrapGenerator:
    """Bootstrap resampling methods for simulation."""
    
    @staticmethod
    def block_bootstrap(data: np.ndarray, block_length: int = None, 
                       n_samples: int = 1000) -> List[np.ndarray]:
        """
        Generate block bootstrap samples.
        
        Args:
            data: Original time series data
            block_length: Length of blocks (auto-selected if None)
            n_samples: Number of bootstrap samples
            
        Returns:
            List of bootstrap samples
        """
        n = len(data)
        
        if block_length is None:
            # Automatic block length selection
            block_length = max(1, int(np.sqrt(n)))
        
        bootstrap_samples = []
        
        for _ in range(n_samples):
            sample = BootstrapGenerator._generate_single_block_bootstrap(
                data, block_length, n
            )
            bootstrap_samples.append(sample)
        
        return bootstrap_samples
    
    @staticmethod
    def _generate_single_block_bootstrap(data: np.ndarray, 
                                       block_length: int, target_length: int) -> np.ndarray:
        """Generate a single block bootstrap sample."""
        n_blocks = int(np.ceil(target_length / block_length))
        
        # Generate random starting points
        starts = np.random.randint(0, len(data) - block_length + 1, size=n_blocks)
        
        # Construct sample
        sample = []
        for start in starts:
            sample.extend(data[start:start + block_length])
        
        # Trim to target length
        return np.array(sample[:target_length])
    
    @staticmethod
    def circular_block_bootstrap(data: np.ndarray, block_length: int = None,
                                n_samples: int = 1000) -> List[np.ndarray]:
        """
        Generate circular block bootstrap samples.
        
        Args:
            data: Original time series data
            block_length: Length of blocks
            n_samples: Number of bootstrap samples
            
        Returns:
            List of bootstrap samples
        """
        n = len(data)
        
        if block_length is None:
            block_length = max(1, int(np.sqrt(n)))
        
        # Create circular data (wrap around)
        circular_data = np.concatenate([data, data[:block_length-1]])
        
        bootstrap_samples = []
        
        for _ in range(n_samples):
            n_blocks = int(np.ceil(n / block_length))
            starts = np.random.randint(0, n, size=n_blocks)
            
            sample = []
            for start in starts:
                end = start + block_length
                sample.extend(circular_data[start:end])
            
            bootstrap_samples.append(np.array(sample[:n]))
        
        return bootstrap_samples


def run_comprehensive_simulation_analysis(historical_data: np.ndarray,
                                        simulation_config: Dict) -> Dict:
    """
    Run comprehensive simulation analysis pipeline.
    
    Args:
        historical_data: Historical price/return data
        simulation_config: Configuration for simulation analysis
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    logger.info("Starting comprehensive simulation analysis")
    
    # Calculate historical returns
    if historical_data.ndim == 1:
        historical_returns = np.diff(np.log(historical_data))
    else:
        historical_returns = historical_data
    
    # Parameter calibration
    gbm_params = ParameterCalibration.calibrate_gbm_parameters(historical_returns)
    heston_params = ParameterCalibration.calibrate_heston_parameters(historical_returns)
    
    # Stress test scenarios
    stress_scenarios = StressTestGenerator.generate_market_crash_scenarios(gbm_params)
    regime_scenarios = StressTestGenerator.generate_regime_change_scenarios(gbm_params)
    
    # Bootstrap samples
    bootstrap_samples = BootstrapGenerator.block_bootstrap(
        historical_returns, 
        n_samples=simulation_config.get('n_bootstrap_samples', 1000)
    )
    
    analysis_results = {
        'calibrated_parameters': {
            'gbm': gbm_params,
            'heston': heston_params
        },
        'stress_scenarios': stress_scenarios,
        'regime_scenarios': regime_scenarios,
        'bootstrap_statistics': {
            'n_samples': len(bootstrap_samples),
            'bootstrap_means': [np.mean(sample) for sample in bootstrap_samples],
            'bootstrap_stds': [np.std(sample) for sample in bootstrap_samples]
        },
        'historical_statistics': SimulationAnalyzer.calculate_path_statistics(
            historical_data.reshape(1, -1) if historical_data.ndim == 1 else historical_data
        )
    }
    
    logger.info("Comprehensive simulation analysis completed")
    
    return analysis_results


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample historical data
    n_days = 252
    historical_returns = np.random.normal(0.001, 0.02, n_days)
    historical_prices = 100 * np.exp(np.cumsum(historical_returns))
    
    # Run analysis
    config = {
        'n_bootstrap_samples': 500
    }
    
    results = run_comprehensive_simulation_analysis(historical_prices, config)
    
    print("Calibrated GBM parameters:", results['calibrated_parameters']['gbm'])
    print("Number of stress scenarios:", len(results['stress_scenarios']))
    print("Number of regime scenarios:", len(results['regime_scenarios'])) 