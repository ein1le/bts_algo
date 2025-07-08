"""
Monte Carlo simulators for financial time series.

This module implements various stochastic models for generating synthetic price paths:
1. Geometric Brownian Motion (GBM)
2. Heston Stochastic Volatility Model
3. Merton Jump Diffusion Model
4. Tick-to-path converter for real market data

These are used for:
- Data augmentation during training
- Backtesting stress scenarios
- Monte Carlo permutation testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy.stats import norm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeometricBrownianMotion:
    """
    Geometric Brownian Motion (GBM) simulator.
    
    dS_t = μ S_t dt + σ S_t dW_t
    
    Where:
    - S_t is the asset price at time t
    - μ is the drift (expected return)
    - σ is the volatility
    - dW_t is a Wiener process (Brownian motion)
    """
    
    def __init__(self, mu: float, sigma: float):
        """
        Initialize GBM parameters.
        
        Args:
            mu: Drift parameter (annualized)
            sigma: Volatility parameter (annualized)
        """
        self.mu = mu
        self.sigma = sigma
    
    def simulate_path(self, S0: float, T: float, n_steps: int, n_paths: int = 1) -> np.ndarray:
        """
        Simulate GBM paths.
        
        Args:
            S0: Initial asset price
            T: Time horizon (in years)
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
            
        Returns:
            Array of shape [n_paths, n_steps+1] with simulated paths
        """
        dt = T / n_steps
        
        # Generate random shocks
        Z = np.random.normal(size=(n_paths, n_steps))
        
        # Calculate price increments
        # Using exact solution: S_t = S_0 * exp((μ - σ²/2)t + σW_t)
        drift_term = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion_term = self.sigma * np.sqrt(dt) * Z
        
        # Cumulative log returns
        log_returns = np.cumsum(drift_term + diffusion_term, axis=1)
        
        # Convert to prices
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        paths[:, 1:] = S0 * np.exp(log_returns)
        
        return paths


class HestonModel:
    """
    Heston Stochastic Volatility Model.
    
    Asset price: dS_t = μ S_t dt + √v_t S_t dW₁_t
    Variance:    dv_t = κ(θ - v_t)dt + σ_v √v_t dW₂_t
    
    Where:
    - v_t is the variance at time t
    - κ is the mean reversion speed
    - θ is the long-term variance
    - σ_v is the volatility of variance
    - ρ is the correlation between W₁ and W₂
    """
    
    def __init__(self, mu: float, kappa: float, theta: float, sigma_v: float, rho: float, v0: float):
        """
        Initialize Heston model parameters.
        
        Args:
            mu: Asset drift
            kappa: Mean reversion speed of variance
            theta: Long-term variance level
            sigma_v: Volatility of variance
            rho: Correlation between asset and variance Brownian motions
            v0: Initial variance
        """
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.v0 = v0
        
        # Ensure Feller condition is satisfied
        if 2 * kappa * theta < sigma_v**2:
            logger.warning("Feller condition not satisfied - variance may become negative")
    
    def simulate_path(self, S0: float, T: float, n_steps: int, n_paths: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston model paths using Euler-Maruyama scheme.
        
        Args:
            S0: Initial asset price
            T: Time horizon (in years)
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
            
        Returns:
            Tuple of (price_paths, variance_paths)
        """
        dt = T / n_steps
        
        # Initialize arrays
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        
        S[:, 0] = S0
        v[:, 0] = self.v0
        
        # Generate correlated random numbers
        for i in range(n_steps):
            # Independent normal random variables
            Z1 = np.random.normal(size=n_paths)
            Z2 = np.random.normal(size=n_paths)
            
            # Create correlated Brownian motions
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            # Variance process (with Feller reflection to ensure positivity)
            v[:, i+1] = np.maximum(
                v[:, i] + self.kappa * (self.theta - v[:, i]) * dt + 
                self.sigma_v * np.sqrt(np.maximum(v[:, i], 0)) * np.sqrt(dt) * W2,
                1e-8  # Small positive number to avoid numerical issues
            )
            
            # Asset price process
            S[:, i+1] = S[:, i] * (
                1 + self.mu * dt + 
                np.sqrt(np.maximum(v[:, i], 0)) * np.sqrt(dt) * W1
            )
        
        return S, v


class MertonJumpDiffusion:
    """
    Merton Jump Diffusion Model.
    
    dS_t = μ S_t dt + σ S_t dW_t + S_t dJ_t
    
    Where J_t is a compound Poisson process:
    - Jump arrivals follow Poisson process with intensity λ
    - Jump sizes are log-normal with parameters μ_J and σ_J
    """
    
    def __init__(self, mu: float, sigma: float, lambda_j: float, mu_j: float, sigma_j: float):
        """
        Initialize jump diffusion parameters.
        
        Args:
            mu: Drift parameter
            sigma: Diffusion volatility
            lambda_j: Jump intensity (jumps per year)
            mu_j: Mean of log jump size
            sigma_j: Standard deviation of log jump size
        """
        self.mu = mu
        self.sigma = sigma
        self.lambda_j = lambda_j
        self.mu_j = mu_j
        self.sigma_j = sigma_j
    
    def simulate_path(self, S0: float, T: float, n_steps: int, n_paths: int = 1) -> np.ndarray:
        """
        Simulate jump diffusion paths.
        
        Args:
            S0: Initial asset price
            T: Time horizon (in years)
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
            
        Returns:
            Array of shape [n_paths, n_steps+1] with simulated paths
        """
        dt = T / n_steps
        
        # Initialize price array
        S = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = S0
        
        for i in range(n_steps):
            # Diffusion component (GBM)
            Z_diffusion = np.random.normal(size=n_paths)
            diffusion_return = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z_diffusion
            
            # Jump component
            # Number of jumps in each path during time interval dt
            n_jumps = np.random.poisson(self.lambda_j * dt, size=n_paths)
            
            # Jump sizes (log-normal)
            jump_return = np.zeros(n_paths)
            for path_idx in range(n_paths):
                if n_jumps[path_idx] > 0:
                    jump_sizes = np.random.normal(self.mu_j, self.sigma_j, size=n_jumps[path_idx])
                    jump_return[path_idx] = np.sum(jump_sizes)
            
            # Total return
            total_return = diffusion_return + jump_return
            
            # Update prices
            S[:, i+1] = S[:, i] * np.exp(total_return)
        
        return S


class MonteCarloSimulator:
    """
    Main Monte Carlo simulator class that orchestrates different models.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize simulator with configuration.
        
        Args:
            config: Configuration dictionary with Monte Carlo parameters
        """
        self.config = config
        self.mc_config = config['monte_carlo']
        
        # Initialize model instances
        self.gbm = GeometricBrownianMotion(
            mu=self.mc_config['models']['gbm']['mu'],
            sigma=self.mc_config['models']['gbm']['sigma']
        )
        
        self.heston = HestonModel(
            mu=self.mc_config['models']['heston'].get('mu', 0.05),
            kappa=self.mc_config['models']['heston']['kappa'],
            theta=self.mc_config['models']['heston']['theta'],
            sigma_v=self.mc_config['models']['heston']['sigma_v'],
            rho=self.mc_config['models']['heston']['rho'],
            v0=self.mc_config['models']['heston']['v0']
        )
        
        self.jump_diffusion = MertonJumpDiffusion(
            mu=self.mc_config['models']['jump_diffusion'].get('mu', 0.05),
            sigma=self.mc_config['models']['jump_diffusion'].get('sigma', 0.2),
            lambda_j=self.mc_config['models']['jump_diffusion']['lambda_j'],
            mu_j=self.mc_config['models']['jump_diffusion']['mu_j'],
            sigma_j=self.mc_config['models']['jump_diffusion']['sigma_j']
        )
    
    def generate_path(self, model_type: str, n_steps: int = None, S0: float = 100.0, 
                     T: float = 1.0, n_paths: int = 1) -> np.ndarray:
        """
        Generate price path using specified model.
        
        Args:
            model_type: Type of model ('gbm', 'heston', 'jump_diffusion')
            n_steps: Number of time steps (defaults to config value)
            S0: Initial price
            T: Time horizon in years
            n_paths: Number of paths
            
        Returns:
            Simulated price paths
        """
        if n_steps is None:
            n_steps = self.mc_config['n_steps']
        
        if model_type == 'gbm':
            paths = self.gbm.simulate_path(S0, T, n_steps, n_paths)
        elif model_type == 'heston':
            price_paths, variance_paths = self.heston.simulate_path(S0, T, n_steps, n_paths)
            paths = price_paths
        elif model_type == 'jump_diffusion':
            paths = self.jump_diffusion.simulate_path(S0, T, n_steps, n_paths)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Return single path if n_paths=1
        if n_paths == 1:
            return paths[0]
        
        return paths
    
    def generate_bootstrap_sample(self, original_returns: np.ndarray, block_length: Optional[int] = None) -> np.ndarray:
        """
        Generate bootstrap sample using block bootstrap to preserve serial correlation.
        
        Args:
            original_returns: Original return series
            block_length: Length of blocks for bootstrap (if None, uses automatic selection)
            
        Returns:
            Bootstrap sample of returns
        """
        n = len(original_returns)
        
        if block_length is None:
            # Automatic block length selection (simple heuristic)
            block_length = max(1, int(np.sqrt(n)))
        
        # Number of blocks needed
        n_blocks = int(np.ceil(n / block_length))
        
        # Generate random starting points for blocks
        starts = np.random.randint(0, n - block_length + 1, size=n_blocks)
        
        # Construct bootstrap sample
        bootstrap_sample = []
        for start in starts:
            bootstrap_sample.extend(original_returns[start:start + block_length])
        
        # Trim to original length
        return np.array(bootstrap_sample[:n])
    
    def simulate_scenarios(self, base_params: Dict, n_scenarios: int = 1000) -> List[np.ndarray]:
        """
        Simulate multiple scenarios with parameter uncertainty.
        
        Args:
            base_params: Base parameters for simulation
            n_scenarios: Number of scenarios to generate
            
        Returns:
            List of simulated price paths
        """
        scenarios = []
        
        for _ in range(n_scenarios):
            # Add parameter uncertainty
            if 'model_type' not in base_params:
                model_type = np.random.choice(['gbm', 'heston', 'jump_diffusion'])
            else:
                model_type = base_params['model_type']
            
            # Parameter perturbation (add noise to parameters)
            if model_type == 'gbm':
                mu_perturbed = self.gbm.mu + np.random.normal(0, 0.02)
                sigma_perturbed = max(0.01, self.gbm.sigma + np.random.normal(0, 0.05))
                
                # Temporarily modify parameters
                original_mu, original_sigma = self.gbm.mu, self.gbm.sigma
                self.gbm.mu, self.gbm.sigma = mu_perturbed, sigma_perturbed
                
                path = self.generate_path(model_type, **{k: v for k, v in base_params.items() if k != 'model_type'})
                
                # Restore original parameters
                self.gbm.mu, self.gbm.sigma = original_mu, original_sigma
                
            else:
                # For other models, use base parameters without perturbation
                path = self.generate_path(model_type, **{k: v for k, v in base_params.items() if k != 'model_type'})
            
            scenarios.append(path)
        
        return scenarios
    
    def calibrate_to_data(self, price_data: np.ndarray, model_type: str = 'gbm') -> Dict:
        """
        Calibrate model parameters to historical data.
        
        Args:
            price_data: Historical price data
            model_type: Model to calibrate ('gbm', 'heston', 'jump_diffusion')
            
        Returns:
            Calibrated parameters
        """
        # Calculate returns
        returns = np.diff(np.log(price_data))
        
        if model_type == 'gbm':
            # Simple method of moments for GBM
            mu_est = np.mean(returns) * 252 + 0.5 * np.var(returns) * 252  # Annualized drift
            sigma_est = np.std(returns) * np.sqrt(252)  # Annualized volatility
            
            return {'mu': mu_est, 'sigma': sigma_est}
        
        elif model_type == 'heston':
            # Simplified calibration for Heston (in practice, would use MLE or method of moments)
            sigma_est = np.std(returns) * np.sqrt(252)
            
            # Use simple estimates
            return {
                'mu': np.mean(returns) * 252,
                'kappa': 2.0,  # Default value
                'theta': sigma_est**2,  # Long-term variance
                'sigma_v': 0.3,  # Default value
                'rho': -0.7,  # Typical negative correlation
                'v0': sigma_est**2  # Initial variance
            }
        
        else:
            raise NotImplementedError(f"Calibration for {model_type} not implemented")
    
    def generate_stress_scenarios(self, base_price: float = 100.0, stress_factors: List[float] = None) -> Dict[str, np.ndarray]:
        """
        Generate stress test scenarios.
        
        Args:
            base_price: Base price level
            stress_factors: List of stress factors to apply
            
        Returns:
            Dictionary of stress scenarios
        """
        if stress_factors is None:
            stress_factors = [0.5, 0.7, 1.0, 1.5, 2.0]  # Volatility multipliers
        
        scenarios = {}
        
        for factor in stress_factors:
            # Stress GBM volatility
            original_sigma = self.gbm.sigma
            self.gbm.sigma = original_sigma * factor
            
            scenario_paths = []
            for _ in range(100):  # Generate multiple paths for each stress level
                path = self.generate_path('gbm', S0=base_price)
                scenario_paths.append(path)
            
            scenarios[f'stress_{factor}x'] = np.array(scenario_paths)
            
            # Restore original volatility
            self.gbm.sigma = original_sigma
        
        return scenarios


class TickToPath:
    """
    Convert real market tick data to continuous price paths.
    """
    
    @staticmethod
    def aggregate_ticks(tick_data: pd.DataFrame, frequency: str = '1min') -> pd.DataFrame:
        """
        Aggregate tick data to specified frequency.
        
        Args:
            tick_data: DataFrame with tick data (timestamp, price, volume)
            frequency: Aggregation frequency ('1min', '5min', '1H', etc.)
            
        Returns:
            Aggregated OHLCV data
        """
        # Ensure timestamp is datetime
        tick_data['timestamp'] = pd.to_datetime(tick_data['timestamp'])
        tick_data.set_index('timestamp', inplace=True)
        
        # Aggregate to OHLCV
        ohlcv = tick_data['price'].resample(frequency).ohlc()
        ohlcv['volume'] = tick_data['volume'].resample(frequency).sum()
        
        # Forward fill missing values
        ohlcv = ohlcv.fillna(method='ffill')
        
        return ohlcv
    
    @staticmethod
    def interpolate_prices(price_data: pd.Series, target_frequency: str = '1s') -> pd.Series:
        """
        Interpolate prices to higher frequency.
        
        Args:
            price_data: Time series of prices
            target_frequency: Target frequency for interpolation
            
        Returns:
            Interpolated price series
        """
        # Create target time index
        target_index = pd.date_range(
            start=price_data.index[0],
            end=price_data.index[-1],
            freq=target_frequency
        )
        
        # Reindex and interpolate
        interpolated = price_data.reindex(target_index).interpolate(method='linear')
        
        return interpolated


if __name__ == "__main__":
    # Example usage
    config = {
        'monte_carlo': {
            'n_paths': 1000,
            'n_steps': 252,
            'models': {
                'gbm': {'mu': 0.08, 'sigma': 0.2},
                'heston': {
                    'kappa': 2.0, 'theta': 0.04, 'sigma_v': 0.3,
                    'rho': -0.7, 'v0': 0.04
                },
                'jump_diffusion': {
                    'lambda_j': 0.1, 'mu_j': -0.05, 'sigma_j': 0.15
                }
            }
        }
    }
    
    # Initialize simulator
    simulator = MonteCarloSimulator(config)
    
    # Generate sample paths
    gbm_path = simulator.generate_path('gbm', n_steps=100, S0=100.0)
    heston_path = simulator.generate_path('heston', n_steps=100, S0=100.0)
    jump_path = simulator.generate_path('jump_diffusion', n_steps=100, S0=100.0)
    
    print(f"GBM path shape: {gbm_path.shape}")
    print(f"Heston path shape: {heston_path.shape}")
    print(f"Jump diffusion path shape: {jump_path.shape}")
    
    print(f"GBM final price: {gbm_path[-1]:.2f}")
    print(f"Heston final price: {heston_path[-1]:.2f}")
    print(f"Jump diffusion final price: {jump_path[-1]:.2f}") 