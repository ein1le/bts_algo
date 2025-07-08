"""
Data loader and preprocessing for algorithmic trading.

This module handles:
- Financial time series data loading from various sources
- Technical indicator computation
- Feature engineering for deep learning models
- Data preprocessing and normalization
- Train/validation/test splitting with temporal preservation
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import yaml
import talib
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Technical indicator calculations using TA-Lib."""
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index."""
        try:
            return talib.RSI(prices, timeperiod=period)
        except:
            # Fallback manual calculation
            deltas = np.diff(prices)
            seed = deltas[:period+1]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            rs = up / down if down != 0 else 100
            rsi = np.zeros_like(prices)
            rsi[:period] = 100 - (100 / (1 + rs))
            
            for i in range(period, len(prices)):
                delta = deltas[i-1]
                if delta > 0:
                    upval = delta
                    downval = 0.0
                else:
                    upval = 0.0
                    downval = -delta
                
                up = (up * (period - 1) + upval) / period
                down = (down * (period - 1) + downval) / period
                rs = up / down if down != 0 else 100
                rsi[i] = 100 - (100 / (1 + rs))
            
            return rsi
    
    @staticmethod
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD, MACD Signal, and MACD Histogram."""
        try:
            macd_line, macd_signal, macd_hist = talib.MACD(prices, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return macd_line, macd_signal, macd_hist
        except:
            # Fallback manual calculation
            ema_fast = pd.Series(prices).ewm(span=fast).mean().values
            ema_slow = pd.Series(prices).ewm(span=slow).mean().values
            macd_line = ema_fast - ema_slow
            macd_signal = pd.Series(macd_line).ewm(span=signal).mean().values
            macd_hist = macd_line - macd_signal
            return macd_line, macd_signal, macd_hist
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        try:
            upper, middle, lower = talib.BBANDS(prices, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0)
            return upper, middle, lower
        except:
            # Fallback manual calculation
            sma = pd.Series(prices).rolling(window=period).mean().values
            std = pd.Series(prices).rolling(window=period).std().values
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper, sma, lower
    
    @staticmethod
    def moving_averages(prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate various moving averages."""
        return {
            'sma_10': pd.Series(prices).rolling(window=10).mean().values,
            'sma_20': pd.Series(prices).rolling(window=20).mean().values,
            'sma_50': pd.Series(prices).rolling(window=50).mean().values,
            'ema_10': pd.Series(prices).ewm(span=10).mean().values,
            'ema_20': pd.Series(prices).ewm(span=20).mean().values,
            'ema_50': pd.Series(prices).ewm(span=50).mean().values,
        }
    
    @staticmethod
    def volume_profile(prices: np.ndarray, volumes: np.ndarray, period: int = 20) -> Dict[str, np.ndarray]:
        """Calculate volume-based indicators."""
        # Volume-weighted average price
        vwap = pd.Series(prices * volumes).rolling(window=period).sum() / pd.Series(volumes).rolling(window=period).sum()
        
        # On-balance volume
        obv = np.zeros_like(prices)
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif prices[i] < prices[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]
        
        return {
            'vwap': vwap.values,
            'obv': obv,
            'volume_sma': pd.Series(volumes).rolling(window=period).mean().values
        }


class VolatilityFeatures:
    """Volatility-based feature calculations."""
    
    @staticmethod
    def realized_volatility(prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate realized volatility."""
        returns = np.diff(np.log(prices))
        return pd.Series(returns).rolling(window=period).std().values * np.sqrt(252)
    
    @staticmethod
    def garch_volatility(returns: np.ndarray) -> np.ndarray:
        """Simple GARCH(1,1) volatility estimation."""
        # Simplified GARCH estimation
        variance = np.zeros_like(returns)
        variance[0] = np.var(returns)
        
        # Simple GARCH parameters (in practice, these would be estimated)
        omega, alpha, beta = 0.000001, 0.1, 0.85
        
        for i in range(1, len(returns)):
            variance[i] = omega + alpha * returns[i-1]**2 + beta * variance[i-1]
        
        return np.sqrt(variance) * np.sqrt(252)  # Annualized
    
    @staticmethod
    def volatility_clustering(returns: np.ndarray, period: int = 20) -> np.ndarray:
        """Detect volatility clustering patterns."""
        rolling_vol = pd.Series(returns).rolling(window=period).std()
        vol_ratio = rolling_vol / rolling_vol.rolling(window=period*2).mean()
        return vol_ratio.fillna(1.0).values


class MarketMicrostructure:
    """Market microstructure features."""
    
    @staticmethod
    def bid_ask_spread(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Estimate bid-ask spread from OHLC data."""
        # Simplified spread estimation
        return (high - low) / close
    
    @staticmethod
    def order_flow_imbalance(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Estimate order flow imbalance."""
        # Simplified calculation based on price and volume
        price_changes = np.diff(close)
        volume_changes = np.diff(volume)
        
        imbalance = np.zeros_like(close)
        imbalance[1:] = price_changes * volume_changes[:-1] if len(volume_changes) > 0 else price_changes
        
        return imbalance
    
    @staticmethod
    def trade_size_distribution(volume: np.ndarray, period: int = 20) -> Dict[str, np.ndarray]:
        """Analyze trade size distribution."""
        volume_mean = pd.Series(volume).rolling(window=period).mean()
        volume_std = pd.Series(volume).rolling(window=period).std()
        
        return {
            'volume_zscore': ((volume - volume_mean) / volume_std).fillna(0).values,
            'volume_percentile': pd.Series(volume).rolling(window=period).rank(pct=True).fillna(0.5).values
        }


class DataLoader:
    """Main data loader class for financial time series."""
    
    def __init__(self, config: Dict):
        """
        Initialize data loader with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config['data']
        self.scaler = None
        self.feature_names = []
        
    def fetch_price_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch price data from specified source.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTC-USD')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        source = self.data_config['sources']['price_data']
        
        if source == 'yfinance':
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                data = data.dropna()
                logger.info(f"Fetched {len(data)} records for {symbol} from yfinance")
                return data
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                return pd.DataFrame()
        else:
            raise NotImplementedError(f"Data source {source} not implemented")
    
    def compute_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        features = pd.DataFrame(index=data.index)
        close_prices = data['Close'].values
        
        # RSI
        if 'rsi' in self.data_config['features']['technical_indicators']:
            features['rsi'] = TechnicalIndicators.rsi(close_prices)
        
        # MACD
        if 'macd' in self.data_config['features']['technical_indicators']:
            macd_line, macd_signal, macd_hist = TechnicalIndicators.macd(close_prices)
            features['macd_line'] = macd_line
            features['macd_signal'] = macd_signal
            features['macd_hist'] = macd_hist
        
        # Bollinger Bands
        if 'bollinger_bands' in self.data_config['features']['technical_indicators']:
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(close_prices)
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_position'] = (close_prices - bb_lower) / (bb_upper - bb_lower)
        
        # Moving Averages
        if 'moving_averages' in self.data_config['features']['technical_indicators']:
            ma_dict = TechnicalIndicators.moving_averages(close_prices)
            for name, values in ma_dict.items():
                features[name] = values
        
        # Volume Profile
        if 'volume_profile' in self.data_config['features']['technical_indicators']:
            volume_dict = TechnicalIndicators.volume_profile(close_prices, data['Volume'].values)
            for name, values in volume_dict.items():
                features[name] = values
        
        return features
    
    def compute_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volatility-based features.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=data.index)
        close_prices = data['Close'].values
        returns = np.diff(np.log(close_prices))
        returns = np.concatenate([[0], returns])  # Pad to match length
        
        # Realized Volatility
        if 'realized_volatility' in self.data_config['features']['volatility_features']:
            features['realized_vol'] = VolatilityFeatures.realized_volatility(close_prices)
        
        # GARCH Volatility
        if 'garch_volatility' in self.data_config['features']['volatility_features']:
            features['garch_vol'] = VolatilityFeatures.garch_volatility(returns)
        
        # Volatility Clustering
        features['vol_clustering'] = VolatilityFeatures.volatility_clustering(returns)
        
        return features
    
    def compute_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute market microstructure features.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with microstructure features
        """
        features = pd.DataFrame(index=data.index)
        
        # Bid-Ask Spread Proxy
        if 'bid_ask_spread' in self.data_config['features']['market_microstructure']:
            features['spread_proxy'] = MarketMicrostructure.bid_ask_spread(
                data['High'].values, data['Low'].values, data['Close'].values
            )
        
        # Order Flow Imbalance
        if 'order_flow_imbalance' in self.data_config['features']['market_microstructure']:
            features['order_flow'] = MarketMicrostructure.order_flow_imbalance(
                data['Close'].values, data['Volume'].values
            )
        
        # Trade Size Distribution
        if 'trade_size_distribution' in self.data_config['features']['market_microstructure']:
            trade_dict = MarketMicrostructure.trade_size_distribution(data['Volume'].values)
            for name, values in trade_dict.items():
                features[name] = values
        
        return features
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all features for the dataset.
        
        Args:
            data: Raw OHLCV DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering...")
        
        # Base features
        features = pd.DataFrame(index=data.index)
        features['close'] = data['Close']
        features['volume'] = data['Volume']
        features['returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Technical indicators
        tech_features = self.compute_technical_indicators(data)
        features = pd.concat([features, tech_features], axis=1)
        
        # Volatility features
        vol_features = self.compute_volatility_features(data)
        features = pd.concat([features, vol_features], axis=1)
        
        # Microstructure features
        micro_features = self.compute_microstructure_features(data)
        features = pd.concat([features, micro_features], axis=1)
        
        # Remove NaN values
        features = features.dropna()
        
        self.feature_names = features.columns.tolist()
        logger.info(f"Engineered {len(self.feature_names)} features: {self.feature_names}")
        
        return features
    
    def preprocess_data(self, features: pd.DataFrame) -> np.ndarray:
        """
        Preprocess features with normalization and outlier handling.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Preprocessed feature array
        """
        logger.info("Preprocessing data...")
        
        # Handle outliers
        if self.data_config['preprocessing']['outlier_handling'] == 'winsorize':
            features = features.clip(lower=features.quantile(0.01), upper=features.quantile(0.99), axis=1)
        elif self.data_config['preprocessing']['outlier_handling'] == 'clip':
            features = features.clip(lower=features.quantile(0.05), upper=features.quantile(0.95), axis=1)
        
        # Normalization
        normalization = self.data_config['preprocessing']['normalization']
        
        if normalization == 'standard':
            self.scaler = StandardScaler()
        elif normalization == 'minmax':
            self.scaler = MinMaxScaler()
        elif normalization == 'robust_scaler':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")
        
        # Fit and transform
        normalized_features = self.scaler.fit_transform(features.values)
        
        logger.info(f"Preprocessed data shape: {normalized_features.shape}")
        
        return normalized_features
    
    def create_sequences(self, features: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            features: Feature array
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            # Target is next period's return (for hedging)
            y.append(features[i, self.feature_names.index('returns')])
        
        return np.array(X), np.array(y)
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/validation/test sets preserving temporal order.
        
        Args:
            X: Feature sequences
            y: Target sequences
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        split_config = self.data_config['split']
        
        if split_config['method'] == 'time_series_split':
            # Simple time-based split
            n_samples = len(X)
            train_end = int(n_samples * split_config['train_ratio'])
            val_end = int(n_samples * (split_config['train_ratio'] + split_config['validation_ratio']))
            
            X_train = X[:train_end]
            X_val = X[train_end:val_end]
            X_test = X[val_end:]
            
            y_train = y[:train_end]
            y_val = y[train_end:val_end]
            y_test = y[val_end:]
            
        else:
            raise ValueError(f"Unknown split method: {split_config['method']}")
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def load_and_prepare_data(self, symbol: str, start_date: str, end_date: str) -> Dict:
        """
        Complete data loading and preparation pipeline.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with prepared datasets
        """
        logger.info(f"Loading and preparing data for {symbol}")
        
        # Fetch raw data
        raw_data = self.fetch_price_data(symbol, start_date, end_date)
        if raw_data.empty:
            raise ValueError(f"No data fetched for {symbol}")
        
        # Engineer features
        features = self.engineer_features(raw_data)
        
        # Preprocess
        processed_features = self.preprocess_data(features)
        
        # Create sequences
        sequence_length = self.config['model']['sequence_length']
        X, y = self.create_sequences(processed_features, sequence_length)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'raw_data': raw_data,
            'features': features
        }


def load_config(config_path: str = 'python/config.yaml') -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    # Example usage
    config = load_config()
    data_loader = DataLoader(config)
    
    # Load data for a sample symbol
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    
    try:
        data_dict = data_loader.load_and_prepare_data('AAPL', start_date, end_date)
        print(f"Data loaded successfully. Training samples: {len(data_dict['X_train'])}")
        print(f"Feature names: {data_dict['feature_names'][:10]}...")  # Show first 10 features
    except Exception as e:
        logger.error(f"Error loading data: {e}") 