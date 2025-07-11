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
from datetime import datetime, timedelta

# Import feature engineering classes
from preprocessing.feature_engineering import TechnicalIndicators, VolatilityFeatures, MarketMicrostructure

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



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
        close_prices = data['Close'].values.flatten()
        
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
            volume_dict = TechnicalIndicators.volume_profile(close_prices, data['Volume'].values.flatten())
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
        close_prices = data['Close'].values.flatten()
        returns = np.diff(np.log(close_prices))
        returns = np.insert(returns, 0, np.nan)  # pad with nan to match length
        
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
                data['High'].values.flatten(), data['Low'].values.flatten(), data['Close'].values.flatten()
            )
        
        # Order Flow Imbalance
        if 'order_flow_imbalance' in self.data_config['features']['market_microstructure']:
            features['order_flow'] = MarketMicrostructure.order_flow_imbalance(
                data['Close'].values.flatten(), data['Volume'].values.flatten()
            )
        
        # Trade Size Distribution
        if 'trade_size_distribution' in self.data_config['features']['market_microstructure']:
            trade_dict = MarketMicrostructure.trade_size_distribution(data['Volume'].values.flatten())
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


def load_config(config_path: str = 'config/config.yaml') -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    # Import visualization module
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from visualization.dataloader_viz import DataLoaderVisualizer, quick_data_validation, validate_preprocessing_pipeline
    
    # Example usage with visualization
    config = load_config()
    data_loader = DataLoader(config)
    
    # Load data for a sample symbol
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    
    try:
        print("Loading and preparing data...")
        data_dict = data_loader.load_and_prepare_data('AAPL', start_date, end_date)
        print(f"Data loaded successfully. Training samples: {len(data_dict['X_train'])}")
        print(f"Feature names: {data_dict['feature_names'][:10]}...")  # Show first 10 features
        
        # Create visualization directory
        viz_dir = 'visualization_output'
        os.makedirs(viz_dir, exist_ok=True)
        
        # Initialize visualizer
        viz = DataLoaderVisualizer()
        
        print("\n=== Running Data Validation Visualizations ===")
        
        # 1. Raw data validation
        print("1. Plotting raw market data...")
        viz.plot_raw_data(
            data_dict['raw_data'], 
            title="AAPL Raw Market Data"
        )
        
        # 2. Technical indicators validation
        print("2. Plotting technical indicators...")
        # Compute indicators separately for visualization
        tech_indicators = data_loader.compute_technical_indicators(data_dict['raw_data'])
        indicators_dict = {col: tech_indicators[col] for col in tech_indicators.columns}
        
        viz.plot_technical_indicators(
            data_dict['raw_data'],
            indicators_dict,
            title="AAPL Technical Indicators"
        )
        
        # 3. Feature engineering validation
        print("3. Plotting engineered features...")
        viz.plot_feature_engineering(
            data_dict['raw_data'],
            data_dict['features'],
            title="AAPL Feature Engineering"
        )
        
        # 4. Data normalization validation
        print("4. Plotting normalization validation...")
        # Create normalized DataFrame for comparison
        normalized_df = pd.DataFrame(
            data_loader.scaler.transform(data_dict['features'].values),
            columns=data_dict['features'].columns,
            index=data_dict['features'].index
        )
        
        viz.plot_data_normalization(
            data_dict['features'],
            normalized_df,
            title="AAPL Data Normalization"
        )
        
        # 5. Sequence generation validation
        print("5. Plotting sequence generation...")
        viz.plot_sequence_generation(
            data_dict['X_train'],
            data_dict['y_train'],
            sequence_length=config['model']['sequence_length'],
            title="AAPL Sequence Generation"
        )
        
        # 6. Correlation matrix
        print("6. Plotting correlation matrix...")
        viz.plot_correlation_matrix(
            data_dict['features'],
            title="AAPL Feature Correlation Matrix"
        )
        
        # 7. Data quality report
        print("7. Generating data quality report...")
        viz.plot_data_quality_report(
            data_dict['features'],
            title="AAPL Data Quality Report"
        )
        
        # 8. Interactive dashboard (optional - requires plotly)
        print("8. Creating interactive dashboard...")
        try:
            interactive_fig = viz.create_interactive_dashboard(
                data_dict['raw_data'],
                indicators_dict,
                title="AAPL Interactive Dashboard"
            )
            interactive_fig.show()
            print("Interactive dashboard displayed")
        except Exception as e:
            print(f"Could not create interactive dashboard: {e}")
        
        # Quick validation function
        print("\n=== Running Quick Validation ===")
        quick_data_validation(
            data_dict['raw_data'],
            indicators_dict
        )
        
        # Comprehensive pipeline validation
        print("\n=== Running Comprehensive Pipeline Validation ===")
        validate_preprocessing_pipeline(
            data_dict['raw_data'],
            data_dict['features'],
            data_dict['X_train'],
            data_dict['y_train']
        )
        
        print(f"\n=== Visualization Complete ===")
        print(f"All plots saved to: {viz_dir}/")
        print("Check the generated plots to validate your preprocessing pipeline!")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        import traceback
        traceback.print_exc() 