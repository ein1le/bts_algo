"""
Feature Engineering Pipeline for Financial Time Series.

This module handles advanced feature engineering specifically designed
for financial time series data including technical indicators, market
microstructure features, and alternative data integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TechnicalFeatureEngine:
    """Advanced technical indicator feature engineering."""
    
    def __init__(self, config: Dict):
        """
        Initialize feature engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
    def create_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create momentum-based features.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=data.index)
        close = data['Close']
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = close.pct_change(period)
        
        # Momentum
        for period in [10, 20, 50]:
            features[f'momentum_{period}'] = close / close.shift(period) - 1
        
        # Relative Strength Index (RSI)
        features['rsi_14'] = self._calculate_rsi(close, 14)
        features['rsi_21'] = self._calculate_rsi(close, 21)
        
        # Stochastic Oscillator
        features['stoch_k'], features['stoch_d'] = self._calculate_stochastic(data)
        
        # Williams %R
        features['williams_r'] = self._calculate_williams_r(data)
        
        return features
    
    def create_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create trend-following features.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with trend features
        """
        features = pd.DataFrame(index=data.index)
        close = data['Close']
        
        # Moving Averages
        ma_periods = [5, 10, 20, 50, 100, 200]
        for period in ma_periods:
            features[f'sma_{period}'] = close.rolling(window=period).mean()
            features[f'ema_{period}'] = close.ewm(span=period).mean()
        
        # Moving Average Convergence Divergence (MACD)
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        sma_20 = close.rolling(window=bb_period).mean()
        rolling_std = close.rolling(window=bb_period).std()
        features['bb_upper'] = sma_20 + (rolling_std * bb_std)
        features['bb_lower'] = sma_20 - (rolling_std * bb_std)
        features['bb_middle'] = sma_20
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Average Directional Index (ADX)
        features['adx'] = self._calculate_adx(data)
        
        return features
    
    def create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create volatility-based features.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=data.index)
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # Historical Volatility
        returns = close.pct_change()
        for period in [10, 20, 30, 60]:
            features[f'hist_vol_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
        
        # True Range and Average True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['atr'] = true_range.rolling(window=14).mean()
        
        # Volatility Ratios
        features['vol_ratio_short_long'] = (
            returns.rolling(window=10).std() / 
            returns.rolling(window=30).std()
        )
        
        # Garman-Klass Volatility
        features['gk_volatility'] = self._calculate_garman_klass_volatility(data)
        
        # Parkinson Volatility
        features['parkinson_vol'] = self._calculate_parkinson_volatility(data)
        
        return features
    
    def create_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-based features.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=data.index)
        close = data['Close']
        volume = data['Volume']
        high = data['High']
        low = data['Low']
        
        # Volume Moving Averages
        for period in [10, 20, 50]:
            features[f'vol_sma_{period}'] = volume.rolling(window=period).mean()
        
        # Volume Rate of Change
        features['vol_roc'] = volume.pct_change()
        
        # On-Balance Volume (OBV)
        features['obv'] = self._calculate_obv(close, volume)
        
        # Volume-Weighted Average Price (VWAP)
        features['vwap'] = self._calculate_vwap(data)
        
        # Accumulation/Distribution Line
        features['ad_line'] = self._calculate_ad_line(data)
        
        # Money Flow Index
        features['mfi'] = self._calculate_money_flow_index(data)
        
        # Volume Profile
        features['volume_profile'] = self._calculate_volume_profile(close, volume)
        
        return features
    
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, 
                            d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (simplified)."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = (high - high.shift(1)).where(
            (high - high.shift(1)) > (low.shift(1) - low), 0
        ).where((high - high.shift(1)) > 0, 0)
        
        dm_minus = (low.shift(1) - low).where(
            (low.shift(1) - low) > (high - high.shift(1)), 0
        ).where((low.shift(1) - low) > 0, 0)
        
        # Smoothed values
        atr = true_range.rolling(window=period).mean()
        di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)
        
        # ADX calculation
        dx = 100 * (abs(di_plus - di_minus) / (di_plus + di_minus))
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _calculate_garman_klass_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Garman-Klass volatility estimator."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        open_price = data['Open'] if 'Open' in data.columns else close.shift(1)
        
        gk = (
            0.5 * (np.log(high / low))**2 - 
            (2 * np.log(2) - 1) * (np.log(close / open_price))**2
        )
        
        return np.sqrt(gk * 252)  # Annualized
    
    def _calculate_parkinson_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Parkinson volatility estimator."""
        high = data['High']
        low = data['Low']
        
        parkinson = (1 / (4 * np.log(2))) * (np.log(high / low))**2
        
        return np.sqrt(parkinson * 252)  # Annualized
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        price_change = close.diff()
        obv = volume.copy()
        obv[price_change < 0] = -volume[price_change < 0]
        obv[price_change == 0] = 0
        return obv.cumsum()
    
    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume-Weighted Average Price."""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        return (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    def _calculate_ad_line(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']
        
        clv = ((close - low) - (high - close)) / (high - low)
        ad_line = (clv * volume).cumsum()
        
        return ad_line
    
    def _calculate_money_flow_index(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfr = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + mfr))
        
        return mfi
    
    def _calculate_volume_profile(self, close: pd.Series, volume: pd.Series, 
                                period: int = 20) -> pd.Series:
        """Calculate Volume Profile indicator."""
        volume_profile = pd.Series(index=close.index, dtype=float)
        
        for i in range(period, len(close)):
            period_data = pd.DataFrame({
                'price': close.iloc[i-period:i],
                'volume': volume.iloc[i-period:i]
            })
            
            # Calculate volume-weighted price percentile
            total_volume = period_data['volume'].sum()
            if total_volume > 0:
                current_price = close.iloc[i]
                below_volume = period_data[period_data['price'] <= current_price]['volume'].sum()
                percentile = below_volume / total_volume
                volume_profile.iloc[i] = percentile
            else:
                volume_profile.iloc[i] = 0.5
        
        return volume_profile


class MarketMicrostructureFeatures:
    """Market microstructure feature engineering."""
    
    @staticmethod
    def create_microstructure_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Create market microstructure features.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with microstructure features
        """
        features = pd.DataFrame(index=data.index)
        
        # Bid-Ask Spread Proxy
        features['spread_proxy'] = (data['High'] - data['Low']) / data['Close']
        
        # Price Impact
        returns = data['Close'].pct_change()
        volume = data['Volume']
        features['price_impact'] = returns / np.log(1 + volume)
        
        # Amihud Illiquidity
        features['amihud_illiquidity'] = abs(returns) / volume
        
        # Roll Spread Estimator
        features['roll_spread'] = MarketMicrostructureFeatures._calculate_roll_spread(returns)
        
        # High-Low Spread
        features['hl_spread'] = 2 * (np.exp(
            np.sqrt(2) * np.sqrt(np.log(data['High'] / data['Low']))
        ) - 1)
        
        return features
    
    @staticmethod
    def _calculate_roll_spread(returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Roll spread estimator."""
        # Simplified Roll spread calculation
        covariance = returns.rolling(window=window).apply(
            lambda x: np.cov(x[:-1], x[1:])[0, 1], raw=True
        )
        roll_spread = 2 * np.sqrt(-covariance)
        return roll_spread.fillna(0)


class AlternativeDataFeatures:
    """Alternative data feature engineering."""
    
    @staticmethod
    def create_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Create calendar-based features.
        
        Args:
            data: DataFrame with datetime index
            
        Returns:
            DataFrame with calendar features
        """
        features = pd.DataFrame(index=data.index)
        
        # Time-based features
        features['hour'] = data.index.hour
        features['day_of_week'] = data.index.dayofweek
        features['day_of_month'] = data.index.day
        features['month'] = data.index.month
        features['quarter'] = data.index.quarter
        features['is_weekend'] = data.index.dayofweek >= 5
        features['is_month_end'] = data.index.is_month_end
        features['is_quarter_end'] = data.index.is_quarter_end
        
        # Market session features (assuming US markets)
        features['is_market_open'] = (
            (data.index.hour >= 9) & (data.index.hour < 16) &
            (data.index.dayofweek < 5)
        )
        features['is_pre_market'] = (
            (data.index.hour >= 4) & (data.index.hour < 9) &
            (data.index.dayofweek < 5)
        )
        features['is_after_hours'] = (
            (data.index.hour >= 16) & (data.index.hour < 20) &
            (data.index.dayofweek < 5)
        )
        
        return features


class FeatureEngineeringPipeline:
    """Main feature engineering pipeline."""
    
    def __init__(self, config: Dict):
        """
        Initialize feature engineering pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.technical_engine = TechnicalFeatureEngine(config)
        self.scaler = None
        
    def create_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features for the dataset.
        
        Args:
            data: Raw OHLCV DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting comprehensive feature engineering")
        
        all_features = pd.DataFrame(index=data.index)
        
        # Basic price features
        all_features['close'] = data['Close']
        all_features['volume'] = data['Volume']
        all_features['returns'] = data['Close'].pct_change()
        all_features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Technical features
        momentum_features = self.technical_engine.create_momentum_features(data)
        trend_features = self.technical_engine.create_trend_features(data)
        volatility_features = self.technical_engine.create_volatility_features(data)
        volume_features = self.technical_engine.create_volume_features(data)
        
        # Microstructure features
        microstructure_features = MarketMicrostructureFeatures.create_microstructure_features(data)
        
        # Calendar features
        calendar_features = AlternativeDataFeatures.create_calendar_features(data)
        
        # Combine all features
        feature_groups = [
            momentum_features,
            trend_features,
            volatility_features,
            volume_features,
            microstructure_features,
            calendar_features
        ]
        
        for feature_group in feature_groups:
            all_features = pd.concat([all_features, feature_group], axis=1)
        
        # Remove duplicate columns
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        
        # Handle infinite values
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill NaN values
        all_features = all_features.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Feature engineering completed. Created {len(all_features.columns)} features")
        
        return all_features
    
    def preprocess_features(self, features: pd.DataFrame, 
                          fit_scaler: bool = True) -> pd.DataFrame:
        """
        Preprocess features with scaling and outlier handling.
        
        Args:
            features: Feature DataFrame
            fit_scaler: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Preprocessed features
        """
        logger.info("Preprocessing features")
        
        # Handle outliers
        outlier_method = self.config.get('feature_preprocessing', {}).get('outlier_handling', 'winsorize')
        
        if outlier_method == 'winsorize':
            features = features.clip(
                lower=features.quantile(0.01),
                upper=features.quantile(0.99),
                axis=1
            )
        elif outlier_method == 'clip':
            features = features.clip(
                lower=features.quantile(0.05),
                upper=features.quantile(0.95),
                axis=1
            )
        
        # Scaling
        scaling_method = self.config.get('feature_preprocessing', {}).get('scaling', 'robust')
        
        if fit_scaler:
            if scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            else:  # robust
                self.scaler = RobustScaler()
            
            scaled_features = pd.DataFrame(
                self.scaler.fit_transform(features),
                index=features.index,
                columns=features.columns
            )
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted yet")
            
            scaled_features = pd.DataFrame(
                self.scaler.transform(features),
                index=features.index,
                columns=features.columns
            )
        
        logger.info("Feature preprocessing completed")
        
        return scaled_features


if __name__ == "__main__":
    # Example usage
    from datetime import datetime, timedelta
    
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n_days = len(dates)
    
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.normal(0, 1, n_days)),
        'High': 100 + np.cumsum(np.random.normal(0, 1, n_days)) + np.random.uniform(0, 2, n_days),
        'Low': 100 + np.cumsum(np.random.normal(0, 1, n_days)) - np.random.uniform(0, 2, n_days),
        'Close': 100 + np.cumsum(np.random.normal(0, 1, n_days)),
        'Volume': np.random.uniform(1000000, 5000000, n_days)
    }, index=dates)
    
    # Initialize pipeline
    config = {
        'feature_preprocessing': {
            'outlier_handling': 'winsorize',
            'scaling': 'robust'
        }
    }
    
    pipeline = FeatureEngineeringPipeline(config)
    
    # Create features
    features = pipeline.create_all_features(sample_data)
    processed_features = pipeline.preprocess_features(features)
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Processed features shape: {processed_features.shape}")
    print(f"Feature names: {list(features.columns[:10])}...")  # Show first 10 