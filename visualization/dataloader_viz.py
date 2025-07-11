"""
Data Loader Visualization Module.

This module provides visualization functions to validate the preprocessing pipeline
and ensure data transformations are working correctly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Set style
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        pass  # Use default style if seaborn not available
sns.set_palette("husl")


class DataLoaderVisualizer:
    """
    Visualizer for data loading and preprocessing pipeline validation.
    
    Provides comprehensive visualization functions to validate:
    - Raw data loading
    - Technical indicators
    - Feature engineering
    - Data normalization
    - Sequence generation
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
    def plot_raw_data(self, data: pd.DataFrame, title: str = "Raw Market Data", 
                      save_path: Optional[str] = None) -> None:
        """
        Plot raw OHLCV data to validate data loading.
        
        Args:
            data: Raw market data DataFrame with OHLCV columns
            title: Plot title
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # Price data
        axes[0].plot(data.index, data['Open'], label='Open', alpha=0.7)
        axes[0].plot(data.index, data['High'], label='High', alpha=0.7)
        axes[0].plot(data.index, data['Low'], label='Low', alpha=0.7)
        axes[0].plot(data.index, data['Close'], label='Close', linewidth=2)
        axes[0].set_title(f'{title} - Price Data')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Volume data - ensure it's properly formatted
        volume_data = data['Volume'].values
        if volume_data.ndim > 1:
            volume_data = volume_data.flatten()
        
        # Remove any NaN values for plotting
        valid_idx = ~np.isnan(volume_data)
        plot_dates = data.index[valid_idx]
        plot_volume = volume_data[valid_idx]
        
        axes[1].bar(plot_dates, plot_volume, alpha=0.6, color='orange')
        axes[1].set_title('Volume')
        axes[1].set_ylabel('Volume')
        axes[1].set_xlabel('Date')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_technical_indicators(self, data: pd.DataFrame, indicators: Dict,
                                title: str = "Technical Indicators",
                                save_path: Optional[str] = None) -> None:
        """
        Plot technical indicators to validate their calculation.
        
        Args:
            data: Original price data
            indicators: Dictionary of calculated indicators
            title: Plot title
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(4, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5), 
                                sharex=True)
        
        # Price with moving averages
        axes[0].plot(data.index, data['Close'], label='Close Price', linewidth=2)
        if 'sma_20' in indicators:
            axes[0].plot(data.index, indicators['sma_20'], label='SMA 20', alpha=0.8)
        if 'sma_50' in indicators:
            axes[0].plot(data.index, indicators['sma_50'], label='SMA 50', alpha=0.8)
        if 'ema_12' in indicators:
            axes[0].plot(data.index, indicators['ema_12'], label='EMA 12', alpha=0.8)
        axes[0].set_title(f'{title} - Price & Moving Averages')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RSI
        if 'rsi' in indicators:
            axes[1].plot(data.index, indicators['rsi'], color='purple', linewidth=2)
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
            axes[1].set_title('RSI (Relative Strength Index)')
            axes[1].set_ylabel('RSI')
            axes[1].set_ylim(0, 100)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # MACD
        if 'macd' in indicators and 'macd_signal' in indicators:
            axes[2].plot(data.index, indicators['macd'], label='MACD', linewidth=2)
            axes[2].plot(data.index, indicators['macd_signal'], label='Signal', linewidth=2)
            if 'macd_histogram' in indicators:
                axes[2].bar(data.index, indicators['macd_histogram'], 
                           label='Histogram', alpha=0.6)
            axes[2].set_title('MACD')
            axes[2].set_ylabel('MACD')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # Bollinger Bands
        if all(col in indicators for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            axes[3].plot(data.index, data['Close'], label='Close Price', linewidth=2)
            axes[3].plot(data.index, indicators['bb_upper'], label='Upper Band', 
                        alpha=0.8, linestyle='--')
            axes[3].plot(data.index, indicators['bb_middle'], label='Middle Band', 
                        alpha=0.8)
            axes[3].plot(data.index, indicators['bb_lower'], label='Lower Band', 
                        alpha=0.8, linestyle='--')
            axes[3].fill_between(data.index, indicators['bb_upper'], 
                               indicators['bb_lower'], alpha=0.2)
            axes[3].set_title('Bollinger Bands')
            axes[3].set_ylabel('Price')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Date')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_feature_engineering(self, original_data: pd.DataFrame, 
                                engineered_features: pd.DataFrame,
                                title: str = "Feature Engineering Validation",
                                save_path: Optional[str] = None) -> None:
        """
        Plot engineered features to validate feature engineering pipeline.
        
        Args:
            original_data: Original price data
            engineered_features: Engineered features DataFrame
            title: Plot title
            save_path: Optional path to save the figure
        """
        n_features = min(len(engineered_features.columns), 8)  # Limit to 8 features
        n_rows = (n_features + 1) // 2
        
        fig, axes = plt.subplots(n_rows, 2, figsize=(self.figsize[0], n_rows * 4))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each feature
        for i, feature in enumerate(engineered_features.columns[:n_features]):
            row, col = i // 2, i % 2
            
            axes[row, col].plot(engineered_features.index, engineered_features[feature], 
                               linewidth=2, color=self.colors[i])
            axes[row, col].set_title(f'{feature}')
            axes[row, col].set_ylabel('Value')
            axes[row, col].grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = engineered_features[feature].mean()
            std_val = engineered_features[feature].std()
            axes[row, col].axhline(y=mean_val, color='red', linestyle='--', 
                                  alpha=0.7, label=f'Mean: {mean_val:.3f}')
            axes[row, col].legend()
        
        # Hide unused subplots
        for i in range(n_features, n_rows * 2):
            row, col = i // 2, i % 2
            axes[row, col].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_data_normalization(self, original_data: pd.DataFrame, 
                               normalized_data: pd.DataFrame,
                               title: str = "Data Normalization Validation",
                               save_path: Optional[str] = None) -> None:
        """
        Plot before/after normalization to validate normalization.
        
        Args:
            original_data: Original data before normalization
            normalized_data: Normalized data
            title: Plot title
            save_path: Optional path to save the figure
        """
        n_features = min(len(original_data.columns), 6)  # Limit to 6 features
        
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        axes = axes.flatten()
        
        for i, feature in enumerate(original_data.columns[:n_features]):
            # Original data
            axes[i].hist(original_data[feature].dropna(), bins=50, alpha=0.7, 
                        label='Original', color='blue')
            
            # Normalized data
            if feature in normalized_data.columns:
                axes[i].hist(normalized_data[feature].dropna(), bins=50, alpha=0.7, 
                            label='Normalized', color='red')
            
            axes[i].set_title(f'{feature}')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_sequence_generation(self, sequences: np.ndarray, targets: np.ndarray,
                                sequence_length: int = 60,
                                title: str = "Sequence Generation Validation",
                                save_path: Optional[str] = None) -> None:
        """
        Plot generated sequences to validate sequence generation.
        
        Args:
            sequences: Generated sequences [n_samples, seq_len, n_features]
            targets: Target values [n_samples, n_targets]
            sequence_length: Length of sequences
            title: Plot title
            save_path: Optional path to save the figure
        """
        # Sample a few sequences for visualization
        n_samples = min(5, len(sequences))
        sample_indices = np.random.choice(len(sequences), n_samples, replace=False)
        
        fig, axes = plt.subplots(2, 1, figsize=self.figsize)
        
        # Plot sample sequences
        for i, idx in enumerate(sample_indices):
            # Plot first feature of each sequence
            axes[0].plot(range(sequence_length), sequences[idx, :, 0], 
                        label=f'Sequence {idx}', alpha=0.8)
        
        axes[0].set_title(f'{title} - Sample Sequences (Feature 0)')
        axes[0].set_ylabel('Normalized Value')
        axes[0].set_xlabel('Time Step')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot target distribution
        axes[1].hist(targets.flatten(), bins=50, alpha=0.7, color='green')
        axes[1].set_title('Target Distribution')
        axes[1].set_ylabel('Frequency')
        axes[1].set_xlabel('Target Value')
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics
        mean_target = np.mean(targets)
        std_target = np.std(targets)
        axes[1].axvline(x=mean_target, color='red', linestyle='--', 
                       label=f'Mean: {mean_target:.3f}')
        axes[1].axvline(x=mean_target + std_target, color='orange', linestyle='--', 
                       label=f'+1 Std: {mean_target + std_target:.3f}')
        axes[1].axvline(x=mean_target - std_target, color='orange', linestyle='--', 
                       label=f'-1 Std: {mean_target - std_target:.3f}')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_correlation_matrix(self, data: pd.DataFrame,
                               title: str = "Feature Correlation Matrix",
                               save_path: Optional[str] = None) -> None:
        """
        Plot correlation matrix to validate feature relationships.
        
        Args:
            data: DataFrame with features
            title: Plot title
            save_path: Optional path to save the figure
        """
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create heatmap
        plt.figure(figsize=self.figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_data_quality_report(self, data: pd.DataFrame,
                                title: str = "Data Quality Report",
                                save_path: Optional[str] = None) -> None:
        """
        Generate comprehensive data quality report.
        
        Args:
            data: DataFrame to analyze
            title: Report title
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Missing values
        missing_counts = data.isnull().sum()
        missing_pct = (missing_counts / len(data)) * 100
        
        axes[0, 0].bar(range(len(missing_pct)), missing_pct.values)
        axes[0, 0].set_title('Missing Values (%)')
        axes[0, 0].set_ylabel('Percentage')
        axes[0, 0].set_xticks(range(len(missing_pct)))
        axes[0, 0].set_xticklabels(missing_pct.index, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Data types
        dtype_counts = data.dtypes.value_counts()
        axes[0, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Data Types Distribution')
        
        # Outliers (using IQR method)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_counts = []
        
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            outlier_counts.append(outliers)
        
        axes[1, 0].bar(range(len(outlier_counts)), outlier_counts)
        axes[1, 0].set_title('Outlier Counts (IQR Method)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xticks(range(len(numeric_cols)))
        axes[1, 0].set_xticklabels(numeric_cols, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Data distribution summary
        summary_stats = data.describe()
        axes[1, 1].text(0.1, 0.9, f"Dataset Shape: {data.shape}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.8, f"Total Missing: {data.isnull().sum().sum()}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.7, f"Numeric Columns: {len(numeric_cols)}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.6, f"Date Range: {data.index.min()} to {data.index.max()}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Dataset Summary')
        axes[1, 1].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_interactive_dashboard(self, data: pd.DataFrame, 
                                   indicators: Optional[Dict] = None,
                                   title: str = "Interactive Data Dashboard") -> go.Figure:
        """
        Create interactive dashboard using Plotly.
        
        Args:
            data: Market data DataFrame
            indicators: Optional dictionary of technical indicators
            title: Dashboard title
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Price & Volume', 'Technical Indicators', 
                          'Returns Distribution', 'Volatility'),
            vertical_spacing=0.08,
            specs=[[{"secondary_y": True}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Price data
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                opacity=0.6
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Technical indicators
        if indicators:
            if 'rsi' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['rsi'],
                        name='RSI',
                        line=dict(color='purple')
                    ),
                    row=2, col=1
                )
        
        # Returns distribution
        returns = data['Close'].pct_change().dropna()
        fig.add_trace(
            go.Histogram(
                x=returns,
                name='Returns Distribution',
                nbinsx=50
            ),
            row=3, col=1
        )
        
        # Rolling volatility
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(
                x=data.index[20:],
                y=rolling_vol[20:],
                name='20-Day Volatility',
                line=dict(color='red')
            ),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=1200,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig


# Convenience functions for quick visualization
def quick_data_validation(data: pd.DataFrame, indicators: Optional[Dict] = None,
                         save_dir: Optional[str] = None) -> None:
    """
    Quick validation of data loading and preprocessing.
    
    Args:
        data: Market data DataFrame
        indicators: Optional technical indicators
        save_dir: Optional directory to save plots
    """
    viz = DataLoaderVisualizer()
    
    # Raw data plot
    save_path = f"{save_dir}/raw_data.png" if save_dir else None
    viz.plot_raw_data(data, save_path=save_path)
    
    # Technical indicators
    if indicators:
        save_path = f"{save_dir}/technical_indicators.png" if save_dir else None
        viz.plot_technical_indicators(data, indicators, save_path=save_path)
    
    # Data quality report
    save_path = f"{save_dir}/data_quality.png" if save_dir else None
    viz.plot_data_quality_report(data, save_path=save_path)


def validate_preprocessing_pipeline(original_data: pd.DataFrame,
                                  processed_data: pd.DataFrame,
                                  sequences: np.ndarray,
                                  targets: np.ndarray,
                                  save_dir: Optional[str] = None) -> None:
    """
    Comprehensive validation of the entire preprocessing pipeline.
    
    Args:
        original_data: Original market data
        processed_data: Processed features
        sequences: Generated sequences
        targets: Target values
        save_dir: Optional directory to save plots
    """
    viz = DataLoaderVisualizer()
    
    # Feature engineering validation
    save_path = f"{save_dir}/feature_engineering.png" if save_dir else None
    viz.plot_feature_engineering(original_data, processed_data, save_path=save_path)
    
    # Normalization validation
    save_path = f"{save_dir}/normalization.png" if save_dir else None
    viz.plot_data_normalization(original_data, processed_data, save_path=save_path)
    
    # Sequence generation validation
    save_path = f"{save_dir}/sequences.png" if save_dir else None
    viz.plot_sequence_generation(sequences, targets, save_path=save_path)
    
    # Correlation matrix
    save_path = f"{save_dir}/correlation_matrix.png" if save_dir else None
    viz.plot_correlation_matrix(processed_data, save_path=save_path) 