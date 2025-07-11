# Data Loader Visualization Guide

## Overview

The data loader visualization system provides comprehensive visual validation of the preprocessing pipeline to ensure data transformations are working correctly. This system generates multiple types of plots to validate each step of the data processing workflow.

## Files Created

### 1. `visualization/dataloader_viz.py`
Main visualization module containing:
- `DataLoaderVisualizer` class with comprehensive plotting methods
- Convenience functions for quick validation
- Interactive dashboard generation capabilities

### 2. `test_dataloader_viz.py`
Standalone test script that runs the complete visualization pipeline

### 3. Updated `preprocessing/data_loader.py`
Enhanced main loop with integrated visualization testing

## Visualization Types

### 1. **Raw Data Validation** (`plot_raw_data`)
- **Purpose**: Validate data loading from external sources
- **Plot Type**: Line plots for OHLC data + bar chart for volume
- **What to Look For**:
  - Continuous price series without gaps
  - Reasonable price ranges
  - Volume spikes correlating with price movements
  - No obvious data quality issues

### 2. **Technical Indicators** (`plot_technical_indicators`)
- **Purpose**: Validate technical indicator calculations
- **Plot Type**: Multi-panel plot with:
  - Price + Moving Averages (SMA, EMA)
  - RSI with overbought/oversold levels
  - MACD with signal line and histogram
  - Bollinger Bands with price envelope
- **What to Look For**:
  - Moving averages following price trends
  - RSI oscillating between 0-100
  - MACD crossovers at trend changes
  - Bollinger Bands expanding/contracting with volatility

### 3. **Feature Engineering** (`plot_feature_engineering`)
- **Purpose**: Validate engineered feature calculations
- **Plot Type**: Grid of time series plots for each feature
- **What to Look For**:
  - Features showing reasonable ranges
  - No constant or flat-line features
  - Features responding to market conditions
  - Statistical summaries (mean, std) displayed

### 4. **Data Normalization** (`plot_data_normalization`)
- **Purpose**: Validate normalization/scaling transformations
- **Plot Type**: Histogram comparisons (before/after normalization)
- **What to Look For**:
  - Normalized data centered around 0 (for standard scaling)
  - Normalized data in [0,1] range (for min-max scaling)
  - Distribution shapes preserved after normalization
  - No extreme outliers after scaling

### 5. **Sequence Generation** (`plot_sequence_generation`)
- **Purpose**: Validate time series sequence creation
- **Plot Type**: Sample sequence plots + target distribution
- **What to Look For**:
  - Sequences showing temporal patterns
  - Target distribution looking reasonable
  - No obvious sequence boundary issues
  - Proper sequence length

### 6. **Correlation Matrix** (`plot_correlation_matrix`)
- **Purpose**: Validate feature relationships
- **Plot Type**: Heatmap of feature correlations
- **What to Look For**:
  - Expected correlations (e.g., price features)
  - No perfect correlations (multicollinearity)
  - Reasonable correlation patterns
  - Features providing diverse information

### 7. **Data Quality Report** (`plot_data_quality_report`)
- **Purpose**: Comprehensive data quality assessment
- **Plot Type**: Multi-panel quality dashboard
- **What to Look For**:
  - Low missing value percentages
  - Reasonable outlier counts
  - Proper data type distribution
  - Complete dataset summary

### 8. **Interactive Dashboard** (`create_interactive_dashboard`)
- **Purpose**: Exploratory data analysis
- **Plot Type**: Interactive Plotly dashboard
- **Features**:
  - Candlestick charts with volume
  - Technical indicators overlay
  - Returns distribution
  - Volatility analysis

## Usage

### Method 1: Run the Test Script
```bash
python test_dataloader_viz.py
```

### Method 2: Run Data Loader Directly
```bash
python preprocessing/data_loader.py
```

### Method 3: Integrate into Your Code
```python
from visualization.dataloader_viz import DataLoaderVisualizer

# Initialize visualizer
viz = DataLoaderVisualizer()

# Plot raw data
viz.plot_raw_data(data, title="My Data", save_path="output.png")

# Quick validation
quick_data_validation(data, indicators, save_dir="validation_plots")
```

## Output Structure

Running the visualization system creates the following directory structure:

```
visualization_output/
├── raw_data.png                    # Raw OHLCV data
├── technical_indicators.png        # Technical indicators
├── feature_engineering.png         # Engineered features
├── normalization.png               # Normalization validation
├── sequences.png                   # Sequence generation
├── correlation_matrix.png          # Feature correlations
├── data_quality.png               # Data quality report
├── interactive_dashboard.html      # Interactive dashboard
├── quick_validation/               # Quick validation plots
│   ├── raw_data.png
│   ├── technical_indicators.png
│   └── data_quality.png
└── pipeline_validation/            # Comprehensive validation
    ├── feature_engineering.png
    ├── normalization.png
    ├── sequences.png
    └── correlation_matrix.png
```

## Validation Checklist

Use this checklist when reviewing the generated plots:

### ✅ Raw Data Validation
- [ ] Price data shows continuous time series
- [ ] Volume data has reasonable spikes
- [ ] No obvious data gaps or anomalies
- [ ] Date range covers expected period

### ✅ Technical Indicators
- [ ] Moving averages smooth price movements
- [ ] RSI oscillates properly (0-100 range)
- [ ] MACD shows trend changes
- [ ] Bollinger Bands envelope price action

### ✅ Feature Engineering
- [ ] All features show variation over time
- [ ] No constant or near-constant features
- [ ] Features respond to market conditions
- [ ] Statistical summaries are reasonable

### ✅ Data Normalization
- [ ] Normalized data has expected distribution
- [ ] No extreme outliers after scaling
- [ ] Original data patterns preserved
- [ ] Scaling method applied correctly

### ✅ Sequence Generation
- [ ] Sequences show temporal structure
- [ ] Target distribution looks reasonable
- [ ] Sequence length matches configuration
- [ ] No boundary artifacts

### ✅ Data Quality
- [ ] Missing values < 5%
- [ ] Outlier counts reasonable
- [ ] Data types correct
- [ ] Dataset size sufficient

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed: `pip install matplotlib seaborn plotly`
   - Check Python path includes project root

2. **Empty Plots**
   - Verify data is loaded correctly
   - Check date ranges are valid
   - Ensure ticker symbol exists

3. **Style Warnings**
   - Matplotlib/seaborn version compatibility
   - Code handles missing style gracefully

4. **Memory Issues**
   - Large datasets may require sampling
   - Reduce plot complexity for big data

### Configuration

The visualization system uses the same configuration as the data loader:
- `config/config.yaml` for model parameters
- Sequence length, feature settings, etc.

## Customization

### Adding New Visualizations

1. Add method to `DataLoaderVisualizer` class
2. Follow naming convention: `plot_<feature_name>`
3. Include save_path parameter for file output
4. Add error handling and logging

### Modifying Existing Plots

1. Edit methods in `DataLoaderVisualizer`
2. Adjust plot parameters (colors, sizes, layouts)
3. Add/remove subplots as needed
4. Update documentation

## Best Practices

1. **Always run visualizations** before training models
2. **Save plots** for documentation and debugging
3. **Review all validation types** - each serves a purpose
4. **Compare plots** across different datasets/periods
5. **Use interactive dashboard** for detailed exploration

## Integration with Training Pipeline

The visualization system integrates seamlessly with the training pipeline:

1. **Data Loading**: Validate raw data quality
2. **Feature Engineering**: Confirm feature calculations
3. **Preprocessing**: Verify normalization/scaling
4. **Sequence Generation**: Check model inputs
5. **Training**: Use validated data with confidence

This comprehensive validation ensures your preprocessing pipeline is working correctly before expensive model training begins. 