"""
Graph visualization for trade execution and computational graphs.

This module creates:
1. GraphViz trade-execution graph (nodes: data → model → decision → risk-guard → venue)
2. TensorFlow computational graph dumping for debugging
3. Network analysis of trading dependencies
4. Visual performance analysis graphs
"""

import networkx as nx
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import os
from datetime import datetime
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
import pydot
from tensorflow.python.platform import gfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TradeExecutionGraph:
    """
    Creates GraphViz visualizations of the trade execution pipeline.
    
    Shows the flow: Data → Model → Decision → Risk Guard → Venue
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trade execution graph.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.graph = graphviz.Digraph(comment='Trade Execution Pipeline')
        self.graph.attr(rankdir='LR', size='12,8')
        self.graph.attr('node', shape='box', style='filled', fontname='Arial')
        
    def create_execution_graph(self) -> graphviz.Digraph:
        """
        Create the main trade execution graph.
        
        Returns:
            GraphViz Digraph object
        """
        # Data Sources
        self.graph.node('market_data', 'Market Data\\n(Price, Volume)', fillcolor='lightblue')
        self.graph.node('features', 'Feature\\nEngineering', fillcolor='lightblue')
        self.graph.node('indicators', 'Technical\\nIndicators', fillcolor='lightblue')
        
        # Model Components
        self.graph.node('model_input', 'Model Input\\n(Sequences)', fillcolor='lightgreen')
        self.graph.node('lstm_layers', 'LSTM/SigFormer\\nLayers', fillcolor='lightgreen')
        self.graph.node('attention', 'Attention\\nMechanism', fillcolor='lightgreen')
        self.graph.node('output_layer', 'Delta Hedge\\nOutput', fillcolor='lightgreen')
        
        # Decision Making
        self.graph.node('position_calc', 'Position\\nCalculation', fillcolor='lightyellow')
        self.graph.node('signal_gen', 'Signal\\nGeneration', fillcolor='lightyellow')
        
        # Risk Management
        self.graph.node('risk_check', 'Risk Guard\\n(Hard/Soft Limits)', fillcolor='lightcoral')
        self.graph.node('portfolio_risk', 'Portfolio\\nRisk Assessment', fillcolor='lightcoral')
        
        # Execution
        self.graph.node('order_sizing', 'Order\\nSizing', fillcolor='lightgray')
        self.graph.node('venue_routing', 'Venue\\nRouting', fillcolor='lightgray')
        self.graph.node('execution', 'Trade\\nExecution', fillcolor='lightgray')
        
        # Feedback Loop
        self.graph.node('fill_data', 'Fill Data\\n(Kafka/Parquet)', fillcolor='lightsalmon')
        self.graph.node('performance', 'Performance\\nTracking', fillcolor='lightsalmon')
        
        # Add edges (data flow)
        self._add_data_flow_edges()
        
        return self.graph
    
    def _add_data_flow_edges(self) -> None:
        """Add edges representing data flow."""
        # Data preprocessing flow
        self.graph.edge('market_data', 'features')
        self.graph.edge('market_data', 'indicators')
        self.graph.edge('features', 'model_input')
        self.graph.edge('indicators', 'model_input')
        
        # Model flow
        self.graph.edge('model_input', 'lstm_layers')
        self.graph.edge('lstm_layers', 'attention')
        self.graph.edge('attention', 'output_layer')
        
        # Decision flow
        self.graph.edge('output_layer', 'position_calc')
        self.graph.edge('position_calc', 'signal_gen')
        
        # Risk management flow
        self.graph.edge('signal_gen', 'risk_check')
        self.graph.edge('signal_gen', 'portfolio_risk')
        self.graph.edge('portfolio_risk', 'risk_check')
        
        # Execution flow
        self.graph.edge('risk_check', 'order_sizing')
        self.graph.edge('order_sizing', 'venue_routing')
        self.graph.edge('venue_routing', 'execution')
        
        # Feedback loop
        self.graph.edge('execution', 'fill_data')
        self.graph.edge('fill_data', 'performance')
        self.graph.edge('performance', 'portfolio_risk', style='dashed', color='red')
    
    def add_latency_metrics(self, latencies: Dict[str, float]) -> None:
        """
        Add latency metrics to nodes.
        
        Args:
            latencies: Dictionary mapping node names to latency values (ms)
        """
        for node, latency in latencies.items():
            if node in ['model_input', 'lstm_layers', 'attention', 'output_layer']:
                color = 'red' if latency > 10 else 'orange' if latency > 5 else 'green'
                self.graph.node(node, f'{node}\\n({latency:.1f}ms)', fillcolor=color)
    
    def save_graph(self, filepath: str, format: str = 'png') -> None:
        """
        Save graph to file.
        
        Args:
            filepath: Output file path
            format: Output format ('png', 'pdf', 'svg')
        """
        try:
            self.graph.render(filepath, format=format, cleanup=True)
            logger.info(f"Trade execution graph saved to {filepath}.{format}")
        except Exception as e:
            logger.error(f"Error saving graph: {e}")


class ComputationalGraphAnalyzer:
    """
    Analyzes and visualizes TensorFlow computational graphs.
    """
    
    def __init__(self, model: tf.keras.Model):
        """
        Initialize with TensorFlow model.
        
        Args:
            model: TensorFlow/Keras model
        """
        self.model = model
        
    def export_graph_def(self, filepath: str) -> None:
        """
        Export model graph definition.
        
        Args:
            filepath: Output file path for GraphDef
        """
        try:
            # Convert to concrete function
            concrete_func = tf.function(lambda x: self.model(x))
            concrete_func = concrete_func.get_concrete_function(
                tf.TensorSpec(shape=self.model.input_shape, dtype=tf.float32)
            )
            
            # Get graph def
            graph_def = concrete_func.graph.as_graph_def()
            
            # Save to file
            with gfile.GFile(filepath, 'wb') as f:
                f.write(graph_def.SerializeToString())
            
            logger.info(f"Graph definition saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting graph definition: {e}")
    
    def visualize_model_architecture(self, filepath: str) -> None:
        """
        Visualize model architecture using Keras plot_model.
        
        Args:
            filepath: Output file path
        """
        try:
            tf.keras.utils.plot_model(
                self.model,
                to_file=filepath,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=True,
                dpi=150
            )
            logger.info(f"Model architecture saved to {filepath}")
        except Exception as e:
            logger.error(f"Error visualizing model architecture: {e}")
    
    def analyze_layer_complexity(self) -> pd.DataFrame:
        """
        Analyze computational complexity of model layers.
        
        Returns:
            DataFrame with layer complexity metrics
        """
        layer_info = []
        
        for i, layer in enumerate(self.model.layers):
            info = {
                'layer_index': i,
                'layer_name': layer.name,
                'layer_type': type(layer).__name__,
                'trainable_params': layer.count_params(),
                'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A',
                'input_shape': str(layer.input_shape) if hasattr(layer, 'input_shape') else 'N/A'
            }
            
            # Add layer-specific metrics
            if hasattr(layer, 'units'):
                info['units'] = layer.units
            if hasattr(layer, 'activation'):
                info['activation'] = str(layer.activation)
            if hasattr(layer, 'dropout'):
                info['dropout'] = layer.dropout
            
            layer_info.append(info)
        
        return pd.DataFrame(layer_info)


class NetworkAnalyzer:
    """
    Network analysis of trading system dependencies and relationships.
    """
    
    def __init__(self):
        """Initialize network analyzer."""
        self.G = nx.DiGraph()
        
    def create_dependency_graph(self, components: Dict[str, List[str]]) -> nx.DiGraph:
        """
        Create dependency graph of system components.
        
        Args:
            components: Dictionary mapping component names to their dependencies
            
        Returns:
            NetworkX directed graph
        """
        for component, dependencies in components.items():
            self.G.add_node(component)
            for dep in dependencies:
                self.G.add_edge(dep, component)
        
        return self.G
    
    def analyze_centrality(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze network centrality metrics.
        
        Returns:
            Dictionary of centrality metrics
        """
        metrics = {
            'degree_centrality': nx.degree_centrality(self.G),
            'betweenness_centrality': nx.betweenness_centrality(self.G),
            'closeness_centrality': nx.closeness_centrality(self.G),
            'eigenvector_centrality': nx.eigenvector_centrality(self.G)
        }
        
        return metrics
    
    def find_critical_path(self) -> List[str]:
        """
        Find critical path in the dependency graph.
        
        Returns:
            List of nodes in the critical path
        """
        try:
            # Find longest path (critical path)
            return nx.dag_longest_path(self.G)
        except nx.NetworkXError:
            logger.warning("Graph contains cycles - cannot find critical path")
            return []
    
    def visualize_network(self, filepath: str, layout: str = 'spring') -> None:
        """
        Visualize network graph.
        
        Args:
            filepath: Output file path
            layout: Layout algorithm ('spring', 'circular', 'hierarchical')
        """
        plt.figure(figsize=(12, 8))
        
        if layout == 'spring':
            pos = nx.spring_layout(self.G, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.G)
        elif layout == 'hierarchical':
            pos = nx.nx_agraph.graphviz_layout(self.G, prog='dot')
        else:
            pos = nx.spring_layout(self.G)
        
        # Draw network
        nx.draw(self.G, pos, with_labels=True, node_color='lightblue',
                node_size=3000, font_size=10, font_weight='bold',
                arrows=True, arrowsize=20, edge_color='gray')
        
        plt.title("System Dependency Network", fontsize=16)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Network visualization saved to {filepath}")


class PerformanceVisualizer:
    """
    Creates various performance visualization charts.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize performance visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
    def plot_equity_curve(self, returns: np.ndarray, benchmark_returns: np.ndarray = None,
                         dates: pd.DatetimeIndex = None, filepath: str = None) -> go.Figure:
        """
        Plot interactive equity curve.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns (optional)
            dates: Date index
            filepath: Output file path (optional)
            
        Returns:
            Plotly figure object
        """
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + returns) - 1
        
        if dates is None:
            dates = pd.date_range(start='2020-01-01', periods=len(returns), freq='D')
        
        fig = go.Figure()
        
        # Strategy equity curve
        fig.add_trace(go.Scatter(
            x=dates,
            y=cum_returns,
            mode='lines',
            name='Strategy',
            line=dict(color='blue', width=2)
        ))
        
        # Benchmark equity curve
        if benchmark_returns is not None:
            bench_cum_returns = np.cumprod(1 + benchmark_returns) - 1
            fig.add_trace(go.Scatter(
                x=dates,
                y=bench_cum_returns,
                mode='lines',
                name='Benchmark',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            hovermode='x unified',
            template='plotly_white'
        )
        
        if filepath:
            fig.write_html(filepath)
            logger.info(f"Equity curve saved to {filepath}")
        
        return fig
    
    def plot_performance_distribution(self, returns: np.ndarray, filepath: str = None) -> go.Figure:
        """
        Plot return distribution analysis.
        
        Args:
            returns: Strategy returns
            filepath: Output file path (optional)
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Return Distribution', 'QQ Plot', 'Rolling Volatility', 'Drawdown'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Return distribution
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns',
            opacity=0.7
        ), row=1, col=1)
        
        # QQ plot
        from scipy import stats
        (osm, osr), (slope, intercept, r) = stats.probplot(returns, dist="norm", plot=None)
        fig.add_trace(go.Scatter(
            x=osm,
            y=osr,
            mode='markers',
            name='QQ Plot',
            marker=dict(color='red', size=4)
        ), row=1, col=2)
        
        # Rolling volatility
        rolling_vol = pd.Series(returns).rolling(window=30).std() * np.sqrt(252)
        dates = pd.date_range(start='2020-01-01', periods=len(returns), freq='D')
        fig.add_trace(go.Scatter(
            x=dates,
            y=rolling_vol,
            mode='lines',
            name='30D Vol',
            line=dict(color='orange')
        ), row=2, col=1)
        
        # Drawdown
        cum_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / peak
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdown * 100,
            mode='lines',
            name='Drawdown',
            fill='tonexty',
            line=dict(color='red')
        ), row=2, col=2)
        
        fig.update_layout(
            title='Performance Analysis',
            template='plotly_white',
            height=600
        )
        
        if filepath:
            fig.write_html(filepath)
            logger.info(f"Performance distribution saved to {filepath}")
        
        return fig
    
    def plot_hyperparameter_importance(self, study_results: Dict, filepath: str = None) -> go.Figure:
        """
        Plot hyperparameter importance from optimization study.
        
        Args:
            study_results: Optuna study results
            filepath: Output file path (optional)
            
        Returns:
            Plotly figure object
        """
        if 'study' not in study_results:
            logger.warning("No study object found in results")
            return None
        
        study = study_results['study']
        
        try:
            importance = study.get_param_importances()
            
            fig = go.Figure([go.Bar(
                x=list(importance.values()),
                y=list(importance.keys()),
                orientation='h'
            )])
            
            fig.update_layout(
                title='Hyperparameter Importance',
                xaxis_title='Importance',
                yaxis_title='Parameter',
                template='plotly_white'
            )
            
            if filepath:
                fig.write_html(filepath)
                logger.info(f"Hyperparameter importance saved to {filepath}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting hyperparameter importance: {e}")
            return None
    
    def create_risk_dashboard(self, returns: np.ndarray, positions: np.ndarray = None,
                            filepath: str = None) -> go.Figure:
        """
        Create comprehensive risk dashboard.
        
        Args:
            returns: Strategy returns
            positions: Position data (optional)
            filepath: Output file path (optional)
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Value at Risk', 'Expected Shortfall', 'Return Autocorrelation',
                          'Volatility Clustering', 'Position Exposure', 'Risk Metrics'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # VaR calculation
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Expected Shortfall
        es_95 = returns[returns <= var_95].mean()
        es_99 = returns[returns <= var_99].mean()
        
        # VaR histogram
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns',
            opacity=0.7
        ), row=1, col=1)
        
        fig.add_vline(x=var_95, line_dash="dash", line_color="red", 
                     annotation_text="VaR 95%", row=1, col=1)
        fig.add_vline(x=var_99, line_dash="dash", line_color="darkred",
                     annotation_text="VaR 99%", row=1, col=1)
        
        # Expected Shortfall
        risk_metrics = pd.DataFrame({
            'Metric': ['VaR 95%', 'VaR 99%', 'ES 95%', 'ES 99%'],
            'Value': [var_95, var_99, es_95, es_99]
        })
        
        fig.add_trace(go.Bar(
            x=risk_metrics['Metric'],
            y=risk_metrics['Value'],
            name='Risk Metrics'
        ), row=1, col=2)
        
        # Autocorrelation
        from statsmodels.tsa.stattools import acf
        autocorr = acf(returns, nlags=20, fft=True)
        lags = range(len(autocorr))
        
        fig.add_trace(go.Bar(
            x=lags,
            y=autocorr,
            name='Autocorrelation'
        ), row=2, col=1)
        
        # Volatility clustering
        squared_returns = returns ** 2
        vol_autocorr = acf(squared_returns, nlags=20, fft=True)
        
        fig.add_trace(go.Bar(
            x=lags,
            y=vol_autocorr,
            name='Vol Clustering'
        ), row=2, col=2)
        
        # Position exposure (if available)
        if positions is not None:
            dates = pd.date_range(start='2020-01-01', periods=len(positions), freq='D')
            fig.add_trace(go.Scatter(
                x=dates,
                y=positions,
                mode='lines',
                name='Position'
            ), row=3, col=1)
        
        # Risk metrics table
        from hyperopt import FinancialMetrics
        sharpe = FinancialMetrics.sharpe_ratio(returns)
        sortino = FinancialMetrics.sortino_ratio(returns)
        max_dd = FinancialMetrics.max_drawdown(returns)
        
        metrics_text = f"""
        Sharpe Ratio: {sharpe:.3f}<br>
        Sortino Ratio: {sortino:.3f}<br>
        Max Drawdown: {max_dd:.3f}<br>
        VaR 95%: {var_95:.3f}<br>
        Expected Shortfall 95%: {es_95:.3f}
        """
        
        fig.add_annotation(
            text=metrics_text,
            xref="x domain", yref="y domain",
            x=0.1, y=0.9, showarrow=False,
            row=3, col=2
        )
        
        fig.update_layout(
            title='Risk Dashboard',
            template='plotly_white',
            height=800
        )
        
        if filepath:
            fig.write_html(filepath)
            logger.info(f"Risk dashboard saved to {filepath}")
        
        return fig


def create_comprehensive_report(model: tf.keras.Model, results: Dict, 
                              returns: np.ndarray, config: Dict,
                              output_dir: str = 'reports') -> None:
    """
    Create comprehensive visualization report.
    
    Args:
        model: Trained model
        results: Training/optimization results
        returns: Strategy returns
        config: Configuration dictionary
        output_dir: Output directory for reports
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Trade execution graph
    execution_graph = TradeExecutionGraph(config)
    graph = execution_graph.create_execution_graph()
    execution_graph.save_graph(os.path.join(output_dir, f'execution_graph_{timestamp}'))
    
    # 2. Model architecture
    model_analyzer = ComputationalGraphAnalyzer(model)
    model_analyzer.visualize_model_architecture(
        os.path.join(output_dir, f'model_architecture_{timestamp}.png')
    )
    
    # 3. Performance visualizations
    perf_viz = PerformanceVisualizer(config)
    
    # Equity curve
    equity_fig = perf_viz.plot_equity_curve(
        returns,
        filepath=os.path.join(output_dir, f'equity_curve_{timestamp}.html')
    )
    
    # Performance distribution
    dist_fig = perf_viz.plot_performance_distribution(
        returns,
        filepath=os.path.join(output_dir, f'performance_dist_{timestamp}.html')
    )
    
    # Risk dashboard
    risk_fig = perf_viz.create_risk_dashboard(
        returns,
        filepath=os.path.join(output_dir, f'risk_dashboard_{timestamp}.html')
    )
    
    # 4. Hyperparameter importance (if available)
    if 'study' in results:
        importance_fig = perf_viz.plot_hyperparameter_importance(
            results,
            filepath=os.path.join(output_dir, f'hyperopt_importance_{timestamp}.html')
        )
    
    # 5. Network analysis
    network_analyzer = NetworkAnalyzer()
    
    # Define system components and dependencies
    components = {
        'market_data': [],
        'feature_engineering': ['market_data'],
        'model_inference': ['feature_engineering'],
        'signal_generation': ['model_inference'],
        'risk_management': ['signal_generation'],
        'order_execution': ['risk_management'],
        'performance_tracking': ['order_execution']
    }
    
    dependency_graph = network_analyzer.create_dependency_graph(components)
    network_analyzer.visualize_network(
        os.path.join(output_dir, f'dependency_network_{timestamp}.png')
    )
    
    logger.info(f"Comprehensive report created in {output_dir}")


if __name__ == "__main__":
    # Example usage
    from model import create_model, load_config
    
    # Load configuration
    config = load_config()
    
    # Create dummy data for demonstration
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)  # Daily returns
    
    # Create sample model
    model_instance = create_model(config)
    model = model_instance.build_model()
    
    # Create comprehensive report
    results = {'study': None}  # Placeholder for actual study results
    create_comprehensive_report(model, results, returns, config) 