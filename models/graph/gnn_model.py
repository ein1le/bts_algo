"""
Graph Neural Network Model for Time Series Visibility Graphs.

This module implements a GNN architecture that operates on visibility graphs
derived from financial time series data, combining graph structure with
temporal features for enhanced prediction capability.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging

from models.base_model import BaseModel
from models.graph.visibility_utils import (
    VisibilityGraphConverter, 
    GraphFeatureExtractor,
    convert_multivariate_to_graphs,
    DivideConquerVisibilityGraph
)

logger = logging.getLogger(__name__)


class GraphConvolutionLayer(layers.Layer):
    """
    Custom Graph Convolution Layer for visibility graphs.
    
    Implements message passing on the visibility graph structure
    while preserving temporal ordering information.
    """
    
    def __init__(self, units: int, activation: str = 'relu', 
                 use_bias: bool = True, **kwargs):
        """
        Initialize graph convolution layer.
        
        Args:
            units: Number of output features
            activation: Activation function
            use_bias: Whether to use bias
        """
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        
    def build(self, input_shape):
        """Build layer parameters."""
        # Input shape: [batch_size, num_nodes, node_features]
        node_features = input_shape[-1]
        
        # Weight matrices for self and neighbor transformations
        self.W_self = self.add_weight(
            name='W_self',
            shape=(node_features, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.W_neighbor = self.add_weight(
            name='W_neighbor', 
            shape=(node_features, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True
            )
        
        super().build(input_shape)
    
    def call(self, inputs, adjacency_matrix):
        """
        Forward pass through graph convolution.
        
        Args:
            inputs: Node features [batch_size, num_nodes, node_features]
            adjacency_matrix: Adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            Updated node features [batch_size, num_nodes, units]
        """
        # Self transformation
        self_features = tf.matmul(inputs, self.W_self)
        
        # Neighbor aggregation
        neighbor_features = tf.matmul(inputs, self.W_neighbor)
        aggregated = tf.matmul(adjacency_matrix, neighbor_features)
        
        # Combine self and neighbor information
        output = self_features + aggregated
        
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        
        return self.activation(output)


class GraphAttentionLayer(layers.Layer):
    """
    Graph Attention Layer for visibility graphs.
    
    Applies attention mechanism to weight the importance of different
    nodes in the visibility graph based on their structural properties.
    """
    
    def __init__(self, units: int, num_heads: int = 8, dropout_rate: float = 0.1, **kwargs):
        """
        Initialize graph attention layer.
        
        Args:
            units: Number of output features per head
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
        """
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.d_model = units * num_heads
        
    def build(self, input_shape):
        """Build layer parameters."""
        node_features = input_shape[-1]
        
        # Attention weights
        self.W_q = self.add_weight(
            name='W_q',
            shape=(node_features, self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.W_k = self.add_weight(
            name='W_k',
            shape=(node_features, self.d_model), 
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.W_v = self.add_weight(
            name='W_v',
            shape=(node_features, self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.W_out = self.add_weight(
            name='W_out',
            shape=(self.d_model, self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.dropout = layers.Dropout(self.dropout_rate)
        
        super().build(input_shape)
    
    def call(self, inputs, adjacency_matrix, training=None):
        """
        Forward pass through graph attention.
        
        Args:
            inputs: Node features [batch_size, num_nodes, node_features]
            adjacency_matrix: Adjacency matrix [batch_size, num_nodes, num_nodes]
            training: Whether in training mode
            
        Returns:
            Attended node features [batch_size, num_nodes, d_model]
        """
        batch_size = tf.shape(inputs)[0]
        num_nodes = tf.shape(inputs)[1]
        
        # Compute queries, keys, values
        Q = tf.matmul(inputs, self.W_q)
        K = tf.matmul(inputs, self.W_k)
        V = tf.matmul(inputs, self.W_v)
        
        # Reshape for multi-head attention
        Q = tf.reshape(Q, (batch_size, num_nodes, self.num_heads, self.units))
        K = tf.reshape(K, (batch_size, num_nodes, self.num_heads, self.units))
        V = tf.reshape(V, (batch_size, num_nodes, self.num_heads, self.units))
        
        # Transpose for attention computation
        Q = tf.transpose(Q, [0, 2, 1, 3])  # [batch, heads, nodes, units]
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.transpose(V, [0, 2, 1, 3])
        
        # Scaled dot-product attention
        attention_scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(self.units))
        
        # Apply adjacency mask (only attend to connected nodes)
        # Expand adjacency matrix for multi-head attention
        adjacency_mask = tf.expand_dims(adjacency_matrix, axis=1)  # [batch, 1, nodes, nodes]
        adjacency_mask = tf.tile(adjacency_mask, [1, self.num_heads, 1, 1])
        
        # Apply mask (set attention to very negative value for non-connected nodes)
        masked_scores = tf.where(
            adjacency_mask > 0,
            attention_scores,
            tf.ones_like(attention_scores) * -1e9
        )
        
        # Softmax to get attention weights
        attention_weights = tf.nn.softmax(masked_scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Apply attention to values
        attended = tf.matmul(attention_weights, V)  # [batch, heads, nodes, units]
        
        # Transpose and reshape to combine heads
        attended = tf.transpose(attended, [0, 2, 1, 3])  # [batch, nodes, heads, units]
        attended = tf.reshape(attended, (batch_size, num_nodes, self.d_model))
        
        # Final linear transformation
        output = tf.matmul(attended, self.W_out)
        
        return output


class VisibilityGraphGNN(BaseModel):
    """
    Graph Neural Network model for visibility graphs from time series.
    
    This model:
    1. Converts time series sequences to visibility graphs
    2. Applies graph convolution/attention layers
    3. Aggregates graph-level features
    4. Predicts hedge ratios for financial trading
    """
    
    def __init__(self, config: Dict):
        """
        Initialize GNN model.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.model_config = config['model']['gnn']
        self.graph_converter = VisibilityGraphConverter(
            weighted=self.model_config.get('weighted_graphs', True)
        )
        self.feature_extractor = GraphFeatureExtractor()
        
    def build_model(self) -> Model:
        """Build the GNN model architecture."""
        
        # Input layers
        # Time series input for graph construction
        ts_input = layers.Input(
            shape=(
                self.config['model']['sequence_length'],
                self.config['model']['input_features']
            ),
            name='time_series_input'
        )
        
        # Pre-computed graph inputs (alternative approach)
        node_features_input = layers.Input(
            shape=(None, self.model_config.get('node_feature_dim', 8)),
            name='node_features'
        )
        
        adjacency_input = layers.Input(
            shape=(None, None),
            name='adjacency_matrix'
        )
        
        # Method 1: Direct time series processing
        ts_features = self._process_time_series_to_graphs(ts_input)
        
        # Method 2: Pre-computed graph processing  
        graph_features = self._process_precomputed_graphs(
            node_features_input, adjacency_input
        )
        
        # Combine approaches
        combined_features = layers.Concatenate(name='combine_features')([
            ts_features, graph_features
        ])
        
        # Final prediction layers
        x = layers.Dense(
            units=self.model_config.get('dense_units', 128),
            activation='relu',
            name='dense_1'
        )(combined_features)
        
        x = layers.Dropout(
            rate=self.model_config.get('dropout_rate', 0.2),
            name='dropout_1'
        )(x)
        
        x = layers.Dense(
            units=self.model_config.get('dense_units', 128) // 2,
            activation='relu',
            name='dense_2'
        )(x)
        
        x = layers.Dropout(
            rate=self.model_config.get('dropout_rate', 0.2),
            name='dropout_2'
        )(x)
        
        # Output layer for hedge ratios
        outputs = layers.Dense(
            units=self.config['model']['output_size'],
            activation='tanh',  # Constrain to [-1, 1]
            name='hedge_output'
        )(x)
        
        # Create model with multiple inputs
        model = Model(
            inputs=[ts_input, node_features_input, adjacency_input],
            outputs=outputs,
            name='VisibilityGraphGNN'
        )
        
        self.model = model
        logger.info(f"Built GNN model with {model.count_params()} parameters")
        
        return model
    
    def _process_time_series_to_graphs(self, ts_input):
        """Process time series input by converting to graphs on-the-fly."""
        
        # Custom layer to convert time series to graph features
        @tf.function
        def ts_to_graph_features(ts_batch):
            """Convert batch of time series to graph features."""
            batch_size = tf.shape(ts_batch)[0]
            seq_len = tf.shape(ts_batch)[1]
            
            # For each sequence in batch, we'll create simplified graph features
            # This is a simplified approach - in practice, you might precompute graphs
            
            # Extract basic temporal features that capture graph-like properties
            # Rolling statistics that capture local graph structure
            
            # Moving averages (capture local trends)
            ma_short = tf.nn.avg_pool1d(
                tf.expand_dims(ts_batch, -1), 
                ksize=5, strides=1, padding='SAME'
            )
            ma_long = tf.nn.avg_pool1d(
                tf.expand_dims(ts_batch, -1),
                ksize=10, strides=1, padding='SAME'
            )
            
            # Local variance (captures volatility clustering)
            squared_diff = tf.square(ts_batch - tf.reduce_mean(ts_batch, axis=1, keepdims=True))
            local_var = tf.nn.avg_pool1d(
                tf.expand_dims(squared_diff, -1),
                ksize=5, strides=1, padding='SAME'
            )
            
            # Combine features
            graph_inspired_features = tf.concat([
                ma_short, ma_long, local_var
            ], axis=-1)
            
            return graph_inspired_features
        
        # Apply the conversion
        graph_features = layers.Lambda(
            ts_to_graph_features,
            name='ts_to_graph_conversion'
        )(ts_input)
        
        # Process with 1D convolutions to simulate graph convolutions
        x = layers.Conv1D(
            filters=self.model_config.get('gnn_units', 64),
            kernel_size=3,
            activation='relu',
            padding='same',
            name='graph_conv_1'
        )(graph_features)
        
        x = layers.Conv1D(
            filters=self.model_config.get('gnn_units', 64),
            kernel_size=3,
            activation='relu', 
            padding='same',
            name='graph_conv_2'
        )(x)
        
        # Global pooling to get sequence-level features
        pooled = layers.GlobalAveragePooling1D(name='graph_global_pool')(x)
        
        return pooled
    
    def _process_precomputed_graphs(self, node_features, adjacency_matrix):
        """Process pre-computed graph inputs through GNN layers."""
        
        # Graph convolution layers
        x = GraphConvolutionLayer(
            units=self.model_config.get('gnn_units', 64),
            activation='relu',
            name='graph_conv_layer_1'
        )(node_features, adjacency_matrix)
        
        x = GraphConvolutionLayer(
            units=self.model_config.get('gnn_units', 64),
            activation='relu',
            name='graph_conv_layer_2'
        )(x, adjacency_matrix)
        
        # Graph attention layer
        if self.model_config.get('use_attention', True):
            x = GraphAttentionLayer(
                units=self.model_config.get('attention_units', 16),
                num_heads=self.model_config.get('attention_heads', 4),
                name='graph_attention'
            )(x, adjacency_matrix)
        
        # Graph-level pooling
        pooling_method = self.model_config.get('graph_pooling', 'mean')
        
        if pooling_method == 'mean':
            pooled = tf.reduce_mean(x, axis=1)
        elif pooling_method == 'max':
            pooled = tf.reduce_max(x, axis=1)
        elif pooling_method == 'sum':
            pooled = tf.reduce_sum(x, axis=1)
        else:  # attention pooling
            # Learned attention weights for pooling
            attention_weights = layers.Dense(1, activation='softmax')(x)
            pooled = tf.reduce_sum(x * attention_weights, axis=1)
        
        return pooled
    
    def compile_model(self) -> None:
        """Compile model with GNN-specific loss function."""
        
        def gnn_visibility_loss(y_true, y_pred):
            """
            Custom loss function for GNN model.
            
            Combines standard deep hedging loss with graph regularization terms.
            """
            # Standard deep hedging loss
            hedge_ratios = tf.clip_by_value(y_pred, -1.0, 1.0)
            portfolio_pnl = hedge_ratios * y_true
            
            # Transaction costs
            transaction_cost = self.config['training']['loss']['transaction_cost']
            costs = transaction_cost * tf.abs(hedge_ratios)
            
            # Net P&L
            net_pnl = portfolio_pnl - costs
            
            # Utility function
            risk_aversion = self.config['training']['loss']['risk_aversion']
            utility = -tf.exp(-risk_aversion * net_pnl)
            
            # Graph regularization (encourage smoother hedge ratios)
            graph_regularization_weight = self.model_config.get('graph_regularization', 0.01)
            
            # L2 regularization on hedge ratios (encourage stability)
            l2_penalty = tf.reduce_mean(tf.square(hedge_ratios))
            
            # Temporal smoothness penalty
            if tf.shape(hedge_ratios)[1] > 1:
                temporal_diff = hedge_ratios[:, 1:] - hedge_ratios[:, :-1]
                smoothness_penalty = tf.reduce_mean(tf.square(temporal_diff))
            else:
                smoothness_penalty = 0.0
            
            # Combined loss
            total_loss = (
                -tf.reduce_mean(utility) +
                graph_regularization_weight * l2_penalty +
                graph_regularization_weight * smoothness_penalty
            )
            
            return total_loss
        
        # Optimizer with learning rate scheduling
        initial_lr = self.config['training']['learning_rate']
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=1000,
            decay_rate=0.95
        )
        
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=gnn_visibility_loss,
            metrics=['mae', 'mse']
        )
        
        logger.info("GNN model compiled with visibility graph loss")
    
    def prepare_graph_inputs(self, time_series_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare graph inputs from time series data.
        
        Args:
            time_series_data: Time series data [batch_size, seq_len, features]
            
        Returns:
            Tuple of (node_features, adjacency_matrices)
        """
        batch_size, seq_len, n_features = time_series_data.shape
        
        # For each sequence, create visibility graph
        batch_node_features = []
        batch_adjacency = []
        
        for i in range(batch_size):
            # Use first feature for visibility graph construction
            ts = time_series_data[i, :, 0]
            
            # Create visibility graph
            graph_method = self.model_config.get('visibility_method', 'natural')
            if graph_method == 'natural':
                vg = self.graph_converter.natural_visibility_graph(ts, use_ts2vg=False)
            elif graph_method == 'horizontal':
                vg = self.graph_converter.horizontal_visibility_graph(ts, use_ts2vg=False)
            else:
                vg = self.graph_converter.limited_penetrable_vg(ts)
            
            # Extract node features
            node_features = self.feature_extractor.extract_node_features(vg, ts)
            
            # Get adjacency matrix
            adj_matrix = self.feature_extractor.get_adjacency_matrix(vg, weighted=True)
            
            batch_node_features.append(node_features)
            batch_adjacency.append(adj_matrix)
        
        # Convert to numpy arrays (pad if necessary)
        max_nodes = max(nf.shape[0] for nf in batch_node_features)
        
        # Pad node features and adjacency matrices
        padded_node_features = np.zeros((batch_size, max_nodes, batch_node_features[0].shape[1]))
        padded_adjacency = np.zeros((batch_size, max_nodes, max_nodes))
        
        for i, (nf, adj) in enumerate(zip(batch_node_features, batch_adjacency)):
            n_nodes = nf.shape[0]
            padded_node_features[i, :n_nodes, :] = nf
            padded_adjacency[i, :n_nodes, :n_nodes] = adj
        
        return padded_node_features, padded_adjacency


def create_visibility_graph_datasets(time_series_data: np.ndarray,
                                   targets: np.ndarray,
                                   config: Dict) -> Dict:
    """
    Create datasets with visibility graph features.
    
    Args:
        time_series_data: Time series sequences
        targets: Target values
        config: Configuration dictionary
        
    Returns:
        Dictionary containing prepared datasets
    """
    logger.info("Creating visibility graph datasets...")
    
    # Initialize GNN model to use its graph preparation methods
    gnn_model = VisibilityGraphGNN(config)
    
    # Prepare graph inputs
    node_features, adjacency_matrices = gnn_model.prepare_graph_inputs(time_series_data)
    
    # Split data
    split_idx = int(len(time_series_data) * config['data']['train_split'])
    
    dataset = {
        'X_train_ts': time_series_data[:split_idx],
        'X_train_nodes': node_features[:split_idx],
        'X_train_adj': adjacency_matrices[:split_idx],
        'y_train': targets[:split_idx],
        
        'X_val_ts': time_series_data[split_idx:],
        'X_val_nodes': node_features[split_idx:],
        'X_val_adj': adjacency_matrices[split_idx:],
        'y_val': targets[split_idx:]
    }
    
    logger.info(f"Dataset created with {len(dataset['X_train_ts'])} training samples")
    
    return dataset


if __name__ == "__main__":
    # Example usage and testing
    from models.model_factory import load_config
    
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Add GNN-specific config
    config['model']['gnn'] = {
        'gnn_units': 64,
        'attention_units': 16,
        'attention_heads': 4,
        'use_attention': True,
        'graph_pooling': 'attention',
        'visibility_method': 'natural',
        'weighted_graphs': True,
        'dense_units': 128,
        'dropout_rate': 0.2,
        'graph_regularization': 0.01
    }
    
    # Create sample data
    np.random.seed(42)
    batch_size = 32
    seq_len = 60
    n_features = 5
    
    sample_ts = np.random.randn(batch_size, seq_len, n_features)
    sample_targets = np.random.randn(batch_size, 1)
    
    # Test GNN model
    gnn_model = VisibilityGraphGNN(config)
    model = gnn_model.build_model()
    
    print(f"GNN Model Summary:")
    print(f"Total parameters: {model.count_params()}")
    
    # Test graph input preparation
    node_features, adj_matrices = gnn_model.prepare_graph_inputs(sample_ts)
    print(f"Node features shape: {node_features.shape}")
    print(f"Adjacency matrices shape: {adj_matrices.shape}")
    
    # Test model compilation
    gnn_model.compile_model()
    print("GNN model compiled successfully!") 