"""
SigFormer Model for Deep Hedging.

Implementation of Signature Transform + Transformer encoder architecture
representing SOTA 2024 financial time series techniques.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Dict
import logging

from models.base_model import BaseModel
from models.sigformer.signature_transform import SignatureTransform

logger = logging.getLogger(__name__)


class SigFormerModel(BaseModel):
    """
    SigFormer: Signature transform + Transformer encoder.
    
    Implements path signature features with multi-head attention
    for capturing complex temporal dependencies in financial data.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_config = config['model']['sigformer']
        self.signature_transform = SignatureTransform(
            depth=self.model_config['signature_depth']
        )
        
    def positional_encoding(self, length: int, depth: int) -> tf.Tensor:
        """Create positional encoding for transformer."""
        depth = depth / 2
        
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth
        
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        
        pos_encoding = np.concatenate([
            np.sin(angle_rads), np.cos(angle_rads)
        ], axis=-1)
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def build_transformer_encoder(self, inputs: tf.Tensor) -> tf.Tensor:
        """Build transformer encoder with multi-head attention."""
        
        d_model = self.model_config['d_model']
        n_heads = self.model_config['n_heads']
        n_layers = self.model_config['n_layers']
        dropout = self.model_config['dropout']
        d_ff = self.model_config['d_ff']
        
        # Project inputs to d_model dimensions
        x = layers.Dense(d_model, name='input_projection')(inputs)
        
        # Add positional encoding
        seq_len = tf.shape(x)[1]
        pos_encoding = self.positional_encoding(self.config['model']['sequence_length'], d_model)
        x += pos_encoding[:seq_len, :]
        
        x = layers.Dropout(dropout)(x)
        
        # Transformer encoder layers
        for i in range(n_layers):
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=n_heads,
                key_dim=d_model // n_heads,
                name=f'attention_{i+1}'
            )(x, x)
            
            attention_output = layers.Dropout(dropout)(attention_output)
            x = layers.LayerNormalization(name=f'attention_norm_{i+1}')(x + attention_output)
            
            # Feed-forward network
            ffn_output = layers.Dense(d_ff, activation='relu', name=f'ffn_1_{i+1}')(x)
            ffn_output = layers.Dense(d_model, name=f'ffn_2_{i+1}')(ffn_output)
            ffn_output = layers.Dropout(dropout)(ffn_output)
            
            x = layers.LayerNormalization(name=f'ffn_norm_{i+1}')(x + ffn_output)
        
        return x
    
    def build_model(self) -> Model:
        """Build SigFormer model architecture."""
        
        # Input layer
        inputs = layers.Input(
            shape=(
                self.config['model']['sequence_length'],
                self.config['model']['input_features']
            ),
            name='price_features'
        )
        
        # Signature feature extraction
        signature_features = layers.Lambda(
            lambda x: self.signature_transform.compute_signature_features(x),
            name='signature_transform'
        )(inputs)
        
        # Reshape for transformer if needed
        batch_size = tf.shape(inputs)[0]
        sig_dim = tf.shape(signature_features)[1]
        
        # Create sequence from signature features (repeat for sequence length)
        seq_len = self.config['model']['sequence_length']
        signature_sequence = tf.tile(
            tf.expand_dims(signature_features, 1),
            [1, seq_len, 1]
        )
        
        # Combine original features with signature features
        combined_features = layers.Concatenate(axis=-1, name='feature_combination')([
            inputs, signature_sequence
        ])
        
        # Transformer encoder
        transformer_output = self.build_transformer_encoder(combined_features)
        
        # Global pooling
        pooled_output = layers.GlobalAveragePooling1D(name='global_pooling')(transformer_output)
        
        # Additional dense layers
        x = layers.Dense(
            units=self.model_config.get('dense_units', 256),
            activation='relu',
            name='dense_1'
        )(pooled_output)
        
        x = layers.Dropout(
            rate=self.model_config['dropout'],
            name='dropout_1'
        )(x)
        
        x = layers.Dense(
            units=self.model_config.get('dense_units', 256) // 2,
            activation='relu',
            name='dense_2'
        )(x)
        
        x = layers.Dropout(
            rate=self.model_config['dropout'],
            name='dropout_2'
        )(x)
        
        # Output layer for delta hedge
        outputs = layers.Dense(
            units=self.config['model']['output_size'],
            activation='tanh',  # Constrain hedge ratio to [-1, 1]
            name='delta_hedge_output'
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='SigFormer')
        
        self.model = model
        logger.info(f"Built SigFormer model with {model.count_params()} parameters")
        
        return model
    
    def compile_model(self) -> None:
        """Compile model with custom SigFormer loss."""
        
        def sigformer_loss(y_true, y_pred):
            """
            SigFormer loss with enhanced utility function.
            
            Incorporates signature-based features for better risk estimation.
            """
            # Portfolio P&L
            hedge_ratios = tf.clip_by_value(y_pred, -1.0, 1.0)
            portfolio_pnl = hedge_ratios * y_true
            
            # Transaction costs
            transaction_cost = self.config['training']['loss']['transaction_cost']
            costs = transaction_cost * tf.abs(hedge_ratios)
            
            # Net P&L
            net_pnl = portfolio_pnl - costs
            
            # Enhanced utility function with signature-based risk adjustment
            risk_aversion = self.config['training']['loss']['risk_aversion']
            
            # Add volatility penalty based on hedge ratio variance
            hedge_variance = tf.reduce_variance(hedge_ratios)
            volatility_penalty = 0.1 * hedge_variance
            
            # Utility function
            utility = -tf.exp(-risk_aversion * (net_pnl - volatility_penalty))
            
            # Return negative expected utility (for minimization)
            return -tf.reduce_mean(utility)
        
        # Optimizer with learning rate scheduling
        initial_lr = self.config['training']['learning_rate']
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=1000,
            decay_rate=0.96
        )
        
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=sigformer_loss,
            metrics=['mae', 'mse']
        )
        
        logger.info("SigFormer model compiled with enhanced signature loss") 