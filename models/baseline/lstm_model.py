"""
Baseline LSTM Model for Deep Hedging.

Implementation of stacked LSTM + dense layers for δ-hedge outputs
following the Deep Hedging paper methodology.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Dict
import logging

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class BaselineLSTMModel(BaseModel):
    """
    Baseline model: Stacked LSTM + dense layers for δ-hedge outputs.
    
    This implementation follows the Deep Hedging paper architecture
    with multiple LSTM layers and dense layers for hedge ratio prediction.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_config = config['model']['baseline']
        
    def build_model(self) -> Model:
        """Build stacked LSTM model architecture."""
        
        # Input layer
        inputs = layers.Input(
            shape=(
                self.config['model']['sequence_length'],
                self.config['model']['input_features']
            ),
            name='price_features'
        )
        
        x = inputs
        
        # Stacked LSTM layers
        for i, units in enumerate(self.model_config['lstm_units']):
            return_sequences = i < len(self.model_config['lstm_units']) - 1
            
            x = layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.model_config['dropout_rate'],
                recurrent_dropout=self.model_config['recurrent_dropout'],
                activation=self.model_config['activation'],
                name=f'lstm_{i+1}'
            )(x)
            
            # Add batch normalization for training stability
            x = layers.BatchNormalization(name=f'batch_norm_lstm_{i+1}')(x)
        
        # Dense layers for feature processing
        for i, units in enumerate(self.model_config['dense_units']):
            x = layers.Dense(
                units=units,
                activation='relu',
                name=f'dense_{i+1}'
            )(x)
            
            x = layers.Dropout(
                rate=self.model_config['dropout_rate'],
                name=f'dropout_dense_{i+1}'
            )(x)
            
            x = layers.BatchNormalization(name=f'batch_norm_dense_{i+1}')(x)
        
        # Output layer for delta hedge
        outputs = layers.Dense(
            units=self.config['model']['output_size'],
            activation='tanh',  # Constrain hedge ratio to [-1, 1]
            name='delta_hedge_output'
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='BaselineLSTM')
        
        self.model = model
        logger.info(f"Built Baseline LSTM model with {model.count_params()} parameters")
        
        return model
    
    def compile_model(self) -> None:
        """Compile model with custom deep hedging loss."""
        
        # Custom loss function for deep hedging
        def deep_hedging_loss(y_true, y_pred):
            """
            Deep hedging loss function from Buehler et al. (2019).
            
            Maximizes expected utility while accounting for transaction costs.
            """
            # Portfolio P&L
            hedge_ratios = tf.clip_by_value(y_pred, -1.0, 1.0)
            portfolio_pnl = hedge_ratios * y_true
            
            # Transaction costs
            transaction_cost = self.config['training']['loss']['transaction_cost']
            costs = transaction_cost * tf.abs(hedge_ratios)
            
            # Net P&L
            net_pnl = portfolio_pnl - costs
            
            # Utility function (exponential)
            risk_aversion = self.config['training']['loss']['risk_aversion']
            utility = -tf.exp(-risk_aversion * net_pnl)
            
            # Return negative expected utility (for minimization)
            return -tf.reduce_mean(utility)
        
        # Optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config['training']['learning_rate']
        )
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=deep_hedging_loss,
            metrics=['mae', 'mse']
        )
        
        logger.info("Baseline LSTM model compiled with deep hedging loss") 