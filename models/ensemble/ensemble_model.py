"""
Ensemble Model for Deep Hedging.

Combines Baseline LSTM and SigFormer models for enhanced performance
through ensemble learning techniques.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Dict, Tuple
import logging

from models.base_model import BaseModel
from models.baseline.lstm_model import BaselineLSTMModel
from models.sigformer.sigformer_model import SigFormerModel
from models.graph.gnn_model import VisibilityGraphGNN

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """
    Ensemble model combining LSTM, SigFormer, and GNN architectures.
    
    Uses weighted averaging of predictions from multiple models
    with learnable ensemble weights and graph-enhanced features.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_config = config['model']['ensemble']
        self.baseline_model = None
        self.sigformer_model = None
        self.gnn_model = None
        
    def build_individual_models(self) -> Tuple[Model, Model, Model]:
        """Build the individual LSTM, SigFormer, and GNN models."""
        
        # Create individual model instances
        baseline_instance = BaselineLSTMModel(self.config)
        sigformer_instance = SigFormerModel(self.config)
        gnn_instance = VisibilityGraphGNN(self.config)
        
        # Build models
        self.baseline_model = baseline_instance.build_model()
        self.sigformer_model = sigformer_instance.build_model()
        self.gnn_model = gnn_instance.build_model()
        
        logger.info("Built individual models for ensemble (LSTM, SigFormer, GNN)")
        
        return self.baseline_model, self.sigformer_model, self.gnn_model
    
    def build_model(self) -> Model:
        """Build ensemble model architecture."""
        
        # Build individual models first
        baseline_model, sigformer_model, gnn_model = self.build_individual_models()
        
        # Input layers
        # Time series input (for LSTM and SigFormer)
        ts_inputs = layers.Input(
            shape=(
                self.config['model']['sequence_length'],
                self.config['model']['input_features']
            ),
            name='ensemble_ts_input'
        )
        
        # Graph inputs (for GNN)
        node_features_input = layers.Input(
            shape=(None, self.config.get('model', {}).get('gnn', {}).get('node_feature_dim', 8)),
            name='ensemble_node_features'
        )
        
        adjacency_input = layers.Input(
            shape=(None, None),
            name='ensemble_adjacency_matrix'
        )
        
        # Get predictions from all models
        baseline_output = baseline_model(ts_inputs)
        sigformer_output = sigformer_model(ts_inputs)
        gnn_output = gnn_model([ts_inputs, node_features_input, adjacency_input])
        
        # Ensemble combination methods
        combination_method = self.model_config.get('combination_method', 'weighted_average')
        
        if combination_method == 'weighted_average':
            # Learnable weights for ensemble (now 3 models)
            ensemble_weights = layers.Dense(
                units=3,
                activation='softmax',
                name='ensemble_weights'
            )(layers.GlobalAveragePooling1D()(ts_inputs))
            
            # Apply weights
            weight_1 = ensemble_weights[:, 0:1]
            weight_2 = ensemble_weights[:, 1:2]
            weight_3 = ensemble_weights[:, 2:3]
            
            ensemble_output = (weight_1 * baseline_output + 
                             weight_2 * sigformer_output + 
                             weight_3 * gnn_output)
            
        elif combination_method == 'stacking':
            # Stacking ensemble with meta-learner
            combined_features = layers.Concatenate(name='stacking_concat')([
                baseline_output, sigformer_output, gnn_output
            ])
            
            # Meta-learner
            meta_features = layers.Dense(
                units=64,
                activation='relu',
                name='meta_dense_1'
            )(combined_features)
            
            meta_features = layers.Dropout(0.2)(meta_features)
            
            meta_features = layers.Dense(
                units=32,
                activation='relu',
                name='meta_dense_2'
            )(meta_features)
            
            ensemble_output = layers.Dense(
                units=self.config['model']['output_size'],
                activation='tanh',
                name='meta_output'
            )(meta_features)
            
        elif combination_method == 'attention':
            # Attention-based ensemble
            attention_input = layers.Concatenate(name='attention_input')([
                layers.Reshape((1, 1))(baseline_output),
                layers.Reshape((1, 1))(sigformer_output),
                layers.Reshape((1, 1))(gnn_output)
            ])
            
            attention_weights = layers.Dense(
                units=3,
                activation='softmax',
                name='attention_weights'
            )(layers.Flatten()(attention_input))
            
            weight_1 = attention_weights[:, 0:1]
            weight_2 = attention_weights[:, 1:2]
            weight_3 = attention_weights[:, 2:3]
            
            ensemble_output = (weight_1 * baseline_output + 
                             weight_2 * sigformer_output + 
                             weight_3 * gnn_output)
            
        else:  # simple_average
            ensemble_output = layers.Average(name='simple_average')([
                baseline_output, sigformer_output, gnn_output
            ])
        
        # Final output processing
        ensemble_output = layers.Dense(
            units=self.config['model']['output_size'],
            activation='tanh',
            name='ensemble_final_output'
        )(ensemble_output)
        
        # Create ensemble model
        model = Model(
            inputs=[ts_inputs, node_features_input, adjacency_input], 
            outputs=ensemble_output, 
            name='EnsembleModel'
        )
        
        self.model = model
        logger.info(f"Built Ensemble model with {model.count_params()} parameters")
        
        return model
    
    def compile_model(self) -> None:
        """Compile ensemble model with custom loss."""
        
        def ensemble_loss(y_true, y_pred):
            """
            Ensemble loss function combining multiple objectives.
            
            Incorporates both individual model losses and ensemble-specific terms.
            """
            # Portfolio P&L
            hedge_ratios = tf.clip_by_value(y_pred, -1.0, 1.0)
            portfolio_pnl = hedge_ratios * y_true
            
            # Transaction costs
            transaction_cost = self.config['training']['loss']['transaction_cost']
            costs = transaction_cost * tf.abs(hedge_ratios)
            
            # Net P&L
            net_pnl = portfolio_pnl - costs
            
            # Base utility
            risk_aversion = self.config['training']['loss']['risk_aversion']
            base_utility = -tf.exp(-risk_aversion * net_pnl)
            
            # Ensemble diversity penalty (encourage diversity between models)
            diversity_weight = self.model_config.get('diversity_weight', 0.01)
            
            # Get individual predictions for diversity calculation
            # This is a simplified version - in practice, you'd extract individual outputs
            diversity_penalty = diversity_weight * tf.reduce_variance(hedge_ratios)
            
            # Combined utility with diversity
            ensemble_utility = base_utility - diversity_penalty
            
            # Return negative expected utility (for minimization)
            return -tf.reduce_mean(ensemble_utility)
        
        # Optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config['training']['learning_rate']
        )
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=ensemble_loss,
            metrics=['mae', 'mse']
        )
        
        logger.info("Ensemble model compiled with ensemble loss")
    
    def get_individual_predictions(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions from individual models.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (baseline_predictions, sigformer_predictions)
        """
        if self.baseline_model is None or self.sigformer_model is None:
            raise ValueError("Individual models not built yet")
        
        baseline_pred = self.baseline_model.predict(X, verbose=0)
        sigformer_pred = self.sigformer_model.predict(X, verbose=0)
        
        return baseline_pred, sigformer_pred
    
    def analyze_ensemble_weights(self, X: np.ndarray) -> Dict:
        """
        Analyze ensemble weights and individual model contributions.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with ensemble analysis
        """
        if self.model is None:
            raise ValueError("Ensemble model not built yet")
        
        # Get ensemble prediction
        ensemble_pred = self.model.predict(X, verbose=0)
        
        # Get individual predictions
        baseline_pred, sigformer_pred = self.get_individual_predictions(X)
        
        # Calculate effective weights (simplified)
        baseline_weight = np.corrcoef(ensemble_pred.flatten(), baseline_pred.flatten())[0, 1]
        sigformer_weight = np.corrcoef(ensemble_pred.flatten(), sigformer_pred.flatten())[0, 1]
        
        # Normalize weights
        total_weight = abs(baseline_weight) + abs(sigformer_weight)
        if total_weight > 0:
            baseline_weight_norm = abs(baseline_weight) / total_weight
            sigformer_weight_norm = abs(sigformer_weight) / total_weight
        else:
            baseline_weight_norm = sigformer_weight_norm = 0.5
        
        analysis = {
            'ensemble_correlation_baseline': baseline_weight,
            'ensemble_correlation_sigformer': sigformer_weight,
            'normalized_weight_baseline': baseline_weight_norm,
            'normalized_weight_sigformer': sigformer_weight_norm,
            'ensemble_std': np.std(ensemble_pred),
            'baseline_std': np.std(baseline_pred),
            'sigformer_std': np.std(sigformer_pred),
            'prediction_diversity': np.std([np.std(baseline_pred), np.std(sigformer_pred)])
        }
        
        return analysis 