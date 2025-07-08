"""
Abstract base model class for trading models.

Defines the common interface that all model architectures must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict
import tensorflow as tf
from tensorflow.keras import Model


class BaseModel(ABC):
    """Abstract base class for trading models."""
    
    def __init__(self, config: Dict):
        """
        Initialize base model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.history = None
        
    @abstractmethod
    def build_model(self) -> Model:
        """
        Build the model architecture.
        
        Returns:
            Compiled TensorFlow/Keras model
        """
        pass
    
    @abstractmethod
    def compile_model(self) -> None:
        """Compile the model with optimizer and loss function."""
        pass
    
    def get_model(self) -> Model:
        """
        Get the built model.
        
        Returns:
            TensorFlow/Keras model
        """
        if self.model is None:
            self.model = self.build_model()
            self.compile_model()
        return self.model
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is not None:
            self.model.save(filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from file.
        
        Args:
            filepath: Path to load the model from
        """
        self.model = tf.keras.models.load_model(filepath)
    
    def get_model_summary(self) -> str:
        """
        Get model summary.
        
        Returns:
            String representation of model architecture
        """
        if self.model is None:
            return "Model not built yet"
        
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list) 