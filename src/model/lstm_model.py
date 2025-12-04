"""
LSTM Model Implementation for Ransomware Detection
Implements bidirectional LSTM network for sequence pattern detection

Author: Molla Samser
Designer & Tester: Rima Khatun
Organization: RSK World
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Tuple, Optional
import numpy as np


class RansomwareDetectionLSTM:
    """
    Bidirectional LSTM model for ransomware detection.
    
    This model analyzes sequences of system calls and file operations
    to detect ransomware patterns using deep learning.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int] = (100, 50),
        lstm_units: List[int] = [128, 64],
        dropout_rate: float = 0.3,
        dense_units: List[int] = [64, 32],
        output_units: int = 1,
        activation: str = 'sigmoid',
        optimizer: str = 'adam',
        learning_rate: float = 0.001
    ):
        """
        Initialize the LSTM model.
        
        Args:
            input_shape: Shape of input sequences (sequence_length, features)
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            dense_units: List of units for dense layers
            output_units: Number of output units
            activation: Activation function for output layer
            optimizer: Optimizer name
            learning_rate: Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.output_units = output_units
        self.activation = activation
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.model = None
        
    def build_model(self) -> keras.Model:
        """
        Build the LSTM model architecture.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='input_sequences')
        
        # First bidirectional LSTM layer
        x = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units[0],
                return_sequences=True,
                name='lstm_1'
            )
        )(inputs)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        
        # Second bidirectional LSTM layer (if multiple LSTM layers)
        if len(self.lstm_units) > 1:
            x = layers.Bidirectional(
                layers.LSTM(
                    self.lstm_units[1],
                    return_sequences=False,
                    name='lstm_2'
                )
            )(x)
            x = layers.Dropout(self.dropout_rate, name='dropout_2')(x)
        
        # Dense layers
        for i, units in enumerate(self.dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dense_dropout_{i+1}')(x)
        
        # Output layer
        outputs = layers.Dense(
            self.output_units,
            activation=self.activation,
            name='output'
        )(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='ransomware_detection_lstm')
        
        # Compile model
        optimizer = self._get_optimizer()
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        self.model = model
        return model
    
    def _get_optimizer(self) -> keras.optimizers.Optimizer:
        """Get the optimizer instance."""
        if self.optimizer_name.lower() == 'adam':
            return keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'sgd':
            return keras.optimizers.SGD(learning_rate=self.learning_rate)
        else:
            return keras.optimizers.Adam(learning_rate=self.learning_rate)
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            self.build_model()
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        return '\n'.join(summary)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input sequences of shape (n_samples, sequence_length, features)
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        return self.model.predict(X, verbose=0)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input sequences
            
        Returns:
            Probability scores
        """
        return self.predict(X)
    
    def save_model(self, filepath: str):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model from disk."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

