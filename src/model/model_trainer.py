"""
Model Training Module
Handles model training, validation, and checkpointing

Author: Molla Samser
Designer & Tester: Rima Khatun
Organization: RSK World
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
from typing import Tuple, Optional, Dict
import yaml
from pathlib import Path


class ModelTrainer:
    """
    Handles training of the ransomware detection LSTM model.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.history = None
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model: Optional[keras.Model] = None
    ) -> keras.Model:
        """
        Train the model.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences (optional)
            y_val: Validation labels (optional)
            model: Pre-built model (optional)
            
        Returns:
            Trained model
        """
        # Get model configuration
        model_config = self.config.get('model', {})
        training_config = self.config.get('training', {})
        
        # Build model if not provided
        if model is None:
            from .lstm_model import RansomwareDetectionLSTM
            lstm_model = RansomwareDetectionLSTM(
                input_shape=tuple(model_config.get('input_shape', [100, 50])),
                lstm_units=model_config.get('lstm_units', [128, 64]),
                dropout_rate=model_config.get('dropout_rate', 0.3),
                dense_units=model_config.get('dense_units', [64, 32]),
                output_units=model_config.get('output_units', 1),
                activation=model_config.get('activation', 'sigmoid'),
                optimizer=model_config.get('optimizer', 'adam')
            )
            model = lstm_model.build_model()
        
        # Prepare callbacks
        callbacks = self._prepare_callbacks(training_config)
        
        # Training parameters
        batch_size = training_config.get('batch_size', 32)
        epochs = training_config.get('epochs', 50)
        validation_split = training_config.get('validation_split', 0.2)
        
        # Train model
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        print("Starting model training...")
        print(f"Training samples: {len(X_train)}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        
        self.history = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            validation_split=validation_split if validation_data is None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return model
    
    def _prepare_callbacks(self, training_config: Dict) -> list:
        """Prepare training callbacks."""
        callbacks = []
        
        # Model checkpoint
        model_save_path = self.config.get('data', {}).get('model_save_path', './data/models')
        os.makedirs(model_save_path, exist_ok=True)
        
        checkpoint_config = training_config.get('checkpoint', {})
        checkpoint_path = os.path.join(
            model_save_path,
            'best_model_{epoch:02d}_{val_loss:.2f}.h5'
        )
        
        callbacks.append(
            ModelCheckpoint(
                checkpoint_path,
                monitor=checkpoint_config.get('monitor', 'val_loss'),
                save_best_only=checkpoint_config.get('save_best_only', True),
                verbose=1
            )
        )
        
        # Early stopping
        early_stop_config = training_config.get('early_stopping', {})
        callbacks.append(
            EarlyStopping(
                monitor=early_stop_config.get('monitor', 'val_loss'),
                patience=early_stop_config.get('patience', 5),
                restore_best_weights=early_stop_config.get('restore_best_weights', True),
                verbose=1
            )
        )
        
        # Learning rate reduction
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        )
        
        return callbacks
    
    def evaluate(self, model: keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            model: Trained model
            X_test: Test sequences
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("Evaluating model...")
        results = model.evaluate(X_test, y_test, verbose=1)
        
        metrics = {}
        if len(model.metrics_names) == len(results):
            for name, value in zip(model.metrics_names, results):
                metrics[name] = value
        
        print("\nEvaluation Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def get_training_history(self) -> Optional[Dict]:
        """Get training history."""
        if self.history is None:
            return None
        return self.history.history

