#!/usr/bin/env python
"""
Main Training Script for Ransomware Detection Model

Author: Molla Samser
Designer & Tester: Rima Khatun
Organization: RSK World
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
"""

import numpy as np
import argparse
from pathlib import Path
import sys

from src.model.lstm_model import RansomwareDetectionLSTM
from src.model.model_trainer import ModelTrainer
from src.data.data_processor import DataProcessor
from src.utils.helpers import load_config, create_directories


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Ransomware Detection Model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to training data CSV file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create necessary directories
    data_config = config.get('data', {})
    create_directories([
        data_config.get('raw_data_path', './data/raw'),
        data_config.get('processed_data_path', './data/processed'),
        data_config.get('model_save_path', './data/models')
    ])
    
    # Initialize data processor
    processor = DataProcessor(
        sequence_length=data_config.get('sequence_length', 100),
        feature_count=data_config.get('feature_count', 50),
        normalize=data_config.get('normalization', True)
    )
    
    # Load or generate data
    if args.data and Path(args.data).exists():
        print(f"Loading data from {args.data}...")
        df = processor.load_data(args.data)
        
        # Extract features and labels
        # Adjust these based on your actual data structure
        feature_columns = [col for col in df.columns if col != 'label']
        X = processor.preprocess_features(df, feature_columns)
        y = df['label'].values if 'label' in df.columns else np.zeros(len(df))
        
        # Create sequences
        X, y = processor.create_sequences(X, y)
    else:
        print("No data file provided or file not found. Generating sample data...")
        # Generate sample data for demonstration
        n_samples = 1000
        n_features = data_config.get('feature_count', 50)
        sequence_length = data_config.get('sequence_length', 100)
        
        X = np.random.randn(n_samples, sequence_length, n_features)
        y = np.random.randint(0, 2, n_samples)
        print(f"Generated sample data: {X.shape}")
    
    # Split data
    training_config = config.get('training', {})
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
        X, y,
        test_size=training_config.get('test_split', 0.1),
        val_size=training_config.get('validation_split', 0.2)
    )
    
    # Build model
    model_config = config.get('model', {})
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
    print("\nModel Architecture:")
    print(lstm_model.get_model_summary())
    
    # Train model
    trainer = ModelTrainer(args.config)
    
    # Override epochs if provided
    if args.epochs:
        training_config['epochs'] = args.epochs
    
    trained_model = trainer.train(
        X_train, y_train,
        X_val, y_val,
        model=model
    )
    
    # Evaluate model
    print("\nEvaluating on test set...")
    metrics = trainer.evaluate(trained_model, X_test, y_test)
    
    # Save model
    model_save_path = Path(data_config.get('model_save_path', './data/models'))
    model_save_path.mkdir(parents=True, exist_ok=True)
    final_model_path = model_save_path / 'ransomware_detection_model.h5'
    lstm_model.save_model(str(final_model_path))
    
    print(f"\nTraining completed! Model saved to {final_model_path}")
    print("\nFinal Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main()

