#!/usr/bin/env python
"""
Example Usage Script for Ransomware Detection System

This script demonstrates how to use the ransomware detection system
for training, prediction, and real-time monitoring.

Author: Molla Samser
Designer & Tester: Rima Khatun
Organization: RSK World
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
"""

import numpy as np
from pathlib import Path

from src.model.lstm_model import RansomwareDetectionLSTM
from src.model.model_trainer import ModelTrainer
from src.data.data_processor import DataProcessor
from src.data.feature_extractor import FeatureExtractor
from src.monitoring.real_time_monitor import RealTimeMonitor
from src.monitoring.alert_system import AlertSystem


def example_training():
    """Example: Train a model."""
    print("="*60)
    print("EXAMPLE: Training a Ransomware Detection Model")
    print("="*60)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    n_samples = 1000
    sequence_length = 100
    n_features = 50
    
    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    print(f"   Data shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    
    # Split data
    print("\n2. Splitting data...")
    processor = DataProcessor(
        sequence_length=sequence_length,
        feature_count=n_features,
        normalize=True
    )
    
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
        X, y, test_size=0.2, val_size=0.1
    )
    
    # Build model
    print("\n3. Building model...")
    lstm_model = RansomwareDetectionLSTM(
        input_shape=(sequence_length, n_features),
        lstm_units=[128, 64],
        dropout_rate=0.3,
        dense_units=[64, 32]
    )
    
    model = lstm_model.build_model()
    print("   Model built successfully!")
    
    # Train model (with fewer epochs for example)
    print("\n4. Training model...")
    trainer = ModelTrainer()
    trained_model = trainer.train(
        X_train, y_train,
        X_val, y_val,
        model=model
    )
    
    # Evaluate
    print("\n5. Evaluating model...")
    metrics = trainer.evaluate(trained_model, X_test, y_test)
    
    print("\nTraining example completed!")
    return lstm_model, trained_model


def example_prediction(model):
    """Example: Make predictions with trained model."""
    print("\n" + "="*60)
    print("EXAMPLE: Making Predictions")
    print("="*60)
    
    # Generate sample test data
    print("\n1. Preparing test data...")
    test_sequences = np.random.randn(10, 100, 50)
    print(f"   Test sequences shape: {test_sequences.shape}")
    
    # Make predictions
    print("\n2. Making predictions...")
    predictions = model.predict_proba(test_sequences)
    
    print("\n3. Prediction results:")
    for i, pred in enumerate(predictions):
        status = "RANSOMWARE DETECTED" if pred[0] > 0.5 else "Normal"
        print(f"   Sequence {i+1}: {pred[0]:.4f} - {status}")
    
    print("\nPrediction example completed!")


def example_feature_extraction():
    """Example: Extract features from file operations."""
    print("\n" + "="*60)
    print("EXAMPLE: Feature Extraction")
    print("="*60)
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Simulate file operations
    print("\n1. Simulating file operations...")
    file_operations = [
        {'operation': 'read', 'file_path': '/path/to/file1.txt', 'timestamp': 1000.0},
        {'operation': 'write', 'file_path': '/path/to/file2.txt', 'timestamp': 1001.0},
        {'operation': 'delete', 'file_path': '/path/to/file3.txt', 'timestamp': 1002.0},
        {'operation': 'encrypt', 'file_path': '/path/to/file4.encrypted', 'timestamp': 1003.0},
    ]
    
    system_calls = ['open', 'read', 'write', 'close', 'unlink']
    file_paths = [op['file_path'] for op in file_operations]
    access_times = [op['timestamp'] for op in file_operations]
    
    # Extract features
    print("\n2. Extracting features...")
    features = extractor.extract_all_features(
        file_operations=file_operations,
        system_calls=system_calls,
        file_paths=file_paths,
        file_access_times=access_times
    )
    
    print(f"   Extracted {len(features)} features")
    print(f"   Feature values: {features[:10]}...")  # Show first 10
    
    print("\nFeature extraction example completed!")
    return features


def example_alert_system():
    """Example: Using the alert system."""
    print("\n" + "="*60)
    print("EXAMPLE: Alert System")
    print("="*60)
    
    # Initialize alert system
    print("\n1. Initializing alert system...")
    alert_system = AlertSystem(log_path="./logs")
    
    # Create sample alerts
    print("\n2. Creating sample alerts...")
    alert1 = alert_system.create_alert(
        confidence=0.85,
        severity="high",
        details={
            'suspicious_files': 15,
            'encryption_detected': True,
            'location': '/home/user/documents'
        }
    )
    
    alert2 = alert_system.create_alert(
        confidence=0.65,
        severity="medium",
        details={
            'suspicious_files': 5,
            'encryption_detected': False,
            'location': '/tmp'
        }
    )
    
    # Display alerts
    print("\n3. Recent alerts:")
    recent_alerts = alert_system.get_recent_alerts(limit=5)
    for alert in recent_alerts:
        print(f"\n{alert_system.format_alert_message(alert)}")
    
    print("\nAlert system example completed!")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("RANSOMWARE DETECTION SYSTEM - EXAMPLE USAGE")
    print("="*70)
    print("\nAuthor: Molla Samser")
    print("Designer & Tester: Rima Khatun")
    print("Organization: RSK World")
    print("Website: https://rskworld.in")
    print("="*70)
    
    try:
        # Example 1: Training
        lstm_model, trained_model = example_training()
        
        # Example 2: Prediction
        example_prediction(lstm_model)
        
        # Example 3: Feature Extraction
        example_feature_extraction()
        
        # Example 4: Alert System
        example_alert_system()
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

