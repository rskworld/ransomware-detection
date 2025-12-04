#!/usr/bin/env python
"""
Advanced Features Example Script
Demonstrates advanced features and data generation capabilities

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

# Import all modules
from src.data.data_generator import RansomwareDataGenerator
from src.data.advanced_features import AdvancedFeatureExtractor
from src.data.feature_extractor import FeatureExtractor
from src.data.data_augmentation import DataAugmenter
from src.monitoring.enhanced_detector import EnhancedRansomwareDetector


def example_data_generation():
    """Example: Generate synthetic dataset."""
    print("="*70)
    print("EXAMPLE: Data Generation")
    print("="*70)
    
    generator = RansomwareDataGenerator(random_seed=42)
    
    # Generate small dataset
    print("\n1. Generating dataset...")
    df = generator.generate_dataset(
        n_normal=50,
        n_ransomware=50,
        sequence_length=20,
        save_path='data/raw/example_dataset.csv'
    )
    
    print(f"\nDataset generated: {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df


def example_advanced_features():
    """Example: Advanced feature extraction."""
    print("\n" + "="*70)
    print("EXAMPLE: Advanced Feature Extraction")
    print("="*70)
    
    # Generate sample operations
    generator = RansomwareDataGenerator()
    operations = generator.generate_ransomware_sequence(length=50)
    
    # Extract advanced features
    print("\n1. Extracting advanced features...")
    advanced_extractor = AdvancedFeatureExtractor()
    
    timestamps = [op.get('timestamp', 0) for op in operations]
    features = advanced_extractor.extract_all_advanced_features(operations, timestamps)
    
    print(f"Extracted {len(features)} advanced features")
    print(f"Feature values (first 10): {features[:10]}")
    
    # Statistical features
    print("\n2. Statistical features from file sizes:")
    file_sizes = [op.get('file_size', 0) for op in operations]
    if file_sizes:
        stats_features = advanced_extractor.extract_statistical_features(np.array(file_sizes))
        print(f"  Mean: {stats_features[0]:.2f}")
        print(f"  Std: {stats_features[1]:.2f}")
        print(f"  Min: {stats_features[3]:.2f}")
        print(f"  Max: {stats_features[4]:.2f}")
    
    # Frequency domain features
    print("\n3. Frequency domain features:")
    if len(file_sizes) > 1:
        freq_features = advanced_extractor.extract_frequency_domain_features(np.array(file_sizes))
        print(f"  Mean magnitude: {freq_features[0]:.2f}")
        print(f"  Max magnitude: {freq_features[2]:.2f}")
    
    # Temporal patterns
    print("\n4. Temporal pattern features:")
    temporal_features = advanced_extractor.extract_temporal_patterns(timestamps, 
                                                                     [op.get('operation', '') for op in operations])
    print(f"  Mean interval: {temporal_features[0]:.4f}")
    print(f"  Burst count: {temporal_features[4]:.0f}")
    print(f"  Operation rate: {temporal_features[6]:.2f}")


def example_enhanced_detection():
    """Example: Enhanced ransomware detection."""
    print("\n" + "="*70)
    print("EXAMPLE: Enhanced Ransomware Detection")
    print("="*70)
    
    generator = RansomwareDataGenerator()
    detector = EnhancedRansomwareDetector()
    
    # Generate ransomware sequence
    print("\n1. Analyzing ransomware sequence...")
    operations = generator.generate_ransomware_sequence(length=100)
    timestamps = [op.get('timestamp', 0) for op in operations]
    
    # Detect patterns
    pattern_scores = detector.detect_patterns(operations, timestamps)
    
    print("\n2. Pattern Detection Scores:")
    for pattern, score in pattern_scores.items():
        print(f"  {pattern}: {score:.4f}")
    
    # Calculate combined threat score
    combined_score = detector.calculate_combined_score(pattern_scores)
    print(f"\n3. Combined Threat Score: {combined_score:.4f}")
    
    if combined_score > 0.5:
        print("  ⚠️  HIGH THREAT DETECTED!")
    else:
        print("  ✓ Normal activity")


def example_data_augmentation():
    """Example: Data augmentation."""
    print("\n" + "="*70)
    print("EXAMPLE: Data Augmentation")
    print("="*70)
    
    # Generate sample sequences
    generator = RansomwareDataGenerator()
    X, y = generator.generate_sequences_for_lstm(
        n_normal=10,
        n_ransomware=10,
        sequence_length=20,
        n_features=15
    )
    
    print(f"\n1. Original data: X shape {X.shape}, y shape {y.shape}")
    
    # Apply augmentation
    print("\n2. Applying data augmentation...")
    augmenter = DataAugmenter(random_seed=42)
    
    X_aug, y_aug = augmenter.augment_dataset(
        X, y,
        methods=['noise', 'time_shift', 'scale'],
        augmentation_factor=2
    )
    
    print(f"3. Augmented data: X shape {X_aug.shape}, y shape {y_aug.shape}")
    print(f"   Augmentation factor: {len(X_aug) / len(X):.1f}x")
    
    # Show augmentation effects
    print("\n4. Augmentation effects on first sample:")
    print(f"   Original mean: {X[0].mean():.4f}")
    print(f"   Augmented mean (noise): {augmenter.add_noise(X[:1])[0].mean():.4f}")


def example_complete_workflow():
    """Example: Complete workflow with all features."""
    print("\n" + "="*70)
    print("EXAMPLE: Complete Workflow")
    print("="*70)
    
    # 1. Generate data
    print("\n1. Generating dataset...")
    generator = RansomwareDataGenerator(random_seed=42)
    X, y = generator.generate_sequences_for_lstm(
        n_normal=20,
        n_ransomware=20,
        sequence_length=30,
        n_features=20
    )
    
    # 2. Augment data
    print("\n2. Augmenting data...")
    augmenter = DataAugmenter()
    X_aug, y_aug = augmenter.augment_dataset(X, y, augmentation_factor=1)
    
    # 3. Extract advanced features
    print("\n3. Extracting advanced features...")
    advanced_extractor = AdvancedFeatureExtractor()
    operations = generator.generate_ransomware_sequence(length=30)
    timestamps = [op.get('timestamp', 0) for op in operations]
    advanced_features = advanced_extractor.extract_all_advanced_features(operations, timestamps)
    
    print(f"   Extracted {len(advanced_features)} advanced features")
    
    # 4. Enhanced detection
    print("\n4. Running enhanced detection...")
    detector = EnhancedRansomwareDetector()
    pattern_scores = detector.detect_patterns(operations, timestamps)
    threat_score = detector.calculate_combined_score(pattern_scores)
    
    print(f"   Threat score: {threat_score:.4f}")
    
    print("\n✓ Complete workflow executed successfully!")


def main():
    """Run all advanced feature examples."""
    print("\n" + "="*70)
    print("RANSOMWARE DETECTION - ADVANCED FEATURES DEMONSTRATION")
    print("="*70)
    print("\nAuthor: Molla Samser")
    print("Designer & Tester: Rima Khatun")
    print("Organization: RSK World")
    print("Website: https://rskworld.in")
    print("="*70)
    
    try:
        # Example 1: Data Generation
        example_data_generation()
        
        # Example 2: Advanced Features
        example_advanced_features()
        
        # Example 3: Enhanced Detection
        example_enhanced_detection()
        
        # Example 4: Data Augmentation
        example_data_augmentation()
        
        # Example 5: Complete Workflow
        example_complete_workflow()
        
        print("\n" + "="*70)
        print("All advanced feature examples completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

