#!/usr/bin/env python
"""
Data Generation Script
Generates synthetic ransomware detection dataset

Author: Molla Samser
Designer & Tester: Rima Khatun
Organization: RSK World
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
"""

import argparse
from pathlib import Path
import sys

from src.data.data_generator import RansomwareDataGenerator


def main():
    """Main data generation function."""
    parser = argparse.ArgumentParser(description='Generate Ransomware Detection Dataset')
    parser.add_argument('--normal', type=int, default=500,
                       help='Number of normal sequences (default: 500)')
    parser.add_argument('--ransomware', type=int, default=500,
                       help='Number of ransomware sequences (default: 500)')
    parser.add_argument('--sequence-length', type=int, default=100,
                       help='Length of each sequence (default: 100)')
    parser.add_argument('--output', type=str, default='data/raw/ransomware_dataset.csv',
                       help='Output file path (default: data/raw/ransomware_dataset.csv)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--lstm-format', action='store_true',
                       help='Generate in LSTM-ready format (sequences)')
    parser.add_argument('--features', type=int, default=50,
                       help='Number of features per timestep (default: 50)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("RANSOMWARE DETECTION - DATA GENERATION")
    print("="*70)
    print(f"Normal sequences: {args.normal}")
    print(f"Ransomware sequences: {args.ransomware}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Output: {args.output}")
    print("="*70)
    
    # Initialize generator
    generator = RansomwareDataGenerator(random_seed=args.seed)
    
    if args.lstm_format:
        # Generate sequences for LSTM
        print("\nGenerating LSTM-ready sequences...")
        X, y = generator.generate_sequences_for_lstm(
            n_normal=args.normal,
            n_ransomware=args.ransomware,
            sequence_length=args.sequence_length,
            n_features=args.features
        )
        
        # Save as numpy arrays
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import numpy as np
        np.save(str(output_path).replace('.csv', '_X.npy'), X)
        np.save(str(output_path).replace('.csv', '_y.npy'), y)
        
        print(f"\nSequences saved:")
        print(f"  Features: {output_path.parent / (output_path.stem + '_X.npy')}")
        print(f"  Labels: {output_path.parent / (output_path.stem + '_y.npy')}")
        print(f"  Shape: X={X.shape}, y={y.shape}")
    else:
        # Generate CSV dataset
        df = generator.generate_dataset(
            n_normal=args.normal,
            n_ransomware=args.ransomware,
            sequence_length=args.sequence_length,
            save_path=args.output
        )
        
        print(f"\nDataset generation completed!")
        print(f"Total records: {len(df)}")
        print(f"Normal: {len(df[df['label'] == 0])}")
        print(f"Ransomware: {len(df[df['label'] == 1])}")
    
    print("\n" + "="*70)
    print("Data generation completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()

