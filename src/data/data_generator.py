"""
Data Generator Module
Generates synthetic ransomware detection data for training and testing

Author: Molla Samser
Designer & Tester: Rima Khatun
Organization: RSK World
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import random
import time
from pathlib import Path


class RansomwareDataGenerator:
    """
    Generates synthetic data for ransomware detection training.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Normal operation patterns
        self.normal_operations = ['read', 'write', 'open', 'close', 'stat']
        self.normal_file_extensions = ['.txt', '.pdf', '.doc', '.jpg', '.png', '.mp4']
        
        # Ransomware operation patterns
        self.ransomware_operations = ['encrypt', 'rename', 'delete', 'write', 'read']
        self.ransomware_extensions = ['.encrypted', '.locked', '.crypto', '.vault', '.ecc']
        
    def generate_normal_sequence(
        self,
        length: int = 100,
        n_files: int = 10
    ) -> List[Dict]:
        """
        Generate a normal file operation sequence.
        
        Args:
            length: Sequence length
            n_files: Number of unique files
            
        Returns:
            List of file operation dictionaries
        """
        operations = []
        files = [f"/home/user/file_{i}{random.choice(self.normal_file_extensions)}" 
                for i in range(n_files)]
        
        base_time = time.time()
        
        for i in range(length):
            operation = {
                'operation': random.choice(self.normal_operations),
                'file_path': random.choice(files),
                'timestamp': base_time + i * random.uniform(0.1, 2.0),
                'file_size': random.randint(1000, 1000000),
                'entropy': random.uniform(3.0, 7.0),  # Normal entropy range
                'access_frequency': random.uniform(0.1, 5.0),
                'file_age': random.randint(1, 365),
                'is_encrypted': 0,
                'is_system_file': random.choice([0, 1]),
                'process_id': random.randint(1000, 9999),
                'user_id': random.randint(100, 999)
            }
            operations.append(operation)
        
        return operations
    
    def generate_ransomware_sequence(
        self,
        length: int = 100,
        n_files: int = 50
    ) -> List[Dict]:
        """
        Generate a ransomware file operation sequence.
        
        Args:
            length: Sequence length
            n_files: Number of files to encrypt
            
        Returns:
            List of file operation dictionaries
        """
        operations = []
        files = [f"/home/user/document_{i}{random.choice(self.normal_file_extensions)}" 
                for i in range(n_files)]
        
        base_time = time.time()
        encrypted_count = 0
        
        for i in range(length):
            # Ransomware behavior: rapid encryption of multiple files
            if i < n_files and encrypted_count < n_files:
                # Encryption phase
                operation = {
                    'operation': 'encrypt',
                    'file_path': files[encrypted_count],
                    'timestamp': base_time + i * random.uniform(0.01, 0.5),  # Very fast
                    'file_size': random.randint(50000, 5000000),
                    'entropy': random.uniform(7.5, 8.0),  # High entropy (encrypted)
                    'access_frequency': random.uniform(10.0, 100.0),  # High frequency
                    'file_age': random.randint(1, 365),
                    'is_encrypted': 1,
                    'is_system_file': 0,
                    'process_id': random.randint(1000, 9999),
                    'user_id': random.randint(100, 999)
                }
                encrypted_count += 1
            else:
                # Rename encrypted files
                if encrypted_count > 0:
                    file_idx = random.randint(0, min(encrypted_count - 1, len(files) - 1))
                    operation = {
                        'operation': 'rename',
                        'file_path': files[file_idx],
                        'dest_path': files[file_idx].replace(
                            random.choice(self.normal_file_extensions),
                            random.choice(self.ransomware_extensions)
                        ),
                        'timestamp': base_time + i * random.uniform(0.01, 0.5),
                        'file_size': random.randint(50000, 5000000),
                        'entropy': random.uniform(7.5, 8.0),
                        'access_frequency': random.uniform(10.0, 100.0),
                        'file_age': random.randint(1, 365),
                        'is_encrypted': 1,
                        'is_system_file': 0,
                        'process_id': random.randint(1000, 9999),
                        'user_id': random.randint(100, 999)
                    }
                else:
                    operation = {
                        'operation': random.choice(self.ransomware_operations),
                        'file_path': random.choice(files) if files else '/tmp/file.txt',
                        'timestamp': base_time + i * random.uniform(0.01, 0.5),
                        'file_size': random.randint(50000, 5000000),
                        'entropy': random.uniform(7.5, 8.0),
                        'access_frequency': random.uniform(10.0, 100.0),
                        'file_age': random.randint(1, 365),
                        'is_encrypted': 1,
                        'is_system_file': 0,
                        'process_id': random.randint(1000, 9999),
                        'user_id': random.randint(100, 999)
                    }
            
            operations.append(operation)
        
        return operations
    
    def generate_dataset(
        self,
        n_normal: int = 500,
        n_ransomware: int = 500,
        sequence_length: int = 100,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate a complete dataset.
        
        Args:
            n_normal: Number of normal sequences
            n_ransomware: Number of ransomware sequences
            sequence_length: Length of each sequence
            save_path: Path to save the dataset (optional)
            
        Returns:
            DataFrame with generated data
        """
        print(f"Generating dataset: {n_normal} normal + {n_ransomware} ransomware sequences...")
        
        all_data = []
        labels = []
        
        # Generate normal sequences
        print("Generating normal sequences...")
        for i in range(n_normal):
            seq = self.generate_normal_sequence(length=sequence_length)
            all_data.extend(seq)
            labels.extend([0] * len(seq))
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{n_normal} normal sequences")
        
        # Generate ransomware sequences
        print("Generating ransomware sequences...")
        for i in range(n_ransomware):
            seq = self.generate_ransomware_sequence(length=sequence_length)
            all_data.extend(seq)
            labels.extend([1] * len(seq))
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{n_ransomware} ransomware sequences")
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        df['label'] = labels
        
        # Add sequence IDs
        df['sequence_id'] = 0
        seq_id = 0
        for i in range(1, len(df)):
            if df.iloc[i]['label'] != df.iloc[i-1]['label']:
                seq_id += 1
            df.iloc[i, df.columns.get_loc('sequence_id')] = seq_id
        
        print(f"\nDataset generated: {len(df)} total records")
        print(f"  Normal: {len(df[df['label'] == 0])} records")
        print(f"  Ransomware: {len(df[df['label'] == 1])} records")
        
        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"\nDataset saved to {save_path}")
        
        return df
    
    def generate_sequences_for_lstm(
        self,
        n_normal: int = 500,
        n_ransomware: int = 500,
        sequence_length: int = 100,
        n_features: int = 50
    ) -> tuple:
        """
        Generate sequences ready for LSTM training.
        
        Args:
            n_normal: Number of normal sequences
            n_ransomware: Number of ransomware sequences
            sequence_length: Length of each sequence
            n_features: Number of features per timestep
            
        Returns:
            Tuple of (X, y) arrays
        """
        print("Generating sequences for LSTM...")
        
        X = []
        y = []
        
        # Generate normal sequences
        for i in range(n_normal):
            seq = self.generate_normal_sequence(length=sequence_length)
            features = self._extract_features_from_sequence(seq, n_features)
            X.append(features)
            y.append(0)
        
        # Generate ransomware sequences
        for i in range(n_ransomware):
            seq = self.generate_ransomware_sequence(length=sequence_length)
            features = self._extract_features_from_sequence(seq, n_features)
            X.append(features)
            y.append(1)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Generated sequences: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def _extract_features_from_sequence(
        self,
        sequence: List[Dict],
        n_features: int
    ) -> np.ndarray:
        """
        Extract features from a sequence of operations.
        
        Args:
            sequence: List of operation dictionaries
            n_features: Number of features to extract
            
        Returns:
            Feature array of shape (sequence_length, n_features)
        """
        features = []
        
        for op in sequence:
            feature_vector = [
                float(op.get('file_size', 0)),
                float(op.get('entropy', 0)),
                float(op.get('access_frequency', 0)),
                float(op.get('file_age', 0)),
                float(op.get('is_encrypted', 0)),
                float(op.get('is_system_file', 0)),
                float(op.get('process_id', 0)) / 10000.0,  # Normalize
                float(op.get('user_id', 0)) / 1000.0,  # Normalize
                # Operation type encoding
                1.0 if op.get('operation') == 'encrypt' else 0.0,
                1.0 if op.get('operation') == 'read' else 0.0,
                1.0 if op.get('operation') == 'write' else 0.0,
                1.0 if op.get('operation') == 'delete' else 0.0,
                1.0 if op.get('operation') == 'rename' else 0.0,
            ]
            
            # Pad or truncate to n_features
            if len(feature_vector) < n_features:
                feature_vector.extend([0.0] * (n_features - len(feature_vector)))
            else:
                feature_vector = feature_vector[:n_features]
            
            features.append(feature_vector)
        
        return np.array(features)

