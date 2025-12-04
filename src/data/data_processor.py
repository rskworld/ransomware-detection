"""
Data Processing Module
Handles data loading, preprocessing, and sequence generation

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, List
import os
import pickle
from pathlib import Path


class DataProcessor:
    """
    Handles data preprocessing for ransomware detection.
    """
    
    def __init__(
        self,
        sequence_length: int = 100,
        feature_count: int = 50,
        normalize: bool = True,
        scaler_type: str = 'standard'
    ):
        """
        Initialize the data processor.
        
        Args:
            sequence_length: Length of sequences for LSTM
            feature_count: Number of features per timestep
            normalize: Whether to normalize features
            scaler_type: Type of scaler ('standard' or 'minmax')
        """
        self.sequence_length = sequence_length
        self.feature_count = feature_count
        self.normalize = normalize
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.scaler_fitted = False
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Loaded DataFrame
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} samples with {len(df.columns)} features")
        return df
    
    def create_sequences(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        stride: int = 1
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences from time series data.
        
        Args:
            data: Input data array (n_samples, features)
            labels: Labels array (optional)
            stride: Stride for sequence creation
            
        Returns:
            Tuple of (sequences, labels)
        """
        sequences = []
        sequence_labels = []
        
        for i in range(0, len(data) - self.sequence_length + 1, stride):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
            
            if labels is not None:
                # Use the label of the last timestep in the sequence
                sequence_labels.append(labels[i + self.sequence_length - 1])
        
        X = np.array(sequences)
        y = np.array(sequence_labels) if labels is not None else None
        
        print(f"Created {len(X)} sequences of shape {X.shape}")
        return X, y
    
    def preprocess_features(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """
        Preprocess and extract features.
        
        Args:
            df: Input DataFrame
            feature_columns: List of column names to use as features
            
        Returns:
            Preprocessed feature array
        """
        # Select features
        features = df[feature_columns].values
        
        # Handle missing values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize if required
        if self.normalize:
            if not self.scaler_fitted:
                features = self.scaler.fit_transform(features)
                self.scaler_fitted = True
            else:
                features = self.scaler.transform(features)
        
        return features
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Features
            y: Labels
            test_size: Proportion of test set
            val_size: Proportion of validation set (from remaining after test split)
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_scaler(self, filepath: str):
        """Save the fitted scaler to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: str):
        """Load a saved scaler from disk."""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        self.scaler_fitted = True
        print(f"Scaler loaded from {filepath}")
    
    def save_processed_data(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        save_dir: str
    ):
        """Save processed data to disk."""
        os.makedirs(save_dir, exist_ok=True)
        
        np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(save_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
        
        print(f"Processed data saved to {save_dir}")
    
    def load_processed_data(
        self,
        load_dir: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load processed data from disk."""
        X_train = np.load(os.path.join(load_dir, 'X_train.npy'))
        X_val = np.load(os.path.join(load_dir, 'X_val.npy'))
        X_test = np.load(os.path.join(load_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(load_dir, 'y_train.npy'))
        y_val = np.load(os.path.join(load_dir, 'y_val.npy'))
        y_test = np.load(os.path.join(load_dir, 'y_test.npy'))
        
        print(f"Processed data loaded from {load_dir}")
        return X_train, X_val, X_test, y_train, y_val, y_test

