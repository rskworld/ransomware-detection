"""
Advanced Feature Extraction Module
Extracts sophisticated features for ransomware detection

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
from collections import Counter, deque
from scipy import stats
import hashlib


class AdvancedFeatureExtractor:
    """
    Advanced feature extraction for ransomware detection.
    """
    
    def __init__(self, window_size: int = 10):
        """
        Initialize advanced feature extractor.
        
        Args:
            window_size: Size of sliding window for temporal features
        """
        self.window_size = window_size
    
    def extract_statistical_features(
        self,
        values: np.ndarray
    ) -> np.ndarray:
        """
        Extract statistical features from a time series.
        
        Args:
            values: Array of values
            
        Returns:
            Statistical features array
        """
        if len(values) == 0:
            return np.zeros(10)
        
        features = [
            np.mean(values),
            np.std(values),
            np.var(values),
            np.min(values),
            np.max(values),
            np.median(values),
            stats.skew(values) if len(values) > 2 else 0.0,
            stats.kurtosis(values) if len(values) > 2 else 0.0,
            np.percentile(values, 25),
            np.percentile(values, 75)
        ]
        
        return np.array(features)
    
    def extract_frequency_domain_features(
        self,
        time_series: np.ndarray
    ) -> np.ndarray:
        """
        Extract frequency domain features using FFT.
        
        Args:
            time_series: Time series data
            
        Returns:
            Frequency domain features
        """
        if len(time_series) < 2:
            return np.zeros(5)
        
        # Compute FFT
        fft_values = np.fft.fft(time_series)
        fft_magnitude = np.abs(fft_values)
        
        # Extract features
        features = [
            np.mean(fft_magnitude),
            np.std(fft_magnitude),
            np.max(fft_magnitude),
            np.sum(fft_magnitude[:len(fft_magnitude)//2]),  # Low frequency energy
            np.sum(fft_magnitude[len(fft_magnitude)//2:])   # High frequency energy
        ]
        
        return np.array(features)
    
    def extract_temporal_patterns(
        self,
        timestamps: List[float],
        operations: List[str]
    ) -> np.ndarray:
        """
        Extract temporal patterns from operations.
        
        Args:
            timestamps: List of timestamps
            operations: List of operation types
            
        Returns:
            Temporal pattern features
        """
        if len(timestamps) < 2:
            return np.zeros(8)
        
        timestamps = np.array(timestamps)
        intervals = np.diff(timestamps)
        
        features = [
            np.mean(intervals),
            np.std(intervals),
            np.min(intervals),
            np.max(intervals),
            np.sum(intervals < 0.1),  # Burst count (very short intervals)
            np.sum(intervals > 10.0),  # Long pauses
            len(intervals) / (timestamps[-1] - timestamps[0] + 1e-6),  # Operation rate
            self._calculate_autocorrelation(intervals)
        ]
        
        return np.array(features)
    
    def extract_entropy_features(
        self,
        file_paths: List[str],
        file_sizes: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Extract advanced entropy-based features.
        
        Args:
            file_paths: List of file paths
            file_sizes: List of file sizes (optional)
            
        Returns:
            Entropy features
        """
        features = []
        
        # Path entropy
        if file_paths:
            path_string = ''.join(file_paths)
            path_entropy = self._calculate_shannon_entropy(path_string.encode())
            features.append(path_entropy)
        else:
            features.append(0.0)
        
        # Extension diversity
        if file_paths:
            extensions = [p.split('.')[-1] if '.' in p else '' for p in file_paths]
            extension_counts = Counter(extensions)
            extension_entropy = self._calculate_entropy_from_counts(extension_counts)
            features.append(extension_entropy)
        else:
            features.append(0.0)
        
        # File size entropy
        if file_sizes and len(file_sizes) > 0:
            size_entropy = self._calculate_shannon_entropy(
                np.array(file_sizes).tobytes()
            )
            features.append(size_entropy)
        else:
            features.append(0.0)
        
        # Path length statistics
        if file_paths:
            path_lengths = [len(p) for p in file_paths]
            features.extend([
                np.mean(path_lengths),
                np.std(path_lengths)
            ])
        else:
            features.extend([0.0, 0.0])
        
        return np.array(features)
    
    def extract_behavioral_features(
        self,
        operations: List[Dict]
    ) -> np.ndarray:
        """
        Extract behavioral pattern features.
        
        Args:
            operations: List of operation dictionaries
            
        Returns:
            Behavioral features
        """
        features = []
        
        if not operations:
            return np.zeros(15)
        
        # Operation type distribution
        op_types = [op.get('operation', 'unknown') for op in operations]
        op_counts = Counter(op_types)
        total_ops = len(operations)
        
        # Operation ratios
        features.append(op_counts.get('read', 0) / max(total_ops, 1))
        features.append(op_counts.get('write', 0) / max(total_ops, 1))
        features.append(op_counts.get('encrypt', 0) / max(total_ops, 1))
        features.append(op_counts.get('delete', 0) / max(total_ops, 1))
        features.append(op_counts.get('rename', 0) / max(total_ops, 1))
        
        # Unique files vs total operations
        unique_files = len(set(op.get('file_path', '') for op in operations))
        features.append(unique_files / max(total_ops, 1))
        
        # Encryption ratio
        encrypted_ops = sum(1 for op in operations if op.get('is_encrypted', 0) == 1)
        features.append(encrypted_ops / max(total_ops, 1))
        
        # System file ratio
        system_files = sum(1 for op in operations if op.get('is_system_file', 0) == 1)
        features.append(system_files / max(total_ops, 1))
        
        # File size statistics
        file_sizes = [op.get('file_size', 0) for op in operations if op.get('file_size', 0) > 0]
        if file_sizes:
            features.extend([
                np.mean(file_sizes),
                np.std(file_sizes),
                np.max(file_sizes),
                np.min(file_sizes)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Entropy statistics
        entropies = [op.get('entropy', 0) for op in operations]
        if entropies:
            features.append(np.mean(entropies))
            features.append(np.std(entropies))
        else:
            features.extend([0.0, 0.0])
        
        return np.array(features)
    
    def extract_sliding_window_features(
        self,
        sequence: List[Dict],
        feature_func: callable
    ) -> np.ndarray:
        """
        Extract features using sliding window approach.
        
        Args:
            sequence: Sequence of operations
            feature_func: Function to extract features from window
            
        Returns:
            Window-based features
        """
        if len(sequence) < self.window_size:
            return feature_func(sequence)
        
        window_features = []
        for i in range(len(sequence) - self.window_size + 1):
            window = sequence[i:i + self.window_size]
            features = feature_func(window)
            window_features.append(features)
        
        # Aggregate window features
        if window_features:
            return np.mean(window_features, axis=0)
        else:
            return np.zeros(10)
    
    def extract_all_advanced_features(
        self,
        operations: List[Dict],
        timestamps: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Extract all advanced features.
        
        Args:
            operations: List of operation dictionaries
            timestamps: List of timestamps (optional)
            
        Returns:
            Combined advanced features
        """
        all_features = []
        
        if not operations:
            return np.zeros(50)
        
        # Extract file paths and sizes
        file_paths = [op.get('file_path', '') for op in operations]
        file_sizes = [op.get('file_size', 0) for op in operations]
        
        # Get timestamps
        if timestamps is None:
            timestamps = [op.get('timestamp', 0.0) for op in operations]
        
        # Statistical features from file sizes
        if file_sizes:
            size_features = self.extract_statistical_features(np.array(file_sizes))
            all_features.extend(size_features)
        else:
            all_features.extend([0.0] * 10)
        
        # Frequency domain features
        if len(file_sizes) > 1:
            freq_features = self.extract_frequency_domain_features(np.array(file_sizes))
            all_features.extend(freq_features)
        else:
            all_features.extend([0.0] * 5)
        
        # Temporal patterns
        if len(timestamps) > 1:
            temporal_features = self.extract_temporal_patterns(timestamps, 
                                                              [op.get('operation', '') for op in operations])
            all_features.extend(temporal_features)
        else:
            all_features.extend([0.0] * 8)
        
        # Entropy features
        entropy_features = self.extract_entropy_features(file_paths, file_sizes)
        all_features.extend(entropy_features)
        
        # Behavioral features
        behavioral_features = self.extract_behavioral_features(operations)
        all_features.extend(behavioral_features)
        
        # Pad or truncate to fixed size
        target_size = 50
        if len(all_features) < target_size:
            all_features.extend([0.0] * (target_size - len(all_features)))
        else:
            all_features = all_features[:target_size]
        
        return np.array(all_features)
    
    def _calculate_shannon_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy."""
        if len(data) == 0:
            return 0.0
        
        byte_counts = Counter(data)
        entropy = 0.0
        data_len = len(data)
        
        for count in byte_counts.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_entropy_from_counts(self, counts: Counter) -> float:
        """Calculate entropy from count dictionary."""
        if not counts:
            return 0.0
        
        total = sum(counts.values())
        entropy = 0.0
        
        for count in counts.values():
            probability = count / total
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_autocorrelation(self, values: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation."""
        if len(values) < lag + 1:
            return 0.0
        
        mean = np.mean(values)
        variance = np.var(values)
        
        if variance == 0:
            return 0.0
        
        autocorr = np.mean((values[:-lag] - mean) * (values[lag:] - mean)) / variance
        return autocorr

