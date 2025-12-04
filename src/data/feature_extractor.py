"""
Feature Extraction Module
Extracts features from file system operations and system calls

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
from collections import Counter
import os
import hashlib


class FeatureExtractor:
    """
    Extracts features from file system operations for ransomware detection.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_names = []
        
    def extract_file_operation_features(
        self,
        file_operations: List[Dict]
    ) -> np.ndarray:
        """
        Extract features from file operations.
        
        Args:
            file_operations: List of file operation dictionaries
            
        Returns:
            Feature array
        """
        features = []
        
        # Operation type counts
        op_types = [op.get('operation', 'unknown') for op in file_operations]
        op_counts = Counter(op_types)
        
        # File access frequency
        file_accesses = len(file_operations)
        features.append(file_accesses)
        
        # Unique files accessed
        unique_files = len(set(op.get('file_path', '') for op in file_operations))
        features.append(unique_files)
        
        # Read operations
        features.append(op_counts.get('read', 0))
        
        # Write operations
        features.append(op_counts.get('write', 0))
        
        # Delete operations
        features.append(op_counts.get('delete', 0))
        
        # Rename operations
        features.append(op_counts.get('rename', 0))
        
        # Encryption-related operations
        encryption_ops = sum(1 for op in file_operations 
                           if 'encrypt' in op.get('operation', '').lower())
        features.append(encryption_ops)
        
        return np.array(features)
    
    def extract_encryption_patterns(
        self,
        file_paths: List[str],
        file_sizes: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Extract encryption pattern features.
        
        Args:
            file_paths: List of file paths
            file_sizes: List of file sizes (optional)
            
        Returns:
            Feature array
        """
        features = []
        
        # File extensions analysis
        extensions = [os.path.splitext(path)[1].lower() for path in file_paths]
        extension_counts = Counter(extensions)
        
        # Common encrypted file extensions
        encrypted_extensions = ['.encrypted', '.locked', '.crypto', '.vault']
        encrypted_count = sum(extension_counts.get(ext, 0) for ext in encrypted_extensions)
        features.append(encrypted_count)
        
        # Entropy analysis (simplified)
        if file_sizes:
            avg_size = np.mean(file_sizes) if file_sizes else 0
            size_variance = np.var(file_sizes) if len(file_sizes) > 1 else 0
            features.extend([avg_size, size_variance])
        else:
            features.extend([0, 0])
        
        # File path patterns
        suspicious_patterns = ['ransom', 'encrypt', 'decrypt', 'key', 'lock']
        suspicious_count = sum(
            1 for path in file_paths
            if any(pattern in path.lower() for pattern in suspicious_patterns)
        )
        features.append(suspicious_count)
        
        return np.array(features)
    
    def extract_system_call_features(
        self,
        system_calls: List[str]
    ) -> np.ndarray:
        """
        Extract features from system calls.
        
        Args:
            system_calls: List of system call names
            
        Returns:
            Feature array
        """
        features = []
        
        call_counts = Counter(system_calls)
        
        # Total system calls
        features.append(len(system_calls))
        
        # Unique system calls
        features.append(len(set(system_calls)))
        
        # File-related system calls
        file_calls = ['open', 'read', 'write', 'close', 'unlink', 'rename']
        file_call_count = sum(call_counts.get(call, 0) for call in file_calls)
        features.append(file_call_count)
        
        # Network-related system calls
        network_calls = ['connect', 'send', 'recv', 'socket']
        network_call_count = sum(call_counts.get(call, 0) for call in network_calls)
        features.append(network_call_count)
        
        # Process-related system calls
        process_calls = ['fork', 'exec', 'clone', 'kill']
        process_call_count = sum(call_counts.get(call, 0) for call in process_calls)
        features.append(process_call_count)
        
        return np.array(features)
    
    def extract_file_access_frequency(
        self,
        file_access_times: List[float],
        window_size: float = 60.0
    ) -> np.ndarray:
        """
        Extract file access frequency features.
        
        Args:
            file_access_times: List of timestamps
            window_size: Time window in seconds
            
        Returns:
            Feature array
        """
        features = []
        
        if not file_access_times:
            return np.zeros(5)
        
        access_times = np.array(file_access_times)
        
        # Total accesses
        features.append(len(access_times))
        
        # Access rate (accesses per second)
        time_span = access_times[-1] - access_times[0] if len(access_times) > 1 else 1.0
        access_rate = len(access_times) / max(time_span, 1.0)
        features.append(access_rate)
        
        # Burst detection (high frequency in short time)
        if len(access_times) > 1:
            intervals = np.diff(access_times)
            min_interval = np.min(intervals)
            avg_interval = np.mean(intervals)
            features.extend([min_interval, avg_interval])
        else:
            features.extend([0, 0])
        
        # Accesses in recent window
        if len(access_times) > 0:
            recent_time = access_times[-1]
            recent_accesses = np.sum((recent_time - access_times) <= window_size)
            features.append(recent_accesses)
        else:
            features.append(0)
        
        return np.array(features)
    
    def extract_entropy_features(
        self,
        file_data_samples: List[bytes]
    ) -> np.ndarray:
        """
        Extract entropy-based features.
        
        Args:
            file_data_samples: List of file data byte samples
            
        Returns:
            Feature array
        """
        features = []
        
        if not file_data_samples:
            return np.zeros(3)
        
        entropies = []
        for data in file_data_samples:
            if len(data) == 0:
                continue
            entropy = self._calculate_entropy(data)
            entropies.append(entropy)
        
        if entropies:
            features.append(np.mean(entropies))
            features.append(np.std(entropies))
            features.append(np.max(entropies))
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def _calculate_entropy(self, data: bytes) -> float:
        """
        Calculate Shannon entropy of data.
        
        Args:
            data: Byte data
            
        Returns:
            Entropy value
        """
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
    
    def extract_all_features(
        self,
        file_operations: List[Dict],
        system_calls: List[str],
        file_paths: List[str],
        file_access_times: Optional[List[float]] = None,
        use_advanced: bool = True
    ) -> np.ndarray:
        """
        Extract all features and combine into single feature vector.
        
        Args:
            file_operations: List of file operation dictionaries
            system_calls: List of system calls
            file_paths: List of file paths
            file_access_times: List of access timestamps (optional)
            use_advanced: Whether to use advanced feature extraction
            
        Returns:
            Combined feature array
        """
        features = []
        
        # File operation features
        op_features = self.extract_file_operation_features(file_operations)
        features.extend(op_features)
        
        # Encryption pattern features
        enc_features = self.extract_encryption_patterns(file_paths)
        features.extend(enc_features)
        
        # System call features
        sys_features = self.extract_system_call_features(system_calls)
        features.extend(sys_features)
        
        # File access frequency features
        if file_access_times:
            freq_features = self.extract_file_access_frequency(file_access_times)
        else:
            freq_features = np.zeros(5)
        features.extend(freq_features)
        
        # Advanced features if enabled
        if use_advanced:
            try:
                from .advanced_features import AdvancedFeatureExtractor
                advanced_extractor = AdvancedFeatureExtractor()
                advanced_features = advanced_extractor.extract_all_advanced_features(
                    file_operations,
                    file_access_times
                )
                features.extend(advanced_features)
            except ImportError:
                pass  # Advanced features not available
        
        return np.array(features)

