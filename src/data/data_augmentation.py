"""
Data Augmentation Module
Augments training data to improve model robustness

Author: Molla Samser
Designer & Tester: Rima Khatun
Organization: RSK World
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
"""

import numpy as np
from typing import Tuple, Optional
import random


class DataAugmenter:
    """
    Data augmentation utilities for ransomware detection.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize data augmenter.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
    
    def add_noise(
        self,
        X: np.ndarray,
        noise_level: float = 0.01
    ) -> np.ndarray:
        """
        Add Gaussian noise to sequences.
        
        Args:
            X: Input sequences
            noise_level: Standard deviation of noise
            
        Returns:
            Augmented sequences
        """
        noise = np.random.normal(0, noise_level, X.shape)
        return X + noise
    
    def time_shift(
        self,
        X: np.ndarray,
        max_shift: int = 5
    ) -> np.ndarray:
        """
        Apply time shifting to sequences.
        
        Args:
            X: Input sequences
            max_shift: Maximum shift amount
            
        Returns:
            Time-shifted sequences
        """
        augmented = []
        
        for seq in X:
            shift = random.randint(-max_shift, max_shift)
            if shift > 0:
                # Shift right (pad at beginning)
                shifted = np.pad(seq, ((shift, 0), (0, 0)), mode='edge')[:len(seq)]
            elif shift < 0:
                # Shift left (pad at end)
                shifted = np.pad(seq, ((0, -shift), (0, 0)), mode='edge')[-len(seq):]
            else:
                shifted = seq
            
            augmented.append(shifted)
        
        return np.array(augmented)
    
    def scale_features(
        self,
        X: np.ndarray,
        scale_range: Tuple[float, float] = (0.9, 1.1)
    ) -> np.ndarray:
        """
        Scale features by random factors.
        
        Args:
            X: Input sequences
            scale_range: Range for scaling factors
            
        Returns:
            Scaled sequences
        """
        augmented = []
        
        for seq in X:
            scale_factor = random.uniform(scale_range[0], scale_range[1])
            scaled = seq * scale_factor
            augmented.append(scaled)
        
        return np.array(augmented)
    
    def mixup(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup augmentation.
        
        Args:
            X: Input sequences
            y: Labels
            alpha: Mixup parameter
            
        Returns:
            Mixed sequences and labels
        """
        if len(X) < 2:
            return X, y
        
        indices = np.random.permutation(len(X))
        lam = np.random.beta(alpha, alpha, len(X))
        
        X_mixed = lam[:, None, None] * X + (1 - lam[:, None, None]) * X[indices]
        y_mixed = lam * y + (1 - lam) * y[indices]
        
        return X_mixed, y_mixed
    
    def augment_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        methods: list = ['noise', 'time_shift', 'scale'],
        augmentation_factor: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply multiple augmentation methods.
        
        Args:
            X: Input sequences
            y: Labels
            methods: List of augmentation methods to apply
            augmentation_factor: How many times to augment each sample
            
        Returns:
            Augmented sequences and labels
        """
        X_augmented = [X]
        y_augmented = [y]
        
        for _ in range(augmentation_factor):
            X_batch = X.copy()
            y_batch = y.copy()
            
            if 'noise' in methods:
                X_batch = self.add_noise(X_batch)
            
            if 'time_shift' in methods:
                X_batch = self.time_shift(X_batch)
            
            if 'scale' in methods:
                X_batch = self.scale_features(X_batch)
            
            X_augmented.append(X_batch)
            y_augmented.append(y_batch)
        
        X_final = np.concatenate(X_augmented, axis=0)
        y_final = np.concatenate(y_augmented, axis=0)
        
        return X_final, y_final

