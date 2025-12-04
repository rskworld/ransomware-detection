"""
Enhanced Detection Module
Advanced pattern detection for ransomware behavior

Author: Molla Samser
Designer & Tester: Rima Khatun
Organization: RSK World
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
"""

import numpy as np
from typing import List, Dict, Optional
from collections import deque
import time


class EnhancedRansomwareDetector:
    """
    Enhanced ransomware detection with multiple detection patterns.
    """
    
    def __init__(
        self,
        encryption_threshold: float = 0.7,
        frequency_threshold: float = 10.0,
        entropy_threshold: float = 7.5
    ):
        """
        Initialize enhanced detector.
        
        Args:
            encryption_threshold: Threshold for encryption pattern detection
            frequency_threshold: Threshold for operation frequency (ops/sec)
            entropy_threshold: Threshold for file entropy
        """
        self.encryption_threshold = encryption_threshold
        self.frequency_threshold = frequency_threshold
        self.entropy_threshold = entropy_threshold
        
        self.detection_patterns = {
            'rapid_encryption': self._detect_rapid_encryption,
            'high_entropy_files': self._detect_high_entropy,
            'suspicious_renaming': self._detect_suspicious_renaming,
            'burst_operations': self._detect_burst_operations,
            'file_extension_changes': self._detect_extension_changes,
            'mass_file_access': self._detect_mass_file_access
        }
    
    def detect_patterns(
        self,
        operations: List[Dict],
        timestamps: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Detect multiple ransomware patterns.
        
        Args:
            operations: List of file operations
            timestamps: List of timestamps (optional)
            
        Returns:
            Dictionary of pattern scores
        """
        if timestamps is None:
            timestamps = [op.get('timestamp', time.time()) for op in operations]
        
        pattern_scores = {}
        
        for pattern_name, detector_func in self.detection_patterns.items():
            try:
                score = detector_func(operations, timestamps)
                pattern_scores[pattern_name] = score
            except Exception as e:
                pattern_scores[pattern_name] = 0.0
        
        return pattern_scores
    
    def _detect_rapid_encryption(
        self,
        operations: List[Dict],
        timestamps: List[float]
    ) -> float:
        """Detect rapid encryption pattern."""
        if len(operations) < 5:
            return 0.0
        
        encryption_ops = [op for op in operations if op.get('operation') == 'encrypt']
        
        if len(encryption_ops) < 3:
            return 0.0
        
        # Calculate encryption rate
        if len(timestamps) > 1:
            time_span = timestamps[-1] - timestamps[0]
            encryption_rate = len(encryption_ops) / max(time_span, 0.1)
            
            # Normalize to 0-1 scale
            score = min(encryption_rate / self.frequency_threshold, 1.0)
            return score
        
        return 0.0
    
    def _detect_high_entropy(
        self,
        operations: List[Dict],
        timestamps: List[float]
    ) -> float:
        """Detect high entropy files (encrypted content)."""
        if not operations:
            return 0.0
        
        entropies = [op.get('entropy', 0) for op in operations if op.get('entropy', 0) > 0]
        
        if not entropies:
            return 0.0
        
        high_entropy_count = sum(1 for e in entropies if e >= self.entropy_threshold)
        ratio = high_entropy_count / len(entropies)
        
        return ratio
    
    def _detect_suspicious_renaming(
        self,
        operations: List[Dict],
        timestamps: List[float]
    ) -> float:
        """Detect suspicious file renaming patterns."""
        rename_ops = [op for op in operations if op.get('operation') == 'rename']
        
        if len(rename_ops) < 2:
            return 0.0
        
        # Check for ransomware extensions
        suspicious_extensions = ['.encrypted', '.locked', '.crypto', '.vault', '.ecc']
        suspicious_count = 0
        
        for op in rename_ops:
            dest_path = op.get('dest_path', '')
            if any(ext in dest_path.lower() for ext in suspicious_extensions):
                suspicious_count += 1
        
        return suspicious_count / len(rename_ops) if rename_ops else 0.0
    
    def _detect_burst_operations(
        self,
        operations: List[Dict],
        timestamps: List[float]
    ) -> float:
        """Detect burst operation patterns."""
        if len(timestamps) < 2:
            return 0.0
        
        intervals = np.diff(sorted(timestamps))
        
        if len(intervals) == 0:
            return 0.0
        
        # Count rapid operations (intervals < 0.1 seconds)
        burst_count = sum(1 for interval in intervals if interval < 0.1)
        burst_ratio = burst_count / len(intervals)
        
        return burst_ratio
    
    def _detect_extension_changes(
        self,
        operations: List[Dict],
        timestamps: List[float]
    ) -> float:
        """Detect file extension changes."""
        rename_ops = [op for op in operations if op.get('operation') == 'rename']
        
        if len(rename_ops) < 2:
            return 0.0
        
        extension_changes = 0
        
        for op in rename_ops:
            src_path = op.get('file_path', '')
            dest_path = op.get('dest_path', '')
            
            src_ext = src_path.split('.')[-1] if '.' in src_path else ''
            dest_ext = dest_path.split('.')[-1] if '.' in dest_path else ''
            
            if src_ext != dest_ext:
                extension_changes += 1
        
        return extension_changes / len(rename_ops) if rename_ops else 0.0
    
    def _detect_mass_file_access(
        self,
        operations: List[Dict],
        timestamps: List[float]
    ) -> float:
        """Detect mass file access pattern."""
        if not operations:
            return 0.0
        
        unique_files = len(set(op.get('file_path', '') for op in operations))
        total_operations = len(operations)
        
        # High ratio of unique files to operations suggests mass access
        if total_operations > 0:
            ratio = unique_files / total_operations
            # Normalize (typical ransomware accesses many files)
            score = min(ratio * 2, 1.0)  # Scale appropriately
            return score
        
        return 0.0
    
    def calculate_combined_score(
        self,
        pattern_scores: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate combined threat score from pattern scores.
        
        Args:
            pattern_scores: Dictionary of pattern scores
            weights: Optional weights for each pattern
            
        Returns:
            Combined threat score (0-1)
        """
        if not pattern_scores:
            return 0.0
        
        if weights is None:
            # Default equal weights
            weights = {pattern: 1.0 / len(pattern_scores) for pattern in pattern_scores}
        
        combined_score = 0.0
        total_weight = 0.0
        
        for pattern, score in pattern_scores.items():
            weight = weights.get(pattern, 1.0 / len(pattern_scores))
            combined_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            combined_score /= total_weight
        
        return min(combined_score, 1.0)

