"""
Real-time Monitoring Module
Monitors file system operations in real-time for ransomware detection

Author: Molla Samser
Designer & Tester: Rima Khatun
Organization: RSK World
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
"""

import numpy as np
import time
from pathlib import Path
from typing import Optional, List, Dict, Callable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
from collections import deque
import psutil
import os


class FileSystemMonitor(FileSystemEventHandler):
    """
    File system event handler for monitoring file operations.
    """
    
    def __init__(self, callback: Callable):
        """
        Initialize the file system monitor.
        
        Args:
            callback: Callback function to handle events
        """
        super().__init__()
        self.callback = callback
        self.operations = []
        
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self.callback({
                'operation': 'create',
                'file_path': event.src_path,
                'timestamp': time.time()
            })
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self.callback({
                'operation': 'modify',
                'file_path': event.src_path,
                'timestamp': time.time()
            })
    
    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory:
            self.callback({
                'operation': 'delete',
                'file_path': event.src_path,
                'timestamp': time.time()
            })
    
    def on_moved(self, event):
        """Handle file move/rename events."""
        if not event.is_directory:
            self.callback({
                'operation': 'rename',
                'file_path': event.src_path,
                'dest_path': event.dest_path,
                'timestamp': time.time()
            })


class RealTimeMonitor:
    """
    Real-time monitoring system for ransomware detection.
    """
    
    def __init__(
        self,
        model,
        feature_extractor,
        watch_directories: List[str],
        check_interval: float = 1.0,
        sequence_length: int = 100,
        alert_threshold: float = 0.7
    ):
        """
        Initialize the real-time monitor.
        
        Args:
            model: Trained LSTM model
            feature_extractor: Feature extractor instance
            watch_directories: Directories to monitor
            check_interval: Interval between checks (seconds)
            sequence_length: Length of sequences for prediction
            alert_threshold: Confidence threshold for alerts
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.watch_directories = [Path(d) for d in watch_directories]
        self.check_interval = check_interval
        self.sequence_length = sequence_length
        self.alert_threshold = alert_threshold
        
        self.observers = []
        self.is_monitoring = False
        self.operation_buffer = deque(maxlen=sequence_length * 10)
        self.system_calls = deque(maxlen=sequence_length * 10)
        self.file_access_times = deque(maxlen=sequence_length * 10)
        
        self.prediction_callback = None
        
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.is_monitoring:
            print("Monitoring already started")
            return
        
        print("Starting real-time monitoring...")
        self.is_monitoring = True
        
        # Start file system observers
        for directory in self.watch_directories:
            if not directory.exists():
                print(f"Warning: Directory {directory} does not exist, creating it...")
                directory.mkdir(parents=True, exist_ok=True)
            
            event_handler = FileSystemMonitor(self._handle_file_event)
            observer = Observer()
            observer.schedule(event_handler, str(directory), recursive=True)
            observer.start()
            self.observers.append(observer)
            print(f"Monitoring directory: {directory}")
        
        # Start prediction thread
        self.prediction_thread = threading.Thread(target=self._prediction_loop, daemon=True)
        self.prediction_thread.start()
        
        print("Real-time monitoring started successfully")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        if not self.is_monitoring:
            return
        
        print("Stopping real-time monitoring...")
        self.is_monitoring = False
        
        # Stop observers
        for observer in self.observers:
            observer.stop()
            observer.join()
        
        self.observers.clear()
        print("Monitoring stopped")
    
    def _handle_file_event(self, event: Dict):
        """Handle file system events."""
        self.operation_buffer.append(event)
        self.file_access_times.append(event.get('timestamp', time.time()))
        
        # Extract system call information (simulated)
        operation = event.get('operation', 'unknown')
        self.system_calls.append(operation)
    
    def _prediction_loop(self):
        """Main prediction loop running in separate thread."""
        while self.is_monitoring:
            try:
                if len(self.operation_buffer) >= self.sequence_length:
                    # Extract features and make prediction
                    features = self._extract_current_features()
                    
                    if features is not None:
                        prediction = self._make_prediction(features)
                        
                        if prediction is not None and prediction >= self.alert_threshold:
                            self._trigger_alert(prediction, features)
                
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Error in prediction loop: {e}")
                time.sleep(self.check_interval)
    
    def _extract_current_features(self) -> Optional[np.ndarray]:
        """Extract features from current buffer state."""
        if len(self.operation_buffer) < self.sequence_length:
            return None
        
        # Get recent operations
        recent_ops = list(self.operation_buffer)[-self.sequence_length:]
        file_paths = [op.get('file_path', '') for op in recent_ops]
        system_calls = list(self.system_calls)[-self.sequence_length:]
        access_times = list(self.file_access_times)[-self.sequence_length:]
        
        # Extract features
        features = self.feature_extractor.extract_all_features(
            file_operations=recent_ops,
            system_calls=system_calls,
            file_paths=file_paths,
            file_access_times=access_times
        )
        
        return features
    
    def _make_prediction(self, features: np.ndarray) -> Optional[float]:
        """Make prediction using the model."""
        try:
            # Reshape features for model input
            # Assuming features need to be in sequence format
            # This is a simplified version - actual implementation depends on model input shape
            features_reshaped = features.reshape(1, -1, 1)  # (batch, timesteps, features)
            
            # Pad or truncate to match model input shape
            # This is a placeholder - adjust based on actual model requirements
            if features_reshaped.shape[1] < self.sequence_length:
                padding = np.zeros((1, self.sequence_length - features_reshaped.shape[1], 1))
                features_reshaped = np.concatenate([features_reshaped, padding], axis=1)
            elif features_reshaped.shape[1] > self.sequence_length:
                features_reshaped = features_reshaped[:, :self.sequence_length, :]
            
            prediction = self.model.predict(features_reshaped, verbose=0)
            return float(prediction[0][0])
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def _trigger_alert(self, confidence: float, features: np.ndarray):
        """Trigger alert for detected ransomware activity."""
        alert_message = (
            f"⚠️ RANSOMWARE DETECTED!\n"
            f"Confidence: {confidence:.2%}\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Recent operations: {len(self.operation_buffer)}\n"
        )
        
        print("\n" + "="*50)
        print(alert_message)
        print("="*50 + "\n")
        
        if self.prediction_callback:
            self.prediction_callback(confidence, features)
    
    def set_prediction_callback(self, callback: Callable):
        """Set callback function for predictions."""
        self.prediction_callback = callback
    
    def get_statistics(self) -> Dict:
        """Get current monitoring statistics."""
        return {
            'is_monitoring': self.is_monitoring,
            'buffered_operations': len(self.operation_buffer),
            'watched_directories': [str(d) for d in self.watch_directories],
            'alert_threshold': self.alert_threshold
        }

