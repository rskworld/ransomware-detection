#!/usr/bin/env python
"""
Real-time Monitoring Script for Ransomware Detection

Author: Molla Samser
Designer & Tester: Rima Khatun
Organization: RSK World
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
"""

import argparse
import signal
import sys
from pathlib import Path

from src.model.lstm_model import RansomwareDetectionLSTM
from src.monitoring.real_time_monitor import RealTimeMonitor
from src.monitoring.alert_system import AlertSystem
from src.data.feature_extractor import FeatureExtractor
from src.utils.helpers import load_config, create_directories


# Global variables for cleanup
monitor = None
alert_system = None


def signal_handler(sig, frame):
    """Handle interrupt signal."""
    print('\n\nShutting down monitoring system...')
    if monitor:
        monitor.stop_monitoring()
    sys.exit(0)


def alert_callback(confidence, features):
    """Callback function for alerts."""
    global alert_system
    if alert_system:
        alert = alert_system.create_alert(
            confidence=confidence,
            severity='high' if confidence > 0.8 else 'medium',
            details={
                'feature_count': len(features),
                'feature_mean': float(features.mean()) if len(features) > 0 else 0.0
            }
        )
        print(alert_system.format_alert_message(alert))


def main():
    """Main monitoring function."""
    global monitor, alert_system
    
    parser = argparse.ArgumentParser(description='Real-time Ransomware Detection Monitor')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default='data/models/ransomware_detection_model.h5',
                       help='Path to trained model file')
    parser.add_argument('--watch', type=str, nargs='+', default=None,
                       help='Directories to watch (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create necessary directories
    monitoring_config = config.get('monitoring', {})
    create_directories([
        monitoring_config.get('log_path', './logs'),
        './data/test'  # Default test directory
    ])
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using train_model.py")
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    lstm_model = RansomwareDetectionLSTM()
    lstm_model.load_model(str(model_path))
    model = lstm_model.model
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()
    
    # Initialize alert system
    alert_system = AlertSystem(
        log_path=monitoring_config.get('log_path', './logs')
    )
    
    # Get watch directories
    if args.watch:
        watch_directories = args.watch
    else:
        watch_directories = monitoring_config.get('watch_directories', ['./data/test'])
    
    # Initialize monitor
    monitor = RealTimeMonitor(
        model=model,
        feature_extractor=feature_extractor,
        watch_directories=watch_directories,
        check_interval=monitoring_config.get('check_interval', 1.0),
        sequence_length=config.get('data', {}).get('sequence_length', 100),
        alert_threshold=monitoring_config.get('alert_threshold', 0.7)
    )
    
    # Set alert callback
    monitor.set_prediction_callback(alert_callback)
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start monitoring
    print("\n" + "="*60)
    print("RANSOMWARE DETECTION - REAL-TIME MONITORING")
    print("="*60)
    print(f"Watching directories: {watch_directories}")
    print(f"Alert threshold: {monitoring_config.get('alert_threshold', 0.7)}")
    print("\nPress Ctrl+C to stop monitoring...\n")
    
    monitor.start_monitoring()
    
    # Keep running
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == '__main__':
    main()

