# Ransomware Detection with Deep Learning

<!--
Project: Ransomware Detection with Deep Learning
Author: Molla Samser
Designer & Tester: Rima Khatun
Organization: RSK World
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
-->

## Overview

This project implements an advanced ransomware detection system using deep learning to identify ransomware attacks early by analyzing file access patterns, encryption behaviors, and system call sequences. The system helps prevent data loss and system compromise through real-time monitoring and early warning capabilities.

> 📖 **For complete documentation, see [DOCUMENTATION.md](DOCUMENTATION.md)**

## Features

### Core Features
- **File System Behavior Analysis**: Monitors and analyzes file system operations to detect suspicious patterns
- **LSTM Network for Sequence Detection**: Uses bidirectional Long Short-Term Memory networks to detect sequential patterns in system calls
- **Encryption Pattern Recognition**: Identifies encryption patterns that are characteristic of ransomware attacks
- **Real-time Monitoring**: Continuously monitors system activities for immediate threat detection
- **Early Warning System**: Provides alerts before significant damage occurs

### Advanced Features
- **Synthetic Data Generation**: Generate realistic ransomware and normal operation datasets for training
- **Advanced Feature Extraction**: Statistical, frequency domain, temporal, and entropy-based features
- **Enhanced Pattern Detection**: Multiple detection patterns including rapid encryption, high entropy, burst operations, and more
- **Data Augmentation**: Noise injection, time shifting, feature scaling, and mixup augmentation
- **Multi-Pattern Analysis**: Combined threat scoring from multiple detection patterns
- **Behavioral Analysis**: File access patterns, operation ratios, and system call analysis

## Technologies

- Python 3.8+
- TensorFlow 2.x
- Keras
- LSTM (Long Short-Term Memory)
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- Matplotlib
- Seaborn

## Project Structure

```
ransomware-detection/
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── lstm_model.py
│   │   └── model_trainer.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_processor.py
│   │   └── feature_extractor.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── real_time_monitor.py
│   │   └── alert_system.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── config/
│   └── config.yaml
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rskworld/ransomware-detection.git
cd ransomware-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generate Sample Data

Generate synthetic dataset for training:
```bash
# Generate CSV dataset
python generate_data.py --normal 500 --ransomware 500 --sequence-length 100

# Generate LSTM-ready sequences
python generate_data.py --normal 500 --ransomware 500 --lstm-format --features 50
```

### Training the Model

1. Generate or prepare your dataset in the `data/raw/` directory
2. Run data preprocessing:
```bash
python src/data/data_processor.py
```

3. Train the model:
```bash
python train_model.py

# With custom data
python train_model.py --data data/raw/ransomware_dataset.csv --epochs 50
```

Or use the Jupyter notebook:
```bash
jupyter notebook notebooks/model_training.ipynb
```

### Real-time Monitoring

Start the real-time monitoring system:
```bash
python run_monitoring.py

# Monitor specific directories
python run_monitoring.py --watch ./data/test ./data/samples
```

### Advanced Features Examples

Run examples demonstrating advanced features:
```bash
# Basic examples
python example_usage.py

# Advanced features
python example_advanced_features.py
```

## Model Architecture

The model uses a bidirectional LSTM architecture with:
- Input layer for sequence data
- Embedding layer for feature representation
- Bidirectional LSTM layers for pattern detection
- Dense layers for classification
- Dropout layers for regularization

## Performance Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is for educational purposes only.

## Contact

**RSK World**
- Founder: Molla Samser
- Designer & Tester: Rima Khatun
- Website: https://rskworld.in
- Email: help@rskworld.in, support@rskworld.in
- Phone: +91 93305 39277
- Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147

## Disclaimer

This project is created for educational and research purposes only. The authors are not responsible for any misuse of this software.

