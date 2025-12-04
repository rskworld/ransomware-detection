"""
Alert System Module
Handles alert generation and notification for ransomware detection

Author: Molla Samser
Designer & Tester: Rima Khatun
Organization: RSK World
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
"""

import time
from typing import Dict, List, Optional
from datetime import datetime
import json
import os
from pathlib import Path


class AlertSystem:
    """
    Alert system for ransomware detection warnings.
    """
    
    def __init__(self, log_path: str = "./logs"):
        """
        Initialize the alert system.
        
        Args:
            log_path: Path to store alert logs
        """
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.alerts = []
        
    def create_alert(
        self,
        confidence: float,
        severity: str = "high",
        details: Optional[Dict] = None
    ) -> Dict:
        """
        Create an alert.
        
        Args:
            confidence: Detection confidence score
            severity: Alert severity level
            details: Additional alert details
            
        Returns:
            Alert dictionary
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'confidence': confidence,
            'severity': severity,
            'details': details or {},
            'alert_id': f"ALERT_{int(time.time())}"
        }
        
        self.alerts.append(alert)
        self._log_alert(alert)
        
        return alert
    
    def _log_alert(self, alert: Dict):
        """Log alert to file."""
        log_file = self.log_path / f"alerts_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Load existing alerts
        if log_file.exists():
            with open(log_file, 'r') as f:
                existing_alerts = json.load(f)
        else:
            existing_alerts = []
        
        # Append new alert
        existing_alerts.append(alert)
        
        # Save to file
        with open(log_file, 'w') as f:
            json.dump(existing_alerts, f, indent=2)
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """
        Get recent alerts.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of recent alerts
        """
        return self.alerts[-limit:] if len(self.alerts) > limit else self.alerts
    
    def get_alerts_by_severity(self, severity: str) -> List[Dict]:
        """
        Get alerts filtered by severity.
        
        Args:
            severity: Severity level to filter by
            
        Returns:
            List of filtered alerts
        """
        return [alert for alert in self.alerts if alert['severity'] == severity]
    
    def format_alert_message(self, alert: Dict) -> str:
        """
        Format alert as human-readable message.
        
        Args:
            alert: Alert dictionary
            
        Returns:
            Formatted message string
        """
        message = f"""
╔═══════════════════════════════════════════════════════════╗
║              RANSOMWARE DETECTION ALERT                    ║
╠═══════════════════════════════════════════════════════════╣
║ Alert ID: {alert['alert_id']:<45} ║
║ Timestamp: {alert['timestamp']:<43} ║
║ Confidence: {alert['confidence']:.2%}                                    ║
║ Severity: {alert['severity'].upper():<47} ║
╚═══════════════════════════════════════════════════════════╝

Details:
{json.dumps(alert['details'], indent=2)}

⚠️  IMMEDIATE ACTION REQUIRED ⚠️
"""
        return message
    
    def clear_alerts(self):
        """Clear all stored alerts."""
        self.alerts = []
        print("All alerts cleared")

