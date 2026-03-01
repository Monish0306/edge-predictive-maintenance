import numpy as np
import json
import os
import sys
import subprocess
from datetime import datetime
sys.path.append('.')

class ModelMonitor:
    """
    MLOps monitoring system that:
    1. Tracks prediction drift over time
    2. Auto-triggers retraining when needed
    3. Logs all monitoring events
    """

    def __init__(self, drift_threshold=0.15, alert_rate_threshold=0.4):
        self.drift_threshold = drift_threshold
        self.alert_rate_threshold = alert_rate_threshold
        self.prediction_log = []
        self.monitoring_log = []
        os.makedirs('data/monitoring', exist_ok=True)
        self._load_baseline()

    def _load_baseline(self):
        baseline_path = 'data/monitoring/baseline.json'
        if os.path.exists(baseline_path):
            with open(baseline_path) as f:
                self.baseline = json.load(f)
        else:
            # First time — set baseline
            self.baseline = {
                'mean_anomaly_prob': 0.35,
                'alert_rate': 0.15,
                'created_at': datetime.now().isoformat()
            }
            self._save_baseline()

    def _save_baseline(self):
        with open('data/monitoring/baseline.json', 'w') as f:
            json.dump(self.baseline, f, indent=2)

    def log_prediction(self, anomaly_prob, actual_label=None):
        """Call this every time model makes a prediction"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'anomaly_prob': float(anomaly_prob),
            'actual_label': actual_label
        }
        self.prediction_log.append(entry)

        # Keep only last 500 predictions in memory
        if len(self.prediction_log) > 500:
            self.prediction_log = self.prediction_log[-500:]

    def check_drift(self):
        """
        Check if model predictions are drifting from baseline.
        Drift = model behavior has changed significantly.
        This means new data patterns have emerged → retrain needed.
        """
        if len(self.prediction_log) < 50:
            return {
                'drift_detected': False,
                'reason': f'Not enough data yet ({len(self.prediction_log)}/50 predictions)',
                'action': 'Keep monitoring'
            }

        recent_probs = [p['anomaly_prob'] for p in self.prediction_log[-50:]]
        current_mean = np.mean(recent_probs)
        current_alert_rate = sum(1 for p in recent_probs if p > 0.5) / len(recent_probs)

        mean_drift = abs(current_mean - self.baseline['mean_anomaly_prob'])
        alert_drift = abs(current_alert_rate - self.baseline['alert_rate'])

        drift_detected = (mean_drift > self.drift_threshold or
                         alert_drift > self.alert_rate_threshold)

        result = {
            'drift_detected': drift_detected,
            'current_mean_prob': round(current_mean, 4),
            'baseline_mean_prob': self.baseline['mean_anomaly_prob'],
            'mean_drift': round(mean_drift, 4),
            'current_alert_rate': round(current_alert_rate, 4),
            'baseline_alert_rate': self.baseline['alert_rate'],
            'alert_drift': round(alert_drift, 4),
            'drift_threshold': self.drift_threshold,
            'predictions_analyzed': len(recent_probs),
            'timestamp': datetime.now().isoformat()
        }

        if drift_detected:
            result['reason'] = f"DRIFT DETECTED: mean shifted by {mean_drift:.3f}"
            result['action'] = 'RETRAIN MODEL'
            self._save_monitoring_event('DRIFT_DETECTED', result)
        else:
            result['reason'] = 'Model predictions stable'
            result['action'] = 'Continue monitoring'

        return result

    def trigger_retraining(self, reason="Manual trigger"):
        """Auto-trigger model retraining"""
        print(f"\n🔄 RETRAINING TRIGGERED: {reason}")
        print("="*50)

        event = {
            'type': 'RETRAINING_TRIGGERED',
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        self._save_monitoring_event('RETRAINING', event)

        print("Running: python src/model/train.py")
        print("Running: python src/model/convert_to_onnx.py")
        print("\n✅ In production, these would run automatically.")
        print("   For now, run them manually in Anaconda Prompt.")

        # Update baseline after retraining
        if self.prediction_log:
            recent = [p['anomaly_prob'] for p in self.prediction_log[-100:]]
            self.baseline['mean_anomaly_prob'] = float(np.mean(recent))
            self.baseline['alert_rate'] = float(sum(1 for p in recent if p > 0.5) / len(recent))
            self.baseline['last_retrain'] = datetime.now().isoformat()
            self._save_baseline()
            print("   Baseline updated ✅")

        return event

    def get_health_report(self):
        """Full system health report"""
        drift_status = self.check_drift()

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(self.prediction_log),
            'drift_status': drift_status,
            'system_healthy': not drift_status['drift_detected'],
            'recommendation': drift_status['action']
        }

        # Load model metadata
        meta_path = 'data/processed/model_metadata.json'
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                report['model_metadata'] = json.load(f)

        return report

    def _save_monitoring_event(self, event_type, data):
        log_path = 'data/monitoring/events.jsonl'
        os.makedirs('data/monitoring', exist_ok=True)
        
        # Fix: convert all values to JSON-serializable types
        def make_serializable(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            return obj

        clean_data = make_serializable(data)
        with open(log_path, 'a') as f:
            f.write(json.dumps(clean_data) + '\n')


if __name__ == '__main__':
    print("Testing MLOps Monitor...")
    monitor = ModelMonitor()

    # Simulate predictions
    print("\nSimulating 60 normal predictions...")
    for i in range(60):
        monitor.log_prediction(np.random.normal(0.3, 0.05))

    report = monitor.get_health_report()
    print(f"Drift detected: {report['drift_status']['drift_detected']}")
    print(f"Recommendation: {report['recommendation']}")

    # Simulate drift
    print("\nSimulating drift (anomaly rate increases)...")
    for i in range(60):
        monitor.log_prediction(np.random.normal(0.75, 0.1))

    report = monitor.get_health_report()
    print(f"Drift detected: {report['drift_status']['drift_detected']}")
    print(f"Recommendation: {report['recommendation']}")

    if report['drift_status']['drift_detected']:
        monitor.trigger_retraining("Drift detected in simulation")