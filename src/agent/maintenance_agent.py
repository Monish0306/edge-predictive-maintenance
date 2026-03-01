import numpy as np
from datetime import datetime, timedelta

class MaintenanceAgent:
    """
    Rule-based + ML Agent that:
    1. Detects anomalies from model predictions
    2. Suggests fixes based on which sensors triggered
    3. Schedules maintenance
    4. Auto-triggers retraining if needed
    """
    
    def __init__(self):
        self.alert_history = []
        self.false_positive_count = 0
        self.total_predictions = 0
        
        # Sensor → possible cause mapping
        self.sensor_rules = {
            'sensor2':  {'name': 'Fan Speed', 
                         'issue': 'Fan bearing wear', 
                         'fix': 'Lubricate or replace fan bearing'},
            'sensor3':  {'name': 'Core Speed', 
                         'issue': 'Rotor imbalance', 
                         'fix': 'Balance rotor, check vibration'},
            'sensor4':  {'name': 'Bypass Ratio', 
                         'issue': 'Compressor degradation', 
                         'fix': 'Inspect compressor blades'},
            'sensor7':  {'name': 'Static Pressure', 
                         'issue': 'Seal leak', 
                         'fix': 'Inspect and replace seals'},
            'sensor8':  {'name': 'Fuel Flow', 
                         'issue': 'Fuel injector clog', 
                         'fix': 'Clean or replace fuel injectors'},
            'sensor11': {'name': 'Bypass Temperature', 
                         'issue': 'Cooling system issue', 
                         'fix': 'Check cooling passages'},
            'sensor12': {'name': 'Burner Fuel-Air', 
                         'issue': 'Combustion anomaly', 
                         'fix': 'Check igniter and fuel mix'},
            'sensor15': {'name': 'Bleed Enthalpy', 
                         'issue': 'Bleed valve leak', 
                         'fix': 'Inspect bleed valves'},
        }
    
    def analyze_anomaly(self, anomaly_prob, sensor_readings, sensor_names):
        """
        Main agent function — analyze and respond to anomaly
        
        Args:
            anomaly_prob: float (0-1) from model
            sensor_readings: dict of {sensor_name: value}
            sensor_names: list of sensor names
        
        Returns:
            action_plan: dict with full agent response
        """
        self.total_predictions += 1
        
        severity = self._get_severity(anomaly_prob)
        triggered_sensors = self._find_triggered_sensors(sensor_readings)
        root_cause = self._diagnose_root_cause(triggered_sensors)
        maintenance_schedule = self._schedule_maintenance(severity)
        recommended_actions = self._get_actions(triggered_sensors, severity)
        
        action_plan = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'anomaly_probability': round(float(anomaly_prob), 4),
            'severity': severity,
            'alert': severity in ['HIGH', 'CRITICAL'],
            'triggered_sensors': triggered_sensors,
            'root_cause': root_cause,
            'recommended_actions': recommended_actions,
            'maintenance_schedule': maintenance_schedule,
            'estimated_downtime': self._estimate_downtime(severity),
            'estimated_cost_saved': self._estimate_cost_saving(severity),
        }
        
        if action_plan['alert']:
            self.alert_history.append(action_plan)
        
        return action_plan
    
    def _get_severity(self, prob):
        if prob < 0.3:
            return 'NORMAL'
        elif prob < 0.5:
            return 'LOW'
        elif prob < 0.7:
            return 'MEDIUM'
        elif prob < 0.9:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def _find_triggered_sensors(self, sensor_readings):
        """Find which sensors are abnormal (above 0.8 after normalization)"""
        triggered = []
        for sensor_name, value in sensor_readings.items():
            if value > 0.8 or value < 0.1:  # extreme values
                if sensor_name in self.sensor_rules:
                    triggered.append(sensor_name)
        return triggered
    
    def _diagnose_root_cause(self, triggered_sensors):
        if not triggered_sensors:
            return "No specific sensor anomaly detected. General degradation pattern."
        
        causes = []
        for sensor in triggered_sensors:
            if sensor in self.sensor_rules:
                rule = self.sensor_rules[sensor]
                causes.append(f"{rule['name']}: {rule['issue']}")
        
        return " | ".join(causes) if causes else "Multiple sensor correlation detected"
    
    def _get_actions(self, triggered_sensors, severity):
        actions = []
        
        if severity == 'NORMAL':
            actions.append("Continue normal operation. Next scheduled check in 30 days.")
        
        elif severity == 'LOW':
            actions.append("Monitor closely. Increase sensor polling frequency to every hour.")
            actions.append("Log readings for trend analysis.")
        
        elif severity == 'MEDIUM':
            actions.append("Schedule inspection within 7 days.")
            for sensor in triggered_sensors:
                if sensor in self.sensor_rules:
                    actions.append(f"→ {self.sensor_rules[sensor]['fix']}")
        
        elif severity == 'HIGH':
            actions.append("⚠️  IMMEDIATE INSPECTION REQUIRED within 48 hours!")
            actions.append("Reduce operational load by 20%.")
            for sensor in triggered_sensors:
                if sensor in self.sensor_rules:
                    actions.append(f"→ PRIORITY FIX: {self.sensor_rules[sensor]['fix']}")
        
        elif severity == 'CRITICAL':
            actions.append("🚨 SHUTDOWN RECOMMENDED — Failure imminent!")
            actions.append("Alert maintenance team immediately.")
            actions.append("Prepare replacement parts.")
            for sensor in triggered_sensors:
                if sensor in self.sensor_rules:
                    actions.append(f"→ EMERGENCY: {self.sensor_rules[sensor]['fix']}")
        
        return actions
    
    def _schedule_maintenance(self, severity):
        now = datetime.now()
        schedule = {
            'NORMAL':   (now + timedelta(days=30)).strftime('%Y-%m-%d'),
            'LOW':      (now + timedelta(days=14)).strftime('%Y-%m-%d'),
            'MEDIUM':   (now + timedelta(days=7)).strftime('%Y-%m-%d'),
            'HIGH':     (now + timedelta(days=2)).strftime('%Y-%m-%d'),
            'CRITICAL': now.strftime('%Y-%m-%d'),  # today
        }
        return schedule[severity]
    
    def _estimate_downtime(self, severity):
        downtimes = {
            'NORMAL': '0 hours',
            'LOW': '2-4 hours',
            'MEDIUM': '8-12 hours',
            'HIGH': '1-2 days',
            'CRITICAL': '3-5 days (unplanned failure avoided)'
        }
        return downtimes[severity]
    
    def _estimate_cost_saving(self, severity):
        # Rough estimates based on industry data
        savings = {
            'NORMAL': '$0',
            'LOW': '$500-1,000',
            'MEDIUM': '$5,000-15,000',
            'HIGH': '$50,000-100,000',
            'CRITICAL': '$200,000-500,000 (catastrophic failure avoided)'
        }
        return savings[severity]
    
    def should_retrain(self):
        """Check if model needs retraining — MLOps trigger"""
        if self.total_predictions < 100:
            return False, "Not enough predictions yet"
        
        # If too many alerts recently, might be drift
        recent_alerts = len([a for a in self.alert_history[-50:] 
                             if a['severity'] == 'CRITICAL'])
        
        if recent_alerts > 10:
            return True, "High critical alert rate — possible data drift"
        
        return False, "Model performing normally"