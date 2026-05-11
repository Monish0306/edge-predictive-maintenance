from datetime import datetime, timedelta
import numpy as np

# Average cycles per day for NASA Turbofan dataset
# Based on dataset: engines run ~1 cycle per flight
CYCLES_PER_DAY = 1.0

def predict_failure_timeline(rul_cycles, anomaly_prob, engine_id=1):
    """
    Convert RUL cycles into human-readable failure timeline.
    
    Args:
        rul_cycles: predicted remaining cycles (from model)
        anomaly_prob: anomaly probability (0-1)
        engine_id: which engine number
    
    Returns:
        timeline dict with dates, confidence, recommendations
    """
    today = datetime.now()
    
    # Convert cycles to days
    rul_days = max(0, rul_cycles / CYCLES_PER_DAY)
    
    # Confidence interval based on anomaly probability
    # Higher anomaly prob = less certain about RUL
    uncertainty = 0.2 + (anomaly_prob * 0.3)
    lower_days = max(0, rul_days * (1 - uncertainty))
    upper_days = rul_days * (1 + uncertainty)
    
    # Calculate dates
    predicted_failure = today + timedelta(days=rul_days)
    earliest_failure  = today + timedelta(days=lower_days)
    latest_failure    = today + timedelta(days=upper_days)
    
    # Recommended maintenance window (before failure)
    if rul_days > 30:
        maintenance_date = today + timedelta(days=rul_days - 14)
        urgency = "PLANNED"
        urgency_color = "green"
    elif rul_days > 14:
        maintenance_date = today + timedelta(days=rul_days - 7)
        urgency = "SOON"
        urgency_color = "orange"
    elif rul_days > 7:
        maintenance_date = today + timedelta(days=2)
        urgency = "URGENT"
        urgency_color = "red"
    else:
        maintenance_date = today
        urgency = "CRITICAL — TODAY"
        urgency_color = "darkred"
    
    # Degradation rate (% per day)
    if rul_days > 0:
        degradation_rate = round((anomaly_prob / max(rul_days, 1)) * 100, 3)
    else:
        degradation_rate = 100.0
    
    return {
        'engine_id': engine_id,
        'rul_cycles': round(float(rul_cycles), 1),
        'rul_days': round(rul_days, 1),
        'predicted_failure_date': predicted_failure.strftime('%Y-%m-%d'),
        'earliest_failure_date': earliest_failure.strftime('%Y-%m-%d'),
        'latest_failure_date': latest_failure.strftime('%Y-%m-%d'),
        'recommended_maintenance': maintenance_date.strftime('%Y-%m-%d'),
        'urgency': urgency,
        'urgency_color': urgency_color,
        'confidence_pct': round((1 - uncertainty) * 100, 1),
        'degradation_rate_per_day': degradation_rate,
        'anomaly_prob': round(float(anomaly_prob), 4),
        'days_until_maintenance': max(0, (maintenance_date - today).days),
    }


def get_timeline_milestones(rul_days):
    """
    Generate key milestone dates for visual timeline.
    """
    today = datetime.now()
    milestones = []
    
    milestones.append({
        'label': '📍 Today',
        'date': today.strftime('%b %d'),
        'days': 0,
        'color': '#3b82f6'
    })
    
    if rul_days > 14:
        inspection = today + timedelta(days=min(7, rul_days * 0.3))
        milestones.append({
            'label': '🔍 Inspect',
            'date': inspection.strftime('%b %d'),
            'days': round(inspection.day - today.day),
            'color': '#f59e0b'
        })
    
    if rul_days > 7:
        parts_order = today + timedelta(days=max(1, rul_days - 10))
        milestones.append({
            'label': '📦 Order Parts',
            'date': parts_order.strftime('%b %d'),
            'days': max(1, round(rul_days - 10)),
            'color': '#8b5cf6'
        })
    
    maintenance = today + timedelta(days=max(1, rul_days - 5))
    milestones.append({
        'label': '🔧 Maintain',
        'date': maintenance.strftime('%b %d'),
        'days': max(1, round(rul_days - 5)),
        'color': '#22c55e'
    })
    
    failure = today + timedelta(days=rul_days)
    milestones.append({
        'label': '💀 Est. Failure',
        'date': failure.strftime('%b %d'),
        'days': round(rul_days),
        'color': '#ef4444'
    })
    
    return milestones