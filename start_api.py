import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
import json
from datetime import datetime
from src.agent.maintenance_agent import MaintenanceAgent
from src.agent.timeline import predict_failure_timeline

app = FastAPI(title="Edge AI Predictive Maintenance API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
MODEL_PATH = 'models/onnx/model_fp32.onnx'
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
agent = MaintenanceAgent()

@app.get("/")
def root():
    return {
        "status": "running",
        "model": "Dual-Head Transformer",
        "dataset": "NASA Turbofan FD001-FD004",
        "version": "2.0.0"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/simulate")
def simulate(mode: str = "normal", engine_id: int = 1):
    num_sensors = 15
    if mode == "fault":
        data = np.clip(np.random.normal(0.88, 0.05, (1, 30, num_sensors)), 0, 1)
        data[0, :, 1] = np.random.normal(0.95, 0.03, 30)
    elif mode == "warning":
        data = np.clip(np.random.normal(0.55, 0.1, (1, 30, num_sensors)), 0, 1)
    else:
        data = np.clip(np.random.normal(0.3, 0.07, (1, 30, num_sensors)), 0, 1)

    result = session.run(None, {input_name: data.astype(np.float32)})
    prob = float(result[0][0])
    rul = float(result[1][0]) if len(result) > 1 else 50.0
    health_score = round((1 - prob) * 100, 1)

    if prob < 0.3:   severity = 'NORMAL'
    elif prob < 0.5: severity = 'LOW'
    elif prob < 0.7: severity = 'MEDIUM'
    elif prob < 0.9: severity = 'HIGH'
    else:            severity = 'CRITICAL'

    sensor_dict = {f'sensor{i+1}': float(data[0, -1, i]) for i in range(num_sensors)}
    action = agent.analyze_anomaly(prob, sensor_dict, list(sensor_dict.keys()))
    timeline = predict_failure_timeline(rul, prob, engine_id)

    return {
        "engine_id": engine_id,
        "anomaly_probability": round(prob, 4),
        "rul_cycles": round(rul, 1),
        "health_score": health_score,
        "severity": severity,
        "root_cause": action["root_cause"],
        "maintenance_schedule": action["maintenance_schedule"],
        "estimated_downtime": action["estimated_downtime"],
        "cost_saved": action["estimated_cost_saved"],
        "recommended_actions": action["recommended_actions"][:4],
        "timeline": timeline,
        "sensor_data": data[0, -1, :].tolist(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/fleet")
def get_fleet(count: int = 20):
    engines = []
    num_sensors = 15
    for i in range(1, count + 1):
        noise = np.random.random()
        if noise > 0.85:
            data = np.clip(np.random.normal(0.85, 0.07, (1, 30, num_sensors)), 0, 1)
        elif noise > 0.7:
            data = np.clip(np.random.normal(0.55, 0.1, (1, 30, num_sensors)), 0, 1)
        else:
            data = np.clip(np.random.normal(0.3, 0.07, (1, 30, num_sensors)), 0, 1)

        result = session.run(None, {input_name: data.astype(np.float32)})
        prob = float(result[0][0])
        rul = float(result[1][0]) if len(result) > 1 else 50.0

        if prob < 0.3:   sev = 'NORMAL'
        elif prob < 0.5: sev = 'LOW'
        elif prob < 0.7: sev = 'MEDIUM'
        elif prob < 0.9: sev = 'HIGH'
        else:            sev = 'CRITICAL'

        engines.append({
            "engine_id": i,
            "anomaly_probability": round(prob, 4),
            "rul_cycles": round(rul, 1),
            "health_score": round((1 - prob) * 100, 1),
            "severity": sev
        })

    engines.sort(key=lambda x: x["anomaly_probability"], reverse=True)
    return {"engines": engines, "total": count}

@app.get("/metadata")
def get_metadata():
    path = 'data/processed/model_metadata.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

@app.get("/evaluation")
def get_evaluation():
    path = 'data/processed/evaluation_results.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}