import torch
import onnx
import onnxruntime as ort
import numpy as np
import pickle
import os
import sys
sys.path.append('.')

from src.model.transformer_model import PredMaintenanceTransformer

def convert_to_onnx():
    # ── AUTO-DETECT number of sensors from saved data ──
    print("Detecting number of sensors from processed data...")
    X = np.load('data/processed/X_train.npy')
    num_sensors = X.shape[2]  # shape is (samples, 30, num_sensors)
    print(f"  Detected: {num_sensors} sensors")

    print("Loading trained model...")
    model = PredMaintenanceTransformer(num_sensors=num_sensors)
    
    # Fix for newer PyTorch versions
    model.load_state_dict(
        torch.load('models/saved/best_model.pth', 
                   map_location=torch.device('cpu'),
                   weights_only=True)
    )
    model.eval()
    
    # Dummy input using correct sensor count
    dummy_input = torch.randn(1, 30, num_sensors)
    
    os.makedirs('models/onnx', exist_ok=True)
    onnx_path = 'models/onnx/model.onnx'
    
    print("Converting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['sensor_data'],
        output_names=['anomaly_prob'],
        dynamic_axes={
            'sensor_data': {0: 'batch_size'},
            'anomaly_prob': {0: 'batch_size'}
        },
        opset_version=11
    )
    
    print(f"ONNX model saved: {onnx_path}")
    
    # Verify it works
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified ✓")
    
    # Compare file sizes
    pytorch_size = os.path.getsize('models/saved/best_model.pth') / 1024
    onnx_size = os.path.getsize(onnx_path) / 1024
    reduction = (1 - onnx_size/pytorch_size) * 100
    
    print(f"\n📊 File Size Comparison:")
    print(f"  PyTorch model : {pytorch_size:.1f} KB")
    print(f"  ONNX model    : {onnx_size:.1f} KB")
    print(f"  Size reduction: {reduction:.1f}%")
    
    # Speed test — this proves edge readiness
    print(f"\n⚡ Inference Speed Test (100 runs)...")
    ort_session = ort.InferenceSession(onnx_path)
    test_input = np.random.randn(1, 30, num_sensors).astype(np.float32)
    
    import time
    times = []
    for _ in range(100):
        start = time.time()
        ort_session.run(None, {'sensor_data': test_input})
        times.append((time.time() - start) * 1000)
    
    avg_ms = np.mean(times)
    print(f"  Average latency : {avg_ms:.2f} ms")
    print(f"  Min latency     : {np.min(times):.2f} ms")
    print(f"  Edge requirement: <50ms")
    print(f"  Status: {'✅ PASSES' if avg_ms < 50 else '❌ too slow'} edge requirement!")
    
    # Save num_sensors for dashboard to use
    with open('data/processed/num_sensors.txt', 'w') as f:
        f.write(str(num_sensors))
    
    print(f"\n✅ ONNX conversion complete!")
    print(f"   num_sensors={num_sensors} saved for dashboard")

if __name__ == '__main__':
    convert_to_onnx()