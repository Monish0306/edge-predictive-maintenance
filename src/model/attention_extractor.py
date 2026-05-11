import numpy as np
import sys
sys.path.append('.')

# Lazy imports — torch may not be available in ONNX-only (Dash) environments
try:
    import torch
    from src.model.transformer_model import PredMaintenanceTransformer
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# NASA Turbofan sensor names for FD001
SENSOR_NAMES = {
    'sensor1':  'Fan Inlet Temp',
    'sensor2':  'LPC Outlet Temp', 
    'sensor3':  'HPC Outlet Temp',
    'sensor4':  'LPT Outlet Temp',
    'sensor5':  'Fan Inlet Pressure',
    'sensor6':  'Bypass Duct Pressure',
    'sensor7':  'HPC Outlet Pressure',
    'sensor8':  'Physical Fan Speed',
    'sensor9':  'Physical Core Speed',
    'sensor10': 'Engine Pressure Ratio',
    'sensor11': 'HPC Outlet Static Pressure',
    'sensor12': 'Fuel Flow Ratio',
    'sensor13': 'Corrected Fan Speed',
    'sensor14': 'Corrected Core Speed',
    'sensor15': 'Bypass Ratio',
}

class AttentionExtractor:
    """
    Extracts attention weights from Transformer to show
    WHICH sensors caused the anomaly detection.
    This is the explainability layer — like BMW's heatmap.
    """
    
    def __init__(self, num_sensors=15):
        self.num_sensors = num_sensors
        self.attention_weights = []
        self.model = None
        self._load_model()
    
    def _load_model(self):
        if not _TORCH_AVAILABLE:
            self.model = None
            return
        try:
            self.model = PredMaintenanceTransformer(
                num_sensors=self.num_sensors
            )
            self.model.load_state_dict(
                torch.load('models/saved/best_model.pth',
                          map_location='cpu',
                          weights_only=True)
            )
            self.model.eval()
            self._register_hooks()
        except Exception as e:
            print(f"Model load error: {e}")
            self.model = None
    
    def _register_hooks(self):
        """Register hooks to capture attention weights during forward pass"""
        self.attention_weights = []
        
        def hook_fn(module, input, output):
            # output[1] contains attention weights when need_weights=True
            if isinstance(output, tuple) and len(output) > 1:
                if output[1] is not None:
                    self.attention_weights.append(
                        output[1].detach().numpy()
                    )
        
        # Register on each attention layer
        for layer in self.model.transformer.layers:
            layer.self_attn.register_forward_hook(hook_fn)
    
    def get_sensor_importance(self, sensor_data):
        """
        Get importance score for each sensor.
        
        Args:
            sensor_data: numpy array (1, 30, num_sensors)
        
        Returns:
            dict with sensor importance scores 0-1
        """
        self.attention_weights = []
        
        if self.model is None:
            # Fallback: use input variance as proxy for importance
            return self._variance_based_importance(sensor_data)
        
        try:
            if not _TORCH_AVAILABLE:
                return self._variance_based_importance(sensor_data)
            with torch.no_grad():
                x = torch.FloatTensor(sensor_data)
                # Enable attention weight output
                for layer in self.model.transformer.layers:
                    layer.self_attn.need_weights = True
                
                _ = self.model(x)
            
            if self.attention_weights:
                # Average attention across all layers and heads
                # Shape: (layers, batch, heads, seq, seq)
                all_attn = np.stack(self.attention_weights)
                avg_attn = all_attn.mean(axis=(0, 1, 2))  # (seq, seq)
                
                # Get attention each time step receives
                time_importance = avg_attn.mean(axis=0)  # (seq,)
                
                # Map to sensors using input correlation
                sensor_importance = self._map_to_sensors(
                    sensor_data, time_importance
                )
            else:
                sensor_importance = self._variance_based_importance(
                    sensor_data
                )
        
        except Exception:
            sensor_importance = self._variance_based_importance(
                sensor_data
            )
        
        return sensor_importance
    
    def _map_to_sensors(self, sensor_data, time_importance):
        """Map time-step importance to sensor importance"""
        # sensor_data: (1, 30, num_sensors)
        data = sensor_data[0]  # (30, num_sensors)
        
        # Weight each sensor by how much it varies in
        # high-attention time steps
        weights = time_importance / (time_importance.sum() + 1e-8)
        weighted_data = (data.T * weights).T  # (30, sensors)
        
        # Sensor importance = weighted variance
        sensor_scores = weighted_data.var(axis=0)
        
        # Normalize to 0-1
        if sensor_scores.max() > 0:
            sensor_scores = sensor_scores / sensor_scores.max()
        
        return self._format_scores(sensor_scores)
    
    def _variance_based_importance(self, sensor_data):
        """Fallback: use sensor variance as importance proxy"""
        data = sensor_data[0]  # (30, num_sensors)
        scores = data.var(axis=0)
        if scores.max() > 0:
            scores = scores / scores.max()
        return self._format_scores(scores)
    
    def _format_scores(self, scores):
        """Format scores with sensor names"""
        result = {}
        for i, score in enumerate(scores):
            sensor_key = f'sensor{i+1}'
            name = SENSOR_NAMES.get(
                sensor_key, f'Sensor {i+1}'
            )
            result[sensor_key] = {
                'name': name,
                'importance': round(float(score), 4),
                'importance_pct': round(float(score) * 100, 1)
            }
        
        # Sort by importance
        result = dict(sorted(
            result.items(),
            key=lambda x: x[1]['importance'],
            reverse=True
        ))
        return result


if __name__ == '__main__':
    extractor = AttentionExtractor(num_sensors=15)
    test_data = np.random.randn(1, 30, 15).astype(np.float32)
    scores = extractor.get_sensor_importance(test_data)
    print("\nSensor Importance Scores:")
    for sensor, info in list(scores.items())[:5]:
        print(f"  {info['name']}: {info['importance_pct']}%")