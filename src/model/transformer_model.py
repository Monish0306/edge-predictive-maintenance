import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PredMaintenanceTransformer(nn.Module):
    """
    ONNX-compatible Dual-Head Transformer:
    - Head 1: Anomaly Detection (binary)
    - Head 2: RUL Prediction (regression)
    """
    def __init__(self, num_sensors=15, d_model=32,
                 nhead=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.input_projection = nn.Linear(num_sensors, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        # ← KEY: use_default_feed_forward=False for ONNX compatibility
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=dropout,
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False  # ← critical for ONNX export
        )

        # Head 1 — Anomaly classification
        self.anomaly_head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

        # Head 2 — RUL regression
        self.rul_head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        anomaly_logit = self.anomaly_head(x).squeeze(-1)
        rul_pred      = self.rul_head(x).squeeze(-1)
        return anomaly_logit, rul_pred

    def predict(self, x):
        logit, rul = self.forward(x)
        return torch.sigmoid(logit), rul * 125.0


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    model = PredMaintenanceTransformer(num_sensors=15)
    print(f"Parameters: {count_parameters(model):,}")
    x = torch.randn(4, 30, 15)
    a, r = model(x)
    print(f"Anomaly: {a.shape}, RUL: {r.shape}")