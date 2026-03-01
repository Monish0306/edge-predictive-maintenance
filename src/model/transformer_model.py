import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Helps Transformer understand time order"""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PredMaintenanceTransformer(nn.Module):
    """
    Lightweight Transformer for anomaly detection
    Input: (batch, seq_len, num_sensors)
    Output: (batch, 1) — anomaly probability
    """
    def __init__(self, 
                 num_sensors=14,    # input features
                 d_model=32,        # small = lightweight for edge
                 nhead=4,           # attention heads
                 num_layers=2,      # transformer layers
                 dropout=0.1):
        super().__init__()
        
        # Project sensors to d_model dimension
        self.input_projection = nn.Linear(num_sensors, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,  # small for edge
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, 
                                                  num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()  # output between 0-1
        )
        
    def forward(self, x):
        # x: (batch, seq_len, num_sensors)
        x = self.input_projection(x)      # → (batch, seq_len, d_model)
        x = self.pos_encoding(x)           # add position info
        x = self.transformer(x)            # → (batch, seq_len, d_model)
        x = x.mean(dim=1)                  # average over time → (batch, d_model)
        out = self.classifier(x)           # → (batch, 1)
        return out.squeeze(-1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # Quick test
    model = PredMaintenanceTransformer(num_sensors=14)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Fake input test
    x = torch.randn(8, 30, 14)  # batch=8, seq=30, sensors=14
    out = model(x)
    print(f"Output shape: {out.shape}")  # should be (8,)
    print(f"Output values: {out}")