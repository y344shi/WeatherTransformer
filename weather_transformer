import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        # Create a long enough PEs matrix once
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_length, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return x

class WeatherTransformer(nn.Module):
    def __init__(self, feature_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, forecast_horizon=1):
        """
        feature_size: number of input features per time step (e.g., 4)
        forecast_horizon: number of future time steps to predict
        """
        super().__init__()
        self.d_model = d_model
        # Input embedding: project the raw features to d_model dimensions
        self.input_linear = nn.Linear(feature_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # For forecasting: predict forecast_horizon * feature_size values per sample.
        self.fc_out = nn.Linear(d_model, forecast_horizon * feature_size)
        
    def forward(self, src):
        """
        src: Tensor of shape (batch_size, seq_length, feature_size)
        Returns: Tensor of shape (batch_size, forecast_horizon, feature_size)
        """
        # Embed input
        x = self.input_linear(src)  # (batch_size, seq_length, d_model)
        # Add positional encoding
        x = self.pos_encoder(x)
        # Transformer in PyTorch expects shape (seq_length, batch_size, d_model)
        x = x.transpose(0, 1)
        # Pass through transformer encoder
        encoded = self.transformer_encoder(x)
        # Take the last time step's output as summary (or you can pool over time)
        summary = encoded[-1, :, :]  # (batch_size, d_model)
        out = self.fc_out(summary)  # (batch_size, forecast_horizon * feature_size)
        # Reshape to (batch_size, forecast_horizon, feature_size)
        out = out.view(-1, self.fc_out.out_features // src.size(-1), src.size(-1))
        return out

# Example usage:
batch_size = 16
seq_length = 24  # past 24 hours
feature_size = 4
forecast_horizon = 1

# Create dummy input (for example, a batch from your dataset)
dummy_input = torch.randn(batch_size, seq_length, feature_size)
model = WeatherTransformer(feature_size=feature_size, forecast_horizon=forecast_horizon)
dummy_output = model(dummy_input)
print("Output shape:", dummy_output.shape)  # Expected: (batch_size, forecast_horizon, feature_size)
