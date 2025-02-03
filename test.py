#!/usr/bin/env python3
"""
A complete script that fetches weather data, preprocesses it,
builds a transformer-based model, trains the model, and saves the training results.
"""

import requests
import time
from datetime import datetime
import pandas as pd
import numpy as np
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# =============================================================================
# 1. Data Acquisition: WeatherDataFetcher Class
# =============================================================================
class WeatherDataFetcher:
    def __init__(self, lat, lon, api_key):
        self.lat = lat
        self.lon = lon
        self.api_key = api_key

    def fetch(self):
        url = (
            f"http://api.openweathermap.org/data/2.5/weather?lat={self.lat}&lon={self.lon}&appid={self.api_key}&units=metric"
        )
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather_info = {
                "timestamp": datetime.utcfromtimestamp(data["dt"]),
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"],
                "wind_deg": data["wind"].get("deg", 0),
            }
            return weather_info
        else:
            print("Error fetching data:", response.status_code)
            return None


# =============================================================================
# 2. Data Preprocessing: DataPreprocessor Class
# =============================================================================
class DataPreprocessor:
    @staticmethod
    def create_dataframe(data_records):
        df = pd.DataFrame(data_records)
        df.sort_values(by="timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def create_sequences(df, input_window=24, forecast_horizon=1):
        features = ["temperature", "humidity", "wind_speed", "wind_deg"]
        data = df[features].values
        X, y = [], []
        total_length = len(data)
        for i in range(total_length - input_window - forecast_horizon + 1):
            X.append(data[i: i + input_window])
            y.append(data[i + input_window: i + input_window + forecast_horizon])
        return np.array(X), np.array(y)


# =============================================================================
# 3. Transformer Model: PositionalEncoding & WeatherTransformer Classes
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class WeatherTransformer(nn.Module):
    def __init__(self, feature_size, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1, forecast_horizon=1):
        super().__init__()
        self.d_model = d_model
        self.feature_size = feature_size
        self.forecast_horizon = forecast_horizon
        
        self.input_linear = nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, forecast_horizon * feature_size)

    def forward(self, src):
        x = self.input_linear(src)  # (batch_size, seq_length, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (seq_length, batch_size, d_model)
        encoded = self.transformer_encoder(x)
        summary = encoded[-1, :, :]  # (batch_size, d_model)
        out = self.fc_out(summary)  # (batch_size, forecast_horizon * feature_size)
        out = out.view(-1, self.forecast_horizon, self.feature_size)
        return out


# =============================================================================
# 4. Training: Trainer Class with Saving/Loading Functionality
# =============================================================================
class Trainer:
    def __init__(self, model, train_loader, criterion, optimizer, device, checkpoint_dir="checkpoints"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, num_epochs, save_every=5):
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            # Save checkpoint every `save_every` epochs
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1, avg_loss)

    def save_checkpoint(self, epoch, loss, filename=None):
        if filename is None:
            filename = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint.get("epoch", 0)
        loss = checkpoint.get("loss", None)
        print(f"Loaded checkpoint from {filename} (Epoch {epoch}, Loss: {loss})")
        return epoch, loss


# =============================================================================
# Main Function: Putting It All Together
# =============================================================================
def main():
    # ---------------------------
    # Parameters & Setup
    # ---------------------------
    lat = 40.7128   # New York City latitude
    lon = -74.0060  # New York City longitude
    api_key = "816f139db383e0f0d8d40761194e4a36"  # Replace with your actual API key

    input_window = 24     # Number of past hours as input
    forecast_horizon = 1  # Number of future hours to predict
    num_records = 24      # Total records to fetch (must be >= input_window + forecast_horizon)
    
    # ---------------------------
    # Data Acquisition
    # ---------------------------
    fetcher = WeatherDataFetcher(lat, lon, api_key)
    data_records = []
    print("Fetching weather data records...")
    for i in range(num_records):
        record = fetcher.fetch()
        if record:
            data_records.append(record)
            print(f"record {i} fetched as {record}")
        time.sleep(1800)  # For simulation/demo purposes only

    if len(data_records) < input_window + forecast_horizon:
        print("Not enough data to create sequences.")
        return

    # ---------------------------
    # Data Preprocessing
    # ---------------------------
    df = DataPreprocessor.create_dataframe(data_records)
    X, y = DataPreprocessor.create_sequences(df, input_window=input_window, forecast_horizon=forecast_horizon)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # ---------------------------
    # Model Setup
    # ---------------------------
    feature_size = 4  # [temperature, humidity, wind_speed, wind_deg]
    model = WeatherTransformer(feature_size=feature_size, forecast_horizon=forecast_horizon)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # Training
    # ---------------------------
    trainer = Trainer(model, train_loader, criterion, optimizer, device)
    num_epochs = 10
    trainer.train(num_epochs)

    # Example: Load a checkpoint later (if needed)
    # checkpoint_path = "checkpoints/checkpoint_epoch_5.pth"
    # trainer.load_checkpoint(checkpoint_path)

if __name__ == "__main__":
    main()
