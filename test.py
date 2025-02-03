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
# Project Defined Functions:
# - data fetching based on lat lon API_KEY
# - pandas data transformation
# - transformer model definition
# - training loop Trainer
# =============================================================================

from src.data_fetcher import WeatherDataFetcher
from src.data_preprocessor import DataPreprocessor
from src.model import WeatherTransformer
from src.trainer import Trainer

# =============================================================================
# Main Function: Putting It All Together
# =============================================================================
def main():
    # ---------------------------
    # Parameters & Setup
    # ---------------------------
    lat = 40.7128   # New York City latitude
    lon = -74.0060  # New York City longitude
    api_key = "YOUR_API_KEY"  # Replace with your actual API key

    input_window = 24     # Number of past hours as input
    forecast_horizon = 1  # Number of future hours to predict
    num_records = input_window + forecast_horizon      # Total records to fetch (must be >= input_window + forecast_horizon)
    
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
        time.sleep(1)  # For simulation/demo purposes only

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
