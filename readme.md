# Weather Forecasting Transformer Project

This project is an initial implementation of a transformer-based neural network for weather forecasting. It fetches weather data from an API, preprocesses the data into time-series sequences, builds a transformer model in PyTorch, and includes a training loop with checkpoint saving and loading capabilities.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Modules and Their Functions](#modules-and-their-functions)
  - [1. WeatherDataFetcher](#1-weatherdatafetcher)
  - [2. DataPreprocessor](#2-datapreprocessor)
  - [3. Transformer Model](#3-transformer-model)
    - [PositionalEncoding](#positionalencoding)
    - [WeatherTransformer](#weathertransformer)
  - [4. Trainer](#4-trainer)
  - [5. Main Script](#5-main-script)
- [Adjustable Parameters and How to Modify](#adjustable-parameters-and-how-to-modify)
- [Sample Usage](#sample-usage)
- [Next Steps](#next-steps)

## Overview

This project aims to predict the next few hours of weather conditions (temperature, humidity, wind speed, and wind direction) by using the past 24 hours of weather data. The solution is modularized into data fetching, preprocessing, model building, and training components. It is designed for easy modification and expansion.

## Project Structure

All the code is contained within a single Python script (for simplicity in this initial version). The main subparts include:
- **WeatherDataFetcher:** Retrieves weather data from an external API.
- **DataPreprocessor:** Converts the raw data into a Pandas DataFrame and creates input/output sequences.
- **Transformer Model:** Contains the `PositionalEncoding` and `WeatherTransformer` classes, which build a transformer model for forecasting.
- **Trainer:** Manages the training loop, saving, and loading of checkpoints.
- **Main Function:** Ties all the parts together.

## Modules and Their Functions

### 1. WeatherDataFetcher

- **Purpose:**  
  Fetches current weather data from OpenWeatherMap (or similar APIs).  
- **Key Functionality:**  
  - Uses HTTP GET requests to obtain weather details (temperature, humidity, wind speed, wind direction, etc.).
  - Parses JSON responses and extracts relevant information.
- **Modifiable Parameters:**  
  - API endpoint, latitude, longitude, and API key.
- **Sample Code:**  
  ```python
  fetcher = WeatherDataFetcher(lat=40.7128, lon=-74.0060, api_key="YOUR_ACTUAL_API_KEY")
  record = fetcher.fetch()
  
### 2. DataPreprocessor
- **Purpose:**  
Processes the collected weather records into a structured format.
- **Key Functionality:**  
Creates a Pandas DataFrame from raw data.
Sorts the data by timestamp.
Creates input (past 24 hours) and output sequences (next hour) for the model.
- **Modifiable Parameters:**  
input_window: Number of past hours to consider.
forecast_horizon: Number of future hours to predict.
- **Sample Code:**  
df = DataPreprocessor.create_dataframe(data_records)
X, y = DataPreprocessor.create_sequences(df, input_window=24, forecast_horizon=1)

### 3. Transformer Model
PositionalEncoding
- **Purpose:**  
Adds positional information to the input embeddings.
- **Key Functionality:**  
Generates sinusoidal positional encodings.
Ensures the transformer can understand the order of the time series.
- **Modifiable Parameters:**  
d_model: Embedding dimension.
max_len: Maximum length of the input sequence.

### WeatherTransformer
- **Purpose:**  
Implements the transformer-based model for weather forecasting.
- **Key Functionality:**  
Maps raw input features to a higher-dimensional space.
Applies positional encoding and transformer encoder layers.
Predicts future weather features from the encoded sequence.
- **Modifiable Parameters:**  
d_model, nhead, num_layers, dim_feedforward, and dropout.
forecast_horizon: How many future time steps to predict.
- **Sample Code:**  
model = WeatherTransformer(feature_size=4, forecast_horizon=1, d_model=64, nhead=4, num_layers=2)

### 4. Trainer
- **Purpose:**
Encapsulates the training loop, checkpoint saving, and loading.
- **Key Functionality:**
Trains the model on the provided data.
Saves checkpoints at configurable intervals.
Loads a saved checkpoint to resume training or for inference.
- **Modifiable Parameters:**
num_epochs: Number of training epochs.
save_every: Frequency of checkpoint saving.
- **Sample Code:**
trainer = Trainer(model, train_loader, criterion, optimizer, device)
trainer.train(num_epochs=10, save_every=5)
# To load a checkpoint:
# trainer.load_checkpoint("checkpoints/checkpoint_epoch_5.pth")

