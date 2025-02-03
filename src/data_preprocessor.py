import pandas as pd
import numpy as np

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