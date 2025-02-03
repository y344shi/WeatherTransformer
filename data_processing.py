import pandas as pd

# Suppose you collected data into a list:
data_records = []
for _ in range(24):  # In practice, collect 24 hours of data (or load historical data)
    record = fetch_weather_data(lat, lon, api_key)
    if record:
        data_records.append(record)
    # Sleep or schedule next fetch
    # time.sleep(3600)

df = pd.DataFrame(data_records)
df.sort_values(by="timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head())

