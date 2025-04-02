import requests
import time
from datetime import datetime
import pandas as pd
from datetime import datetime
import pytz

FILE = "/mnt/d/446Project/WeatherTransformer/data_processing/raw_data/openweather/no_fires_augmented_fuel1.csv"
OUTPUT_PATH = "/mnt/d/446Project/WeatherTransformer/data_processing/raw_data/openweather/OpenWeatherHistoryNonFireData.csv"

def toronto_to_utc(toronto_time_str, time_format="%Y-%m-%d %H:%M:%S"):
    """
    Converts Toronto local time string to UTC datetime.
    
    Parameters:
    - toronto_time_str: A string like "2020-07-01 00:00:00"
    - time_format: Format of the input string

    Returns:
    - A UTC datetime object
    """
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H",
        "%Y-%m-%d"
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(str(toronto_time_str).strip(), fmt)
            break
        except ValueError:
            continue
    else:
        return 0

    # Localize to Toronto and convert to UTC
    toronto_tz = pytz.timezone("America/Toronto")
    local_dt = toronto_tz.localize(dt)
    utc_dt = local_dt.astimezone(pytz.utc)

    return int(utc_dt.timestamp())


def fetch_weather_history(api_key, locations, start_hour_unix, save_path = OUTPUT_PATH, count=24, save_every=100):
    """
    Fetch historical weather data from OpenWeatherMap API for a list of coordinates.

    Parameters:
    -----------
    api_key : str
        Your OpenWeatherMap API key.
    locations : list of tuples
        List of (latitude, longitude, time) pairs.
    start_hour_unix : int
        Start time in Unix timestamp (seconds).
    count : int
        Number of hourly data points to retrieve (max = 24).

    Returns:
    --------
    pd.DataFrame
        Combined DataFrame of results.
    """
    base_url = "https://history.openweathermap.org/data/2.5/history/city"

    all_data = []

    index = 0

    for lat, lon, time in locations:
        try:
            if (toronto_to_utc(time) == 0):
                print(f"Skipped {lat, lon, time} due to timestamp: {time}")
                continue
            
            params = {
                "lat": lat,
                "lon": lon,
                "type": "hour",
                "start": toronto_to_utc(time),
                "cnt": count,
                "appid": api_key
            }

            print(f"Fetching data for ({lat}, {lon})...")
            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()

                if 'list' in data:
                    for entry in data['list']:
                        all_data.append({
                            "lat": lat,
                            "lon": lon,
                            "timestamp": entry["dt"],
                            "datetime": datetime.utcfromtimestamp(entry["dt"]),
                            "temp": entry["main"].get("temp"),
                            "pressure": entry["main"].get("pressure"),
                            "humidity": entry["main"].get("humidity"),
                            "wind_speed": entry["wind"].get("speed"),
                            "wind_deg": entry["wind"].get("deg"),
                            "weather": entry["weather"][0]["description"] if entry["weather"] else None
                        })
                    if (index + 1) % save_every == 0:
                        results = pd.DataFrame(all_data)
                        results.to_csv(save_path, index=False)
                        print(f"Saved progress at row {index+1} to {save_path}")
                    index += 1
                else:
                    print(f"No data returned for ({lat}, {lon})")

            except Exception as e:
                print(f"Error fetching data for ({lat}, {lon}): {e}")
        except:
            print(f"Skipped {lat, lon, time} due to timestamp: {time}")
            continue


    return pd.DataFrame(all_data)



if __name__ == "__main__":
    # Replace with your OpenWeatherMap API key
    API_KEY = "1cb3b3071f7874253f660c2be5032a2a"

    df = pd.read_csv(FILE, encoding='utf-8', low_memory=False, skiprows=0)

    # List of (lat, lon)
    toronto_time = "2024-08-04 15:02:00"
    utc_time = toronto_to_utc(toronto_time)
    print("UTC Time:", utc_time)


    location_list = list(df[["LATITUDE", "LONGITUDE", "FIRE_START_DATE"]].itertuples(index=False, name=None))
    location_list[:len(location_list) // 2]
    print(location_list)


    # Start time (e.g., 24 hours ago)
    start_time_unix = int(time.time()) - (24 * 3600)

    df_weather = fetch_weather_history(
        api_key=API_KEY,
        locations=location_list,
        start_hour_unix=start_time_unix,
        count=24  # Max = 24 hours
    )

    print(toronto_to_utc("2024-08-20 15:30:00"))

    # # Display result
    print(df_weather)



    # location_list = [
    #     (55.724083, -119.269417, "2023-08-04 15:02:00"),
    #     (56.019633, -113.366133, "2023-08-20 15:30:00")
    # ]

    # # Start time (e.g., 24 hours ago)
    # start_time_unix = int(time.time()) - (24 * 3600)

    # df_weather = fetch_weather_history(
    #     api_key=API_KEY,
    #     locations=location_list,
    #     start_hour_unix=start_time_unix,
    #     count=24  # Max = 24 hours
    # )

    # print(toronto_to_utc("2024-08-20 15:30:00"))

    # # # Display result
    # print(df_weather)

    # print()
