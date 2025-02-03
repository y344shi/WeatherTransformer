import requests
import json
from datetime import datetime

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

# Example usage:
lat = 40.7128  # New York City latitude
lon = -74.0060  # New York City longitude
api_key = "816f139db383e0f0d8d40761194e4a36"  # Replace with your API key
weather_data = WeatherDataFetcher(lat, lon, api_key)
print(weather_data)
# curl "http://api.openweathermap.org/data/2.5/weather?lat=40.7128&lon=-74.0060&appid=816f139db383e0f0d8d40761194e4a36&units=metric"
