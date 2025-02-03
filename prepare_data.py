import requests
import json
from datetime import datetime

def fetch_weather_data(lat, lon, api_key):
    """
    Fetch current weather data from OpenWeatherMap API for the given latitude and longitude.
    """
    url = (
        # f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Extract some example fields (expand as needed)
        weather_info = {
            "timestamp": datetime.utcfromtimestamp(data["dt"]),
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "wind_deg": data["wind"].get("deg", 0)  # wind direction
        }
        return weather_info
    else:
        print("Error fetching data:", response.status_code)
        return None

# Example usage:
lat = 40.7128  # New York City latitude
lon = -74.0060  # New York City longitude
api_key = "816f139db383e0f0d8d40761194e4a36"  # Replace with your API key
weather_data = fetch_weather_data(lat, lon, api_key)
print(weather_data)
# curl "http://api.openweathermap.org/data/2.5/weather?lat=40.7128&lon=-74.0060&appid=816f139db383e0f0d8d40761194e4a36&units=metric"
