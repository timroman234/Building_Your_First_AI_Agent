"""
tools.py — The actual functions the agent can call.

Each function here corresponds to a tool definition that we'll give the model.
The model never calls these directly — our agent loop does, based on the
model's tool_call requests.
"""

import json
import requests


def get_weather(city: str) -> str:
    """
    Fetch current weather for a city using the Open-Meteo API.
    
    The process is two steps:
    1. Convert city name → latitude/longitude (geocoding)
    2. Use coordinates → fetch weather data
    
    Returns a JSON string because that's what we'll send back to the model.
    The model is very good at reading JSON and summarizing it naturally.
    """
    
    # --- Step 1: Geocode the city name to coordinates ---
    geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_response = requests.get(geocode_url, params={
        "name": city,
        "count": 1  # We only need the top match
    })
    geo_data = geo_response.json()
    
    # Handle the case where the city isn't found
    if "results" not in geo_data or len(geo_data["results"]) == 0:
        return json.dumps({"error": f"City '{city}' not found."})
    
    location = geo_data["results"][0]
    lat = location["latitude"]
    lon = location["longitude"]
    resolved_name = location.get("name", city)
    country = location.get("country", "")
    
    # --- Step 2: Fetch weather using coordinates ---
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_response = requests.get(weather_url, params={
        "latitude": lat,
        "longitude": lon,
        "current_weather": True  # We only need current conditions
    })
    weather_data = weather_response.json()
    current = weather_data.get("current_weather", {})
    
    # Package the result as a clean JSON string
    result = {
        "city": resolved_name,
        "country": country,
        "temperature_celsius": current.get("temperature"),
        "windspeed_kmh": current.get("windspeed"),
        "weather_code": current.get("weathercode"),
        "description": _weather_code_to_text(current.get("weathercode", -1))
    }
    
    return json.dumps(result)


def _weather_code_to_text(code: int) -> str:
    """
    Convert Open-Meteo's numeric weather code to a human-readable description.
    
    This is a helper function (note the underscore prefix — a Python convention
    meaning 'this is internal, not part of the public interface').
    
    We only map the most common codes. For a full list, see:
    https://open-meteo.com/en/docs
    """
    descriptions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return descriptions.get(code, f"Unknown (code: {code})")

def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert a temperature between Celsius, Fahrenheit, and Kelvin.
    
    This is a 'pure' function — it doesn't call any external API,
    it just does math. Not every tool needs to be an API call.
    Tools can be anything: calculations, file operations, database
    queries, or even calls to other AI models.
    """
    
    # Normalize to lowercase so "Celsius", "celsius", "C" all work
    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()
    
    # Map common variations to standard names
    unit_aliases = {
        "c": "celsius", "celsius": "celsius",
        "f": "fahrenheit", "fahrenheit": "fahrenheit",
        "k": "kelvin", "kelvin": "kelvin",
    }
    
    from_std = unit_aliases.get(from_unit)
    to_std = unit_aliases.get(to_unit)
    
    if not from_std or not to_std:
        return json.dumps({
            "error": f"Unknown unit. Supported: celsius (C), fahrenheit (F), kelvin (K)"
        })
    
    # Convert input to Celsius first (a common pattern — normalize to one
    # unit, then convert out. This avoids writing 6 separate formulas.)
    if from_std == "celsius":
        celsius = value
    elif from_std == "fahrenheit":
        celsius = (value - 32) * 5 / 9
    elif from_std == "kelvin":
        celsius = value - 273.15
    
    # Convert from Celsius to the target unit
    if to_std == "celsius":
        result = celsius
    elif to_std == "fahrenheit":
        result = (celsius * 9 / 5) + 32
    elif to_std == "kelvin":
        result = celsius + 273.15
    
    return json.dumps({
        "input_value": value,
        "input_unit": from_std,
        "output_value": round(result, 2),
        "output_unit": to_std
    })