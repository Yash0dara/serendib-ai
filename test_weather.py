# test_weather.py

import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENWEATHER_API_KEY")

if not api_key:
    print("❌ OPENWEATHER_API_KEY not found in .env")
    exit()

def get_weather(city: str):
    url = f"http://api.openweathermap.org/data/2.5/weather"
    
    params = {
        "q": f"{city},LK",
        "appid": api_key,
        "units": "metric"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if response.status_code == 200:
        print(f"\n✅ Weather in {city}, Sri Lanka:")
        print(f"   Condition  : {data['weather'][0]['description']}")
        print(f"   Temperature: {data['main']['temp']}°C")
        print(f"   Humidity   : {data['main']['humidity']}%")
        print(f"   Wind Speed : {data['wind']['speed']} m/s")
    else:
        print(f"❌ Error: {data.get('message', 'Unknown error')}")

# Test with Sri Lanka cities
get_weather("Colombo")
get_weather("Kandy")
get_weather("Galle")