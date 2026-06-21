# Handles real time weather queries for Sri Lanka

import os
import requests
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Sri Lanka city coordinates for weather API
SL_CITIES = {
    "colombo":      {"q": "Colombo,LK"},
    "kandy":        {"q": "Kandy,LK"},
    "galle":        {"q": "Galle,LK"},
    "ella":         {"q": "Ella,LK"},
    "nuwara eliya": {"q": "Nuwara Eliya,LK"},
    "trincomalee":  {"q": "Trincomalee,LK"},
    "jaffna":       {"q": "Jaffna,LK"},
    "mirissa":      {"q": "Mirissa,LK"},
    "negombo":      {"q": "Negombo,LK"},
}

DEFAULT_CITY = "Colombo,LK"

WEATHER_PROMPT = """
You are Serendib AI, a Sri Lanka travel assistant.

Based on the weather data and Sri Lanka seasonal knowledge,
provide helpful travel advice.

Include:
- Current conditions summary
- What this means for the traveler
- Any activities affected by weather
- Alternative suggestions if weather is bad
- Best areas to visit given current conditions

Be practical and helpful.
Keep response concise and clear.

Weather data:
{weather_data}

User question:
{query}

Conversation history:
{history}
"""


class WeatherAgent:
    """
    Handles weather queries using:
    - OpenWeatherMap API for real time data
    - LLM for intelligent weather advice
    """

    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0.5,
            max_tokens=512
        )

    def get_weather(self, city_query: str) -> dict:
        """Fetches real time weather from OpenWeatherMap."""
        try:
            url = f"{self.base_url}/weather"
            params = {
                "q": city_query,
                "appid": self.api_key,
                "units": "metric"
            }
            response = requests.get(url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "city": data["name"],
                    "condition": data["weather"][0]["description"],
                    "temperature": data["main"]["temp"],
                    "feels_like": data["main"]["feels_like"],
                    "humidity": data["main"]["humidity"],
                    "wind_speed": data["wind"]["speed"],
                    "visibility": data.get("visibility", "N/A")
                }
            else:
                return {"success": False, "error": "City not found"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def find_city(self, locations: list) -> str:
        """Finds best matching Sri Lanka city from entities."""
        if not locations:
            return DEFAULT_CITY

        for location in locations:
            location_lower = location.lower()
            for city, config in SL_CITIES.items():
                if city in location_lower:
                    return config["q"]

        return DEFAULT_CITY

    def format_weather_data(self, weather: dict) -> str:
        """Formats weather data for LLM prompt."""
        if not weather.get("success"):
            return "Weather data unavailable."

        return f"""
City        : {weather['city']}, Sri Lanka
Condition   : {weather['condition']}
Temperature : {weather['temperature']}°C
Feels Like  : {weather['feels_like']}°C
Humidity    : {weather['humidity']}%
Wind Speed  : {weather['wind_speed']} m/s
"""

    async def process(
        self,
        query: str,
        nlp_result: dict,
        history: str
    ) -> dict:
        """
        Fetches weather and generates
        travel advice based on conditions.
        """
        entities = nlp_result.get("entities", {})
        locations = entities.get("locations", [])

        # Find city to check
        city_query = self.find_city(locations)

        # Fetch real time weather
        weather = self.get_weather(city_query)
        weather_data = self.format_weather_data(weather)

        # Generate travel advice with LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", WEATHER_PROMPT),
            ("human", "{query}")
        ])

        chain = prompt | self.llm

        response = chain.invoke({
            "weather_data": weather_data,
            "query": query,
            "history": history
        })

        return {
            "answer": response.content,
            "sources": ["OpenWeatherMap API", "Sri Lanka Travel Guide"],
            "agent": "weather",
            "weather_data": weather
        }


weather_agent = WeatherAgent()