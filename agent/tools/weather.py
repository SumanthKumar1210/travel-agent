from typing import Any, Dict, List

from langchain_core.tools import BaseTool, tool

from ..deps import TravelDeps


def make_weather_tools(deps: TravelDeps) -> List[BaseTool]:
    """Build the Google Weather tools, closed over this run's `deps`."""

    @tool
    async def get_current_weather(latitude: float, longitude: float) -> Dict[str, Any]:
        """Get current weather conditions for given coordinates."""
        try:
            params = {
                "key": deps.google_maps_api_key,
                "location.latitude": latitude,
                "location.longitude": longitude,
            }
            url = "https://weather.googleapis.com/v1/currentConditions:lookup"
            response = await deps.http.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Weather API error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Weather request failed: {str(e)}"}

    @tool
    async def get_weather_forecast(latitude: float, longitude: float, days: int = 7) -> Dict[str, Any]:
        """Get weather forecast for the next `days` days for given coordinates."""
        try:
            params = {
                "key": deps.google_maps_api_key,
                "location.latitude": latitude,
                "location.longitude": longitude,
                "days": days,
            }
            url = "https://weather.googleapis.com/v1/forecast/days:lookup"
            response = await deps.http.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Weather API error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Weather forecast failed: {str(e)}"}

    return [get_current_weather, get_weather_forecast]
