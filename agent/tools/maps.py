from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool, tool

from ..deps import TravelDeps

_NOT_CONFIGURED = {"error": "Google Maps is not configured (set GOOGLE_MAPS_API_KEY)"}


def make_maps_tools(deps: TravelDeps) -> List[BaseTool]:
    """Build the Google Maps tools (geocoding, directions, places, address
    validation), each closed over this run's `deps` for dependency injection."""

    @tool
    def geocode_address(address: str) -> List[Dict[str, Any]]:
        """Geocode an address to get coordinates and location details."""
        if deps.gmaps is None:
            return [_NOT_CONFIGURED]
        try:
            return deps.gmaps.geocode(address)
        except Exception as e:
            return [{"error": f"Geocoding failed: {str(e)}"}]

    @tool
    def reverse_geocode_coordinates(latitude: float, longitude: float) -> List[Dict[str, Any]]:
        """Reverse geocode coordinates to get address information."""
        if deps.gmaps is None:
            return [_NOT_CONFIGURED]
        try:
            return deps.gmaps.reverse_geocode((latitude, longitude))
        except Exception as e:
            return [{"error": f"Reverse geocoding failed: {str(e)}"}]

    @tool
    def get_directions(origin: str, destination: str, mode: str = "driving") -> List[Dict[str, Any]]:
        """Get directions between two locations. Mode: driving, walking, bicycling, or transit."""
        if deps.gmaps is None:
            return [_NOT_CONFIGURED]
        try:
            return deps.gmaps.directions(
                origin, destination, mode=mode, departure_time=datetime.now()
            )
        except Exception as e:
            return [{"error": f"Directions failed: {str(e)}"}]

    @tool
    def validate_address(addresses: List[str], region_code: str = "US") -> Dict[str, Any]:
        """Validate addresses using Google Maps Address Validation API."""
        if deps.gmaps is None:
            return _NOT_CONFIGURED
        try:
            return deps.gmaps.addressvalidation(addresses, regionCode=region_code)
        except Exception as e:
            return {"error": f"Address validation error: {str(e)}"}

    @tool
    def find_places(
        query: str, location: Optional[str] = None, radius: int = 5000
    ) -> List[Dict[str, Any]]:
        """Search for places of interest (e.g. "restaurants", "museums") via Google Places.
        `location`, if given, is "latitude,longitude" to center the search."""
        if deps.gmaps is None:
            return [_NOT_CONFIGURED]
        try:
            if location:
                lat, lng = map(float, location.split(","))
                return deps.gmaps.places(
                    query=query, location=(lat, lng), radius=radius
                ).get("results", [])
            else:
                return deps.gmaps.places(query=query).get("results", [])
        except Exception as e:
            return [{"error": f"Places search failed: {str(e)}"}]

    return [geocode_address, reverse_geocode_coordinates, get_directions, validate_address, find_places]
