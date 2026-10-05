from datetime import datetime
from typing import Any, Dict, List

from langchain_core.tools import BaseTool, tool

from ..deps import TravelDeps


def make_misc_tools(deps: TravelDeps) -> List[BaseTool]:
    """Build the location/time utility tools, closed over this run's `deps`."""

    @tool
    async def get_current_location() -> Dict[str, Any]:
        """Get the current location of the user based on IP address."""
        try:
            response = await deps.http.get("https://ipinfo.io/json")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Could not get current location: {str(e)}"}

    @tool
    def get_current_date_time() -> str:
        """Get the current date and time in ISO format."""
        return datetime.now().isoformat()

    return [get_current_location, get_current_date_time]
