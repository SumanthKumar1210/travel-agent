import os
from typing import Optional

from langgraph.prebuilt import create_react_agent

from .. import prompts
from ..deps import TravelDeps
from ..tools import make_maps_tools, make_misc_tools, pick

LOCATION_MODEL = os.environ.get("LOCATION_MODEL", "openai:gpt-4o")


def build_location_worker(deps: TravelDeps, model: Optional[object] = None):
    """Geographic intelligence & spatial analysis specialist."""
    tools = [
        *make_maps_tools(deps),  # geocode, reverse_geocode, directions, validate_address, find_places
        *pick(make_misc_tools(deps), "get_current_location"),
    ]
    return create_react_agent(
        model=model or LOCATION_MODEL,
        tools=tools,
        name="location",
        prompt=prompts.load("location"),
    )
