import os
from typing import Optional

from langgraph.prebuilt import create_react_agent

from .. import prompts
from ..deps import TravelDeps
from ..tools import make_maps_tools, make_misc_tools, pick

PLANNING_MODEL = os.environ.get("PLANNING_MODEL", "openai:gpt-4o")


def build_planning_worker(deps: TravelDeps, model: Optional[object] = None):
    """Itinerary optimization & logistics specialist. Assembles research,
    location, and timing findings (already in the shared conversation) into
    a day-by-day itinerary."""
    tools = [
        *pick(make_maps_tools(deps), "get_directions", "geocode_address", "find_places", "validate_address"),
        *pick(make_misc_tools(deps), "get_current_location"),
    ]
    return create_react_agent(
        model=model or PLANNING_MODEL,
        tools=tools,
        name="planning",
        prompt=prompts.load("planning"),
    )
