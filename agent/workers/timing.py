import os
from typing import Optional

from langgraph.prebuilt import create_react_agent

from .. import prompts
from ..deps import TravelDeps
from ..tools import make_maps_tools, make_misc_tools, make_weather_tools, pick

TIMING_MODEL = os.environ.get("TIMING_MODEL", "openai:gpt-4o")


def build_timing_worker(deps: TravelDeps, model: Optional[object] = None):
    """Temporal optimization & weather analysis specialist."""
    tools = [
        *make_weather_tools(deps),
        *pick(make_maps_tools(deps), "geocode_address"),
        *make_misc_tools(deps),  # get_current_location, get_current_date_time
    ]
    return create_react_agent(
        model=model or TIMING_MODEL,
        tools=tools,
        name="timing",
        prompt=prompts.load("timing"),
    )
