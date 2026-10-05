import os
from typing import Optional

from langgraph.prebuilt import create_react_agent

from .. import prompts
from ..deps import TravelDeps
from ..tools import make_maps_tools, make_misc_tools, make_web_tools, pick

RESEARCH_MODEL = os.environ.get("RESEARCH_MODEL", "openai:gpt-4o")


def build_research_worker(deps: TravelDeps, model: Optional[object] = None):
    """Destination research & cultural analysis specialist."""
    tools = [
        *make_web_tools(deps),
        *pick(make_maps_tools(deps), "find_places", "geocode_address"),
        *pick(make_misc_tools(deps), "get_current_location"),
    ]
    return create_react_agent(
        model=model or RESEARCH_MODEL,
        tools=tools,
        name="research",
        prompt=prompts.load("research"),
    )
