from typing import List

from langchain_core.tools import BaseTool

from .maps import make_maps_tools
from .misc import make_misc_tools
from .weather import make_weather_tools
from .web import make_web_tools

__all__ = [
    "make_maps_tools",
    "make_weather_tools",
    "make_web_tools",
    "make_misc_tools",
    "pick",
]


def pick(tools: List[BaseTool], *names: str) -> List[BaseTool]:
    """Select specific tools by name from a factory's output, in the given order."""
    by_name = {t.name: t for t in tools}
    return [by_name[name] for name in names]
