from typing import Dict, List

from langchain_core.tools import BaseTool, tool

from ..deps import TravelDeps


def make_web_tools(deps: TravelDeps) -> List[BaseTool]:
    """Build the web-search tool, closed over this run's `deps`."""

    @tool
    def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Search the web using Google Custom Search."""
        if deps.search is None:
            return [{"error": "Web search is not configured (set GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID)"}]
        try:
            return deps.search.results(query, num_results=num_results)
        except Exception as e:
            return [{"error": f"Web search failed: {str(e)}"}]

    return [search_web]
