import asyncio
import os
from typing import Optional

import httpx

from .deps import TravelDeps, build_gmaps_client, build_search_client
from .supervisor import build_supervisor


class TravelAgent:
    def __init__(self):
        # Built once per TravelAgent instance, lazily — never at module import
        # time — so `import agent` never requires API keys to be set.
        self._gmaps = build_gmaps_client()
        self._search = build_search_client()

    def run(self, query: str, model: Optional[str] = None):
        """
        Run the travel agent with the given query.

        Args:
            query: User's travel query
            model: Optional model override for the Supervisor only.
                Worker agents keep their own configured models.

        Returns:
            Tuple of (response, full_message_history)
        """
        try:
            return asyncio.run(self._run_async(query, model))
        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            return error_msg, []

    async def _run_async(self, query: str, model: Optional[str]):
        async with httpx.AsyncClient(timeout=30) as http:
            deps = TravelDeps(
                gmaps=self._gmaps,
                search=self._search,
                http=http,
                google_maps_api_key=os.environ.get("GOOGLE_MAPS_API_KEY", ""),
            )
            supervisor = build_supervisor(deps, model=model)
            result = await supervisor.ainvoke({"messages": [{"role": "user", "content": query}]})
            return result["messages"][-1].content, result["messages"]
