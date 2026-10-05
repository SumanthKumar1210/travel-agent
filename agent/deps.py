import os
from dataclasses import dataclass
from typing import Optional

import googlemaps
import httpx
from langchain_google_community import GoogleSearchAPIWrapper


@dataclass
class TravelDeps:
    """Shared dependencies closed over by every tool factory in agent/tools/.

    Built lazily (never at module import time) so that importing the
    `agent` package never requires API keys to already be set.

    `gmaps`/`search` are None when their API keys aren't configured (missing,
    blank, or still the .env.example placeholder text) — both underlying SDKs
    (googlemaps.Client, GoogleSearchAPIWrapper) validate their key at
    *construction* time and raise if it's missing or malformed, so we skip
    constructing them entirely rather than build-then-crash. Tools using them
    check for None and return a normal {"error": ...} dict, same as any other
    tool failure, so agents degrade gracefully instead of the whole run dying.
    """
    gmaps: Optional[googlemaps.Client]
    search: Optional[GoogleSearchAPIWrapper]
    http: httpx.AsyncClient
    google_maps_api_key: str


def build_gmaps_client() -> Optional[googlemaps.Client]:
    key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    if not key:
        return None
    try:
        return googlemaps.Client(key=key)
    except ValueError:
        # Present but malformed/placeholder (e.g. still "your-google-maps-api-key-here")
        return None


def build_search_client() -> Optional[GoogleSearchAPIWrapper]:
    api_key = os.environ.get("GOOGLE_SEARCH_API_KEY", "")
    cse_id = os.environ.get("GOOGLE_CSE_ID", "")
    if not api_key or not cse_id:
        return None
    try:
        return GoogleSearchAPIWrapper(google_api_key=api_key, google_cse_id=cse_id)
    except Exception:
        return None