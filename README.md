# Travel Planning System — Hierarchical Supervisor (LangGraph)

## Overview

A multi-agent travel planner built on [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)'s **Hierarchical Supervisor** pattern ([`langgraph_supervisor.create_supervisor`](https://github.com/langchain-ai/langgraph-supervisor-py)): a Supervisor agent routes to specialist worker agents via auto-generated handoff tools, each worker reports back, and the Supervisor delivers the final plan. Each agent — Supervisor and every worker — is independently configurable to a different model.

This replaces an earlier `pydantic_ai`-based design that used custom `delegate_*` tools. The Supervisor pattern gives the same LLM-driven delegation (the Supervisor still decides who to call and when, including resolving conflicts between specialists) while the team of workers is a plain registry list — **the core design goal is making it cheap to add new specialists** (a Flight agent, a Hotel agent, ...) without touching the orchestration code at all.

## Quick Start

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in your API keys
python run_agent.py --query "Plan a 3-day trip from Boston to Paris"
```

`OPENAI_API_KEY` is the only key actually required — every agent defaults to OpenAI. `GOOGLE_MAPS_API_KEY` (geocoding/directions/places/address-validation/weather) and `GOOGLE_SEARCH_API_KEY`+`GOOGLE_CSE_ID` (web search) are optional: `agent/deps.py` only constructs those clients if a plausible key is present, and every tool that needs one returns a normal `{"error": "... not configured"}` result instead of crashing when it's missing — the plan just falls back to the model's general knowledge instead of live data for that part. See `.env.example` for setup notes.

`--model` overrides only the Supervisor's own model for that run; each worker keeps its own configured model (see `.env.example` for the per-agent override env vars).

## Architecture

```
                         ┌──────────────┐
            user query ─▶│  Supervisor  │◀── decides what to call next, reading
                         │ (openai:gpt- │    the whole shared conversation each
                         │     4o)      │    time a worker hands control back
                         └──────┬───────┘
                 transfer_to_*  │  (one auto-generated handoff tool per worker)
        ┌───────────┬───────────┼───────────┐
        ▼           ▼           ▼           ▼
    research    location     timing     planning       ← agent/workers/WORKERS
   (openai:     (openai:    (openai:    (openai:
    gpt-4o)      gpt-4o)     gpt-4o)     gpt-4o)
```

Each worker is a `create_react_agent` (its own LLM + its own tool subset + its own system prompt) that the Supervisor hands off to via an auto-generated `transfer_to_<name>` tool. A worker sees the *entire* shared conversation when it receives control — including what earlier workers already found — so e.g. the Planning worker automatically has Research/Location/Timing's findings without the Supervisor needing to manually restate them. Workers hand control back to the Supervisor when done; the Supervisor decides the next step (another handoff, or the final answer) by reading the updated conversation.

**Concurrency is real, not just hoped-for**: LangGraph's execution model runs every tool call from a single model response concurrently (verified against `langgraph`'s `ToolNode`, which dispatches via `asyncio.gather`), and `create_supervisor(..., parallel_tool_calls=True)` lets the Supervisor issue multiple handoffs in one turn. Since research/location/timing don't depend on each other, the Supervisor's prompt (`agent/prompts/supervisor.txt`) explicitly tells it it may hand off to all three at once — see `tests/test_agents.py::test_parallel_handoff_runs_concurrently` for a test that actually measures this.

## Extending the team — the main architectural feature

Adding a new specialist (e.g. a Flight agent or Hotel agent) requires **no changes to the Supervisor, the graph wiring, or any existing worker**:

1. Write `agent/prompts/flight.txt` (its system prompt — role, tools, output format).
2. Write `agent/workers/flight.py`, following the exact shape of every other worker file:
   ```python
   import os
   from langgraph.prebuilt import create_react_agent
   from .. import prompts
   from ..deps import TravelDeps
   from ..tools import make_flight_tools  # or reuse existing tool factories

   FLIGHT_MODEL = os.environ.get("FLIGHT_MODEL", "openai:gpt-4o")

   def build_flight_worker(deps: TravelDeps, model=None):
       return create_react_agent(
           model=model or FLIGHT_MODEL,
           tools=make_flight_tools(deps),
           name="flight",
           prompt=prompts.load("flight"),
       )
   ```
3. Add one line to `WORKERS` in `agent/workers/__init__.py`.

That's it — the Supervisor automatically gets a `transfer_to_flight` handoff tool and its prompt already describes the team generically enough to route to it (or update `agent/prompts/supervisor.txt` with a one-line mention of the new specialist, if you want the Supervisor to proactively favor it). `langgraph_supervisor` also supports nesting (a "supervisor of supervisors"), so a future "Booking Team" grouping Flight + Hotel under its own sub-supervisor is a natural extension too, not a rewrite.

## The team today

| Worker | Model (default) | Tools | Role |
|---|---|---|---|
| **research** | `openai:gpt-4o` (`RESEARCH_MODEL`) | `search_web`, `find_places`, `geocode_address`, `get_current_location` | Destination research & cultural analysis |
| **location** | `openai:gpt-4o` (`LOCATION_MODEL`) | `geocode_address`, `reverse_geocode_coordinates`, `get_directions`, `find_places`, `validate_address`, `get_current_location` | Proximity/routing analysis |
| **timing** | `openai:gpt-4o` (`TIMING_MODEL`) | `get_current_weather`, `get_weather_forecast`, `geocode_address`, `get_current_location`, `get_current_date_time` | Weather & seasonal timing |
| **planning** | `openai:gpt-4o` (`PLANNING_MODEL`) | `get_directions`, `geocode_address`, `find_places`, `validate_address`, `get_current_location` | Day-by-day itinerary assembly |

The Supervisor itself defaults to `openai:gpt-4o` (`SUPERVISOR_MODEL`). Every agent defaults to OpenAI so a single `OPENAI_API_KEY` is enough to run the whole system; set any `*_MODEL` env var to an `anthropic:...` string (and add `ANTHROPIC_API_KEY`) to mix providers back in per agent. All tool implementations live in `agent/tools/` as `make_*_tools(deps)` factories (closures over a `TravelDeps` instance — Google Maps client, HTTP client, search client — for dependency injection and testability); each worker picks the specific tools it needs via `agent/tools/pick()`.

**Model-string resolution note**: `create_react_agent` (used for every worker) accepts a `"provider:model"` string directly, but `create_supervisor`'s `model` param expects an actual model instance — it won't resolve a string itself. `agent/supervisor.py` handles this by calling `langchain.chat_models.init_chat_model()` on the Supervisor's model string before passing it in. Worth knowing if you ever bypass `build_supervisor()` and call `create_supervisor` directly.

## Example: "3-day trip from Boston to Paris"

1. Supervisor reads the query, resolves it needs research + location + timing context, and hands off to **research**, **location**, and **timing** together in one turn — they run concurrently.
2. Each reports back (attractions/culture; proximity/routing; weather/seasonal notes) and control returns to the Supervisor.
3. Supervisor hands off to **planning**, which already sees all three findings in the shared conversation and builds the day-by-day itinerary.
4. Supervisor delivers the final answer: a complete markdown plan (route, accommodations, itinerary, weather-based fallbacks, budget), passing through the specialists' details faithfully.

## Testing

`tests/test_agents.py` runs the whole thing offline — every model is a scripted `FakeToolCallingModel` (see `tests/conftest.py`), so no real API keys or network calls are needed. Covers: package import with zero env vars, each worker in isolation, the Supervisor delegating and synthesizing, and the parallel-handoff concurrency claim above (timing-based). Run with `pytest tests/ -q`.
