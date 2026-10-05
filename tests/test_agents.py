"""
Offline tests for the Hierarchical Supervisor travel system. No real API keys
or network access required: every agent's model is a scripted FakeToolCallingModel
(see conftest.py).
"""
import time

import pytest
from langchain_core.messages import AIMessage
from langgraph_supervisor import create_supervisor

from agent import prompts
from agent.workers import (
    build_location_worker,
    build_planning_worker,
    build_research_worker,
    build_timing_worker,
)
from tests.conftest import FakeToolCallingModel, tool_call_message


def test_import_agent_package_without_env_vars(monkeypatch):
    """Importing the package must not require API keys to be set — nothing
    provider-backed is built at module import time (only inside builder
    functions, called from TravelAgent.run())."""
    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_MAPS_API_KEY",
        "GOOGLE_SEARCH_API_KEY",
        "GOOGLE_CSE_ID",
    ):
        monkeypatch.delenv(var, raising=False)

    import importlib

    import agent

    importlib.reload(agent)


@pytest.mark.parametrize(
    "build_worker",
    [build_research_worker, build_location_worker, build_timing_worker, build_planning_worker],
)
def test_worker_runs_under_fake_model(build_worker, fake_deps):
    model = FakeToolCallingModel(
        responses=[
            tool_call_message([{"name": "geocode_address", "args": {"address": "Paris"}}]),
            AIMessage(content="Findings: ok"),
        ]
    )
    worker = build_worker(fake_deps, model=model)
    result = worker.invoke({"messages": [{"role": "user", "content": "Paris, 3 days"}]})
    assert result["messages"][-1].content == "Findings: ok"


def _build_fake_team(fake_deps, *, supervisor_responses, worker_responses=None, worker_sleep=None):
    """Assemble a Supervisor + all 4 workers with fake, fully-scripted models.
    `worker_responses` maps worker name -> list of AIMessages; defaults to a
    single canned finding each. `worker_sleep`, if given, makes every worker's
    fake model sleep that many seconds before responding (for concurrency tests)."""
    worker_responses = worker_responses or {}

    def model_for(name):
        responses = worker_responses.get(name) or [AIMessage(content=f"{name} findings: ok")]
        return FakeToolCallingModel(responses=responses, sleep=worker_sleep)

    workers = [
        build_research_worker(fake_deps, model=model_for("research")),
        build_location_worker(fake_deps, model=model_for("location")),
        build_timing_worker(fake_deps, model=model_for("timing")),
        build_planning_worker(fake_deps, model=model_for("planning")),
    ]
    workflow = create_supervisor(
        workers,
        model=FakeToolCallingModel(responses=supervisor_responses),
        prompt=prompts.load("supervisor"),
        parallel_tool_calls=True,
        output_mode="last_message",
    )
    return workflow.compile()


def test_supervisor_delegates_and_synthesizes(fake_deps):
    graph = _build_fake_team(
        fake_deps,
        supervisor_responses=[
            tool_call_message([{"name": "transfer_to_planning"}]),
            AIMessage(content="Final itinerary: day 1, day 2, day 3."),
        ],
        worker_responses={"planning": [AIMessage(content="Planning says: itinerary built.")]},
    )
    result = graph.invoke({"messages": [{"role": "user", "content": "Plan a 3-day trip to Paris"}]})
    assert "Final itinerary" in result["messages"][-1].content


@pytest.mark.asyncio
async def test_parallel_handoff_runs_concurrently(fake_deps):
    """The Supervisor hands off to research/location/timing in one turn
    (parallel_tool_calls=True); each worker's fake model sleeps 0.3s before
    responding. If they ran serially that's ~0.9s+; concurrently it should be
    well under that."""
    delay = 0.3
    graph = _build_fake_team(
        fake_deps,
        supervisor_responses=[
            tool_call_message(
                [
                    {"name": "transfer_to_research"},
                    {"name": "transfer_to_location"},
                    {"name": "transfer_to_timing"},
                ]
            ),
            AIMessage(content="done"),
        ],
        worker_sleep=delay,
    )

    start = time.perf_counter()
    await graph.ainvoke({"messages": [{"role": "user", "content": "Plan a trip"}]})
    elapsed = time.perf_counter() - start

    assert elapsed < delay * 3 * 0.7, (
        f"Expected concurrent handoff (<{delay * 3 * 0.7:.2f}s), took {elapsed:.2f}s — "
        "research/location/timing may have run serially instead of in parallel."
    )


def test_geocode_address_tool_returns_error_dict_on_failure(fake_deps):
    from agent.tools.maps import make_maps_tools

    fake_deps.gmaps.geocode.side_effect = RuntimeError("boom")
    tools = {t.name: t for t in make_maps_tools(fake_deps)}

    result = tools["geocode_address"].invoke({"address": "1600 Amphitheatre Parkway"})
    assert result == [{"error": "Geocoding failed: boom"}]
