from .location import build_location_worker
from .planning import build_planning_worker
from .research import build_research_worker
from .timing import build_timing_worker

# The team roster. Adding a new specialist (e.g. flight, hotel) means writing
# a new agent/prompts/<name>.txt + agent/workers/<name>.py (same shape as the
# others in this package) and appending its builder here — nothing else in
# the codebase needs to change; the Supervisor picks up the new handoff tool
# automatically (see agent/supervisor.py).
WORKERS = [
    build_research_worker,
    build_location_worker,
    build_timing_worker,
    build_planning_worker,
]

__all__ = [
    "WORKERS",
    "build_research_worker",
    "build_location_worker",
    "build_timing_worker",
    "build_planning_worker",
]
