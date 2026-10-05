from .agent import TravelAgent
from .supervisor import build_supervisor
from .workers import WORKERS

__all__ = [
    "TravelAgent",
    "build_supervisor",
    "WORKERS",
]

# Version info
__version__ = "0.4.0"
__description__ = "Hierarchical Supervisor-Router travel planner (LangGraph): a Supervisor delegating to Research/Location/Timing/Planning workers, extensible via the WORKERS registry"
