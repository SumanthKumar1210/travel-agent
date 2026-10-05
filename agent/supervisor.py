import os
from typing import Optional

from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor

from . import prompts
from .deps import TravelDeps
from .workers import WORKERS

SUPERVISOR_MODEL = os.environ.get("SUPERVISOR_MODEL", "openai:gpt-4o")


def build_supervisor(deps: TravelDeps, model: Optional[object] = None):
    """
    Build the Supervisor graph: the team roster (`agent.workers.WORKERS`) plus
    a Supervisor LLM that gets one auto-generated `transfer_to_<name>` handoff
    tool per worker. Adding a new specialist means adding its builder to
    `WORKERS` — nothing here needs to change.
    """
    chat_model = model or SUPERVISOR_MODEL
    if isinstance(chat_model, str):
        # Unlike create_react_agent, create_supervisor's `model` param expects
        # an actual model instance — it doesn't resolve "provider:model"
        # strings itself, so we do it here.
        chat_model = init_chat_model(chat_model)

    workers = [build(deps) for build in WORKERS]
    workflow = create_supervisor(
        workers,
        model=chat_model,
        prompt=prompts.load("supervisor"),
        parallel_tool_calls=True,
        output_mode="last_message",
    )
    return workflow.compile()
