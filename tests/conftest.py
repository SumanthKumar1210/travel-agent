"""
Shared offline-testing fixtures for the LangGraph-based travel agent.

LangChain's own fake chat models (FakeMessagesListChatModel, GenericFakeChatModel)
don't implement `bind_tools` (the base class raises NotImplementedError), so they
can't stand in for a model inside create_react_agent/create_supervisor, both of
which call `model.bind_tools(tools)`. FakeToolCallingModel below is a minimal fake
that does implement it (a no-op — tool *schemas* don't matter when the model's
responses are fully scripted) plus a scripted, cycling `_generate`, mirroring
FakeMessagesListChatModel's own cycling behavior.
"""
from typing import Any, List, Optional

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeToolCallingModel(BaseChatModel):
    """Cycles through a scripted list of AIMessage responses. `bind_tools` is a
    no-op passthrough since behavior is fully pre-scripted, not actually decided
    from the tool schemas."""

    responses: List[AIMessage]
    sleep: Optional[float] = None
    i: int = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolCallingModel":
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.sleep is not None:
            import time

            time.sleep(self.sleep)
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling-model"


def tool_call_message(tool_calls: List[dict], content: str = "") -> AIMessage:
    """Build an AIMessage that requests one or more tool calls.
    Each entry in `tool_calls` is {"name": ..., "args": {...}}."""
    return AIMessage(
        content=content,
        tool_calls=[
            {"name": tc["name"], "args": tc.get("args", {}), "id": f"call_{i}"}
            for i, tc in enumerate(tool_calls)
        ],
    )


@pytest.fixture
def fake_deps():
    from unittest.mock import MagicMock

    from agent.deps import TravelDeps

    gmaps = MagicMock()
    gmaps.geocode.return_value = []
    gmaps.reverse_geocode.return_value = []
    gmaps.directions.return_value = []
    gmaps.addressvalidation.return_value = {}
    gmaps.places.return_value = {"results": []}

    search = MagicMock()
    search.results.return_value = []

    class _FakeHTTPResponse:
        status_code = 200

        def json(self):
            return {"ok": True}

        def raise_for_status(self):
            pass

    class _FakeHTTPClient:
        async def get(self, *args, **kwargs):
            return _FakeHTTPResponse()

    return TravelDeps(
        gmaps=gmaps,
        search=search,
        http=_FakeHTTPClient(),
        google_maps_api_key="fake-key",
    )
