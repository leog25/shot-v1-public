"""Offline stand-ins, so the voice path can be exercised with no network.

livekit-agents ships no fake LLM -- `livekit.agents.testing` exports exactly one
name, `fake_job_context` -- so the framework's intended pattern is to bring your
own. `llm.LLM` has a single abstract method and `llm.LLMStream` has one too, so
a usable stub is about thirty lines.
"""

from __future__ import annotations

from livekit.agents import llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN


class _StubStream(llm.LLMStream):
    def __init__(self, parent, *, chat_ctx, tools, conn_options, text):
        super().__init__(parent, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._text = text

    async def _run(self) -> None:
        self._event_ch.send_nowait(llm.ChatChunk(
            id="stub", delta=llm.ChoiceDelta(role="assistant", content=self._text)))


class StubLLM(llm.LLM):
    """Replies with fixed text and records the instructions it was given.

    Never pass a model *string* anywhere in an offline test -- `AgentSession(llm="…")`
    and `AMD(llm="…")` both resolve through the LiveKit inference gateway, which
    is a network call.
    """

    def __init__(self, text: str = "ok") -> None:
        super().__init__()
        self._text = text
        self.prompts: list[str] = []

    @property
    def model(self) -> str:
        return "stub"

    @property
    def provider(self) -> str:
        return "stub"

    def chat(self, *, chat_ctx, tools=None, conn_options=DEFAULT_API_CONNECT_OPTIONS,
             parallel_tool_calls=NOT_GIVEN, tool_choice=NOT_GIVEN,
             extra_kwargs=NOT_GIVEN) -> llm.LLMStream:
        for item in chat_ctx.items:
            if getattr(item, "role", None) == "system" and item.text_content:
                self.prompts.append(item.text_content)
        return _StubStream(self, chat_ctx=chat_ctx, tools=tools or [],
                           conn_options=conn_options, text=self._text)


class FakeSip:
    """`ctx.api` is a functools.cached_property on a class with no __slots__, so
    assigning to it shadows the descriptor permanently. Assign before anything
    reads it and there is no way to reach the network by accident."""

    def __init__(self) -> None:
        self.requests: list[object] = []

    @property
    def sip(self) -> FakeSip:
        return self

    async def create_sip_participant(self, req):
        self.requests.append(req)
        return None
