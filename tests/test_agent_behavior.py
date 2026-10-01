"""L1 agent behavior — does the model route work correctly?

Runs the REAL model in text mode: RealtimeModel(modalities=["text"]) is the same
model with audio-token billing skipped, so these cost cents rather than dollars.
Text mode cannot catch turn-taking, barge-in or endpointing — that is what the
audio simulation lane is for.
"""

from __future__ import annotations

import asyncio

import pytest
from livekit.agents import AgentSession
from livekit.agents.testing import fake_job_context
from livekit.agents.voice.run_result import mock_tools
from livekit.plugins import openai as lkopenai
from openai.types.realtime import RealtimeReasoning

from shot_core.settings import get_settings
from shot_voice.agent import ShotAgent

from support import TEST_NAME

pytestmark = pytest.mark.contract

# Tools are MOCKED here. Without this, these tests call the real supervisor,
# which creates real Managed Agents sessions, which finish, which makes the
# reconcile sweep schedule real callbacks -- so running the test suite
# repeatedly phoned the owner unprompted. Behavior tests assert routing; they must
# never do work.
# Tools take RunContext first, so mocks must accept it.
_MOCKS = {
    "start_background_task": lambda ctx, goal, short_title: "Started.",
    "list_my_tasks": lambda ctx: "Nothing is running right now.",
    "get_task_status": lambda ctx, task_ref: f"{task_ref} is running.",
    # Every tool on ShotAgent must appear here. mock_tools patches only the
    # names it is given, so a new tool left out reaches the REAL supervisor
    # from this lane -- and this one cancels things.
    "cancel_task": lambda ctx, task_ref: "Stopped — I won't call you back about it.",
    # Must return something that PLAUSIBLY ANSWERS the query. A mock returning
    # filler makes the model correctly escalate to a background task, which then
    # reads as a routing bug when it is really a bad fixture.
    "web_search": lambda ctx, query: (
        "National Weather Service New York: 64 degrees, partly cloudy, wind 8 mph. | "
        "AccuWeather New York: currently 64F, feels like 63."),
    # Two live network legs -- the supervisor AND the Linear API -- so leaving
    # it out means this lane queries the owner's real workspace on every run.
    "check_my_day": lambda ctx: (
        "On your plate in Linear: Ship the callback fix, due today; Write the "
        "runbook. Background work -- nothing running."),
    # Touches no network and spends nothing, which is why it went unmocked for
    # months while the comment above claimed otherwise. It still matters: left
    # real, a time question in this lane is answered off a live wall clock, so
    # the assertion is not deterministic.
    "echo_time": lambda ctx: "Wednesday 9 September 2026, 8:14 AM in New York",
    # The three Linear tools. Two of these WRITE -- left out of this dict they
    # would create real tickets and real comments in the owner's workspace every
    # time the paid lane ran. test_tools.py enforces that they are all here.
    "find_linear_issue": lambda ctx, query: (
        "Complete the founder profile, Done; Complete the cofounder profile, Done."),
    "write_linear_issue": lambda ctx, title, notes="": (
        f"Written down in Linear: {title}."),
    "comment_on_linear_issue": lambda ctx, query, comment: (
        f"Added to {query}."),
}


# Opening realtime sessions back-to-back gets throttled, and it degrades
# SILENTLY -- no error, just an empty response with no function calls, which
# reads as the model choosing differently. Measured: 3 sessions in a row are
# fine, the 4th comes back empty; 6s of spacing fixes it. Almost certainly the
# account's TPM ceiling rather than a concurrency cap. Real calls are minutes
# apart so this is a test-harness concern, but it is worth knowing that the
# failure mode is silence rather than a 429.
# Raised from 6.0 when this file went from 9 tests to 11: the ceiling is on
# TOKENS per minute, not sessions, so adding tests tightens it for the ones
# already there. The tell that you have hit it is a failure whose "Context
# around failure" is EMPTY -- no events at all -- and a different test failing
# on each run while every one of them passes in isolation. It also shows up as
# `RealtimeError: response failed: [tokens] rate_limit_exceeded` in the log.
_SPACING_S = 12.0


@pytest.fixture(autouse=True)
async def _space_realtime_sessions():
    yield
    await asyncio.sleep(_SPACING_S)


def _session() -> AgentSession:
    s = get_settings()
    return AgentSession(
        llm=lkopenai.realtime.RealtimeModel(
            model=s.openai_realtime_model,
            modalities=["text"],          # same model, no audio billing
            turn_detection=None,
            reasoning=RealtimeReasoning(effort=s.openai_reasoning_effort),
            api_key=s.openai_api_key.get_secret_value(),
        ),
    )


@pytest.mark.asyncio
async def test_slow_work_is_delegated_not_attempted_inline():
    """The single most important routing decision in the product.

    Chained tool reasoning inside a realtime model costs ~6s to first word
    (Full-Duplex-Bench-v3, zero-latency mock APIs), so anything multi-hop MUST
    leave the voice loop.
    """
    with fake_job_context(), mock_tools(ShotAgent, _MOCKS):
        session = _session()
        await session.start(agent=ShotAgent())
        try:
            res = await session.run(
                user_input="Can you go find me a good deal on noise cancelling "
                           "headphones and let me know what you find?")
            res.expect.contains_function_call(name="start_background_task")
        finally:
            await session.aclose()


@pytest.mark.asyncio
async def test_status_question_reads_never_guesses():
    with fake_job_context(), mock_tools(ShotAgent, _MOCKS):
        session = _session()
        await session.start(agent=ShotAgent())
        try:
            res = await session.run(user_input="What are you working on right now?")
            res.expect.contains_function_call(name="list_my_tasks")
        finally:
            await session.aclose()


@pytest.mark.asyncio
async def test_goodbye_ends_the_call():
    with fake_job_context(), mock_tools(ShotAgent, _MOCKS):
        session = _session()
        await session.start(agent=ShotAgent())
        try:
            res = await session.run(user_input="That's all I needed, thanks. Bye!")
            res.expect.contains_function_call(name="end_call")
        finally:
            await session.aclose()


@pytest.mark.asyncio
async def test_greeting_does_not_hang_up():
    """EndCallTool(ignore_on_enter=True) must keep the model from ending the
    call while it is still saying hello."""
    with fake_job_context(), mock_tools(ShotAgent, _MOCKS):
        session = _session()
        await session.start(agent=ShotAgent())
        try:
            res = await session.run(user_input="Hello?")
            with pytest.raises(AssertionError):
                res.expect.contains_function_call(name="end_call")
        finally:
            await session.aclose()


@pytest.mark.asyncio
async def test_quick_fact_uses_search_not_delegation():
    """A question a snippet genuinely answers must not become a callback.

    Note the boundary: "when's the next Lakers home game" is NOT this -- a
    schedule needs a page read, and escalating that to a background task is
    correct. The weather is answerable from the results page itself.
    """
    with fake_job_context(), mock_tools(ShotAgent, _MOCKS):
        session = _session()
        await session.start(agent=ShotAgent())
        try:
            res = await session.run(user_input="What's the weather in New York right now?")
            res.expect.contains_function_call(name="web_search")
            with pytest.raises(AssertionError):
                res.expect.contains_function_call(name="start_background_task")
        finally:
            await session.aclose()


@pytest.mark.asyncio
async def test_multi_step_work_still_delegates():
    """The search tool must not swallow work that genuinely needs a browser."""
    with fake_job_context(), mock_tools(ShotAgent, _MOCKS):
        session = _session()
        await session.start(agent=ShotAgent())
        try:
            res = await session.run(
                user_input="Order me a large pepperoni from the pizza place I "
                           "used last week and pay with my saved card.")
            res.expect.contains_function_call(name="start_background_task")
        finally:
            await session.aclose()


@pytest.mark.asyncio
async def test_chitchat_calls_no_tools_at_all():
    """Conversation should just be conversation."""
    with fake_job_context(), mock_tools(ShotAgent, _MOCKS):
        session = _session()
        await session.start(agent=ShotAgent())
        try:
            res = await session.run(user_input="Pretty tired today, long week.")
            for name in ("web_search", "start_background_task", "end_call"):
                with pytest.raises(AssertionError):
                    res.expect.contains_function_call(name=name)
        finally:
            await session.aclose()


@pytest.mark.asyncio
async def test_explicit_browser_request_delegates_not_searches():
    """The owner said 'use the browser' and got five web searches instead.

    He asks for the browser because he wants a real page read; search only
    returns snippets. Naming the browser must force delegation.
    """
    with fake_job_context(), mock_tools(ShotAgent, _MOCKS):
        session = _session()
        await session.start(agent=ShotAgent())
        try:
            res = await session.run(
                user_input="Use the browser to go look at the Jazz Bistro site "
                           "and find tonight's set times.")
            res.expect.contains_function_call(name="start_background_task")
            with pytest.raises(AssertionError):
                res.expect.contains_function_call(name="web_search")
        finally:
            await session.aclose()


@pytest.mark.asyncio
async def test_page_read_delegates_even_without_the_word_browser():
    """'Check their site' is a page read, not a snippet lookup."""
    with fake_job_context(), mock_tools(ShotAgent, _MOCKS):
        session = _session()
        await session.start(agent=ShotAgent())
        try:
            res = await session.run(
                user_input="Can you check the Old Mill's site and see what's on "
                           "their menu tonight?")
            res.expect.contains_function_call(name="start_background_task")
        finally:
            await session.aclose()


@pytest.mark.asyncio
async def test_a_thin_search_is_not_escalated_to_a_background_task():
    """THE incident, as a behaviour test.

    web_search came back with no number and the model did exactly what the
    prompt told it to: delegate. That opened a paid worker session, which
    finished, which made the sweep ring the owner ninety seconds later to tell him it
    would be hot tomorrow -- while he was still on the phone objecting to it.

    NOTE the mock here is DELIBERATELY thin, which inverts the usual rule in
    this file. Elsewhere a filler mock makes the model correctly escalate and
    reads as a routing bug; here the thin result IS the fixture, because
    web_search now retries itself internally and its answer already reflects two
    passes. Do not "fix" this mock -- doing so deletes the coverage.
    """
    thin = dict(_MOCKS)
    thin["web_search"] = lambda ctx, query: (
        "Two searches for portland forecast turned up nothing that answers it. "
        f"Tell {TEST_NAME} you could not find it and stop there -- do NOT start a "
        "background task unless he asks for a page to be opened.")
    with fake_job_context(), mock_tools(ShotAgent, thin):
        session = _session()
        await session.start(agent=ShotAgent())
        try:
            res = await session.run(
                user_input="What's the weather in Portland tomorrow?")
            res.expect.contains_function_call(name="web_search")
            with pytest.raises(AssertionError):
                res.expect.contains_function_call(name="start_background_task")
        finally:
            await session.aclose()


@pytest.mark.asyncio
async def test_it_does_not_cancel_work_leo_has_not_asked_to_stop():
    """"It's taking forever" is impatience, not a cancellation.

    Cancelling work he still wants loses it silently: there is no callback left
    to tell him, and he finds out by asking for something that no longer exists.
    """
    with fake_job_context(), mock_tools(ShotAgent, _MOCKS):
        session = _session()
        await session.start(agent=ShotAgent())
        try:
            res = await session.run(
                user_input="How's that headphones thing going? "
                           "It's taking forever.")
            with pytest.raises(AssertionError):
                res.expect.contains_function_call(name="cancel_task")
        finally:
            await session.aclose()


@pytest.mark.asyncio
async def test_whats_on_today_goes_to_the_day_tool_in_one_call():
    """One round trip, not two. The model has 65 Linear tools sitting next to
    this one and could assemble the same answer from list_my_tasks plus
    list_issues -- which is two round trips inside a conversational turn, and
    5,159 characters of UUIDs and URLs to read out of."""
    with fake_job_context(), mock_tools(ShotAgent, _MOCKS):
        session = _session()
        await session.start(agent=ShotAgent())
        try:
            res = await session.run(user_input="What's on today?")
            res.expect.contains_function_call(name="check_my_day")
            with pytest.raises(AssertionError):
                res.expect.contains_function_call(name="list_my_tasks")
        finally:
            await session.aclose()


@pytest.mark.asyncio
async def test_an_offer_is_not_a_tool_call():
    """The new persona offers to go and look at things. "Offering is fine; the
    tool call waits until he says yes" is what keeps an enthusiastic agent from
    turning a remark into spend -- and start_background_task rings his phone."""
    with fake_job_context(), mock_tools(ShotAgent, _MOCKS):
        session = _session()
        await session.start(agent=ShotAgent())
        try:
            res = await session.run(
                user_input="I've got way too much on this week, honestly.")
            for name in ("start_background_task", "web_search", "end_call"):
                with pytest.raises(AssertionError):
                    res.expect.contains_function_call(name=name)
        finally:
            await session.aclose()
