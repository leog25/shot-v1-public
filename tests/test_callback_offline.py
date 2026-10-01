"""L2 — the real CallbackAgent, in a real AgentSession, with no network.

`test_callback_flow.py` proves the decision tree. This proves the *wiring*: that
`CallbackAgent.on_enter` actually runs that tree against a live
`AgentSession`, that speech plays, that the outcome is reported, and that the
line is hung up. Every callback bug that reached the owner's phone was a wiring bug in
exactly this layer, and none of it had any coverage at all -- the outbound
branch of the entrypoint was entered by no test, contract or offline.

Offline is possible because an AgentSession needs no room: omit `room=` from
`start()` and no RoomIO is built. Three landmines, all avoided below:

  - omitting `vad=`/`turn_handling=` silently pulls in LiveKit inference
    (a cloud turn detector, or ~108 MB of local weights);
  - `room=None` is NOT the same as omitting it -- `is_given(None)` is True;
  - any string model spec (`llm="openai/..."`) is a network call.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import pytest
from livekit.agents import AgentSession
from livekit.agents.testing import fake_job_context

from shot_voice import agent as agent_mod
from shot_voice import callback_flow as flow_mod
from shot_voice import outbound as outbound_mod
from shot_voice.agent import CallbackAgent
from shot_voice.callback_flow import CallRecord
from shot_voice.outbound import DialOutcome, DialRequest
from shot_voice.pin import PinResult

from support import TEST_NAME, TEST_PIN
from support.fakes import FakeSip, StubLLM

BRIEF = "Jazz Bistro has three shows tonight at seven, nine and eleven."
META = json.dumps({"to_number": "+16505551234", "brief": BRIEF,
                   "callback_id": "11111111-1111-1111-1111-111111111111",
                   "task_id": "22222222-2222-2222-2222-222222222222"})


@pytest.fixture
def offline(monkeypatch):
    """A started CallbackAgent whose dial and PIN are scripted."""

    async def _run(*, outcome: DialOutcome, pin: PinResult = PinResult.VERIFIED,
                   linger: float = 0.0) -> tuple[CallRecord, dict]:
        seen: dict = {}

        async def fake_dial(ctx, session, dial):
            seen["dialed"] = dial.to_number
            return outcome

        async def fake_code(self, limit):
            seen["pin_asked"] = seen.get("pin_asked", 0) + 1
            return pin

        async def fake_report(dial, rec):
            seen["reported"] = rec.outcome

        async def fake_mark(ids):
            seen["marked"] = list(ids)

        monkeypatch.setattr(outbound_mod, "dial_and_classify", fake_dial)
        monkeypatch.setattr(agent_mod._LiveIO, "wait_for_code", fake_code)
        monkeypatch.setattr(agent_mod, "_report_call", fake_report)
        monkeypatch.setattr(agent_mod, "_mark_reported", fake_mark)

        stub = StubLLM("understood")
        with fake_job_context(job_metadata=META) as ctx:
            ctx.api = FakeSip()                    # assign BEFORE any read
            session = AgentSession(
                llm=stub, stt=None, tts=None, vad=None,
                turn_handling={"turn_detection": None},
            )
            agent = CallbackAgent(dial=DialRequest.parse(META))
            async with session:
                await session.start(agent=agent, record=False)   # NO room=
                await asyncio.wait_for(agent.finished.wait(), timeout=10)
                # What prompt is the agent left running under? On a delivered
                # callback the flow hands back and stays on the line, and the
                # FIRST turn after that used to be the first generation in the
                # whole call under a session prompt -- the INBOUND one. Drive
                # one more turn so the swapped prompt is observed on the wire
                # and not merely on the attribute.
                seen["instructions_after"] = agent.instructions
                if not agent._holding:
                    handle = session.generate_reply(
                        user_input="that's all, thanks")
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(asyncio.shield(handle), timeout=5)
                seen["holding_after"] = agent._holding
                if linger:
                    # The idle watchdog is a bare create_task spawned by
                    # hand_back; leaving the session immediately cancels it
                    # before it has run a single poll.
                    await asyncio.sleep(linger)
        seen["said"] = stub.prompts
        return agent.record, seen

    return _run


# --------------------------------------------------------------- the wiring

async def test_a_verified_human_hears_the_brief_and_the_line_is_released(offline):
    rec, seen = await offline(outcome=DialOutcome(kind="human", amd_speech_s=0.3,
                                                  amd_reason="short_greeting"))

    assert rec.outcome == "human"
    assert rec.delivered is True
    assert rec.amd_reason == "short_greeting", (
        "on_enter swallows exceptions into outcome='error' and overwrites "
        "amd_reason with the message -- a hand_back signature mismatch surfaces "
        "here, not as a TypeError, because nothing in this file stubs hand_back")
    assert seen["dialed"] == "+16505551234"
    assert seen["pin_asked"] == 1
    assert seen["reported"] == "human", "the supervisor rendezvous must be released"
    assert seen["marked"] == ["22222222-2222-2222-2222-222222222222"], (
        "a delivered brief must mark the task reported, or he is told twice")
    # the brief actually reached the model as speech instructions
    assert any(BRIEF in p for p in seen["said"])


async def test_a_voicemail_box_is_told_nothing(offline):
    rec, seen = await offline(outcome=DialOutcome(kind="machine-vm", amd_speech_s=3.7))

    assert rec.outcome == "machine-vm"
    assert rec.delivered is False
    assert seen.get("pin_asked", 0) == 0, (
        "a machine must never be made to wait out the code window")
    assert not any(BRIEF in p for p in seen["said"]), "the brief reached voicemail"
    assert seen["said"], (
        "it hears a greeting that names nothing -- AMD is not infallible, and a "
        "misclassified person must get words rather than dead air")
    assert seen["reported"] == "machine-vm"
    assert "marked" not in seen, "nothing was delivered, so nothing is reported"


async def test_an_uncertain_answer_is_greeted_and_pinned(offline):
    """He answered and waited for the agent, which is what it asked of him. That
    used to be the deadlock; now it is just a call that goes to the PIN."""
    rec, seen = await offline(
        outcome=DialOutcome(kind="uncertain", amd_speech_s=0.0,
                            amd_reason="no_speech_timeout"))

    assert rec.outcome == "human"
    assert rec.delivered
    assert seen["pin_asked"] == 1


async def test_a_line_that_never_enters_a_code_gets_a_signoff_but_no_brief(offline):
    rec, seen = await offline(
        outcome=DialOutcome(kind="uncertain", amd_speech_s=0.0),
        pin=PinResult.NO_INPUT)

    assert rec.outcome == "pin_failed"
    assert rec.pin_result == "no_input"
    assert not any(BRIEF in p for p in seen["said"])
    assert seen["said"], "never sit in dead air on a live handset"
    assert seen["reported"] == "pin_failed"


async def test_a_closing_session_reports_abandoned(offline):
    rec, seen = await offline(outcome=DialOutcome(kind="human"),
                              pin=PinResult.ABANDONED)

    assert rec.outcome == "abandoned"
    assert seen["reported"] == "abandoned", (
        "abandoned must reach the supervisor -- pin_failed is terminal and "
        "would bury a finished task forever")


async def test_a_crash_in_the_flow_still_reports(offline, monkeypatch):
    """place_callback blocks on this for four minutes. A path that reports
    nothing leaves the workflow stuck and the line open."""
    async def boom(*a, **kw):
        raise ValueError("kaboom")

    monkeypatch.setattr(agent_mod, "run_callback", boom)
    rec, seen = await offline(outcome=DialOutcome(kind="human"))

    assert rec.outcome == "error"
    assert seen["reported"] == "error"
    assert "kaboom" in (rec.amd_reason or "")


# ------------------------------------------- the real PIN, with real DTMF

@pytest.fixture
def live_pin(monkeypatch):
    """Runs the REAL identity collector, driving keypresses as room events.

    This is the wiring that broke twice under the framework's GetDtmfTask: once
    by awaiting the AgentTask outside on_enter, once by treating a closing
    session as a wrong code. Both reached the owner's phone. The collector that
    replaced it is ours, and it opens its buffer before the dial -- so this also
    covers a code keyed before the agent ever asks.
    """
    from livekit import rtc

    async def _run(keys: str, *, before_greeting: bool = False,
                   window: float = 2.0, wait: float = 30.0) -> CallRecord:
        async def fake_dial(ctx, session, dial):
            return DialOutcome(kind="human", amd_speech_s=0.3)

        async def noop(*a, **kw):
            return None

        monkeypatch.setattr(outbound_mod, "dial_and_classify", fake_dial)
        monkeypatch.setattr(agent_mod, "_report_call", noop)
        monkeypatch.setattr(agent_mod, "_mark_reported", noop)
        # the real window is 25s; nothing here depends on its length
        monkeypatch.setattr(flow_mod, "PIN_WINDOW_S", window)

        with fake_job_context(job_metadata=META) as ctx:
            ctx.api = FakeSip()
            session = AgentSession(llm=StubLLM("ok"), stt=None, tts=None, vad=None,
                                   turn_handling={"turn_detection": None})
            agent = CallbackAgent(dial=DialRequest.parse(META))

            def press() -> None:
                for digit in keys:
                    ctx.room.emit("sip_dtmf_received", rtc.SipDTMF(
                        code=int(digit), digit=digit, participant=None))

            async with session:
                await session.start(agent=agent, record=False)
                if before_greeting:
                    # keyed during the ring, before anything was asked
                    press()
                    await asyncio.sleep(0.2)
                else:
                    await asyncio.sleep(0.5)
                    press()
                try:
                    await asyncio.wait_for(agent.finished.wait(), timeout=wait)
                except TimeoutError:
                    pytest.fail("the callback never reached a terminal outcome")
        return agent.record

    return _run


async def test_real_dtmf_verifies_the_pin_and_delivers(live_pin):
    rec = await live_pin(TEST_PIN)

    assert rec.pin_verified is True
    assert rec.outcome == "human"
    assert rec.delivered is True


async def test_a_code_keyed_before_the_agent_asks_still_counts(live_pin):
    """The owner's request: the buffer opens with the call, not with the question."""
    rec = await live_pin(TEST_PIN, before_greeting=True)

    assert rec.pin_verified is True
    assert rec.outcome == "human"
    assert rec.delivered is True


async def test_real_dtmf_with_the_wrong_code_withholds(live_pin):
    rec = await live_pin("0000")

    assert rec.pin_verified is False
    assert rec.outcome == "pin_failed"
    assert rec.delivered is False


async def test_pressing_nothing_ends_the_call_instead_of_hanging(live_pin):
    """Nobody keys anything. The call must end on a timer rather than holding an
    open line in silence, and the record must say `no_input` rather than
    implying he failed an identity check."""
    rec = await live_pin("")

    assert rec.outcome == "pin_failed"
    assert rec.pin_result == "no_input"
    assert rec.delivered is False


async def test_the_agent_does_not_talk_over_its_own_prepared_lines(live_pin):
    """The owner said his code aloud and the model answered it with a summary of its
    own, BEFORE the prepared brief ran -- so he heard the first show twice.
    While the callback owns the conversation the agent must not reply on its
    own; once the brief is out it must behave normally again."""
    from shot_voice.agent import CallbackAgent as CA

    agent = CA(dial=DialRequest.parse(META))
    assert agent._holding is True, "it must hold the floor from the start"

    with pytest.raises(Exception) as caught:
        await agent.on_user_turn_completed(None, None)
    assert type(caught.value).__name__ == "StopResponse"

    # and once handed back, an ordinary turn is answered normally
    agent._holding = False
    assert await agent.on_user_turn_completed(None, None) is None


async def test_after_delivering_it_stays_on_the_line(live_pin):
    rec = await live_pin(TEST_PIN)

    assert rec.delivered is True
    assert rec.stayed_on_line is True, (
        "the owner asked it to stay ready for more work rather than hang up on him")


# ------------------------------------------- the prompt after the callback

async def test_the_session_prompt_is_swapped_when_the_call_is_handed_back(offline):
    """update_instructions had exactly two call sites, both in scripted.py --
    empty, then restore INSTRUCTIONS. So the first turn after hand_back was the
    FIRST generation in the whole call under a session prompt, and that prompt
    was the INBOUND one: greet the owner, ask what he needs. It did, ninety seconds
    into a call he was already on, over an explicit dismissal."""
    from shot_voice.prompts import INSTRUCTIONS

    rec, seen = await offline(outcome=DialOutcome(kind="human"))

    assert rec.stayed_on_line is True
    after = seen["instructions_after"]
    assert after != INSTRUCTIONS, "the inbound prompt was left in force"
    assert f"already on a call with {TEST_NAME}" in after
    assert "do NOT ask" in after
    # and it actually reached the model, not just the attribute
    assert any(f"already on a call with {TEST_NAME}" in p for p in seen["said"])


def test_the_cut_variant_never_claims_he_heard_it_all():
    """An interrupted brief means he heard the beginning and not the rest.
    Telling him otherwise is how "which one?" gets an apology instead of an
    answer."""
    from shot_voice.agent import _after_prompt

    cut = _after_prompt(False, BRIEF)
    full = _after_prompt(True, BRIEF)

    assert BRIEF in cut, "he must be able to ask for the rest and get it"
    assert "finished reading him the results in full" in full
    assert "finished reading him the results in full" not in cut


async def test_a_failed_prompt_swap_does_not_leave_the_agent_mute(offline, monkeypatch):
    """The fix's own failure mode. A raise inside hand_back would skip the line
    that clears `_holding`, and the floor would be held forever: he speaks, gets
    silence, the line drops. That is worse than a badly-framed prompt."""
    # Patched on _after_prompt, not on update_instructions: the latter is also
    # how scripted() empties and restores the prompt for every spoken line, so
    # breaking it stops the brief playing at all and the flow never reaches
    # hand_back. _after_prompt is evaluated inside hand_back's own try.
    def boom(_delivered, _brief):
        raise RuntimeError("nope")

    monkeypatch.setattr(agent_mod, "_after_prompt", boom)
    rec, seen = await offline(outcome=DialOutcome(kind="human"))

    assert rec.outcome == "human"
    assert rec.stayed_on_line is True
    assert seen["holding_after"] is False, "the floor must be released anyway"


async def test_the_idle_watchdog_says_something_before_it_deletes_the_room(
        offline, monkeypatch):
    """Deleting the room without a word is how a callback becomes a mystery on
    his end -- the same reason a brief that never played says so first."""
    monkeypatch.setattr(agent_mod, "IDLE_HANGUP_S", 0.0)
    monkeypatch.setattr(agent_mod, "IDLE_POLL_S", 0.01)

    _rec, seen = await offline(outcome=DialOutcome(kind="human"), linger=0.4)

    assert any("about to end the call" in p for p in seen["said"]), (
        "the line must not just go dead")
