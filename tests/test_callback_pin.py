"""L3 rung 1 — a callback delivers, over a real room, through a real PIN.

This is the test that would have caught what the owner actually hit: he answered two
callbacks and heard nothing about either finished task. The PIN loop had burned
all three attempts in two milliseconds, because GetDtmfTask was awaited from
the job entrypoint, and it refuses there:

    "GetDtmfTask should only be awaited inside tool_functions or the
     on_enter/on_exit methods of an Agent"

Nothing offline catches that. The task only refuses at await time, inside a
live activity, so proving the fix needs a real room, a SIP-shaped participant,
and real keypad digits.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from livekit import api, rtc
from livekit.agents import inference, testing, utils

from shot_core.settings import get_settings
from shot_voice import agent as agent_mod
from shot_voice import outbound as outbound_mod
from shot_voice.agent import CallbackAgent
from shot_voice.outbound import DialOutcome, DialRequest
from shot_voice.worker import build_session

from support import TEST_PIN

pytestmark = pytest.mark.contract

BRIEF = "Jazz Bistro has three shows tonight: seven, nine and eleven."
PIN = TEST_PIN

META = json.dumps({"to_number": "+16505551234", "brief": BRIEF,
                   "callback_id": "33333333-3333-3333-3333-333333333333",
                   # So the title has something to leak. Offline tests can only
                   # assert what the PROMPT says; this is the one lane where the
                   # model's actual obedience is observable.
                   "task_titles": ["Jazz Bistro schedule"]})


def _patch_dial(monkeypatch):
    """AMD and the PSTN are not what this test is for -- the room, the real
    audio pipeline and real DTMF are. The dial itself is covered offline in
    test_callback_offline.py."""
    async def fake_dial(ctx, session, dial):
        return DialOutcome(kind="human", amd_speech_s=0.3, amd_reason="short_greeting")

    async def noop(*a, **kw):
        return None

    monkeypatch.setattr(outbound_mod, "dial_and_classify", fake_dial)
    monkeypatch.setattr(agent_mod, "_report_call", noop)
    monkeypatch.setattr(agent_mod, "_mark_reported", noop)


def _token(s, identity: str, room: str, *, sip: bool) -> str:
    t = (api.AccessToken(s.livekit_api_key, s.livekit_api_secret.get_secret_value())
         .with_identity(identity)
         .with_grants(api.VideoGrants(room_join=True, room=room,
                                      can_publish=True, can_subscribe=True,
                                      can_publish_data=True)))
    if sip:
        # Both the agent and GetDtmfTask key off a SIP-shaped participant.
        t = t.with_kind("sip").with_attributes({
            "sip.callStatus": "active",
            "sip.phoneNumber": s.owner_phone_number})
    return t.to_jwt()


class _Speech:
    """Tracks agent speech via the session's own state events.

    Audio-energy detection was tried first and does not work here: the agent's
    published track never reads as silent, so "has it stopped talking" was
    false forever even while it was plainly listening. The framework already
    knows, so ask it.
    """

    def __init__(self, session) -> None:
        self.turns = 0
        self.state = "initializing"
        self._since = time.monotonic()
        session.on("agent_state_changed", self._on)

    def _on(self, ev) -> None:
        if ev.new_state == "speaking" and self.state != "speaking":
            self.turns += 1
        self.state = ev.new_state
        self._since = time.monotonic()

    def listening_for(self, secs: float) -> bool:
        return self.state == "listening" and (time.monotonic() - self._since) > secs


async def _wait(pred, limit: float, what: str) -> None:
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if pred():
            return
        await asyncio.sleep(0.2)
    raise AssertionError(f"timed out after {limit}s waiting for {what}")


def _said(session) -> list[str]:
    return [it.text_content or "" for it in session.history.items
            if getattr(it, "role", None) == "assistant" and it.text_content]


@pytest.mark.asyncio
async def test_callback_greets_takes_the_pin_and_delivers_the_brief(monkeypatch):
    _patch_dial(monkeypatch)
    s = get_settings()
    room_name = f"shot-cb-pin-{int(time.time())}"
    caller = rtc.Room()
    agent_room = rtc.Room()
    session = None
    hum = None

    try:
        async with utils.http_context.open():   # plugins need a job-scoped http session
            await caller.connect(s.livekit_url, _token(s, "caller", room_name, sip=True))

            # RoomIO links to a participant's published audio, so the fake
            # caller has to actually publish something.
            src = rtc.AudioSource(24000, 1)
            await caller.local_participant.publish_track(
                rtc.LocalAudioTrack.create_audio_track("mic", src),
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))

            async def silence() -> None:
                frame = rtc.AudioFrame.create(24000, 1, 240)
                while True:
                    await src.capture_frame(frame)

            hum = asyncio.create_task(silence())

            await agent_room.connect(s.livekit_url, _token(s, "agent", room_name, sip=False))

            with testing.fake_job_context(room=agent_room) as ctx:
                # The PRODUCTION session builder, so a config change that breaks
                # endpointing on real calls also breaks this test.
                ctx.proc.userdata["vad"] = inference.VAD(model="silero")
                ctx.proc.userdata["eou"] = inference.TurnDetector(version="v1")
                session = build_session(ctx)
                heard = _Speech(session)

                agent = CallbackAgent(dial=DialRequest.parse(META))
                await session.start(agent=agent, room=agent_room)

                # ONE turn now: the greeting says why it called and asks for the
                # code in the same breath. It used to be two -- greet, then a
                # separate demand for a code -- which the owner found confusing.
                await _wait(lambda: heard.turns >= 1, 60, "the greeting")
                await _wait(lambda: heard.listening_for(1.0), 30, "the agent to stop talking")

                # The base instructions say to greet the owner and ask what he needs.
                # On a callback that is backwards: Shot dialed him, so Shot
                # opens with why. Asserted because the model drifted straight
                # back to "what do you want to talk about today?" without it.
                opening = _said(session)[0]
                assert "call" in opening.lower(), f"greeting isn't a callback: {opening!r}"
                assert "code" in opening.lower(), (
                    f"the greeting must ask for the code in the same breath: {opening!r}")
                for banned in ("what do you want", "what can i", "what do you need",
                               "what would you like", "how can i help"):
                    assert banned not in opening.lower(), f"opened by asking: {opening!r}"

                # THE assertion this lane exists for. Whatever came back is
                # the owner's business and nobody else's, and until the code lands we
                # do not know who is holding the phone. Everything said so far
                # went to an unidentified line.
                for said in _said(session):
                    assert "jazz" not in said.lower(), (
                        f"the subject reached the line before the code: {said!r}")

                for digit in PIN:
                    await caller.local_participant.publish_dtmf(code=int(digit), digit=digit)
                    await asyncio.sleep(0.15)

                await _wait(lambda: agent.finished.is_set(), 90, "a terminal outcome")
                assert agent.record.outcome == "human", f"ended as {agent.record.outcome!r}"

                # And once the code checks out it must say what it called
                # about, in the same turn as the findings.
                await _wait(lambda: any("jazz" in t.lower() for t in _said(session)),
                            45, "the brief to be delivered")
    finally:
        if hum:
            hum.cancel()
        if session:
            for t in _said(session):
                print("  SAID", repr(t)[:160], flush=True)
            await session.aclose()
        await caller.disconnect()
        await agent_room.disconnect()


@pytest.mark.asyncio
async def test_callback_gives_up_when_nobody_presses_anything(monkeypatch):
    """Answer, then stay silent. The PIN must fail on a timer, not hang.

    GetDtmfTask arms its completion debounce on the FIRST digit, so with zero
    digits it never finishes on its own -- the callback would sit on an open
    line saying nothing until the job was killed.
    """
    _patch_dial(monkeypatch)
    s = get_settings()
    room_name = f"shot-cb-silent-{int(time.time())}"
    caller = rtc.Room()
    agent_room = rtc.Room()
    session = None
    hum = None

    try:
        async with utils.http_context.open():
            await caller.connect(s.livekit_url, _token(s, "caller", room_name, sip=True))
            src = rtc.AudioSource(24000, 1)
            await caller.local_participant.publish_track(
                rtc.LocalAudioTrack.create_audio_track("mic", src),
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))

            async def silence() -> None:
                frame = rtc.AudioFrame.create(24000, 1, 240)
                while True:
                    await src.capture_frame(frame)

            hum = asyncio.create_task(silence())
            await agent_room.connect(s.livekit_url, _token(s, "agent", room_name, sip=False))

            with testing.fake_job_context(room=agent_room) as ctx:
                ctx.proc.userdata["vad"] = inference.VAD(model="silero")
                ctx.proc.userdata["eou"] = inference.TurnDetector(version="v1")
                session = build_session(ctx)
                agent = CallbackAgent(dial=DialRequest.parse(META))
                await session.start(agent=agent, room=agent_room)

                from shot_core.budget import PIN_WINDOW_S
                budget = PIN_WINDOW_S * 2 + 60
                await _wait(lambda: agent.finished.is_set(), budget, "the PIN to give up")
                assert agent.record.outcome == "pin_failed", f"got {agent.record.outcome!r}"

                # Withholding is not enough -- it must also say why, or the call
                # is just dead air from the owner's side. Two turns now, not four:
                # the greeting asks once, then the sign-off. It used to re-ask
                # three times with no room to answer.
                await _wait(lambda: len(_said(session)) >= 2, 40, "a sign-off")
                assert not any("jazz" in t.lower() for t in _said(session)), \
                    "brief leaked to an unverified answerer"
    finally:
        if hum:
            hum.cancel()
        if session:
            for t in _said(session):
                print("  SAID", repr(t)[:160], flush=True)
            await session.aclose()
        await caller.disconnect()
        await agent_room.disconnect()
