"""L3 rung 1 — the agent must speak first, into a real room, over real WebRTC.

Text-mode behavior tests cannot catch this: they always supply user input, so
they can never prove the agent opens unprompted.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from livekit import api, rtc

from shot_core.settings import get_settings

pytestmark = pytest.mark.contract


@pytest.mark.asyncio
async def test_agent_greets_a_silent_caller():
    s = get_settings()
    room_name = f"shot-greet-test-{int(time.time())}"
    lk = api.LiveKitAPI(s.livekit_url, s.livekit_api_key,
                        s.livekit_api_secret.get_secret_value())
    room = rtc.Room()
    try:
        await lk.agent_dispatch.create_dispatch(api.CreateAgentDispatchRequest(
            agent_name="shot-voice", room=room_name,
            metadata=json.dumps({"tier": "voice"})))

        tok = (api.AccessToken(s.livekit_api_key, s.livekit_api_secret.get_secret_value())
               .with_identity("caller")
               # AMD and the agent both key off sip.callStatus, so a fake SIP
               # participant is the only way to exercise that path off-PSTN.
               .with_attributes({"sip.callStatus": "active",
                                 "sip.phoneNumber": s.owner_phone_number})
               .with_grants(api.VideoGrants(room_join=True, room=room_name,
                                            can_publish=True, can_subscribe=True))).to_jwt()

        spoke = asyncio.Event()
        loud_frames = {"n": 0}

        @room.on("track_subscribed")
        def _(track, pub, participant):
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return

            async def pump() -> None:
                async for ev in rtc.AudioStream(track):
                    d = ev.frame.data
                    peak = max((abs(int.from_bytes(d[i:i + 2], "little", signed=True))
                                for i in range(0, min(len(d), 640), 2)), default=0)
                    if peak > 300:          # ignore comfort-noise / silence padding
                        loud_frames["n"] += 1
                        spoke.set()
                    if loud_frames["n"] > 40:
                        return

            asyncio.create_task(pump())

        await room.connect(s.livekit_url, tok)
        # deliberately publish NOTHING: the agent must open unprompted
        await asyncio.wait_for(spoke.wait(), timeout=25)
        await asyncio.sleep(3)
        assert loud_frames["n"] > 10, "agent produced a track but barely any audio"
    finally:
        await room.disconnect()
        try:
            await lk.room.delete_room(api.DeleteRoomRequest(room=room_name))
        finally:
            await lk.aclose()


@pytest.mark.asyncio
async def test_inbound_greeting_leads_with_finished_work():
    """The owner called in to "hi, what do you need?" with two finished tasks
    sitting unmentioned.

    The context WAS seeded -- the failure was the opening instruction, which
    made referencing it optional ("you may") while making "ask what he needs"
    mandatory. So the model did the mandatory half. Text mode is enough here:
    this is about what the agent chooses to say first.
    """
    import uuid

    from livekit.agents import AgentSession
    from livekit.agents.llm import ChatContext
    from livekit.agents.testing import fake_job_context
    from livekit.plugins import openai as lkopenai

    from shot_voice.agent import ShotAgent

    s = get_settings()
    ctx = ChatContext()
    ctx.add_message(role="assistant", content=(
        "Context from our earlier calls: Finished 20 minutes ago — Jazz Bistro "
        "schedule: Jazz Bistro has three shows tonight at seven, nine and eleven."))
    # A random id so the reported-marking POST matches no real row.
    news = [{"id": str(uuid.uuid4()), "ref": "jazz-bistro-schedule",
             "title": "Jazz Bistro schedule", "ok": True, "when": "20 minutes ago",
             "gist": "Three shows tonight at seven, nine and eleven."}]

    session = AgentSession(llm=lkopenai.realtime.RealtimeModel(
        model=s.openai_realtime_model, modalities=["text"],
        api_key=s.openai_api_key.get_secret_value()))
    with fake_job_context():
        await session.start(agent=ShotAgent(chat_ctx=ctx, news=news))
        try:
            opening = ""
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline and not opening:
                spoken = [it.text_content for it in session.history.items
                          if getattr(it, "role", None) == "assistant" and it.text_content]
                opening = spoken[0] if spoken else ""
                await asyncio.sleep(0.3)
            assert opening, "the agent never spoke first"
        finally:
            await session.aclose()

    print("  OPENING", repr(opening), flush=True)
    low = opening.lower()
    assert "jazz" in low, f"greeting never mentions the finished work: {opening!r}"
    for banned in ("what do you need", "what can i", "what do you want",
                   "what would you like", "how can i help"):
        assert banned not in low, f"greeting still asks instead of telling: {opening!r}"
    # It parroted the instruction once: "Details are already in the context if
    # you need them." The owner has no idea what "the context" is.
    for leak in ("context", "my notes", "on record", "in my records"):
        assert leak not in low, f"greeting leaks its own plumbing: {opening!r}"
