"""LiveKit voice worker.

One AgentServer, one rtc_session — the framework raises if you register two.
Inbound and outbound therefore branch on ctx.job.metadata rather than being
separate workers.

    uv run python -m shot_voice.worker dev
"""

from __future__ import annotations

import asyncio
import logging

from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
    inference,
)
from livekit.plugins import openai as lkopenai
from openai.types.realtime import RealtimeReasoning

from shot_core.settings import get_settings
from shot_voice.agent import CallbackAgent, ShotAgent
from shot_voice.outbound import (
    DialRequest,
)
from shot_voice.transcript import Transcript

log = logging.getLogger("shot.voice")
AGENT_NAME = "shot-voice"  # must equal the dispatch rule's room_config.agents[].agent_name

# How many calls this worker will take at once. The owner is one person, so the real
# number is one -- plus headroom for a callback dialling while he is already on
# an inbound call, and one more so the cap is a safety net rather than a limit
# we brush against. Crossed, the worker declines; see _call_load.
MAX_CONCURRENT_CALLS = 3


def _call_load(server: AgentServer) -> float:
    """Load as a fraction of the CALLS we will take, not of machine CPU.

    The SDK default is a 2.5s moving average of MACHINE-WIDE cpu_percent
    (`utils/hw/cpu.py`: psutil, or the ROOT cgroup's cpu.max -- never this
    service's own cgroup). At or above `load_threshold` (0.7 in prod) the worker
    answers `available = False` in `_answer_availability` and DECLINES the job
    outright. It does not queue and it does not retry, and `ops.status` requires
    exactly ONE registered worker -- so there is nobody else to take it. The
    caller gets nothing, and because no job ever starts, nothing is logged: no
    `job start`, no room, no row in `shot.calls`. It reads exactly like the
    phone number being broken.

    That threshold is there to stop a worker accepting more concurrent jobs than
    it can serve. On this box it was instead measuring `apt-daily-upgrade`,
    which on 2026-09-09 burned 54s of CPU on 2 vCPU and took the number
    unreachable for 12.5s at 02:26 local, twice, with nothing to say why. The
    ten other daily runs cost 2-7s of CPU and crossed nothing -- which is what
    makes it the kind of hole you find once and can never reproduce.

    Counting jobs measures the thing the threshold is actually for, and it
    cannot be moved by anything that is not a call. Raising CPUWeight on the
    unit would NOT have fixed this: the metric is machine-wide, so apt's cycles
    would still be counted as ours and the worker would still decline.

    Crosses 0.7 at MAX_CONCURRENT_CALLS active jobs, so that many run and the
    next is declined.
    """
    return len(server.active_jobs) / MAX_CONCURRENT_CALLS


def prewarm(proc: JobProcess) -> None:
    # Loaded once per process, not per job. Anything heavy imported on this
    # path is paid for against initialize_process_timeout (10s).
    proc.userdata["vad"] = inference.VAD(model="silero")
    proc.userdata["eou"] = inference.TurnDetector(version="v1")


# Keep processes warm. A cold start puts 10-20s of dead air on the ring,
# which is the worst possible first impression for an inbound call.
#
# load_fnc is ours on purpose -- the default one measures the whole machine and
# let a nightly apt run decline real calls. See _call_load.
server = AgentServer(setup_fnc=prewarm, num_idle_processes=2, load_fnc=_call_load)


def build_session(ctx: JobContext) -> AgentSession:
    s = get_settings()
    return AgentSession(
        llm=lkopenai.realtime.RealtimeModel(
            model=s.openai_realtime_model,
            voice="marin",
            modalities=["audio"],
            # Hard-disable the model's server-side VAD. The framework also
            # auto-disables it when a streaming turn detector is present, but
            # being explicit is what makes the intent legible.
            turn_detection=None,
            # a bare dict here serializes wrong and emits a pydantic warning;
            # the plugin types this as RealtimeReasoning.
            reasoning=RealtimeReasoning(effort=s.openai_reasoning_effort),
            input_audio_noise_reduction="far_field",  # PSTN, not near_field
            api_key=s.openai_api_key.get_secret_value(),
        ),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            # Audio-native detector: needs no STT, so it works with a
            # speech-to-speech model, and resamples 8kHz PSTN internally.
            turn_detection=ctx.proc.userdata["eou"],
            # endpointing is deliberately OMITTED. With a streaming detector the
            # framework selects min_delay=0.3 / max_delay=2.5 automatically, and
            # the VAD default is already min_silence_duration=0.25. Overriding
            # these re-introduces the 0.55+0.5 stacking they were fixed for.
            #
            # "adaptive" must be asked for by name. Every call logged "adaptive
            # interruption is disabled by default in production mode", leaving
            # plain VAD with min_duration=0.5s -- and min_words is STT-only, so
            # it is inert behind a realtime model. A "mm-hmm" over an
            # eighty-second brief therefore counted as a full barge-in.
            # `_resolve_interruption_detection` accepts it here: a RealtimeModel
            # with turn_detection=None can gatekeep without STT, and we have both
            # a VAD and a non-manual turn detector. Everything else keeps its
            # default -- notably resume_false_interruption, which pauses playout
            # and resumes it 2s later if no real turn materialises.
            interruption={"mode": "adaptive"},
            preemptive_generation={"enabled": False},  # inert with a RealtimeModel; say so
        ),
    )


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    meta = ctx.job.metadata or ""
    log.info("job start room=%s metadata=%r", ctx.room.name, meta[:200])

    session = build_session(ctx)
    dial = DialRequest.parse(meta)

    # Record what is said, on every call, before anything is said. The last
    # callback failure was diagnosable down to "the model called end_call" and
    # no further, because the words existed nowhere once the room closed.
    tx = Transcript(room_name=ctx.room.name,
                    direction="outbound" if dial else "inbound",
                    callback_id=dial.callback_id if dial else None)
    tx.listen(session)
    ctx.add_shutdown_callback(tx.flush)

    if dial is None:
        # Inbound. Build the seeded context FIRST, then wait for the caller to
        # actually be in the room before starting the session -- on_enter fires
        # the greeting immediately, and speaking into a half-established SIP
        # audio path means the owner hears only the tail of it, or nothing at all.
        chat_ctx, news = await ShotAgent.seeded_chat_ctx()
        try:
            caller = await asyncio.wait_for(ctx.wait_for_participant(), timeout=15)
            log.info("caller present: identity=%r name=%r", caller.identity, caller.name)
            for k, v in sorted(caller.attributes.items()):
                log.info("  ATTR %s = %r", k, v)
        except TimeoutError:
            log.warning("no caller joined within 15s; greeting anyway")
        await session.start(agent=ShotAgent(chat_ctx=chat_ctx, news=news,
                                           transcript=tx),
                            room=ctx.room)
        return

    # Outbound. Everything -- dial, classify, greet, PIN, deliver, hang up --
    # runs in CallbackAgent.on_enter, because GetDtmfTask is only legal there
    # and splitting the rest across this function is what produced every
    # callback bug that reached the owner's phone. on_enter is its own task, and the
    # job outlives this function returning, so it is safe to just start it.
    chat_ctx, _ = await ShotAgent.seeded_chat_ctx()
    await session.start(agent=CallbackAgent(dial=dial, chat_ctx=chat_ctx,
                                           transcript=tx),
                        room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
