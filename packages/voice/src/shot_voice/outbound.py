"""Outbound callback branch: dial, classify who answered, then gate disclosure.

The supervisor dispatches the agent and stops there. THIS process creates the
SIP participant, because AMD needs a live AgentSession and that only exists
inside the dispatched job.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from google.protobuf.duration_pb2 import Duration
from livekit import api
from livekit.agents import AMD, AgentSession, JobContext
from livekit.plugins import deepgram

from shot_core.settings import get_settings

log = logging.getLogger("shot.outbound")


def _amd_stt():
    """Streaming STT for AMD, or None to let LiveKit Inference pick.

    Returning None selects cartesia/ink-whisper through the Inference gateway,
    which is precisely the configuration that reproduces livekit/agents#6996
    (AMD stalling 17-20s on ~7% of answered calls). Set DEEPGRAM_API_KEY to
    avoid it; the outer asyncio.wait_for is the backstop either way.
    """
    key = get_settings().deepgram_api_key.get_secret_value()
    if not key:
        log.warning("no DEEPGRAM_API_KEY: AMD falls back to the default STT, "
                    "which is the livekit/agents#6996 stall configuration")
        return None
    return deepgram.STT(model="nova-3", api_key=key)


@dataclass(frozen=True)
class DialRequest:
    to_number: str
    brief: str
    callback_id: str | None = None
    require_pin: bool = True
    task_id: str | None = None
    task_titles: tuple[str, ...] = ()

    @staticmethod
    def parse(metadata: str | None) -> DialRequest | None:
        if not metadata:
            return None
        try:
            d = json.loads(metadata)
        except ValueError:
            return None
        if not d.get("to_number"):
            return None
        return DialRequest(
            to_number=d["to_number"],
            brief=d.get("brief", ""),
            callback_id=d.get("callback_id"),
            # Not switchable from the wire. The supervisor already computes
            # this from Settings.pin_required, which returns True for every
            # outbound call under every policy -- so hand-crafted dispatch
            # metadata must not be able to reach a lower bar than the code path.
            # Two independent layers, the same shape as the number allowlist.
            require_pin=True,
            task_id=d.get("task_id"),
            task_titles=tuple(d.get("task_titles") or ()),
        )


# Outcomes where the call never reached a live audio path. There is nobody to
# greet and nothing to withhold from, so the flow stops immediately.
UNANSWERED = frozenset({"no_answer", "busy", "rejected", "trunk_error"})

# Positively identified as not-a-person. These block disclosure outright and are
# NOT eligible for the greet-and-listen second chance: a voicemail greeting
# talks, which is exactly how it got classified.
MACHINE = frozenset({"machine-vm", "machine-ivr", "machine-unavailable"})


@dataclass(frozen=True)
class DialOutcome:
    kind: str                 # human|machine-vm|machine-ivr|machine-unavailable|
                              # uncertain|no_answer|busy|rejected|trunk_error
    sip_status_code: int | None = None
    amd_delay_ms: int | None = None
    amd_speech_s: float | None = None
    amd_reason: str | None = None   # llm|short_greeting|no_speech_timeout|
                                    # detection_timeout|participant_missing

    @property
    def answered(self) -> bool:
        return self.kind not in UNANSWERED


def classify_sip_error(e: api.SipCallError) -> DialOutcome:
    """A ring-out produces NO sip code at all — LiveKit sends CANCEL and raises
    a bare timeout — so that case never reaches here. These are the codes that
    genuinely come back from the far end."""
    code = getattr(e, "sip_status_code", None)
    kind = {486: "busy", 600: "busy", 603: "rejected",
            480: "no_answer", 408: "no_answer", 404: "trunk_error"}.get(code or 0, "trunk_error")
    return DialOutcome(kind=kind, sip_status_code=code)


async def dial_and_classify(ctx: JobContext, session: AgentSession,
                            dial: DialRequest) -> DialOutcome:
    s = get_settings()
    identity = f"sip-{dial.callback_id or 'call'}"

    try:
        # AMD must be entered BEFORE the participant is created, or the greeting
        # audio it classifies on is already gone.
        amd_kwargs = {}
        if (_stt := _amd_stt()) is not None:
            amd_kwargs["stt"] = _stt
        async with AMD(
            session,
            # A RealtimeModel is not an llm.LLM, so AMD cannot fall back to the
            # session's model; it must be given one explicitly.
            llm="openai/gpt-4.1-mini",
            # Streaming STT, pinned deliberately: livekit/agents#6996 (open) has
            # AMD stalling 17-20s on ~7% of answered calls with the DEFAULT
            # config. Without a Deepgram key we fall back to that default and
            # accept the risk, but say so loudly rather than failing closed.
            # make `timeout` a hard cap once speech starts
            wait_until_finished=False,
            detection_options={"timeout": s.amd_detection_timeout_seconds,
                               "no_speech_threshold": s.amd_no_speech_seconds},
            participant_identity=identity,
            **amd_kwargs,
        ) as detector:
            await ctx.api.sip.create_sip_participant(api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=s.livekit_sip_outbound_trunk_id,
                sip_call_to=dial.to_number,
                sip_number=s.twilio_phone_number,
                participant_identity=identity,
                participant_name="callee",
                wait_until_answered=True,
                # Without this the SDK silently pins 30s, and a mobile rolls to
                # voicemail at ~20-25s — so we would reliably reach voicemail.
                ringing_timeout=Duration(seconds=s.ringing_timeout_seconds),
                krisp_enabled=False,   # cancellation belongs on the agent
            ))
            # outer belt for #6996: never let AMD hold the call in dead air
            verdict = await asyncio.wait_for(detector.execute(),
                                         timeout=s.amd_wait_seconds)

    except api.SipCallError as e:
        log.info("sip error: %s", getattr(e, "sip_status_code", None))
        return classify_sip_error(e)
    except TimeoutError:
        # Two different timeouts land here. A ring-out has sip_status_code=None
        # because LiveKit sends CANCEL rather than surfacing a SIP status.
        return DialOutcome(kind="no_answer")

    speech = getattr(verdict, "speech_duration", None)
    cat = getattr(verdict, "category", None)
    reason = getattr(verdict, "reason", None)
    # .value, NOT str(): AMDCategory is a (str, Enum), so str() yields
    # "AMDCategory.HUMAN" and the disclosure gate would never match "human"
    # again -- withholding results from the real user on every callback.
    kind = getattr(cat, "value", None) or (cat if isinstance(cat, str) else "uncertain")
    return DialOutcome(
        kind=kind,
        # the detector's own delay, not wall time since dispatch: t0 includes
        # ring time, which would make every call look like a #6996 stall.
        amd_delay_ms=int(float(getattr(verdict, "delay", 0.0)) * 1000),
        # 0.0 is the documented tell that no VAD boundary ever reached the
        # classifier — the #6996 signature.
        amd_speech_s=float(speech) if speech is not None else None,
        amd_reason=str(reason) if reason else None,
    )


# Answered kinds we will read RESULTS onto. An allowlist, not "anything that is
# not a machine": a new AMDCategory shipped by a future SDK release must
# withhold by default, and a denylist would silently open for it.
DISCLOSABLE = frozenset({"human", "uncertain"})


def disclosure_allowed(outcome: DialOutcome) -> bool:
    """May we read the owner's results onto this line?

    Two gates, two questions. This one asks *is a person there*; the PIN asks
    *is it the owner*. Both must pass, and this one is not switchable off -- turning
    the PIN off for a demo must never let the agent read task results into a
    voicemail box.

    Rules, in order:
      - never before the far end answered;
      - never for a positively-detected machine. That is the invariant the whole
        gate exists for, and it is written out below even though DISCLOSABLE
        already excludes it, because this is where people come looking for it;
      - `human` and `uncertain` open it; anything else does not.

    `uncertain` used to need a live voice answering our greeting. That rule is
    gone with the thing that made it necessary: nothing about the work is said
    before the PIN now, so AMD no longer has to be certain before the agent may
    speak. The PIN is strictly stronger evidence than "a voice answered" -- a
    voicemail box cannot press a key -- and the old rule deadlocked, because the owner
    answers and waits for the agent while AMD waits for the owner.

    Note what this gate does NOT cover any more: the greeting. It names nothing,
    so everyone who answers hears it, a detected machine included. AMD is not
    infallible, and a person misclassified as a machine getting dead air and a
    dropped line is a worse failure than a voicemail box recording an apology.

    NOTE this function must be the only place these rules live. It previously
    existed, was asserted by eight tests, and was called by nothing -- while
    production ran a bare `if outcome.kind != "human"` somewhere else entirely.
    """
    if not outcome.answered:
        return False
    if outcome.kind in MACHINE:
        return False
    return outcome.kind in DISCLOSABLE
