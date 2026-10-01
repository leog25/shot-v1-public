"""The callback, as one linear decision tree.

This used to be split across the job entrypoint and CallbackAgent.on_enter,
coordinated by an asyncio.Event and a polling loop, with the greeting in two
places behind a flag. Every callback bug that reached the owner's phone lived in that
seam, and none of it was reachable by a test: nothing in the suite ever entered
the outbound branch of the entrypoint.

So the decision tree lives here, depending only on `CallbackIO` -- a handful of methods,
no session, no room, no network. Production passes the real implementation;
tests pass a scripted fake and assert the whole tree in milliseconds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from shot_core.budget import (
    BRIEF_PLAYOUT_S,
    GREETING_PLAYOUT_S,
    OFFER_MORE_S,
    PIN_WINDOW_S,
    SIGNOFF_PLAYOUT_S,
)
from shot_voice.outbound import DialOutcome, disclosure_allowed
from shot_voice.pin import PinResult
from shot_voice.prompts import (
    BRIEF_FAILED_CALLBACK,
    DELIVER_CALLBACK,
    OFFER_MORE,
    OPENING_CALLBACK,
    OPENING_CALLBACK_VERIFIED,
    PIN_FAILED_CALLBACK,
    WITHHELD_CALLBACK,
)
from shot_voice.scripted import Spoken

log = logging.getLogger("shot.callback")


@dataclass
class CallRecord:
    """One row of shot.calls. Written on every attempt, including the ones that
    fail early -- the whole point is that a failed callback leaves evidence
    that outlives the log file."""

    outcome: str = "no_report"
    amd_category: str | None = None
    amd_reason: str | None = None
    amd_delay_ms: int | None = None
    amd_speech_s: float | None = None
    sip_status_code: int | None = None
    pin_verified: bool | None = None
    # verified|wrong|no_input|abandoned. Reaches shot.calls.pin_result: "nobody
    # keyed anything" and "he keyed the wrong thing" are different calls.
    pin_result: str | None = None
    delivered: bool = False
    stayed_on_line: bool = False
    # How much of the brief actually reached him before he spoke over it. 0 with
    # delivered=False is the real failure; a large number is just a conversation
    # starting, and used to be answered by hanging up on him.
    heard_chars: int = 0
    interrupted: bool = False
    task_ids: list[str] = field(default_factory=list)


class CallbackIO(Protocol):
    """Everything the flow needs from the outside world."""

    async def dial(self) -> DialOutcome: ...
    async def say(self, script: str, *, limit: float,
                  interruptible: bool = True) -> Spoken: ...
    def code_already_entered(self) -> bool: ...
    async def wait_for_code(self, limit: float) -> PinResult: ...
    async def hang_up(self) -> None: ...
    async def hand_back(self, *, delivered: bool) -> None: ...


def _phrase(titles: list[str]) -> str:
    """What to tell the owner came back, in words he will recognise."""
    titles = [t for t in titles if t]
    if not titles:
        return "what you asked me to look into"
    if len(titles) == 1:
        return titles[0]
    return f"{', '.join(titles[:-1])} and {titles[-1]}"


async def run_callback(io: CallbackIO, *, brief: str, require_pin: bool,
                       task_ids: list[str] | None = None,
                       task_titles: list[str] | None = None) -> CallRecord:
    """Dial, decide who answered, and deliver only if allowed to.

    Always returns a CallRecord; never raises for an ordinary failure.
    """
    rec = CallRecord(task_ids=list(task_ids or []))

    outcome = await io.dial()
    rec.amd_category = outcome.kind
    rec.amd_reason = outcome.amd_reason
    rec.amd_delay_ms = outcome.amd_delay_ms
    rec.amd_speech_s = outcome.amd_speech_s
    rec.sip_status_code = outcome.sip_status_code
    log.info("dial outcome=%s reason=%s speech=%s sip=%s",
             outcome.kind, outcome.amd_reason, outcome.amd_speech_s,
             outcome.sip_status_code)

    if not outcome.answered:
        # Nobody picked up. Nothing to say and nobody to say it to.
        rec.outcome = outcome.kind
        return rec

    # Greet EVERYONE who answered, and name nothing. Nothing about the work is
    # said until the code checks out, so the classifier no longer has to be
    # right before the agent may speak -- which is what let the old flow greet
    # from two places behind a flag, and what made `uncertain` a deadlock: the owner
    # answers and waits for the agent, AMD waits for the owner, six seconds later the
    # call gave up on him. A detected machine hears this too, deliberately:
    # AMD is not infallible, and a misclassified person getting dead air and a
    # dropped line is worse than a voicemail box recording nine seconds of an
    # apology.
    allowed = disclosure_allowed(outcome)
    early = allowed and require_pin and io.code_already_entered()
    opening = (OPENING_CALLBACK_VERIFIED if allowed and (early or not require_pin)
               else OPENING_CALLBACK)
    await io.say(opening, limit=GREETING_PLAYOUT_S, interruptible=False)

    if not allowed:
        # Short-circuit HERE rather than after the PIN. A machine cannot press a
        # key, so waiting out PIN_WINDOW_S would only record another twenty-five
        # seconds of the agent talking to itself onto a voicemail box.
        rec.outcome = outcome.kind
        await io.say(WITHHELD_CALLBACK, limit=SIGNOFF_PLAYOUT_S)
        await io.hang_up()
        return rec

    if require_pin:
        result = PinResult.VERIFIED if early else await io.wait_for_code(PIN_WINDOW_S)
        rec.pin_verified = result is PinResult.VERIFIED
        rec.pin_result = result.value
        if result is PinResult.ABANDONED:
            # The line went away mid-challenge. Not a failed identity check --
            # recording it as one would bury a finished task forever.
            rec.outcome = "abandoned"
            return rec
        if result is not PinResult.VERIFIED:
            rec.outcome = "pin_failed"
            await io.say(PIN_FAILED_CALLBACK, limit=SIGNOFF_PLAYOUT_S)
            await io.hang_up()
            return rec

    rec.outcome = "human"
    # Computed HERE, not at the top of the call: this is the first point at
    # which it may legally be said, and keeping it out of scope until now makes
    # it impossible to reference the titles in an earlier turn by accident.
    what = _phrase(task_titles or [])
    # delivered reflects what actually played. It was previously set to True
    # simply because say() had been called, and a call where the agent never
    # read the brief was filed as a success.
    spoken = await io.say(DELIVER_CALLBACK.format(what=what, brief=brief),
                          limit=BRIEF_PLAYOUT_S)
    rec.delivered = spoken.delivered
    rec.interrupted = spoken.interrupted
    rec.heard_chars = len(spoken.text)

    if not spoken.started:
        # Nothing came out of the speaker at all -- a skipped generation, a
        # failed turn, or a line that went away. This is the only delivery
        # failure worth ending the call over, and even here we say something
        # first rather than dropping him into silence.
        log.error("the brief never played; leaving the task unreported")
        await io.say(BRIEF_FAILED_CALLBACK, limit=SIGNOFF_PLAYOUT_S)
        await io.hang_up()
        return rec

    if spoken.interrupted:
        # He talked over it. That is proof he is on the line and listening --
        # the exact moment to STOP talking and answer him, and the exact moment
        # this used to hang up on him instead, six seconds into an eighty-second
        # brief. `_holding` clears in hand_back(), so the turn he is part-way
        # through gets a real reply, and he can just ask for the rest.
        #
        # `delivered` stays False here, so the task stays unreported and he is
        # told again on his next inbound call. That is deliberate: we cannot
        # prove he heard the part he spoke over, and being told twice is a far
        # cheaper mistake than burying a finished task forever.
        log.info("the brief was interrupted after %d chars; handing back so he "
                 "can be answered", rec.heard_chars)
        rec.stayed_on_line = True
        await io.hand_back(delivered=False)
        return rec

    # The whole brief landed, so hand the call back BEFORE the last line rather
    # than after it. The floor is held so the model cannot improvise over a
    # prepared line -- and the brief IS the prepared line. Once it has landed
    # there is nothing left to protect, and holding on through "anything else?"
    # created a seam that swallowed him whole:
    #
    # he talked over "Is there anything else you'd like me--", the flow resolved
    # the handle and handed back, his turn completed a second later, and NOTHING
    # was generated. Thirteen seconds of silence, then he hung up and rang back.
    # Every callback where this line finished uninterrupted replied to his next
    # turn normally; the one where he cut in did not. Releasing first takes the
    # handback out of the interruption's path entirely: a barge-in here is now
    # an ordinary turn, answered the ordinary way.
    #
    # Hanging up is not the alternative -- the agent has all its tools and he
    # often has the next thing in mind while it is there. He ends the call, or
    # goes quiet long enough that the idle watchdog does.
    rec.stayed_on_line = True
    await io.hand_back(delivered=True)

    # Its own delivery does not matter and is deliberately not measured: asked
    # as the last line of the brief it invited him to answer over the tail, and
    # an interruption there would have marked a brief he heard in full as
    # undelivered.
    await io.say(OFFER_MORE, limit=OFFER_MORE_S)
    return rec
