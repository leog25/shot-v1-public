"""Make the agent say a prepared line instead of improvising one.

`gpt-realtime-2.1` reports `supports_say = False`, so there is no exact-speech
API. And the OpenAI plugin CONCATENATES a per-response instruction onto the
session prompt (realtime_model.py):

    instructions = f"{self._instructions}\n{instructions}"

so a 300-character script arrives as 11% of a 2,500-character prompt whose
other 89% says "just talk with the owner" and "two sentences or fewer per turn".
Replaying a real call, that lost 3 times out of 3: asked to read the owner his
results, the agent instead answered his previous question -- "No, I don't need
it. Your identity is already verified." -- and he never heard the brief.

The lever is that same line: `if is_given(instructions) and self._instructions:`
An EMPTY session prompt is falsy, so the concatenation is skipped and the script
is the entire prompt. Scripted this way, the same replay delivered in full 3
times out of 3.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

log = logging.getLogger("shot.scripted")


@dataclass(frozen=True)
class Spoken:
    """What actually reached the caller's ear.

    Three outcomes, not two. Collapsing them into one `delivered` flag is what
    made the agent hang up on the owner six seconds into an eighty-second brief:
    "he interrupted me" and "nothing came out of the speaker" were the same
    value, and the code answered both by deleting the room.

        started=False                -> he heard nothing. A real failure.
        started, interrupted=True    -> he heard some of it AND he is talking.
        started, interrupted=False   -> he heard all of it.
    """

    started: bool            # an assistant message actually reached the line
    interrupted: bool        # cut off, by his voice or by the line going away
    text: str                # truncated at the playback position, not what was generated

    @property
    def delivered(self) -> bool:
        """Played to the end, uninterrupted. The bar for "the owner heard it"."""
        return self.started and not self.interrupted

    def __bool__(self) -> bool:
        return self.delivered


async def scripted(agent, session, script: str, *, limit: float,
                   interruptible: bool = True) -> Spoken:
    """Say `script`, with the session prompt out of the way.

    Restoring the real instructions must happen AFTER the handle resolves.
    `generate_reply` returns before anything is sent -- `response.create` is
    only issued later, inside the reply task -- so restoring early puts the long
    prompt back on the wire first and the dilution returns.
    """
    original = agent.instructions
    handle = None
    try:
        await agent.update_instructions("")
        handle = session.generate_reply(instructions=script,
                                        allow_interruptions=interruptible)
        try:
            await asyncio.wait_for(asyncio.shield(handle), timeout=limit)
        except TimeoutError:
            log.warning("scripted line did not finish within %ss", limit)
        except Exception as e:                      # the line went away mid-sentence
            log.info("scripted line interrupted: %s: %s", type(e).__name__, e)
    except Exception:
        # One line failing must not abort the call -- a sign-off that cannot be
        # spoken should still let us hang up and record what happened.
        log.exception("scripted line could not be started")
        handle = None
    finally:
        # Always restore. A session left on an empty prompt stays empty across
        # every later turn and gets re-pushed on any agent handoff.
        try:
            await agent.update_instructions(original)
        except Exception:
            log.exception("could not restore instructions after a scripted line")

    return spoken_from(handle)


def spoken_from(handle) -> Spoken:
    """Delivery is a fact on the handle, not an assumption.

    `wait_for_playout()` returns normally after an interruption, which is how a
    call where the agent said nothing about the task was twice recorded as
    delivered. `chat_items` carries the transcript truncated at the playback
    position -- what was heard, not what was generated -- and `skipped` speech
    produces no item at all.

    Public because the inbound news greeting needs exactly this too. It was
    still trusting `wait_for_playout()`, and stamped `reported_at` on a task the owner
    never heard about five seconds into a call the agent then hung up.
    """
    if handle is None:
        return Spoken(False, False, "")
    if not handle.done():
        # exception() RAISES InvalidStateError on a handle that is not finished,
        # and asyncio.wait_for timing out leaves exactly that. Calling it
        # unconditionally turned "the brief is still playing" into a crash that
        # took the whole call down.
        log.warning("scripted line still playing when read back")
        return Spoken(False, False, "")
    if (exc := handle.exception()) is not None:
        log.warning("scripted line failed: %s", exc)
        return Spoken(False, False, "")

    msgs = [i for i in handle.chat_items
            if getattr(i, "type", None) == "message" and getattr(i, "role", None) == "assistant"]
    text = " ".join(m.text_content or "" for m in msgs).strip()
    interrupted = bool(handle.interrupted) or any(
        getattr(m, "interrupted", False) for m in msgs)
    spoken = Spoken(bool(msgs), interrupted, text)
    if not spoken.delivered:
        log.warning("scripted line not fully delivered (started=%s, interrupted=%s, "
                    "items=%d, heard=%d chars)",
                    spoken.started, interrupted, len(msgs), len(text))
    return spoken
