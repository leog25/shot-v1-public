"""Who is on the line, decided from anything they type or say, all call long.

Replaces livekit's `GetDtmfTask`, which caused two of the four callback
failures the owner hit in one evening -- it refuses to be awaited outside `on_enter`,
and it raises "the activity that awaited the inline task is closing" when the
caller hangs up, which the retry loop counted as three wrong codes in the same
millisecond. It also re-prompts once per attempt, so the owner was asked for his code
three times with no room to answer.

The deeper problem was the window. It only listened once the challenge started,
so a code typed during the greeting was thrown away. Here the buffer opens when
the call does: press the digits whenever you like.
"""

from __future__ import annotations

import asyncio
import logging
import re

from shot_voice.pin import PinResult, normalize, verify

log = logging.getLogger("shot.identity")

# "one two three four" as the transcriber renders it. `oh` and `o` are how
# people actually say zero on the phone.
_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "nought": "0",
    "one": "1", "two": "2", "to": "2", "too": "2", "three": "3", "four": "4",
    "for": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "ate": "8",
    "nine": "9",
}
_WORD_RE = re.compile(r"[a-z]+")


def digits_in(text: str) -> str:
    """Every digit in a transcript, however it was written.

    The transcriber may render spoken digits either way -- "1234" or "one two
    three four" -- and often mixes them, so both are read in order.
    """
    out: list[str] = []
    for token in re.findall(r"\d|[A-Za-z]+", text or ""):
        if token.isdigit():
            out.append(token)
        elif (word := token.lower()) in _WORDS:
            out.append(_WORDS[word])
    return "".join(out)


class Identity:
    """Collects keypresses and spoken digits, and says when the code appears.

    Matching is on a rolling window, so a stray keypress before the code -- or
    a menu digit, or a misheard word -- cannot poison the rest of the call.
    """

    def __init__(self, *, expected: str | None = None, length: int = 4) -> None:
        self._expected = expected
        self._length = length
        self._buffer = ""
        self._hit = asyncio.Event()
        self._gone = asyncio.Event()
        self._saw_input = False
        self._room = None
        self._session = None

    # ------------------------------------------------------------ lifecycle

    def listen(self, room, session) -> None:
        """Open the buffer. Called before the dial, so nothing is missed."""
        self._room, self._session = room, session
        room.on("sip_dtmf_received", self._on_dtmf)
        room.on("participant_disconnected", self._on_gone)
        session.on("user_input_transcribed", self._on_transcript)

    def stop(self) -> None:
        if self._room is not None:
            self._room.off("sip_dtmf_received", self._on_dtmf)
            self._room.off("participant_disconnected", self._on_gone)
        if self._session is not None:
            self._session.off("user_input_transcribed", self._on_transcript)
        self._room = self._session = None

    # -------------------------------------------------------------- inputs

    def _on_dtmf(self, ev) -> None:
        self.feed(getattr(ev, "digit", "") or "")

    def _on_gone(self, ev) -> None:
        """The far end left. Only the callee is ever a remote participant here.

        This is what makes ABANDONED real. Without it a caller who hangs up
        mid-challenge times out and is recorded as NO_INPUT -- which becomes
        `pin_failed`, which is TERMINAL, which buries a finished task forever.
        """
        self._gone.set()

    def _on_transcript(self, ev) -> None:
        if getattr(ev, "is_final", False):
            self.feed(digits_in(getattr(ev, "transcript", "") or ""))

    def feed(self, text: str) -> None:
        """Add whatever arrived; never log the digits themselves."""
        got = normalize(text)
        if not got:
            return
        self._saw_input = True
        # Keep a little more than the code so a rolling match still works after
        # a stray press, but not the whole call.
        self._buffer = (self._buffer + got)[-(self._length * 4):]
        log.info("identity: %d digit(s) received", len(got))
        if self._matches():
            self._hit.set()

    def _matches(self) -> bool:
        n = self._length
        return any(verify(self._buffer[i:i + n], expected=self._expected)
                   for i in range(max(0, len(self._buffer) - n + 1)))

    # ------------------------------------------------------------- verdict

    @property
    def already_verified(self) -> bool:
        """Did the code arrive before we even asked? Then do not ask."""
        return self._hit.is_set()

    async def verified(self, within: float) -> PinResult:
        """Wait for the code, returning the moment it appears.

        Races the code against the caller leaving, because "I could not ask" is
        not "he got it wrong". ABANDONED had no producer at all until now: a
        caller who hung up mid-challenge fell through to NO_INPUT, and NO_INPUT
        becomes `pin_failed`, which is terminal and buries a finished task. A
        hangup must leave the task unreported so he is told on his next call.
        """
        if self._hit.is_set():
            return PinResult.VERIFIED
        hit = asyncio.ensure_future(self._hit.wait())
        gone = asyncio.ensure_future(self._gone.wait())
        try:
            done, _ = await asyncio.wait(
                {hit, gone}, timeout=within, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (hit, gone):
                t.cancel()
        # Order matters: a code that landed on the same tick as the disconnect
        # is still a code, and VERIFIED is the kinder of the two readings.
        if hit in done:
            return PinResult.VERIFIED
        if gone in done:
            log.info("identity: the caller hung up during the challenge")
            return PinResult.ABANDONED
        if not self._saw_input:
            log.info("identity: nothing entered in %ss", within)
            return PinResult.NO_INPUT
        log.info("identity: digits received but never matched")
        return PinResult.WRONG
