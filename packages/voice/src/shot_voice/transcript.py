"""What was actually said on the call, written down where it survives.

Every callback failure this system has had was diagnosed from the outside --
Twilio durations, AMD categories, a `delivered=False` flag -- because the words
were never kept anywhere. The last one ended with three provable facts and one
unanswerable question: the model called `end_call` six seconds into an inbound
call, and nothing on this machine or in LiveKit could say what the owner had just
said to it. `enable_recording` produced no egress, journald logs no transcript
at INFO, and the dashboard needs a browser login.

So: one row per call, the turns as jsonb, written at shutdown. And every turn
also goes to the log as it happens, because a flush that never runs (the
supervisor is down, the process is killed) must not take the evidence with it.

`ChatMessage.text_content` on an interrupted assistant turn is the transcript
TRUNCATED AT THE PLAYBACK POSITION -- what reached his ear, not what the model
generated. That is precisely the thing worth recording, so `interrupted` is
kept alongside it: "he heard half of this" is a different fact from "he heard
this".
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger("shot.transcript")

# One turn is at most this many characters in the stored row. A realtime model
# does not produce anything close on a phone call, but a runaway generation
# must not be able to write an unbounded row.
MAX_TURN_CHARS = 4000
MAX_TURNS = 400
# A tool result is noisier per useful byte than speech, and the point is to be
# able to answer "what did the search actually return", not to keep a full copy
# of every payload.
MAX_TOOL_CHARS = 600


class Transcript:
    """Accumulates the conversation, then writes it once.

    Deliberately passive: it only listens. Nothing here may raise into the call
    path -- a bookkeeping failure that drops a call would be a far worse bug
    than the blind spot it is here to close.
    """

    def __init__(self, *, room_name: str, direction: str,
                 callback_id: str | None = None) -> None:
        self.room_name = room_name
        self.direction = direction
        self.callback_id = callback_id
        self.turns: list[dict[str, Any]] = []
        self._started = time.time()
        self._session = None
        # Token headroom, for _on_metrics. Peak rather than total: the ceiling
        # is per minute and per response, not per call.
        self.peak_input_tokens = 0
        self.peak_cached_tokens = 0
        self.responses = 0

    # ------------------------------------------------------------- capture

    def listen(self, session) -> None:
        """Start recording. Safe to call before anyone is on the line.

        TWO listeners, and it has to be two. `conversation_item_added` carries
        only ChatMessages -- tool items go through a separate internal path that
        emits nothing -- so a tool call was invisible here no matter how the
        type filter below was written. That is why nobody could prove what Brave
        actually returned on the call that got escalated to a worker session:
        the only record of that search was the model's own paraphrase of it.
        """
        self._session = session
        for name, fn in (("conversation_item_added", self._on_item),
                         ("function_tools_executed", self._on_tools),
                         ("error", self._on_error),
                         ("metrics_collected", self._on_metrics)):
            try:
                session.on(name, fn)
            except Exception:
                log.warning("could not attach the %s recorder", name, exc_info=True)

    def stop(self) -> None:
        if self._session is None:
            return
        for name, fn in (("conversation_item_added", self._on_item),
                         ("function_tools_executed", self._on_tools),
                         ("error", self._on_error),
                         ("metrics_collected", self._on_metrics)):
            try:
                self._session.off(name, fn)
            except Exception:
                pass
        self._session = None

    def note(self, text: str) -> None:
        """Mark a milestone in line with the words, e.g. `pin verified`.

        The transcript alone cannot explain a hang-up; interleaving what the
        code decided with what was said is what makes one readable end to end.
        """
        self._append(role="note", text=text, interrupted=False)

    def _on_item(self, ev) -> None:
        try:
            item = getattr(ev, "item", None)
            if getattr(item, "type", None) != "message":
                return                      # handoffs and tool items are not speech
            text = (item.text_content or "").strip()
            if not text:
                return
            self._append(role=str(item.role), text=text,
                         interrupted=bool(getattr(item, "interrupted", False)),
                         at=getattr(item, "created_at", None))
        except Exception:
            log.warning("could not record a turn", exc_info=True)

    def _on_tools(self, ev) -> None:
        """One row per tool call: what was asked, and what came back.

        The event carries the calls and their outputs as parallel lists and
        fires on the realtime path, including for calls cut short by an
        interruption.
        """
        try:
            calls = list(getattr(ev, "function_calls", None) or [])
            outs = list(getattr(ev, "function_call_outputs", None) or [])
            # zip defensively: the event validates these as parallel, but a
            # mismatched pair must never raise into a live call.
            # strict=False deliberately: a truncated pair should record what it
            # can, not raise into a live call.
            for call, out in zip(calls, outs, strict=False):
                name = getattr(call, "name", "?")
                args = " ".join(str(getattr(call, "arguments", "") or "").split())
                body = " ".join(str(getattr(out, "output", "") or "").split())
                arrow = "!>" if getattr(out, "is_error", False) else "->"
                self._append(role="tool",
                             text=f"{name}({args[:200]}) {arrow} {body[:MAX_TOOL_CHARS]}",
                             interrupted=False)
        except Exception:
            log.warning("could not record a tool call", exc_info=True)

    def _on_error(self, ev) -> None:
        """A generation the model never got to make.

        On 2026-09-10 five responses in one call came back
        `response failed: [tokens] rate_limit_exceeded` -- the account's 40,000
        TPM ceiling, blown by 78KB of Linear MCP tool schema being re-charged on
        every response. The model cannot see that and cannot recover from it: it
        simply says nothing. The owner asked it to check his day and got thirty-one
        seconds of silence.

        NOTHING recorded it. The plugin sets the exception on a future nobody
        awaits, so it surfaced only as asyncio's "Future exception was never
        retrieved" under the `asyncio` logger -- no shot.* line, no row in
        shot.calls, and no turn here. From the transcript alone the agent had
        simply chosen not to answer, which is indistinguishable from a routing
        bug and sent the investigation to the prompts first.

        So: one note, in line with the words, saying the model was stopped
        rather than silent.
        """
        try:
            err = getattr(ev, "error", None)
            # RealtimeModelError wraps the real exception one deeper; the
            # message is what names the cause, and the bare class name is what
            # once turned a wiring bug into a bland "no input" for two calls.
            inner = getattr(err, "error", None)
            detail = " ".join(str(inner if inner is not None else err).split())
            recoverable = getattr(err, "recoverable", None)
            self._append(role="note",
                         text=f"the model could not answer: {detail[:300]}"
                              + ("" if recoverable is None
                                 else f" (recoverable={recoverable})"),
                         interrupted=False)
        except Exception:
            log.warning("could not record a model error", exc_info=True)

    def _on_metrics(self, ev) -> None:
        """How close this call ran to the token ceiling.

        Accumulated, never appended: this fires once per response, and a row
        each would bury the words. One line at flush is what turns "why did it
        go quiet" into a grep -- the ceiling is per MINUTE and counts the whole
        tool schema plus history on every response, so the peak is the number
        that matters, and cached tokens are charged too.
        """
        try:
            m = getattr(ev, "metrics", ev)
            # An event we cannot read tells us nothing, and counting it would
            # report a peak measured over fewer responses than it claims.
            raw = getattr(m, "input_tokens", None)
            if raw is None:
                return
            got = int(raw or 0)
            if got > self.peak_input_tokens:
                self.peak_input_tokens = got
                d = getattr(m, "input_token_details", None)
                self.peak_cached_tokens = int(getattr(d, "cached_tokens", 0) or 0)
            self.responses += 1
        except Exception:
            log.warning("could not record model metrics", exc_info=True)

    def _append(self, *, role: str, text: str, interrupted: bool,
                at: float | None = None) -> None:
        """`at` is the epoch time the turn actually HAPPENED, when we know it.

        Without it this stamps the moment the event reached us, and for a
        realtime model those are not the same thing. The model answers your
        AUDIO; the text transcription of it comes back on a slower, separate
        channel, and can land after the reply it caused. On 2026-09-10 the
        transcript therefore read:

            06:11:18.779  note  the model called end_call
            06:11:18.879  user  Okay, this one.

        which says the agent hung up and THEN the owner spoke. The truth is the
        reverse -- he dismissed it, the model obeyed, and his words were
        transcribed a tenth of a second later. Read in the recorded order it
        is a report of the agent hanging up on him unprompted, and that is
        exactly how it was read.

        The SDK already solves this and says so at `agent_activity.py`: "a
        provider may withhold the final transcript until its reply has
        finished generating, which would otherwise stamp the turn after the
        reply it prompted". It puts the true time on `ChatMessage.created_at`.
        This used to throw that away.
        """
        if len(self.turns) >= MAX_TURNS:
            return
        text = text[:MAX_TURN_CHARS]
        # Clamped: a bogus provider timestamp must not reorder the call or
        # produce a negative offset. Falls back to "now" outside the call.
        now = time.time()
        when = now if at is None else min(max(at, self._started), now)
        self.turns.append({"at": round(when - self._started, 2),
                           "role": role, "text": text, "interrupted": interrupted})
        # Straight to the log too. A flush that never runs must not take the
        # evidence with it -- which is exactly how the last investigation ended
        # up reconstructing a call from Twilio durations.
        log.info("turn %s%s: %s", role, " [interrupted]" if interrupted else "", text)

    # -------------------------------------------------------------- persist

    async def flush(self) -> None:
        """Write the call down, in the order it happened. Never raises."""
        # Turns arrive out of order -- a user turn stamped at the moment he
        # started speaking can be appended after the reply it prompted. Sort
        # once, here, so the stored row reads chronologically. The per-turn
        # log line above is still emitted as-observed, on purpose: it is the
        # evidence that survives a flush that never runs.
        self.turns.sort(key=lambda t: t["at"])
        if self.responses:
            log.info("tokens: %d response(s), peak input %d (%d cached) -- the "
                     "realtime ceiling is per minute and charges cached tokens too",
                     self.responses, self.peak_input_tokens, self.peak_cached_tokens)
        if not self.turns:
            log.info("no turns to record for %s", self.room_name)
            return
        import httpx

        from shot_core.budget import SUPERVISOR_HTTP_S
        from shot_core.settings import get_settings
        try:
            async with httpx.AsyncClient(base_url=get_settings().supervisor_base_url,
                                         timeout=SUPERVISOR_HTTP_S) as c:
                r = await c.post("/internal/transcripts", json={
                    "room_name": self.room_name,
                    "direction": self.direction,
                    "callback_id": self.callback_id,
                    "turns": self.turns,
                })
                r.raise_for_status()
            log.info("recorded transcript: %d turn(s) for %s",
                     len(self.turns), self.room_name)
        except Exception:
            log.warning("could not record the transcript for %s", self.room_name,
                        exc_info=True)
