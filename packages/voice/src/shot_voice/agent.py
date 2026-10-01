from __future__ import annotations

import asyncio
import logging

from livekit.agents import (
    Agent,
    RunContext,
    StopResponse,
    function_tool,
    get_job_context,
)
from livekit.agents.beta import EndCallTool
from livekit.agents.llm import ChatContext

from shot_core.budget import (
    GREETING_START_S,
    IDLE_HANGUP_S,
    NEWS_GREETING_S,
    NEWS_OFFER_S,
    SIGNOFF_PLAYOUT_S,
)
from shot_core.clock import now_local, part_of_day, spoken_now
from shot_voice.callback_flow import CallRecord, run_callback
from shot_voice.identity import Identity
from shot_voice.outbound import DialOutcome, DialRequest
from shot_voice.pin import PinResult
from shot_voice.prompts import (
    AFTER_CALLBACK,
    AFTER_CALLBACK_CUT,
    IDLE_SIGNOFF,
    IDLE_SIGNOFF_CUT,
    INSTRUCTIONS,
    OPENING_INBOUND,
    OPENING_INBOUND_NEWS,
    OPENING_INBOUND_OFFER,
    OWNER,
    owner_named,
)
from shot_voice.scripted import Spoken, scripted, spoken_from
from shot_voice.tools import (
    cancel_task,
    check_my_day,
    comment_on_linear_issue,
    find_linear_issue,
    get_task_status,
    list_my_tasks,
    start_background_task,
    web_search,
    write_linear_issue,
)
from shot_voice.transcript import Transcript

log = logging.getLogger("shot.agent")

# How often the idle watchdog checks. Its own constant, not a literal inside the
# loop, so an offline test can shrink it -- IDLE_HANGUP_S alone is not enough.
IDLE_POLL_S = 5.0


def _after_prompt(delivered: bool, brief: str) -> str:
    """Which session prompt the agent runs under once the callback is over.

    Split out so the choice is unit-testable: the wiring that installs it is
    only reachable through a live AgentSession, but which one is correct is a
    plain question about a boolean.
    """
    return AFTER_CALLBACK if delivered else AFTER_CALLBACK_CUT.format(brief=brief)

# Attached to end_call itself. The tool description is read right where the
# decision is made, which the session prompt demonstrably is not.
EXTRA_END_CALL = f"""
{OWNER} ends calls himself, almost always by hanging up. Ending one on him while he
still wants something is far worse than staying on a quiet line -- an idle call
closes itself, so there is no cost to waiting.

Do NOT call this:
- because you have finished saying something, or answered his question
- straight after your opening greeting; he called for a reason and has not
  spoken yet
- when he sounds annoyed, confused, or asks what happened
- when anything you told him about is still unexplained, or he cut you off
  part-way through and has not heard the rest
- on any silence, however long

If he interrupts you, that is proof he is listening and it is never a reason to
hang up. Stop talking and answer what he actually said -- do not restart the
line he cut into and do not carry on over him.

DO call it as soon as he signals the call is over, however he phrases it:
"bye", "that's all", "that's it", "dismissed", "nothing else", "we're good",
"talk later", "thanks, that's everything", or telling you to hang up. Say one
short goodbye and call it in the same turn -- do not answer first, do not offer
more, do not ask whether he is sure.

If YOU placed this call and he dismisses you, ending it is the polite thing to
do, not the rude one. Do not hold open a call he has closed.

If you genuinely cannot tell whether he is done, ask one short question. Do NOT
start a new topic and do NOT greet him.

A warmer, funnier voice is not a reason to hold the line. When he closes the
call, close it -- ending on his cue is the polite move, not the abrupt one.
"""


def _news_block(news: list[dict]) -> str:
    """The finished work, with its actual findings, for the opening greeting.

    `/internal/context` has always returned a `gist` -- `result_summary` cut to
    a speakable length -- and on_enter threw it away and passed only titles. So
    the prompt told the model to "say the actual headline result, you already
    know it", while giving it nothing to say. It did the only thing it could:
    "It's done -- Harbor Lights Friday is fully researched and written up. If you
    want, I can read the highlights next." Then the task was stamped
    `reported_at` on a greeting that contained no findings at all, and had he
    hung up there it would have been buried as already told to him.
    """
    lines = []
    for n in news:
        title = (n.get("title") or "something").strip()
        gist = (n.get("gist") or "").strip()
        if not n.get("ok", True):
            lines.append(f"- {title}: did not finish")
        elif gist:
            lines.append(f"- {title}: {gist}")
        else:
            lines.append(f"- {title}: finished, but no summary was saved")
    return "\n".join(lines)


async def _mark_reported(task_ids: list[str]) -> None:
    """Record that the owner actually HEARD about these, as distinct from a callback
    having been scheduled for them -- the distinction that let two finished
    tasks be filed as reported while both callbacks failed."""
    import httpx

    from shot_core.settings import get_settings
    try:
        async with httpx.AsyncClient(
                base_url=get_settings().supervisor_base_url, timeout=5.0) as c:
            await c.post("/internal/tasks/reported", json={"task_ids": task_ids})
        log.info("marked %d task(s) reported", len(task_ids))
    except Exception:
        log.warning("could not mark tasks reported", exc_info=True)


class ShotAgent(Agent):
    def __init__(self, *, chat_ctx: ChatContext | None = None,
                 news: list[dict] | None = None,
                 transcript: Transcript | None = None) -> None:
        self._news = news or []
        # Set before super(), because the tools below bind methods that use it.
        self._tx = transcript
        tools = [
            # ignore_on_enter keeps the model from hanging up during the
            # greeting. The prebuilt tool is used rather than a hand-rolled
            # one because it drains audio correctly for a realtime model:
            # RunContext.wait_for_playout() only waits for speech from
            # BEFORE the tool ran, which is not the goodbye.
            EndCallTool(
                delete_room=True,
                ignore_on_enter=True,
                end_instructions="Say a short, warm goodbye.",
                # Guidance rides on the TOOL, not the session prompt. As one
                # line of a 2,500-character prompt, "when the owner says goodbye"
                # did not stop the model ending a call two seconds after its
                # own opening greeting -- the owner, who had just been hung up on
                # mid-brief and called straight back, was hung up on again.
                extra_description=EXTRA_END_CALL,
                on_tool_called=self._note_end_call,
            ),
            echo_time,
            start_background_task,
            list_my_tasks,
            get_task_status,
            cancel_task,
            check_my_day,
            web_search,
            find_linear_issue,
            write_linear_issue,
            comment_on_linear_issue,
        ]
        # There is deliberately no Linear MCP toolset here any more, and none
        # is passed in. It was 65 tools and 78KB of schema, and the realtime
        # API re-charges the WHOLE tool schema against a 40,000 TPM bucket on
        # every response -- so carrying it left room for two responses a
        # minute, and a tool-using turn needs two. The three Linear tools above
        # do the same job from one GraphQL client for ~1KB. See the ticket
        # section in tools.py.
        super().__init__(
            instructions=INSTRUCTIONS,
            chat_ctx=chat_ctx,
            tools=tools,
        )

    def note(self, text: str) -> None:
        """Put a milestone in the transcript, in line with the words."""
        if self._tx is not None:
            self._tx.note(text)

    async def _note_end_call(self, ev) -> None:
        """Leave a trace when the MODEL hangs up.

        `end_call` deleted a room six seconds into an inbound call, and the only
        evidence was a livekit line reading "deleting the room because the user
        ended the call" -- which means the model called the tool, not that the owner
        did. Distinguishing those took reading the SDK source.
        """
        log.info("executing tool end_call: the MODEL is ending the call")
        self.note("the model called end_call")

    @staticmethod
    async def seeded_chat_ctx() -> tuple[ChatContext, list[dict]]:
        """Seed a SUMMARY, never a transcript.

        Loading extensive history makes gpt-realtime reply in text only, and
        assistant turns must be `output_text` — which add_message() handles.

        Also returns the finished work the owner has NOT been told about, which
        decides how the call opens.
        """
        import httpx

        from shot_core.settings import get_settings
        ctx = ChatContext()

        # ABOVE the try, deliberately. That block swallows every exception and
        # returns an empty ChatContext, so a time line inside it would vanish
        # exactly when the supervisor is flaky -- leaving INSTRUCTIONS asserting
        # "you were told the day and the time" with nothing interpolated. An
        # instruction the model cannot follow is worse than none: that gap is
        # what produced "It's done -- if you want, I can read the highlights
        # next", on a greeting that contained no findings at all.
        #
        # Purely factual, and no imperative. Greeting by time of day is the
        # OPENING prompt's job; an instruction smuggled into an assistant
        # ChatContext message would leak into every later turn.
        #
        # "at the start of this call" so the model does not re-assert it as
        # "right now" twenty minutes in -- echo_time stays the authority for
        # the time NOW, and says so on the tool.
        now = now_local()
        ctx.add_message(
            role="assistant",
            content=(f"{OWNER}'s local time at the start of this call: "
                     f"{spoken_now(now)}. It is {part_of_day(now)} where he is."))
        try:
            async with httpx.AsyncClient(
                    base_url=get_settings().supervisor_base_url, timeout=3.0) as c:
                body = (await c.get("/internal/context")).json()
        except Exception:
            log.warning("no cross-call context available", exc_info=True)
            return ctx, []
        summary = body.get("summary") or ""
        news = body.get("news") or []
        if summary:
            ctx.add_message(role="assistant",
                            content=f"Context from our earlier calls: {summary}")
        log.info("seeded context: %d chars, %d unreported task(s): %s",
                 len(summary), len(news), [n.get("ref") for n in news])
        return ctx, news

    async def _say_opening(self, script: str) -> bool:
        """Say an opening line, and NOTICE when no audio comes out of it.

        This was fire-and-forget, which made the one failure that matters
        completely invisible. On 2026-09-10 an inbound call produced no greeting
        at all: no audio, no chat item, no error anywhere. The owner heard nine
        seconds of dead air, said "hello?", and hung up. The stored transcript
        holds two turns and neither is a greeting, and nothing in the log said
        the agent had failed to speak -- so from the outside it was
        indistinguishable from the phone number being broken.

        gpt-realtime does exactly this: a throttled session returns "no events
        at all -- no error, no 429". `spoken_from` cannot help, because SKIPPED
        speech produces no chat item to read back. The only reliable signal that
        audio actually started is the session entering the `speaking` state, so
        that is what this waits on.

        One retry, because a greeting that made no sound is precisely the thing
        worth saying again, and dead air on the front of a call costs the whole
        call. It does NOT retry once the owner has spoken: at that point the silence
        is his turn to fill, and talking over him is the worse failure.
        """
        for attempt in (1, 2):
            started = asyncio.Event()

            # `started` bound explicitly: the listener outlives the loop
            # iteration only briefly, but a late event from attempt one must
            # never set attempt two's flag.
            def _on_state(ev, _started=started) -> None:
                if getattr(ev, "new_state", None) == "speaking":
                    _started.set()

            self.session.on("agent_state_changed", _on_state)
            try:
                self.session.generate_reply(instructions=script)
                try:
                    await asyncio.wait_for(started.wait(), timeout=GREETING_START_S)
                    return True
                except TimeoutError:
                    pass
            finally:
                try:
                    self.session.off("agent_state_changed", _on_state)
                except Exception:
                    pass

            log.warning("the opening produced NO audio within %ss (attempt %d)",
                        GREETING_START_S, attempt)
            self.note(f"opening produced no audio (attempt {attempt})")

            if self._user_has_spoken():
                log.info("he is already talking; not repeating the opening over him")
                return False
            if attempt == 1:
                # Drop the silent generation before issuing another, or a late
                # one lands on top of the retry and he hears it twice.
                try:
                    self.session.interrupt()
                except Exception:
                    log.debug("could not interrupt the silent opening", exc_info=True)

        log.error("the caller is hearing DEAD AIR: two openings produced no audio")
        self.note("dead air: two openings produced no audio")
        return False

    def _user_has_spoken(self) -> bool:
        try:
            return any(getattr(i, "role", None) == "user"
                       for i in self.session.history.items)
        except Exception:
            return False

    async def on_enter(self) -> None:
        # Voice-side and synchronous. The supervisor round trip is already 3s
        # on the critical path to the first word and must not grow.
        now = now_local()
        when, part = spoken_now(now), part_of_day(now)

        # NOT session.say(): that needs a TTS plugin and raises with a bare
        # realtime model (RealtimeCapabilities.supports_say is False).
        if not self._news:
            # One turn, with the offer inside it. The split below exists only
            # because turn one's `delivered` gates reported_at; with no news
            # there is nothing to mark and nothing an interruption can mis-file.
            await self._say_opening(
                OPENING_INBOUND.format(part=part, when=when))
            return

        handle = self.session.generate_reply(
            instructions=OPENING_INBOUND_NEWS.format(
                news=_news_block(self._news), part=part, when=when))
        # NOT wait_for_playout(): it returns normally after an interruption, so
        # it is not proof of anything. Trusting it here stamped `reported_at` on
        # a task five seconds into a call, and the owner -- who had just been hung up
        # on mid-brief and called straight back -- got "what do you need?" on his
        # next call with the finished work filed as already told to him.
        # shield() so the timeout does not cancel speech that is still playing.
        try:
            await asyncio.wait_for(asyncio.shield(handle), timeout=NEWS_GREETING_S)
        except TimeoutError:
            log.warning("news greeting did not finish within %ss", NEWS_GREETING_S)
        except Exception as e:
            log.info("news greeting ended early: %s: %s", type(e).__name__, e)

        spoken = spoken_from(handle)
        if not spoken.delivered:
            # Interrupted or never played. He is on the line and can ask, and he
            # hears it again next call -- better than filing it as heard.
            #
            # And NO offer turn. If he cut in he is talking, and reading a
            # prepared line over him is what the interruption rule forbids; if
            # nothing played, "what ELSE is on today?" is the first thing he
            # hears, with nothing to be "else" than. Both are the one boolean,
            # so there is no second predicate here to drift out of agreement.
            log.info("news greeting not fully delivered (started=%s interrupted=%s "
                     "heard=%d chars); leaving %d task(s) unreported and not "
                     "offering over the top of him",
                     spoken.started, spoken.interrupted, len(spoken.text),
                     len(self._news))
            self.note(f"greeting: news NOT delivered (started={spoken.started} "
                      f"interrupted={spoken.interrupted}); no offer")
            return
        self.note("greeting: news delivered; offering the day")

        # Turn TWO, its own turn. Issued BEFORE awaiting the reported_at POST so
        # the two overlap: generate_reply returns before anything reaches the
        # wire, and _mark_reported is a 5s-budgeted call. Serialised, a wedged
        # supervisor would put five seconds of silence between the news and the
        # offer -- exactly where silence reads as the call having died.
        offer = self.session.generate_reply(instructions=OPENING_INBOUND_OFFER)
        await _mark_reported([n["id"] for n in self._news])

        # Awaited, and deliberately never read back: no spoken_from, no
        # `delivered`, no consequence. Same as OFFER_MORE on the callback path.
        # Measuring it is the bug -- as the last line of the news turn it
        # invited him to answer over the tail, and an interruption there marked
        # work he had heard IN FULL as undelivered. The await buys one log line
        # if it hangs, and keeps on_enter running, which is what keeps
        # EndCallTool(ignore_on_enter=True) covering this turn too.
        try:
            await asyncio.wait_for(asyncio.shield(offer), timeout=NEWS_OFFER_S)
        except TimeoutError:
            log.warning("the offer turn did not finish within %ss", NEWS_OFFER_S)
        except Exception as e:
            log.info("the offer turn ended early: %s: %s", type(e).__name__, e)


@function_tool
@owner_named
async def echo_time(ctx: RunContext) -> str:
    """The current date and time. Use when the owner asks what time or day it is.

    The time you were given at the start of this call is when the call STARTED.
    For the time right now -- and on any call that has been running a while --
    call this.
    """
    return spoken_now(now_local())


class CallbackAgent(ShotAgent):
    """The agent for an outbound callback.

    The whole sequence lives in on_enter, and it has to: GetDtmfTask raises
    "should only be awaited inside tool_functions or the on_enter/on_exit
    methods of an Agent" anywhere else. Awaiting it from the job entrypoint
    failed three times in two milliseconds, so the owner answered, was never asked
    for a code, and the call dropped without a word.

    on_enter is spawned as its own task rather than awaited by session.start(),
    and returning from the job entrypoint does not end the job -- the runner
    blocks until ctx.shutdown(). So running the entire call from here is safe.
    """

    def __init__(self, *, dial: DialRequest, chat_ctx=None,
                 transcript: Transcript | None = None) -> None:
        super().__init__(chat_ctx=chat_ctx, transcript=transcript)
        self._dial = dial
        self.record = CallRecord()
        self.identity = Identity()
        # While this is set the model does not answer on its own. Without it,
        # the owner said his code and the model replied with a summary of its own
        # BEFORE the prepared brief ran, so he heard the first show twice. The
        # callback's scripted turns do all the talking until it hands back.
        self._holding = True
        # on_enter is spawned as its own task, so there is otherwise no way to
        # know the call is over -- the job just sits until the room closes.
        self.finished = asyncio.Event()

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        """Keep the floor until the callback has said its piece.

        Raising StopResponse drops the model's own reply to a turn. That is what
        stops it improvising over a prepared line -- and once the brief is out,
        `_holding` clears and it behaves like the ordinary assistant again,
        tools and all.
        """
        if self._holding:
            log.debug("holding the floor; not replying to this turn")
            raise StopResponse

    async def on_enter(self) -> None:
        io = _LiveIO(self, self._dial)
        try:
            # Open the digit buffer BEFORE dialling. The owner can key his code during
            # the ring or over the greeting; the old collector only started
            # listening once it asked, and threw away everything before that.
            self.identity.listen(get_job_context().room, self.session)
            self.record = await run_callback(
                io, brief=self._dial.brief, require_pin=self._dial.require_pin,
                task_ids=[self._dial.task_id] if self._dial.task_id else [],
                task_titles=self._dial.task_titles)
        except Exception as e:
            # An unhandled failure must still leave a record and still release
            # the supervisor's rendezvous, or place_callback blocks for its full
            # 240s and the line sits open in silence.
            log.exception("callback flow failed")
            self.record.outcome = "error"
            self.record.amd_reason = f"{type(e).__name__}: {e}"[:200]
        finally:
            rec = self.record
            self.note(
                f"outcome={rec.outcome} delivered={rec.delivered} "
                f"heard={rec.heard_chars} chars"
                + (" (interrupted)" if rec.interrupted else "")
                + (" — stayed on the line" if rec.stayed_on_line else " — hung up"))
            await _report_call(self._dial, rec)
            if rec.delivered and rec.task_ids:
                await _mark_reported(rec.task_ids)
            if not rec.stayed_on_line:
                await io.hang_up()
            self.identity.stop()
            self.finished.set()


class _LiveIO:
    """CallbackIO backed by the real session, room and supervisor.

    Every method here is a thin adapter over something that needs a live
    AgentSession. Keeping them thin is the point: the decisions all live in
    callback_flow, where they can be tested without any of this.
    """

    def __init__(self, agent: CallbackAgent, dial: DialRequest) -> None:
        self._agent = agent
        self._dial = dial
        self._hung_up = False
        self._spoken: list[str] = []

    async def dial(self) -> DialOutcome:
        from livekit.agents import get_job_context

        from shot_voice.outbound import dial_and_classify
        outcome = await dial_and_classify(get_job_context(), self._agent.session,
                                          self._dial)
        # Milestones go in the transcript beside the words. Reading "who
        # answered" and "did the code check out" in line with what was said is
        # what turns a transcript into an explanation.
        self._agent.note(f"dialled: {outcome.kind} ({outcome.amd_reason})")
        return outcome

    async def say(self, script: str, *, limit: float,
                  interruptible: bool = True) -> Spoken:
        """Say a prepared line, and report what actually reached him.

        Scripted rather than merely instructed: as a fraction of the full
        session prompt these lines lost to "just talk with the owner" every time, and
        the agent answered his last question instead of reading his results.

        Returns the whole `Spoken`, not just `delivered`. The flow has to tell
        "he interrupted me" from "nothing played" -- collapsing the two is what
        hung up on him mid-brief.
        """
        spoken = await scripted(self._agent, self._agent.session, script,
                                limit=limit, interruptible=interruptible)
        self._spoken.append(spoken.text)
        return spoken

    def code_already_entered(self) -> bool:
        return self._agent.identity.already_verified

    async def wait_for_code(self, limit: float) -> PinResult:
        result = await self._agent.identity.verified(within=limit)
        self._agent.note(f"code: {result.value}")
        return result

    async def hand_back(self, *, delivered: bool) -> None:
        """Stop scripting and let the owner talk, under a prompt that knows where we are.

        `update_instructions` had exactly two call sites, both in scripted.py --
        empty the session prompt, then restore INSTRUCTIONS. So every turn of a
        callback ran under an emptied prompt with a per-response script, and the
        FIRST turn after this was the first generation in the whole call under a
        session prompt at all. That prompt was the INBOUND one, which says
        nothing about having placed the call or having already read him his
        results, so the model did the only thing it supported: it opened the
        conversation. Ninety seconds in, over an explicit dismissal --
        "Hi there. What can I help you with right now?"

        Installed BEFORE `_holding` clears, because in the window between the
        two a user turn would generate under the old prompt, which is the bug.
        """
        try:
            await self._agent.update_instructions(
                _after_prompt(delivered, self._dial.brief))
        except Exception:
            # A failed swap is just the old behaviour, which is survivable. A
            # raise here would skip the line below and leave the floor held
            # forever -- he speaks, gets silence, the line drops. That is a
            # worse failure than a badly-framed prompt.
            log.exception("could not install the post-callback prompt")
        self._agent._holding = False
        log.info("handed the call back to the owner (delivered=%s); staying on the line",
                 delivered)
        asyncio.create_task(self._idle_hangup(delivered=delivered))

    async def _idle_hangup(self, *, delivered: bool = True) -> None:
        """End a line he has stopped using. He usually just hangs up, and the
        room closing ends the job -- this is for when he does not."""
        last = asyncio.get_running_loop().time()

        def _touch(_ev=None) -> None:
            nonlocal last
            last = asyncio.get_running_loop().time()

        self._agent.session.on("user_state_changed", _touch)
        try:
            while True:
                await asyncio.sleep(IDLE_POLL_S)
                if asyncio.get_running_loop().time() - last > IDLE_HANGUP_S:
                    log.info("idle for %ss; saying goodbye and ending the call",
                             IDLE_HANGUP_S)
                    # Never delete the room without a word -- that is how a
                    # callback becomes a mystery on his end, the same reason a
                    # brief that never played says so before it hangs up. If he
                    # cut in and then went quiet he never heard the rest, and
                    # the sign-off has to say so rather than imply he is done.
                    try:
                        await self.say(
                            IDLE_SIGNOFF if delivered else IDLE_SIGNOFF_CUT,
                            limit=SIGNOFF_PLAYOUT_S)
                    except Exception:
                        log.warning("idle sign-off did not play", exc_info=True)
                    await self.hang_up()
                    return
        except asyncio.CancelledError:
            pass
        finally:
            # The room may already be gone -- this task outlives on_enter, and
            # `Agent.session` raises once the activity is torn down. Detaching a
            # listener from a session that no longer exists is a no-op worth
            # swallowing; letting it raise surfaces as an unhandled task
            # exception at interpreter shutdown and reads like a real fault.
            try:
                self._agent.session.off("user_state_changed", _touch)
            except Exception:
                pass

    async def hang_up(self) -> None:
        """Idempotent: the flow hangs up, then on_enter's finally does too."""
        if self._hung_up:
            return
        self._hung_up = True
        from livekit.agents import get_job_context
        try:
            ctx = get_job_context()
            await ctx.delete_room()
            ctx.shutdown(reason=f"callback:{self._agent.record.outcome}")
        except Exception:
            log.warning("could not hang up cleanly", exc_info=True)


async def _report_call(dial: DialRequest, rec: CallRecord) -> None:
    """Tell the supervisor how it went, and leave a durable row behind.

    shot.calls existed with exactly the right columns and was never once
    written, so every failure had to be reconstructed from a log file that is
    deleted on restart.
    """
    import httpx
    from livekit.agents import get_job_context

    from shot_core.budget import SUPERVISOR_HTTP_S
    from shot_core.settings import get_settings
    try:
        ctx = get_job_context()
        async with httpx.AsyncClient(base_url=get_settings().supervisor_base_url,
                                     timeout=SUPERVISOR_HTTP_S) as c:
            r = await c.post("/internal/calls", json={
                "callback_id": dial.callback_id,
                "direction": "outbound",
                "room_name": ctx.room.name,
                "livekit_job_id": ctx.job.id,
                "outcome": rec.outcome,
                "amd_category": rec.amd_category,
                "amd_reason": rec.amd_reason,
                "amd_delay_ms": rec.amd_delay_ms,
                "amd_speech_s": rec.amd_speech_s,
                "sip_status_code": rec.sip_status_code,
                "pin_verified": rec.pin_verified,
                "pin_result": rec.pin_result,
                "delivered": rec.delivered,
                "heard_chars": rec.heard_chars,
                "interrupted": rec.interrupted,
            })
            # Without this a 500 read as success: the row silently failed a NOT
            # NULL constraint while the agent logged "reported call".
            r.raise_for_status()
            if not r.json().get("recorded"):
                log.error("outcome reported but NOT recorded in shot.calls")
        log.info("reported call outcome=%s delivered=%s", rec.outcome, rec.delivered)
    except Exception:
        log.warning("could not report call outcome %s", rec.outcome, exc_info=True)
